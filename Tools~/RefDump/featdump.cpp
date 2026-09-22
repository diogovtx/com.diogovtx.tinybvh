// featdump.cpp - reference data for the remaining tinybvh (v1.8.0) features ported after
// refdump.cpp's format was frozen: the alternative builders, presplitting, the SBVH bin
// settings, the EPO cost metric, sphere queries, 256-ray packet traversal and the stochastic
// optimizer. Same conventions as refdump.cpp: scalar builders, no threading, MSVC.
//
// Usage: featdump <scene.bin> <out.feat.ref>
//
// Scene file format (input): as refdump.cpp (int32 triCount, then triCount*3 float4 vertices).
//
// Output file format (little-endian, no padding):
//   char[8]  magic = "TBVHFEA1"
//   u32      triCount
//
//   -- Tree blocks --
//   Every tree block below has the same layout:
//   u32      usedNodes
//   f32      sahCost                    (BVH::SAHCost())
//   f32[3]   aabbMin, f32[3] aabbMax    (root bounds)
//   u32      nodeCount (== usedNodes)
//            nodeCount * 32 bytes: raw BVH::BVHNode (aabbMin xyz, leftFirst, aabbMax xyz, triCount)
//   u32      idxCount                   (see each block for what is written)
//            idxCount * u32: bvh.primIdx[]
//   u32      rayCount (65536)
//            per ray: f32[3] O, f32[3] D, f32 t, f32 u, f32 v, u32 prim, u32 occludedFull,
//            u32 occludedHalf (the 48-byte record of refdump.cpp). The rays are the refdump
//            BLAS rays: same generator, same seed, same root bounds, so they are identical to
//            the ones in <scene>.ref, and every block traces the same 65536 rays.
//
//   [1] "quick":      BVH::BuildQuick( verts, triCount ). Mid-point splits, no SAH.
//                     idxCount = bvh.idxCount (== triCount).
//   [2] "fullsweep":  settings.useFullSweep = true, BVH::Build. idxCount = bvh.idxCount.
//   [3] "presplit":   settings.usePresplitting = true (presplitFactor 0.3, presplitPostPass
//                     true), BVH::Build (binned). After Presplit() the C++ sets
//                     triCount = idxCount = fragment count, which is > the input triangle
//                     count; idxCount here is that bvh.idxCount and the whole primIdx array
//                     is written. PresplitPostPass shrinks leaf counts in place without
//                     touching the entries it drops, so every entry is deterministic.
//                     The tree is flagged not refittable.
//   [4] "presplit+fullsweep": usePresplitting and useFullSweep. idxCount = bvh.idxCount.
//   [5] "presplit-nopostpass": usePresplitting with presplitPostPass = false, binned.
//   [6] "hqbins":     SBVH (settings.useSpatialSplits) with hqbvhbins = 32 and
//                     hqbvhoddeven = true (odd levels use 33 bins). idxCount = PrimCount(),
//                     for the reason given in refdump.cpp's SBVH section.
//   [7] "stochastic": a fresh binned BVH::Build followed by
//                     BVH::Optimize( iterations, false, true ), i.e. the stochastic variant of
//                     the tree-rotation optimizer. It is the only tinybvh code that calls the
//                     C runtime's rand(); this tool never calls srand() or rand() itself, so
//                     the sequence is MSVC's rand() from its initial state: a per-thread
//                     holdrand starting at 1, holdrand = holdrand * 214013 + 2531011,
//                     result = ( holdrand >> 16 ) & 0x7fff, RAND_MAX = 32767. This block is
//                     preceded by:
//   u32      iterations (25)
//   f32      sahCostBefore
//                     and its idxCount is PrimCount() (see refdump.cpp's optimizer note).
//
//   -- Cost metrics --
//   f32      sahBinned                  (BVH::SAHCost of a fresh binned build)
//   u32      hasEpo                     (1 when the EPO costs below were computed; EPOCost
//                                        is O( N * overlap ) and is skipped above 100k tris)
//   f32      epoBinned                  (BVH::EPOCost of the binned build; 0 when !hasEpo)
//   f32      epoQuick                   (BVH::EPOCost of the BuildQuick tree; 0 when !hasEpo)
//
//   -- Sphere queries (BVH::IntersectSphere on the binned build) --
//   u32      sphereCount (4096)
//            per query: f32[3] pos, f32 r, u32 hit (0/1)
//            pos is uniform in the root AABB grown by 10% on each side; r = R()*R()*0.1*maxExtent.
//
//   -- Packet traversal (BVH::Intersect256Rays on the binned build) --
//   u32      packetCount (64)
//            per packet:
//            f32[3]   O            (shared by all 256 rays)
//            256 x    f32[3] D, f32 t, f32 u, f32 v, u32 prim     (24 bytes per ray)
//            Ray i of a packet is pixel ( px, py ) of a 16x16 tile with px = x*4+u,
//            py = y*4+v and i = ( ( y*4 + x )*4 + v )*4 + u, the order tinybvh's test app
//            uses, which puts the tile corners at rays 0, 51, 204 and 255 as the traversal
//            requires. The tile is a pinhole frustum: fwd points from O at a random point in
//            the AABB, right = normalize( cross( upHint, fwd ) ), up = cross( fwd, right ),
//            D = normalize( fwd + right*sx*s + up*sy*s ) with sx = ( px + 0.5 )/16*2 - 1,
//            sy = 1 - ( py + 0.5 )/16*2 and s = tan( half fov ), half fov in [1, 12] degrees.
//            The stored D is the Ray's normalized D. Hits are ray.hit after Intersect256Rays
//            with all 256 rays starting at t = BVH_FAR; no occlusion queries.
//
// Determinism: TINYBVH_NO_SIMD and NO_THREADED_BUILDS, exactly as refdump.cpp, and every
// BVH gets settings.useSIMDifavailable = false and a null spawn/barrier/parallel_for.

#define TINYBVH_NO_SIMD
#define NO_THREADED_BUILDS
#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <fstream>

using namespace tinybvh;

static const uint32_t N_RAYS = 65536;
static const uint32_t N_SPHERES = 4096;
static const uint32_t N_PACKETS = 64;
static const uint32_t OPT_ITERATIONS = 25;
static const uint32_t EPO_MAX_TRIS = 100000;
static const float PI = 3.14159265358979323846f;

// Fixed LCG, identical to refdump.cpp - never rand(), which the stochastic optimizer owns.
static uint32_t s = 0x12345678;
static float R()
{
	s = s * 1664525u + 1013904223u;
	return (s >> 8) * (1.0f / 16777216.0f);
}

static bvhvec3 RandomUnitVector()
{
	float z = 2.0f * R() - 1.0f;
	float phi = 2.0f * PI * R();
	float r = sqrtf( 1.0f - z * z );
	return bvhvec3( r * cosf( phi ), r * sinf( phi ), z );
}

static bvhvec3 RandomPointInAABB( const bvhvec3& aabbMin, const bvhvec3& aabbMax )
{
	return bvhvec3(
		aabbMin.x + R() * (aabbMax.x - aabbMin.x),
		aabbMin.y + R() * (aabbMax.y - aabbMin.y),
		aabbMin.z + R() * (aabbMax.z - aabbMin.z) );
}

struct GenRay { bvhvec3 O, D; };

static std::vector<GenRay> GenerateRays( const bvhvec3& aabbMin, const bvhvec3& aabbMax, uint32_t count )
{
	const bvhvec3 center = (aabbMin + aabbMax) * 0.5f;
	const float radius = tinybvh_length( aabbMax - aabbMin ) * 0.5f;
	std::vector<GenRay> rays( count );
	for (uint32_t i = 0; i < count; i++)
	{
		bvhvec3 origin = center + RandomUnitVector() * radius * 1.2f;
		bvhvec3 target = RandomPointInAABB( aabbMin, aabbMax );
		Ray ray( origin, target - origin );
		rays[i].O = ray.O, rays[i].D = ray.D;
	}
	return rays;
}

struct BlasHit { float t, u, v; uint32_t prim, occludedFull, occludedHalf; };

static void TraceBlasRays( BVH& bvh, const std::vector<GenRay>& rays, std::vector<BlasHit>& out )
{
	out.resize( rays.size() );
	for (size_t i = 0; i < rays.size(); i++)
	{
		const GenRay& gr = rays[i];
		Ray ray( gr.O, gr.D );
		bvh.Intersect( ray );
		BlasHit& h = out[i];
		h.t = ray.hit.t, h.u = ray.hit.u, h.v = ray.hit.v, h.prim = ray.hit.prim;
		Ray occRay( gr.O, gr.D );
		h.occludedFull = bvh.IsOccluded( occRay ) ? 1u : 0u;
		float halfT = ray.hit.t < BVH_FAR ? 0.5f * ray.hit.t : BVH_FAR;
		Ray occHalfRay( gr.O, gr.D, halfT );
		h.occludedHalf = bvh.IsOccluded( occHalfRay ) ? 1u : 0u;
	}
}

static void WriteU32( std::ofstream& f, uint32_t v ) { f.write( (const char*)&v, 4 ); }
static void WriteF32( std::ofstream& f, float v ) { f.write( (const char*)&v, 4 ); }
static void WriteVec3( std::ofstream& f, const bvhvec3& v ) { WriteF32( f, v.x ); WriteF32( f, v.y ); WriteF32( f, v.z ); }

static void WriteRayRecords( std::ofstream& f, const std::vector<GenRay>& rays, const std::vector<BlasHit>& hits )
{
	WriteU32( f, (uint32_t)rays.size() );
	for (size_t i = 0; i < rays.size(); i++)
	{
		WriteVec3( f, rays[i].O ); WriteVec3( f, rays[i].D );
		const BlasHit& h = hits[i];
		WriteF32( f, h.t ); WriteF32( f, h.u ); WriteF32( f, h.v );
		WriteU32( f, h.prim ); WriteU32( f, h.occludedFull ); WriteU32( f, h.occludedHalf );
	}
}

// One tree block: nodes, the first idxCount primitive indices, and the traced rays.
static void WriteTreeBlock( std::ofstream& f, BVH& bvh, uint32_t idxCount, const std::vector<GenRay>& rays )
{
	std::vector<BlasHit> hits;
	TraceBlasRays( bvh, rays, hits );
	WriteU32( f, bvh.usedNodes );
	WriteF32( f, bvh.SAHCost() );
	WriteVec3( f, bvh.aabbMin ); WriteVec3( f, bvh.aabbMax );
	WriteU32( f, bvh.usedNodes );
	f.write( (const char*)bvh.bvhNode, (size_t)bvh.usedNodes * sizeof( BVH::BVHNode ) );
	WriteU32( f, idxCount );
	f.write( (const char*)bvh.primIdx, (size_t)idxCount * sizeof( uint32_t ) );
	WriteRayRecords( f, rays, hits );
	uint32_t hitCount = 0;
	for (size_t i = 0; i < hits.size(); i++) if (hits[i].t < BVH_FAR) hitCount++;
	printf( "    nodes %u, idx %u, SAH %f, hit ratio %.2f%% (%u/%u)\n",
		bvh.usedNodes, idxCount, bvh.SAHCost(), 100.0 * hitCount / hits.size(), hitCount, (uint32_t)hits.size() );
}

static void Configure( BVH& bvh )
{
	bvh.settings.useSIMDifavailable = false;
	bvh.context.spawn = nullptr, bvh.context.barrier = nullptr, bvh.context.parallel_for = nullptr;
}

int main( int argc, char** argv )
{
	static_assert (sizeof( BVH::BVHNode ) == 32, "BVH::BVHNode must be 32 bytes");
	if (argc != 3)
	{
		fprintf( stderr, "usage: featdump <scene.bin> <out.feat.ref>\n" );
		return 1;
	}
	std::ifstream sf( argv[1], std::ios::binary );
	if (!sf)
	{
		fprintf( stderr, "cannot open scene file: %s\n", argv[1] );
		return 1;
	}
	int32_t triCount = 0;
	sf.read( (char*)&triCount, 4 );
	const uint32_t vertCount = (uint32_t)triCount * 3;
	bvhvec4* verts = (bvhvec4*)malloc64( vertCount * sizeof( bvhvec4 ) );
	sf.read( (char*)verts, (size_t)vertCount * sizeof( bvhvec4 ) );
	sf.close();

	std::ofstream f( argv[2], std::ios::binary );
	if (!f)
	{
		fprintf( stderr, "cannot open output file: %s\n", argv[2] );
		return 1;
	}
	f.write( "TBVHFEA1", 8 );
	WriteU32( f, (uint32_t)triCount );
	printf( "scene: %s (%d tris)\n", argv[1], triCount );

	// The binned reference build: its root bounds seed the ray generator, exactly as in refdump.
	BVH binned;
	Configure( binned );
	binned.Build( verts, (uint32_t)triCount );
	std::vector<GenRay> rays = GenerateRays( binned.aabbMin, binned.aabbMax, N_RAYS );

	printf( "  [1] BuildQuick\n" );
	BVH quick;
	Configure( quick );
	quick.BuildQuick( verts, (uint32_t)triCount );
	WriteTreeBlock( f, quick, quick.idxCount, rays );

	printf( "  [2] full sweep\n" );
	BVH sweep;
	Configure( sweep );
	sweep.settings.useFullSweep = true;
	sweep.Build( verts, (uint32_t)triCount );
	WriteTreeBlock( f, sweep, sweep.idxCount, rays );

	printf( "  [3] presplit (binned)\n" );
	BVH presplit;
	Configure( presplit );
	presplit.settings.usePresplitting = true;
	presplit.Build( verts, (uint32_t)triCount );
	if (presplit.refittable) { fprintf( stderr, "FATAL: presplit tree flagged refittable.\n" ); return 1; }
	WriteTreeBlock( f, presplit, presplit.idxCount, rays );

	printf( "  [4] presplit + full sweep\n" );
	BVH presweep;
	Configure( presweep );
	presweep.settings.usePresplitting = true;
	presweep.settings.useFullSweep = true;
	presweep.Build( verts, (uint32_t)triCount );
	WriteTreeBlock( f, presweep, presweep.idxCount, rays );

	printf( "  [5] presplit without post pass\n" );
	BVH prenopost;
	Configure( prenopost );
	prenopost.settings.usePresplitting = true;
	prenopost.settings.presplitPostPass = false;
	prenopost.Build( verts, (uint32_t)triCount );
	WriteTreeBlock( f, prenopost, prenopost.idxCount, rays );

	printf( "  [6] SBVH, 32 bins, odd/even\n" );
	BVH hq;
	Configure( hq );
	hq.hqbvhbins = 32;
	hq.hqbvhoddeven = true;
	hq.BuildHQ( verts, (uint32_t)triCount );
	WriteTreeBlock( f, hq, (uint32_t)hq.PrimCount(), rays );

	printf( "  [7] stochastic optimizer, %u iterations\n", OPT_ITERATIONS );
	BVH sto;
	Configure( sto );
	sto.Build( verts, (uint32_t)triCount );
	const float stoBefore = sto.SAHCost();
	clock_t stoStart = clock();
	sto.Optimize( OPT_ITERATIONS, false, true );
	printf( "    SAH %f -> %f, %.1fs\n", stoBefore, sto.SAHCost(), (double)(clock() - stoStart) / CLOCKS_PER_SEC );
	WriteU32( f, OPT_ITERATIONS );
	WriteF32( f, stoBefore );
	WriteTreeBlock( f, sto, (uint32_t)sto.PrimCount(), rays );

	// Cost metrics.
	const float sahBinned = binned.SAHCost();
	const uint32_t hasEpo = (uint32_t)triCount <= EPO_MAX_TRIS ? 1u : 0u;
	float epoBinned = 0, epoQuick = 0;
	if (hasEpo)
	{
		clock_t epoStart = clock();
		epoBinned = binned.EPOCost();
		epoQuick = quick.EPOCost();
		printf( "  EPO: binned %f, quick %f (SAH binned %f), %.1fs\n", epoBinned, epoQuick, sahBinned, (double)(clock() - epoStart) / CLOCKS_PER_SEC );
	}
	else printf( "  EPO: skipped (%d tris > %u)\n", triCount, EPO_MAX_TRIS );
	WriteF32( f, sahBinned );
	WriteU32( f, hasEpo );
	WriteF32( f, epoBinned );
	WriteF32( f, epoQuick );

	// Sphere queries.
	const bvhvec3 ext = binned.aabbMax - binned.aabbMin;
	const float maxExtent = tinybvh_max( tinybvh_max( ext.x, ext.y ), ext.z );
	const bvhvec3 grownMin = binned.aabbMin - ext * 0.1f, grownMax = binned.aabbMax + ext * 0.1f;
	WriteU32( f, N_SPHERES );
	uint32_t sphereHits = 0;
	for (uint32_t i = 0; i < N_SPHERES; i++)
	{
		const bvhvec3 pos = RandomPointInAABB( grownMin, grownMax );
		const float r = R() * R() * 0.1f * maxExtent;
		const uint32_t hit = binned.IntersectSphere( pos, r ) ? 1u : 0u;
		sphereHits += hit;
		WriteVec3( f, pos ); WriteF32( f, r ); WriteU32( f, hit );
	}
	printf( "  spheres: %u/%u overlap\n", sphereHits, N_SPHERES );

	// Packets.
	WriteU32( f, N_PACKETS );
	const bvhvec3 center = (binned.aabbMin + binned.aabbMax) * 0.5f;
	const float radius = tinybvh_length( ext ) * 0.5f;
	uint32_t packetHits = 0;
	ALIGNED( 64 ) Ray packet[256];
	for (uint32_t p = 0; p < N_PACKETS; p++)
	{
		const bvhvec3 O = center + RandomUnitVector() * radius * 1.2f;
		const bvhvec3 target = RandomPointInAABB( binned.aabbMin, binned.aabbMax );
		const bvhvec3 fwd = tinybvh_normalize( target - O );
		const bvhvec3 upHint = fabsf( fwd.y ) < 0.9f ? bvhvec3( 0, 1, 0 ) : bvhvec3( 1, 0, 0 );
		const bvhvec3 right = tinybvh_normalize( tinybvh_cross( upHint, fwd ) );
		const bvhvec3 up = tinybvh_cross( fwd, right );
		const float halfFov = (1.0f + R() * 11.0f) * PI / 180.0f;
		const float sc = tanf( halfFov );
		for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) for (int v = 0; v < 4; v++) for (int u = 0; u < 4; u++)
		{
			const int px = x * 4 + u, py = y * 4 + v, i = ((y * 4 + x) * 4 + v) * 4 + u;
			const float sx = (px + 0.5f) / 16.0f * 2.0f - 1.0f, sy = 1.0f - (py + 0.5f) / 16.0f * 2.0f;
			packet[i] = Ray( O, fwd + right * (sx * sc) + up * (sy * sc) );
		}
		binned.Intersect256Rays( packet );
		WriteVec3( f, O );
		for (int i = 0; i < 256; i++)
		{
			WriteVec3( f, packet[i].D );
			WriteF32( f, packet[i].hit.t ); WriteF32( f, packet[i].hit.u ); WriteF32( f, packet[i].hit.v );
			WriteU32( f, packet[i].hit.prim );
			if (packet[i].hit.t < BVH_FAR) packetHits++;
		}
	}
	printf( "  packets: %u/%u rays hit\n", packetHits, N_PACKETS * 256 );
	f.close();
	printf( "  wrote: %s\n", argv[2] );
	free64( verts );
	return 0;
}
