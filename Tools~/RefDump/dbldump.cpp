// dbldump.cpp - reference data for tinybvh's (v1.8.0) double-precision BVH (BVH_Double,
// BLASInstanceEx, RayEx; DOUBLE_PRECISION_SUPPORT), for validating the C# port's BvhDouble.
// Same conventions as refdump.cpp: scalar builders, no threading, MSVC.
//
// Usage: dbldump <scene.bin> <out.dbl.ref>
//
// Scene file format (input): as refdump.cpp (int32 triCount, then triCount*3 float4 vertices).
// The float vertices are widened to double exactly ( ( double )x ), which is what the consumer
// must do too; nothing else is derived from the float data.
//
// Output file format (little-endian, no padding):
//   char[8]  magic = "TBVHDBL1"
//   u32      triCount
//   -- BLAS (BVH_Double::Build( verts, triCount )) --
//   u64      usedNodes
//   f64      sahCost                    (BVH_Double::SAHCost())
//   f64[3]   aabbMin, f64[3] aabbMax
//   u64      nodeCount (== usedNodes)
//            nodeCount * 64 bytes: raw BVH_Double::BVHNode (aabbMin xyz f64, aabbMax xyz f64,
//            leftFirst u64, triCount u64). Note that the double builder starts allocating
//            children at node 1: there is no unused node 1 in this layout.
//   u64      idxCount
//            idxCount * u64: bvh.primIdx[]
//   u32      rayCount (65536)
//            per ray: f64[3] O, f64[3] D, f64 t, f64 u, f64 v, u64 prim, u32 occludedFull,
//            u32 occludedHalf (88 bytes). Rays: origin = center + RandomUnitVector() * radius
//            * 1.2, target = RandomPointInAABB( root bounds ), both computed in float with the
//            refdump generator and widened to double, then RayEx( origin, target - origin ),
//            which normalizes D in double and sets rD = 1 / D. The stored O and D are the
//            RayEx members. occludedFull is IsOccluded with a fresh RayEx( O, D ); occludedHalf
//            is IsOccluded with tmax = 0.5 * t of the hit (BVH_DBL_FAR for misses).
//   -- Indexed BLAS (BVH_Double::Build( verts, indices, triCount )) --
//   Welded exactly as refdump.cpp does it (on the float bit patterns; the welded vertices are
//   then widened to double), so the consumer can weld the same way or read the arrays.
//   u32      weldedVertCount
//            weldedVertCount * 24 bytes: the welded vertices as f64[3]
//   u32      indexCount (== triCount * 3)
//            indexCount * u32
//   u64      usedNodes; u64 nodeCount; nodes; u64 idxCount; primIdx   (as the BLAS block)
//   u32      rayCount; 88-byte records                                  (the same rays)
//   -- TLAS (BVH_Double::Build( BLASInstanceEx*, 3, BVH_Double**, 1 )) --
//   The same three instances as refdump.cpp's TLAS, with the transforms in double:
//   [0] identity, mask 0xFFFF; [1] translated by extent.x * 1.1 along x, mask 0xFFFF;
//   [2] rotated/scaled (see the code) and translated by extent.z * 1.1 along z, mask 0x0002.
//   u32      instCount (3)
//            per instance: f64[16] transform, f64[16] invTransform, f64[3] aabbMin,
//            f64[3] aabbMax, u64 blasIdx, u64 mask     (after the build updated them)
//   u64      tlasUsedNodes
//   f64[3]   tlasAabbMin, f64[3] tlasAabbMax
//   u64      tlasNodeCount; nodes; u64 tlasIdxCount; primIdx (u64)
//   u32      tlasRayCount (65536)
//            per ray: f64[3] O, f64[3] D, f64 t, f64 u, f64 v, u64 prim, u64 inst,
//            u32 occludedFull (96 bytes). Rays generated over the TLAS bounds the same way.
//
// Determinism: TINYBVH_NO_SIMD, NO_THREADED_BUILDS, useSIMDifavailable = false, null
// spawn/barrier/parallel_for, as in refdump.cpp.

#define TINYBVH_NO_SIMD
#define NO_THREADED_BUILDS
#define DOUBLE_PRECISION_SUPPORT
#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>
#include <array>
#include <map>

using namespace tinybvh;

static const uint32_t N_RAYS = 65536;
static const float PI = 3.14159265358979323846f;

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

struct GenRay { bvhdbl3 O, D; };

// Same generator as refdump (float), widened to double before the RayEx constructor.
static std::vector<GenRay> GenerateRays( const bvhdbl3& aabbMinD, const bvhdbl3& aabbMaxD, uint32_t count )
{
	const bvhvec3 aabbMin( (float)aabbMinD.x, (float)aabbMinD.y, (float)aabbMinD.z );
	const bvhvec3 aabbMax( (float)aabbMaxD.x, (float)aabbMaxD.y, (float)aabbMaxD.z );
	const bvhvec3 center = (aabbMin + aabbMax) * 0.5f;
	const float radius = tinybvh_length( aabbMax - aabbMin ) * 0.5f;
	std::vector<GenRay> rays( count );
	for (uint32_t i = 0; i < count; i++)
	{
		bvhvec3 origin = center + RandomUnitVector() * radius * 1.2f;
		bvhvec3 target = RandomPointInAABB( aabbMin, aabbMax );
		bvhdbl3 O( origin ), T( target );
		RayEx ray( O, T - O );
		rays[i].O = ray.O, rays[i].D = ray.D;
	}
	return rays;
}

struct Hit { double t, u, v; uint64_t prim, inst; uint32_t occludedFull, occludedHalf; };

static void TraceRays( BVH_Double& bvh, const std::vector<GenRay>& rays, std::vector<Hit>& out )
{
	out.resize( rays.size() );
	for (size_t i = 0; i < rays.size(); i++)
	{
		const GenRay& gr = rays[i];
		RayEx ray( gr.O, gr.D );
		bvh.Intersect( ray );
		Hit& h = out[i];
		h.t = ray.hit.t, h.u = ray.hit.u, h.v = ray.hit.v, h.prim = ray.hit.prim, h.inst = ray.hit.inst;
		RayEx occRay( gr.O, gr.D );
		h.occludedFull = bvh.IsOccluded( occRay ) ? 1u : 0u;
		double halfT = ray.hit.t < BVH_DBL_FAR ? 0.5 * ray.hit.t : BVH_DBL_FAR;
		RayEx occHalfRay( gr.O, gr.D, halfT );
		h.occludedHalf = bvh.IsOccluded( occHalfRay ) ? 1u : 0u;
	}
}

static void WriteU32( std::ofstream& f, uint32_t v ) { f.write( (const char*)&v, 4 ); }
static void WriteU64( std::ofstream& f, uint64_t v ) { f.write( (const char*)&v, 8 ); }
static void WriteF64( std::ofstream& f, double v ) { f.write( (const char*)&v, 8 ); }
static void WriteDbl3( std::ofstream& f, const bvhdbl3& v ) { WriteF64( f, v.x ); WriteF64( f, v.y ); WriteF64( f, v.z ); }

static void WriteTree( std::ofstream& f, const BVH_Double& bvh )
{
	WriteU64( f, bvh.usedNodes );
	f.write( (const char*)bvh.bvhNode, (size_t)bvh.usedNodes * sizeof( BVH_Double::BVHNode ) );
	WriteU64( f, bvh.idxCount );
	f.write( (const char*)bvh.primIdx, (size_t)bvh.idxCount * sizeof( uint64_t ) );
}

static void WriteBlasRays( std::ofstream& f, const std::vector<GenRay>& rays, const std::vector<Hit>& hits, const char* label )
{
	WriteU32( f, (uint32_t)rays.size() );
	uint32_t hitCount = 0;
	for (size_t i = 0; i < rays.size(); i++)
	{
		WriteDbl3( f, rays[i].O ); WriteDbl3( f, rays[i].D );
		const Hit& h = hits[i];
		WriteF64( f, h.t ); WriteF64( f, h.u ); WriteF64( f, h.v );
		WriteU64( f, h.prim ); WriteU32( f, h.occludedFull ); WriteU32( f, h.occludedHalf );
		if (h.t < BVH_DBL_FAR) hitCount++;
	}
	printf( "  %s hit ratio: %.2f%% (%u/%u)\n", label, 100.0 * hitCount / rays.size(), hitCount, (uint32_t)rays.size() );
}

static void Configure( BVH_Double& bvh )
{
	bvh.settings.useSIMDifavailable = false;
	bvh.context.spawn = nullptr, bvh.context.barrier = nullptr, bvh.context.parallel_for = nullptr;
}

int main( int argc, char** argv )
{
	static_assert (sizeof( BVH_Double::BVHNode ) == 64, "BVH_Double::BVHNode must be 64 bytes");
	static_assert (sizeof( bvhdbl3 ) == 24, "bvhdbl3 must be 24 bytes");
	if (argc != 3)
	{
		fprintf( stderr, "usage: dbldump <scene.bin> <out.dbl.ref>\n" );
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
	std::vector<bvhvec4> verts( vertCount );
	sf.read( (char*)verts.data(), (size_t)vertCount * sizeof( bvhvec4 ) );
	sf.close();

	// widen to double
	bvhdbl3* dverts = (bvhdbl3*)malloc64( vertCount * sizeof( bvhdbl3 ) );
	for (uint32_t i = 0; i < vertCount; i++) dverts[i] = bvhdbl3( (double)verts[i].x, (double)verts[i].y, (double)verts[i].z );

	std::ofstream f( argv[2], std::ios::binary );
	if (!f)
	{
		fprintf( stderr, "cannot open output file: %s\n", argv[2] );
		return 1;
	}
	f.write( "TBVHDBL1", 8 );
	WriteU32( f, (uint32_t)triCount );
	printf( "scene: %s (%d tris)\n", argv[1], triCount );

	// BLAS
	BVH_Double bvh;
	Configure( bvh );
	bvh.Build( dverts, (uint64_t)triCount );
	std::vector<GenRay> rays = GenerateRays( bvh.aabbMin, bvh.aabbMax, N_RAYS );
	std::vector<Hit> hits;
	TraceRays( bvh, rays, hits );
	WriteU64( f, bvh.usedNodes );
	WriteF64( f, bvh.SAHCost() );
	WriteDbl3( f, bvh.aabbMin ); WriteDbl3( f, bvh.aabbMax );
	WriteTree( f, bvh );
	printf( "  BLAS: nodes %llu, SAH %f\n", (unsigned long long)bvh.usedNodes, bvh.SAHCost() );
	WriteBlasRays( f, rays, hits, "BLAS" );

	// Indexed BLAS
	std::vector<bvhvec4> welded;
	std::vector<uint32_t> indices( vertCount );
	welded.reserve( vertCount );
	{
		std::map<std::array<uint32_t, 3>, uint32_t> seen;
		for (uint32_t i = 0; i < vertCount; i++)
		{
			std::array<uint32_t, 3> key;
			memcpy( key.data(), &verts[i].x, 3 * sizeof( uint32_t ) );
			std::map<std::array<uint32_t, 3>, uint32_t>::const_iterator it = seen.find( key );
			if (it == seen.end())
			{
				const uint32_t slot = (uint32_t)welded.size();
				seen.insert( std::make_pair( key, slot ) );
				welded.push_back( verts[i] );
				indices[i] = slot;
			}
			else indices[i] = it->second;
		}
	}
	const uint32_t weldedCount = (uint32_t)welded.size();
	bvhdbl3* dwelded = (bvhdbl3*)malloc64( weldedCount * sizeof( bvhdbl3 ) );
	for (uint32_t i = 0; i < weldedCount; i++) dwelded[i] = bvhdbl3( (double)welded[i].x, (double)welded[i].y, (double)welded[i].z );
	BVH_Double ibvh;
	Configure( ibvh );
	ibvh.Build( dwelded, indices.data(), (uint64_t)triCount );
	std::vector<Hit> ihits;
	TraceRays( ibvh, rays, ihits );
	WriteU32( f, weldedCount );
	f.write( (const char*)dwelded, (size_t)weldedCount * sizeof( bvhdbl3 ) );
	WriteU32( f, (uint32_t)indices.size() );
	f.write( (const char*)indices.data(), indices.size() * sizeof( uint32_t ) );
	WriteU64( f, ibvh.usedNodes );
	WriteTree( f, ibvh );
	printf( "  indexed: welded %u verts, nodes %llu\n", weldedCount, (unsigned long long)ibvh.usedNodes );
	WriteBlasRays( f, rays, ihits, "indexed" );

	// TLAS
	const bvhdbl3 ext = bvh.aabbMax - bvh.aabbMin;
	BLASInstanceEx inst[3] = { BLASInstanceEx( 0 ), BLASInstanceEx( 0 ), BLASInstanceEx( 0 ) };
	inst[0].mask = 0xFFFF;
	inst[1].transform[3] = ext.x * 1.1;
	inst[1].mask = 0xFFFF;
	inst[2].transform[0] = 0, inst[2].transform[1] = 0, inst[2].transform[2] = 0.5, inst[2].transform[3] = 0;
	inst[2].transform[4] = 0, inst[2].transform[5] = 0.5, inst[2].transform[6] = 0, inst[2].transform[7] = 0;
	inst[2].transform[8] = -0.5, inst[2].transform[9] = 0, inst[2].transform[10] = 0, inst[2].transform[11] = ext.z * 1.1;
	inst[2].mask = 0x0002;
	BVH_Double tlas;
	Configure( tlas );
	BVH_Double* blasList[1] = { &bvh };
	tlas.Build( inst, 3, blasList, 1 );
	std::vector<GenRay> tlasRays = GenerateRays( tlas.aabbMin, tlas.aabbMax, N_RAYS );
	WriteU32( f, 3 );
	for (int i = 0; i < 3; i++)
	{
		f.write( (const char*)inst[i].transform, 16 * sizeof( double ) );
		f.write( (const char*)inst[i].invTransform, 16 * sizeof( double ) );
		WriteDbl3( f, inst[i].aabbMin ); WriteDbl3( f, inst[i].aabbMax );
		WriteU64( f, inst[i].blasIdx ); WriteU64( f, inst[i].mask );
	}
	WriteU64( f, tlas.usedNodes );
	WriteDbl3( f, tlas.aabbMin ); WriteDbl3( f, tlas.aabbMax );
	WriteTree( f, tlas );
	WriteU32( f, N_RAYS );
	uint32_t tlasHitCount = 0;
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		const GenRay& gr = tlasRays[i];
		RayEx ray( gr.O, gr.D );
		tlas.Intersect( ray );
		RayEx occRay( gr.O, gr.D );
		const uint32_t occluded = tlas.IsOccluded( occRay ) ? 1u : 0u;
		WriteDbl3( f, gr.O ); WriteDbl3( f, gr.D );
		WriteF64( f, ray.hit.t ); WriteF64( f, ray.hit.u ); WriteF64( f, ray.hit.v );
		WriteU64( f, ray.hit.prim ); WriteU64( f, ray.hit.inst ); WriteU32( f, occluded );
		if (ray.hit.t < BVH_DBL_FAR) tlasHitCount++;
	}
	printf( "  TLAS: nodes %llu, hit ratio %.2f%% (%u/%u)\n", (unsigned long long)tlas.usedNodes, 100.0 * tlasHitCount / N_RAYS, tlasHitCount, N_RAYS );
	f.close();
	printf( "  wrote: %s\n", argv[2] );
	free64( dverts );
	free64( dwelded );
	return 0;
}
