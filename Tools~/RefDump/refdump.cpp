// refdump.cpp - generates deterministic reference data from tinybvh (v1.8.0)
// for validating the tinybvhcs-unity C# port.
//
// Usage: refdump <scene.bin> <out.ref>
//
// Scene file format (input):
//   int32 triCount
//   triCount*3 vertices, each a 16-byte float4 (x,y,z,w); three consecutive
//   vertices form a triangle.
//
// Output file format (little-endian, no padding):
//   char[8]  magic = "TBVHREF5"
//   u32      triCount
//   -- BLAS section --
//   u32      usedNodes
//   f32      sahCost
//   f32[3]   aabbMin, f32[3] aabbMax            (root bounds)
//   u32      nodeCount (== usedNodes)
//            nodeCount * 32 bytes: raw BVH::BVHNode (aabbMin xyz, leftFirst, aabbMax xyz, triCount)
//   u32      idxCount
//            idxCount * u32: bvh.primIdx[]
//   -- BLAS rays --
//   u32      rayCount (65536)
//            per ray: f32[3] O, f32[3] D, f32 t, f32 u, f32 v, u32 prim, u32 occludedFull, u32 occludedHalf
//            (9 floats + 3 u32 = 48 bytes per ray)
//   -- Refit section --
//   f32[3]   aabbMin, f32[3] aabbMax            (root bounds after refit)
//            per ray (same count/order as BLAS rays): f32 t, f32 u, f32 v, u32 prim   (16 bytes)
//   -- TLAS section --
//   u32      instCount (3)
//            per instance: f32[16] transform (cell[0..15] row-major), f32[16] invTransform,
//                           f32[3] aabbMin, f32[3] aabbMax, u32 mask
//   u32      tlasUsedNodes
//   f32[3]   tlasAabbMin, f32[3] tlasAabbMax
//   u32      tlasNodeCount; tlasNodeCount * 32 bytes raw nodes
//   u32      tlasIdxCount;  tlasIdxCount * u32 primIdx
//   u32      tlasRayCount (65536)
//            per ray: f32[3] O, f32[3] D, f32 t, f32 u, f32 v, u32 prim, u32 inst, u32 occludedFull
//            (9 floats + 3 u32 = 48 bytes per ray)
//   -- SBVH section (added in TBVHREF2) --
//   u32      sbvhUsedNodes
//   f32      sbvhSahCost
//   f32[3]   sbvhAabbMin, f32[3] sbvhAabbMax     (root bounds)
//   u32      sbvhNodeCount (== sbvhUsedNodes)
//            sbvhNodeCount * 32 bytes: raw BVH::BVHNode, same layout as the BLAS nodes
//   u32      sbvhIdxCount
//            sbvhIdxCount * u32: sbvh.primIdx[]
//            NOTE: this is PrimCount(), not bvh.idxCount. BuildHQ ends with Compact(), which
//            rewrites primIdx so that leaf ranges are contiguous from 0 but leaves idxCount at
//            triCount + slack; only the first PrimCount() entries were written, the tail of the
//            freshly allocated array is uninitialized and is therefore not dumped.
//   u32      sbvhRayCount (65536)
//            the same rays as the BLAS ray section, in the same order, traced against the SBVH.
//            per ray: f32[3] O, f32[3] D, f32 t, f32 u, f32 v, u32 prim, u32 occludedFull,
//            u32 occludedHalf (identical 48-byte record to the BLAS rays)
//   -- Optimizer sections (added in TBVHREF3) --
//   Two sections with an identical layout, written back to back:
//     [1] "plain": a fresh binned BVH (BVH::Build) with BVH::Optimize( iterations, false, false )
//         called on it. That is exactly the C++ one-liner: ConvertFrom to BVH_Verbose, optimize,
//         ConvertFrom back with compact = true. No leaf reshaping at all.
//     [2] "split/merge": a fresh binned BVH converted to BVH_Verbose by hand, then
//         SplitLeafs( 1 ), Optimize( iterations, false, false ), MergeLeafs(), and finally
//         BVH::ConvertFrom( verbose ) (compact = true). This is the pipeline tinybvh recommends
//         for the best result; BVH::Optimize does not do it itself in v1.8.0.
//     Both use extreme = false and stochastic = false, i.e. the fully deterministic path.
//   Per section:
//   u32      iterations         (the value passed to Optimize; may be < 25 for large scenes)
//   u32      usedNodes          (bvh.usedNodes after converting back)
//   f32      sahCost            (BVH::SAHCost of the optimized tree)
//   f32      sahCostBefore      (BVH::SAHCost of the same tree before optimizing)
//   f32[3]   aabbMin, f32[3] aabbMax                (root bounds)
//   u32      nodeCount (== usedNodes)
//            nodeCount * 32 bytes: raw BVH::BVHNode, same layout as the BLAS nodes
//   u32      primCount
//            primCount * u32: bvh.primIdx[]
//            NOTE: this is PrimCount(), not bvh.idxCount. BVH::ConvertFrom( BVH_Verbose& ) copies
//            the index pointer through unchanged, and MergeLeafs replaces it with a fresh array of
//            idxCount entries of which only the first PrimCount() were written; the tail is
//            uninitialized and is therefore not dumped. For a binned build PrimCount() == triCount.
//   u32      rayCount (65536)
//            the same rays as the BLAS ray section, in the same order, traced against the
//            optimized tree. Per ray: f32[3] O, f32[3] D, f32 t, f32 u, f32 v, u32 prim,
//            u32 occludedFull, u32 occludedHalf (identical 48-byte record to the BLAS rays)
//   -- Indexed-geometry sections (added in TBVHREF4) --
//   Everything below validates the indexed build path, i.e. BVH::Build( slice, indices, prims )
//   and the vertIdx / GET_PRIM_INDICES_I0_I1_I2 addressing that hangs off it. The triangle soup
//   is welded into an indexed mesh here so the consumer does not have to reproduce the welding:
//   vertices are matched on the exact bit pattern of x, y and z (w is carried over from the first
//   occurrence and never compared), the first occurrence of a position wins, and the soup is
//   walked in order, so the welded array and the index array are bit-reproducible. The welded
//   vertex count is generally not triCount*3, so the builds below pass an explicit
//   bvhvec4slice with the real count: the bvhvec4* convenience overloads of Build/BuildHQ
//   hardcode count = prims*3, which is wrong for an indexed mesh.
//   u32      weldedVertCount
//            weldedVertCount * 16 bytes: the welded vertices, as float4
//   u32      indexCount (== triCount * 3)
//            indexCount * u32: the index array, three per triangle, in triangle order
//   [1] binned SAH build over the indexed mesh (BVH::Build):
//   u32      idxUsedNodes
//   u32      idxNodeCount (== idxUsedNodes)
//            idxNodeCount * 32 bytes: raw BVH::BVHNode, same layout as the BLAS nodes
//   u32      idxPrimCount
//            idxPrimCount * u32: bvh.primIdx[]   (idxCount; == triCount for a binned build)
//   u32      idxRayCount (65536)
//            the same rays as the BLAS ray section, in the same order, traced against the
//            indexed tree. Identical 48-byte record to the BLAS rays.
//   [2] SBVH build over the same indexed mesh (BVH::BuildHQ), same five fields:
//   u32      idxSbvhUsedNodes
//   u32      idxSbvhNodeCount; idxSbvhNodeCount * 32 bytes raw nodes
//   u32      idxSbvhPrimCount; idxSbvhPrimCount * u32 primIdx
//            NOTE: PrimCount(), not idxCount, for the reason given in the SBVH section above.
//   u32      idxSbvhRayCount (65536); the same 48-byte ray records
//   -- Opacity micro map section (added in TBVHREF5) --
//   A procedural opacity map, chosen to be trivial to reproduce exactly in C#: with N = 8, each
//   triangle owns N*N bits, one per micro-triangle, packed low-bit-first into (N*N+31)/32 = 2
//   uint32 words per triangle. Micro-triangle b of triangle t is opaque when
//   ((t * 7 + b * 13) & 3) != 0, i.e. one in four micro-triangles is a hole; all arithmetic is
//   uint32. The map buffer has two extra zeroed uint32 words at the very end: tinybvh computes
//   the micro-triangle bit index from u and v without clamping, so a hit with u + v == 1 exactly
//   indexes one bit past the last triangle's own words, and the slack means both the generator
//   and the C# port read the same zero there instead of running off the end of the allocation.
//   The map is applied to a fresh binned BVH (BVH::Build) built over the pristine, unwelded
//   vertices, via BVH::SetOpacityMicroMaps( opmap.data(), N ), and traced with the same BLAS
//   rays used above, in the same order.
//   u32      opMapN (8)
//   u32      opMapWords (== (opMapN*opMapN+31)/32, i.e. 2 for N = 8)
//   u32      opacityRayCount (65536)
//            the same rays as the BLAS ray section, in the same order, traced against the BVH
//            carrying the opacity map. Identical 48-byte record to the BLAS rays.
//
// NOTE: the task description that spawned this tool wrote "(40 bytes)" next to
// the two ray-record layouts above. Counting the listed fields (3+3 floats for
// O/D, 3 floats for t/u/v, 3 u32) gives 48 bytes, and that is what is written
// here; the field list is unambiguous and matches what's needed downstream
// (O/D must be stored so the consumer doesn't replicate the RNG), so 48 bytes
// is treated as authoritative and the byte count in the spec as a mistake.
//
// Determinism strategy: NO_THREADED_BUILDS is defined before including
// tiny_bvh.h. This compiles ENABLE_THREADED_BUILDS out entirely, so every
// `#ifdef ENABLE_THREADED_BUILDS` branch that spawns worker tasks during a
// build is removed at compile time and `threadedBuild` is unconditionally
// forced to false right before those branches. TINYBVH_NO_SIMD is defined so
// BVH_USEAVX/BVH_USENEON never get set, which routes BVH::Build(verts,triCount)
// through PrepareBuild()+Build() - the scalar reference builder - rather than
// BuildAVX(). bvh.settings.useSIMDifavailable is also set to false, and
// bvh.context.spawn/barrier/parallel_for are nulled explicitly, for belt and
// braces on top of the compile-time switch.

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
#include <array>
#include <map>

using namespace tinybvh;

static const uint32_t N_RAYS = 65536;
static const float PI = 3.14159265358979323846f;

// Fixed LCG, as specified - do not use rand().
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

// One generated ray, kept around so it can be replayed against the refit BVH
// without needing to re-run the RNG.
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
		Ray ray( origin, target - origin ); // constructor normalizes D and computes rD
		rays[i].O = ray.O, rays[i].D = ray.D;
	}
	return rays;
}

static void WriteVec3( std::ofstream& f, const bvhvec3& v )
{
	f.write( (const char*)&v.x, 4 );
	f.write( (const char*)&v.y, 4 );
	f.write( (const char*)&v.z, 4 );
}

// One traced BLAS-style ray record; 48 bytes once the ray's O and D are written in front of it.
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

// One optimizer reference section; see the format comment at the top of this file.
struct OptSection
{
	uint32_t iterations = 0, usedNodes = 0, primCount = 0;
	float sahCost = 0, sahCostBefore = 0;
	bvhvec3 aabbMin, aabbMax;
	std::vector<BVH::BVHNode> nodes;
	std::vector<uint32_t> primIdx;
	std::vector<BlasHit> hits;
};

static void WriteU32( std::ofstream& f, uint32_t v ) { f.write( (const char*)&v, 4 ); }
static void WriteF32( std::ofstream& f, float v ) { f.write( (const char*)&v, 4 ); }

// Writes the raw node pool and primitive-index array for a built BVH (BLAS or TLAS).
static void WriteNodesAndIndices( std::ofstream& f, const BVH& bvh )
{
	WriteU32( f, bvh.usedNodes );
	f.write( (const char*)bvh.bvhNode, (size_t)bvh.usedNodes * sizeof( BVH::BVHNode ) );
	WriteU32( f, bvh.idxCount );
	f.write( (const char*)bvh.primIdx, (size_t)bvh.idxCount * sizeof( uint32_t ) );
}

// Writes one ray block of the indexed-geometry sections: the count, then a 48-byte record per
// ray holding its O and D followed by the hit it produced.
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

int main( int argc, char** argv )
{
	static_assert (sizeof( BVH::BVHNode ) == 32, "BVH::BVHNode must be 32 bytes");

	if (argc != 3)
	{
		fprintf( stderr, "usage: refdump <scene.bin> <out.ref>\n" );
		return 1;
	}
	const char* sceneFile = argv[1];
	const char* outFile = argv[2];

	// --- Load scene -----------------------------------------------------
	std::ifstream sf( sceneFile, std::ios::binary );
	if (!sf)
	{
		fprintf( stderr, "cannot open scene file: %s\n", sceneFile );
		return 1;
	}
	int32_t triCount = 0;
	sf.read( (char*)&triCount, 4 );
	const uint32_t vertCount = (uint32_t)triCount * 3;
	bvhvec4* verts = (bvhvec4*)malloc64( vertCount * sizeof( bvhvec4 ) );
	sf.read( (char*)verts, (size_t)vertCount * sizeof( bvhvec4 ) );
	sf.close();

	// keep a pristine copy of the vertices, for the refit test.
	std::vector<bvhvec4> vertsOriginal( verts, verts + vertCount );

	// --- Determinism check: build the same scene twice, compare byte-for-byte.
	BVH bvhA, bvhB;
	bvhA.settings.useSIMDifavailable = false;
	bvhA.context.spawn = nullptr, bvhA.context.barrier = nullptr, bvhA.context.parallel_for = nullptr;
	bvhB.settings.useSIMDifavailable = false;
	bvhB.context.spawn = nullptr, bvhB.context.barrier = nullptr, bvhB.context.parallel_for = nullptr;
	bvhA.Build( verts, (uint32_t)triCount );
	bvhB.Build( verts, (uint32_t)triCount );

	bool deterministic = bvhA.usedNodes == bvhB.usedNodes && bvhA.idxCount == bvhB.idxCount;
	if (deterministic)
	{
		deterministic = memcmp( bvhA.bvhNode, bvhB.bvhNode, (size_t)bvhA.usedNodes * sizeof( BVH::BVHNode ) ) == 0
			&& memcmp( bvhA.primIdx, bvhB.primIdx, (size_t)bvhA.idxCount * sizeof( uint32_t ) ) == 0;
	}
	if (!deterministic)
	{
		fprintf( stderr, "FATAL: two builds of the same scene produced different BVHs - not deterministic.\n" );
		return 1;
	}
	if (bvhA.may_have_holes)
	{
		fprintf( stderr, "FATAL: bvh.may_have_holes is true; expected a hole-free serial build.\n" );
		return 1;
	}

	// bvhA is the working BLAS from here on; bvhB was only needed for the check above.
	BVH& bvh = bvhA;

	// --- BLAS section -----------------------------------------------------
	const float sahCost = bvh.SAHCost();
	const bvhvec3 blasAabbMin = bvh.aabbMin, blasAabbMax = bvh.aabbMax;

	// snapshot the node/index arrays before Refit() touches them.
	std::vector<BVH::BVHNode> blasNodes( bvh.bvhNode, bvh.bvhNode + bvh.usedNodes );
	std::vector<uint32_t> blasIdx( bvh.primIdx, bvh.primIdx + bvh.idxCount );

	// --- BLAS rays ----------------------------------------------------
	std::vector<GenRay> blasRays = GenerateRays( blasAabbMin, blasAabbMax, N_RAYS );
	std::vector<BlasHit> blasHits( N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		const GenRay& gr = blasRays[i];
		Ray ray( gr.O, gr.D );
		bvh.Intersect( ray );
		BlasHit& h = blasHits[i];
		h.t = ray.hit.t, h.u = ray.hit.u, h.v = ray.hit.v, h.prim = ray.hit.prim;

		Ray occRay( gr.O, gr.D );
		h.occludedFull = bvh.IsOccluded( occRay ) ? 1u : 0u;

		float halfT = ray.hit.t < BVH_FAR ? 0.5f * ray.hit.t : BVH_FAR;
		Ray occHalfRay( gr.O, gr.D, halfT );
		h.occludedHalf = bvh.IsOccluded( occHalfRay ) ? 1u : 0u;
	}

	// --- SBVH section ---------------------------------------------------
	// Built here, while the vertices are still pristine: the refit test below perturbs them.
	BVH sbvh;
	sbvh.settings.useSIMDifavailable = false;
	sbvh.context.spawn = nullptr, sbvh.context.barrier = nullptr, sbvh.context.parallel_for = nullptr;
	sbvh.BuildHQ( verts, (uint32_t)triCount );
	const float sbvhSahCost = sbvh.SAHCost();
	const bvhvec3 sbvhAabbMin = sbvh.aabbMin, sbvhAabbMax = sbvh.aabbMax;
	// BuildHQ ends with Compact(); see the format note above for why idxCount is not usable here.
	const uint32_t sbvhIdxCount = (uint32_t)sbvh.PrimCount();

	std::vector<BlasHit> sbvhHits( N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		const GenRay& gr = blasRays[i];
		Ray ray( gr.O, gr.D );
		sbvh.Intersect( ray );
		BlasHit& h = sbvhHits[i];
		h.t = ray.hit.t, h.u = ray.hit.u, h.v = ray.hit.v, h.prim = ray.hit.prim;

		Ray occRay( gr.O, gr.D );
		h.occludedFull = sbvh.IsOccluded( occRay ) ? 1u : 0u;

		float halfT = ray.hit.t < BVH_FAR ? 0.5f * ray.hit.t : BVH_FAR;
		Ray occHalfRay( gr.O, gr.D, halfT );
		h.occludedHalf = sbvh.IsOccluded( occHalfRay ) ? 1u : 0u;
	}

	// --- Optimizer sections ---------------------------------------------
	// Also built here, while the vertices are still pristine.
	// Large scenes take a while; 25 iterations over Crytek Sponza is a few minutes, which is
	// within budget. Should a bigger scene ever be added, drop the count here - it is written
	// into the file so the consumer uses the same number.
	const uint32_t optIterations = 25;

	// [1] plain BVH::Optimize on a fresh binned build.
	BVH opt;
	opt.settings.useSIMDifavailable = false;
	opt.context.spawn = nullptr, opt.context.barrier = nullptr, opt.context.parallel_for = nullptr;
	opt.Build( verts, (uint32_t)triCount );
	OptSection optPlain;
	optPlain.iterations = optIterations;
	optPlain.sahCostBefore = opt.SAHCost();
	clock_t optStart = clock();
	opt.Optimize( optIterations, false, false );
	const double optSeconds = (double)(clock() - optStart) / CLOCKS_PER_SEC;
	optPlain.usedNodes = opt.usedNodes;
	optPlain.sahCost = opt.SAHCost();
	optPlain.aabbMin = opt.aabbMin, optPlain.aabbMax = opt.aabbMax;
	optPlain.nodes.assign( opt.bvhNode, opt.bvhNode + opt.usedNodes );
	optPlain.primCount = (uint32_t)opt.PrimCount();
	optPlain.primIdx.assign( opt.primIdx, opt.primIdx + optPlain.primCount );
	TraceBlasRays( opt, blasRays, optPlain.hits );

	// [2] SplitLeafs( 1 ) / Optimize / MergeLeafs, driven through BVH_Verbose by hand.
	BVH optm;
	optm.settings.useSIMDifavailable = false;
	optm.context.spawn = nullptr, optm.context.barrier = nullptr, optm.context.parallel_for = nullptr;
	optm.Build( verts, (uint32_t)triCount );
	OptSection optMerged;
	optMerged.iterations = optIterations;
	optMerged.sahCostBefore = optm.SAHCost();
	clock_t optmStart = clock();
	{
		BVH_Verbose verbose;
		verbose.ConvertFrom( optm );
		verbose.SplitLeafs( 1 );
		verbose.Optimize( optIterations, false, false );
		verbose.MergeLeafs(); // frees optm.primIdx and hands over a new array
		optm.ConvertFrom( verbose ); // ..which ConvertFrom copies back into optm
	}
	const double optmSeconds = (double)(clock() - optmStart) / CLOCKS_PER_SEC;
	optMerged.usedNodes = optm.usedNodes;
	optMerged.sahCost = optm.SAHCost();
	optMerged.aabbMin = optm.aabbMin, optMerged.aabbMax = optm.aabbMax;
	optMerged.nodes.assign( optm.bvhNode, optm.bvhNode + optm.usedNodes );
	optMerged.primCount = (uint32_t)optm.PrimCount();
	optMerged.primIdx.assign( optm.primIdx, optm.primIdx + optMerged.primCount );
	TraceBlasRays( optm, blasRays, optMerged.hits );

	// --- Indexed-geometry sections ---------------------------------------
	// Built here, while the vertices are still pristine. See the format comment at the top of
	// this file for the welding rule; the key is the raw bit pattern of x, y and z, and std::map
	// is used rather than a hash map so nothing about the result can depend on bucket layout.
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
	// give the welded vertices the same 64-byte aligned storage the soup gets.
	bvhvec4* weldedVerts = (bvhvec4*)malloc64( weldedCount * sizeof( bvhvec4 ) );
	memcpy( weldedVerts, welded.data(), (size_t)weldedCount * sizeof( bvhvec4 ) );
	const bvhvec4slice weldedSlice( weldedVerts, weldedCount, sizeof( bvhvec4 ) );

	BVH ibvh;
	ibvh.settings.useSIMDifavailable = false;
	ibvh.context.spawn = nullptr, ibvh.context.barrier = nullptr, ibvh.context.parallel_for = nullptr;
	ibvh.Build( weldedSlice, indices.data(), (uint32_t)triCount );
	std::vector<BlasHit> idxHits;
	TraceBlasRays( ibvh, blasRays, idxHits );

	BVH isbvh;
	isbvh.settings.useSIMDifavailable = false;
	isbvh.context.spawn = nullptr, isbvh.context.barrier = nullptr, isbvh.context.parallel_for = nullptr;
	isbvh.BuildHQ( weldedSlice, indices.data(), (uint32_t)triCount );
	const uint32_t idxSbvhPrimCount = (uint32_t)isbvh.PrimCount();
	std::vector<BlasHit> idxSbvhHits;
	TraceBlasRays( isbvh, blasRays, idxSbvhHits );

	// --- Opacity micro map section ---------------------------------------
	// Procedural map, easy to reproduce exactly in C#: with N = OPMAP_N, each triangle owns
	// N*N consecutive bits, packed low-bit-first into ( N * N + 31 ) / 32 uint32 words.
	// Micro-triangle b of triangle t is opaque when ((t * 7 + b * 13) & 3) != 0, i.e. one in
	// four micro-triangles is a hole. All arithmetic is uint32.
	const uint32_t OPMAP_N = 8;
	const uint32_t opmapWords = (OPMAP_N * OPMAP_N + 31) >> 5;
	// Two zeroed slack words at the end: tinybvh computes the bit index from u and v without
	// clamping, so a hit with u + v == 1 exactly indexes past the triangle's own words. The C#
	// port reproduces the same read, and the slack makes both sides read the same zeroes
	// instead of running off the end of the allocation.
	std::vector<uint32_t> opmap( (size_t)triCount * opmapWords + 2, 0u );
	for (uint32_t t = 0; t < (uint32_t)triCount; t++)
		for (uint32_t b = 0; b < OPMAP_N * OPMAP_N; b++)
			if (((t * 7u + b * 13u) & 3u) != 0u)
				opmap[(size_t)t * opmapWords + (b >> 5)] |= 1u << (b & 31);

	// Built here, while the vertices are still pristine: a fresh binned BVH carrying the map.
	BVH opbvh;
	opbvh.settings.useSIMDifavailable = false;
	opbvh.context.spawn = nullptr, opbvh.context.barrier = nullptr, opbvh.context.parallel_for = nullptr;
	opbvh.Build( verts, (uint32_t)triCount );
	opbvh.SetOpacityMicroMaps( opmap.data(), OPMAP_N );
	std::vector<BlasHit> opacityHits;
	TraceBlasRays( opbvh, blasRays, opacityHits );

	// --- Refit section ------------------------------------------------
	// deterministic per-vertex perturbation.
	const bvhvec3 ext = blasAabbMax - blasAabbMin;
	for (uint32_t i = 0; i < vertCount; i++)
	{
		bvhvec4& v = verts[i];
		v.x = v.x + (float)((int)(i % 7) - 3) * 0.005f * ext.x;
		v.y = v.y + (float)((int)(i % 5) - 2) * 0.005f * ext.y;
	}
	bvh.Refit();
	const bvhvec3 refitAabbMin = bvh.aabbMin, refitAabbMax = bvh.aabbMax;

	struct RefitHit { float t, u, v; uint32_t prim; };
	std::vector<RefitHit> refitHits( N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		const GenRay& gr = blasRays[i];
		Ray ray( gr.O, gr.D );
		bvh.Intersect( ray );
		refitHits[i] = { ray.hit.t, ray.hit.u, ray.hit.v, ray.hit.prim };
	}

	// restore original vertices for the TLAS test.
	memcpy( verts, vertsOriginal.data(), (size_t)vertCount * sizeof( bvhvec4 ) );
	bvh.Refit(); // restore node bounds to match the un-perturbed geometry, for the TLAS test

	// --- TLAS section ---------------------------------------------------
	BLASInstance inst[3] = { BLASInstance( 0 ), BLASInstance( 0 ), BLASInstance( 0 ) };
	inst[0].mask = 0xFFFF; // identity, default transform

	inst[1].transform.cell[3] = ext.x * 1.1f;
	inst[1].mask = 0xFFFF;

	inst[2].transform.cell[0] = 0, inst[2].transform.cell[1] = 0, inst[2].transform.cell[2] = 0.5f, inst[2].transform.cell[3] = 0;
	inst[2].transform.cell[4] = 0, inst[2].transform.cell[5] = 0.5f, inst[2].transform.cell[6] = 0, inst[2].transform.cell[7] = 0;
	inst[2].transform.cell[8] = -0.5f, inst[2].transform.cell[9] = 0, inst[2].transform.cell[10] = 0, inst[2].transform.cell[11] = ext.z * 1.1f;
	inst[2].mask = 0x0002;

	BVH tlas;
	tlas.settings.useSIMDifavailable = false;
	tlas.context.spawn = nullptr, tlas.context.barrier = nullptr, tlas.context.parallel_for = nullptr;
	BVHBase* blasList[1] = { &bvh };
	tlas.Build( inst, 3, blasList, 1 );

	std::vector<GenRay> tlasRays = GenerateRays( tlas.aabbMin, tlas.aabbMax, N_RAYS );
	struct TlasHit { float t, u, v; uint32_t prim, inst, occludedFull; };
	std::vector<TlasHit> tlasHits( N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		const GenRay& gr = tlasRays[i];
		Ray ray( gr.O, gr.D );
		tlas.Intersect( ray );
		TlasHit& h = tlasHits[i];
		h.t = ray.hit.t, h.u = ray.hit.u, h.v = ray.hit.v, h.prim = ray.hit.prim, h.inst = ray.hit.inst;

		Ray occRay( gr.O, gr.D );
		h.occludedFull = tlas.IsOccluded( occRay ) ? 1u : 0u;
	}

	// --- Write output file ----------------------------------------------
	std::ofstream f( outFile, std::ios::binary );
	if (!f)
	{
		fprintf( stderr, "cannot open output file: %s\n", outFile );
		return 1;
	}
	f.write( "TBVHREF5", 8 );
	WriteU32( f, (uint32_t)triCount );

	// BLAS section
	WriteU32( f, bvh.usedNodes );
	WriteF32( f, sahCost );
	WriteVec3( f, blasAabbMin ); WriteVec3( f, blasAabbMax );
	WriteU32( f, (uint32_t)blasNodes.size() );
	f.write( (const char*)blasNodes.data(), blasNodes.size() * sizeof( BVH::BVHNode ) );
	WriteU32( f, (uint32_t)blasIdx.size() );
	f.write( (const char*)blasIdx.data(), blasIdx.size() * sizeof( uint32_t ) );

	// BLAS rays
	WriteU32( f, N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		WriteVec3( f, blasRays[i].O ); WriteVec3( f, blasRays[i].D );
		const BlasHit& h = blasHits[i];
		WriteF32( f, h.t ); WriteF32( f, h.u ); WriteF32( f, h.v );
		WriteU32( f, h.prim ); WriteU32( f, h.occludedFull ); WriteU32( f, h.occludedHalf );
	}

	// Refit section
	WriteVec3( f, refitAabbMin ); WriteVec3( f, refitAabbMax );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		const RefitHit& h = refitHits[i];
		WriteF32( f, h.t ); WriteF32( f, h.u ); WriteF32( f, h.v ); WriteU32( f, h.prim );
	}

	// TLAS section
	WriteU32( f, 3 );
	for (int i = 0; i < 3; i++)
	{
		f.write( (const char*)inst[i].transform.cell, 16 * sizeof( float ) );
		f.write( (const char*)inst[i].invTransform.cell, 16 * sizeof( float ) );
		WriteVec3( f, inst[i].aabbMin ); WriteVec3( f, inst[i].aabbMax );
		WriteU32( f, inst[i].mask );
	}
	WriteU32( f, tlas.usedNodes );
	WriteVec3( f, tlas.aabbMin ); WriteVec3( f, tlas.aabbMax );
	WriteNodesAndIndices( f, tlas ); // writes tlasNodeCount/nodes then tlasIdxCount/idx
	WriteU32( f, N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		WriteVec3( f, tlasRays[i].O ); WriteVec3( f, tlasRays[i].D );
		const TlasHit& h = tlasHits[i];
		WriteF32( f, h.t ); WriteF32( f, h.u ); WriteF32( f, h.v );
		WriteU32( f, h.prim ); WriteU32( f, h.inst ); WriteU32( f, h.occludedFull );
	}

	// SBVH section
	WriteU32( f, sbvh.usedNodes );
	WriteF32( f, sbvhSahCost );
	WriteVec3( f, sbvhAabbMin ); WriteVec3( f, sbvhAabbMax );
	WriteU32( f, sbvh.usedNodes );
	f.write( (const char*)sbvh.bvhNode, (size_t)sbvh.usedNodes * sizeof( BVH::BVHNode ) );
	WriteU32( f, sbvhIdxCount );
	f.write( (const char*)sbvh.primIdx, (size_t)sbvhIdxCount * sizeof( uint32_t ) );
	WriteU32( f, N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		WriteVec3( f, blasRays[i].O ); WriteVec3( f, blasRays[i].D );
		const BlasHit& h = sbvhHits[i];
		WriteF32( f, h.t ); WriteF32( f, h.u ); WriteF32( f, h.v );
		WriteU32( f, h.prim ); WriteU32( f, h.occludedFull ); WriteU32( f, h.occludedHalf );
	}

	// Optimizer sections
	const OptSection* optSections[2] = { &optPlain, &optMerged };
	for (int s = 0; s < 2; s++)
	{
		const OptSection& o = *optSections[s];
		WriteU32( f, o.iterations );
		WriteU32( f, o.usedNodes );
		WriteF32( f, o.sahCost );
		WriteF32( f, o.sahCostBefore );
		WriteVec3( f, o.aabbMin ); WriteVec3( f, o.aabbMax );
		WriteU32( f, (uint32_t)o.nodes.size() );
		f.write( (const char*)o.nodes.data(), o.nodes.size() * sizeof( BVH::BVHNode ) );
		WriteU32( f, o.primCount );
		f.write( (const char*)o.primIdx.data(), (size_t)o.primCount * sizeof( uint32_t ) );
		WriteU32( f, N_RAYS );
		for (uint32_t i = 0; i < N_RAYS; i++)
		{
			WriteVec3( f, blasRays[i].O ); WriteVec3( f, blasRays[i].D );
			const BlasHit& h = o.hits[i];
			WriteF32( f, h.t ); WriteF32( f, h.u ); WriteF32( f, h.v );
			WriteU32( f, h.prim ); WriteU32( f, h.occludedFull ); WriteU32( f, h.occludedHalf );
		}
	}

	// Indexed-geometry sections
	WriteU32( f, weldedCount );
	f.write( (const char*)weldedVerts, (size_t)weldedCount * sizeof( bvhvec4 ) );
	WriteU32( f, (uint32_t)indices.size() );
	f.write( (const char*)indices.data(), indices.size() * sizeof( uint32_t ) );
	WriteU32( f, ibvh.usedNodes );
	WriteU32( f, ibvh.usedNodes );
	f.write( (const char*)ibvh.bvhNode, (size_t)ibvh.usedNodes * sizeof( BVH::BVHNode ) );
	WriteU32( f, ibvh.idxCount );
	f.write( (const char*)ibvh.primIdx, (size_t)ibvh.idxCount * sizeof( uint32_t ) );
	WriteRayRecords( f, blasRays, idxHits );
	WriteU32( f, isbvh.usedNodes );
	WriteU32( f, isbvh.usedNodes );
	f.write( (const char*)isbvh.bvhNode, (size_t)isbvh.usedNodes * sizeof( BVH::BVHNode ) );
	WriteU32( f, idxSbvhPrimCount );
	f.write( (const char*)isbvh.primIdx, (size_t)idxSbvhPrimCount * sizeof( uint32_t ) );
	WriteRayRecords( f, blasRays, idxSbvhHits );

	// Opacity micro map section
	WriteU32( f, OPMAP_N );
	WriteU32( f, opmapWords );
	WriteU32( f, N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		WriteVec3( f, blasRays[i].O ); WriteVec3( f, blasRays[i].D );
		const BlasHit& h = opacityHits[i];
		WriteF32( f, h.t ); WriteF32( f, h.u ); WriteF32( f, h.v );
		WriteU32( f, h.prim ); WriteU32( f, h.occludedFull ); WriteU32( f, h.occludedHalf );
	}
	f.close();

	// --- Summary ----------------------------------------------------------
	uint32_t blasHitCount = 0, refitHitCount = 0, tlasHitCount = 0;
	for (uint32_t i = 0; i < N_RAYS; i++) if (blasHits[i].t < BVH_FAR) blasHitCount++;
	for (uint32_t i = 0; i < N_RAYS; i++) if (refitHits[i].t < BVH_FAR) refitHitCount++;
	for (uint32_t i = 0; i < N_RAYS; i++) if (tlasHits[i].t < BVH_FAR) tlasHitCount++;

	printf( "scene: %s\n", sceneFile );
	printf( "  triCount:       %d\n", triCount );
	printf( "  usedNodes:      %u\n", bvh.usedNodes );
	printf( "  SAH cost:       %f\n", sahCost );
	printf( "  BLAS hit ratio:  %.2f%% (%u/%u)\n", 100.0 * blasHitCount / N_RAYS, blasHitCount, N_RAYS );
	printf( "  refit hit ratio: %.2f%% (%u/%u)\n", 100.0 * refitHitCount / N_RAYS, refitHitCount, N_RAYS );
	printf( "  TLAS hit ratio:  %.2f%% (%u/%u)\n", 100.0 * tlasHitCount / N_RAYS, tlasHitCount, N_RAYS );
	uint32_t sbvhHitCount = 0;
	for (uint32_t i = 0; i < N_RAYS; i++) if (sbvhHits[i].t < BVH_FAR) sbvhHitCount++;
	printf( "  SBVH usedNodes: %u, prims %u (%.2fx), SAH cost %f\n",
		sbvh.usedNodes, sbvhIdxCount, (double)sbvhIdxCount / triCount, sbvhSahCost );
	printf( "  SBVH hit ratio:  %.2f%% (%u/%u)\n", 100.0 * sbvhHitCount / N_RAYS, sbvhHitCount, N_RAYS );
	uint32_t optHitCount = 0, optmHitCount = 0;
	for (uint32_t i = 0; i < N_RAYS; i++) if (optPlain.hits[i].t < BVH_FAR) optHitCount++;
	for (uint32_t i = 0; i < N_RAYS; i++) if (optMerged.hits[i].t < BVH_FAR) optmHitCount++;
	printf( "  Optimize(%u): nodes %u, SAH %f -> %f (%.1f%%), %.1fs\n",
		optPlain.iterations, optPlain.usedNodes, optPlain.sahCostBefore, optPlain.sahCost,
		100.0 * (optPlain.sahCost / optPlain.sahCostBefore - 1.0), optSeconds );
	printf( "  Optimize hit ratio: %.2f%% (%u/%u)\n", 100.0 * optHitCount / N_RAYS, optHitCount, N_RAYS );
	printf( "  SplitLeafs(1)+Optimize(%u)+MergeLeafs: nodes %u, prims %u, SAH %f -> %f (%.1f%%), %.1fs\n",
		optMerged.iterations, optMerged.usedNodes, optMerged.primCount, optMerged.sahCostBefore, optMerged.sahCost,
		100.0 * (optMerged.sahCost / optMerged.sahCostBefore - 1.0), optmSeconds );
	printf( "  split/merge hit ratio: %.2f%% (%u/%u)\n", 100.0 * optmHitCount / N_RAYS, optmHitCount, N_RAYS );
	uint32_t idxHitCount = 0, idxSbvhHitCount = 0;
	for (uint32_t i = 0; i < N_RAYS; i++) if (idxHits[i].t < BVH_FAR) idxHitCount++;
	for (uint32_t i = 0; i < N_RAYS; i++) if (idxSbvhHits[i].t < BVH_FAR) idxSbvhHitCount++;
	printf( "  indexed: welded %u verts from %u (%.2fx), nodes %u\n",
		weldedCount, vertCount, (double)weldedCount / vertCount, ibvh.usedNodes );
	printf( "  indexed hit ratio: %.2f%% (%u/%u)\n", 100.0 * idxHitCount / N_RAYS, idxHitCount, N_RAYS );
	printf( "  indexed SBVH: nodes %u, prims %u; hit ratio %.2f%% (%u/%u)\n",
		isbvh.usedNodes, idxSbvhPrimCount, 100.0 * idxSbvhHitCount / N_RAYS, idxSbvhHitCount, N_RAYS );
	uint32_t opacityHitCount = 0;
	for (uint32_t i = 0; i < N_RAYS; i++) if (opacityHits[i].t < BVH_FAR) opacityHitCount++;
	printf( "  opacity(N=%u) hit ratio: %.2f%% (%u/%u)\n", OPMAP_N, 100.0 * opacityHitCount / N_RAYS, opacityHitCount, N_RAYS );
	printf( "  determinism check: PASS (two independent builds are byte-identical)\n" );
	printf( "  may_have_holes: %s\n", bvh.may_have_holes ? "true (UNEXPECTED)" : "false" );
	printf( "  wrote: %s\n", outFile );

	free64( verts );
	free64( weldedVerts );
	return 0;
}
