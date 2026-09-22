// simddump.cpp - reference data for the tinybvh (v1.8.0) code that only exists in a SIMD build:
// the CPU traversal of the CWBVH layout (BVH8_CWBVH::Intersect, inside BVH_USEAVX), the BVH_SoA
// layout (ENABLE_BVH_SOA, traversal inside BVH_USEAVX), the AVX binned builder (BVH::BuildAVX),
// and a TLAS whose BLASes use the SIMD layouts (BVH4_CPU, BVH8_CPU, BVH_SoA). Compiled with
// /arch:AVX2, unlike refdump/featdump/layoutdump which are compiled TINYBVH_NO_SIMD; every BVH
// except the AVX-build block still gets settings.useSIMDifavailable = false so the scalar binned
// builder produces the trees, and the base tree is written out so the consumer can assert that
// this build did not drift from the scalar one before trusting the traversal results.
//
// Usage: simddump <scene.bin> <out.simd.ref>
//
// Output file format (little-endian, no padding):
//   char[8]  magic = "TBVHSIM2"
//   u32      triCount
//   -- Base tree (scalar binned BVH::Build; must equal the BLAS section of <scene>.ref) --
//   u32      usedNodes
//            usedNodes * 32 bytes: raw BVH::BVHNode
//   u32      idxCount
//            idxCount * u32: bvh.primIdx[]
//   -- Rays --
//   u32      rayCount (65536)
//            per ray: f32[3] O, f32[3] D. Same generator, seed and root bounds as refdump, so
//            these are the refdump BLAS rays.
//   -- CWBVH, traced on the CPU --
//   BVH8_CWBVH::Build over the soup (which is bvh8.bvh.Build + Compact + SplitLeafs( 3 ) +
//   MBVH<8>::ConvertFrom + BVH8_CWBVH::ConvertFrom, all scalar code), then per ray
//   BVH8_CWBVH::Intersect and the two FALLBACK_SHADOW_QUERY occlusion tests refdump uses
//   (full-length, then at half the hit distance).
//   u32      usedBlocks                 (16-byte blocks of bvh8Data)
//   u64      nodeHash                   (FNV-1a 64 over usedBlocks * 16 bytes of bvh8Data)
//   u64      triHash                    (FNV-1a 64 over bvh8.idxCount * 64 bytes of bvh8Tris)
//   u32      rayCount (65536)
//            per ray: f32 t, f32 u, f32 v, u32 prim, u32 occludedFull, u32 occludedHalf (24 bytes)
//   -- BVH_SoA, traced on the CPU --
//   BVH_SoA::Build over the soup (bvh.Build + ConvertFrom( bvh, false )), then per ray
//   BVH_SoA::Intersect and BVH_SoA::IsOccluded (full-length and half-distance).
//   u32      usedNodes
//            usedNodes * 64 bytes: raw BVH_SoA::BVHNode (xxxx, yyyy, zzzz as 4 floats each,
//            then u32 left, right, triCount, firstTri)
//   u32      rayCount (65536)
//            per ray: f32 t, f32 u, f32 v, u32 prim, u32 occludedFull, u32 occludedHalf (24 bytes)
//   -- AVX binned builder (added in TBVHSIM2) --
//   A fresh BVH with the default settings.useSIMDifavailable = true, so BVH::Build routes to
//   BuildAVX (PrepareAVXBuild + BuildAVXSubtree + BuildAVXFinalize), serial (NO_THREADED_BUILDS).
//   u32      usedNodes
//   f32      sahCost
//   f32[3]   aabbMin, f32[3] aabbMax
//   u32      nodeCount (== usedNodes); nodeCount * 32 bytes raw BVH::BVHNode
//   u32      idxCount; idxCount * u32 primIdx
//   u32      rayCount (65536); per ray the 24-byte hit record above, for the rays listed above
//   -- Mixed-layout TLAS (added in TBVHSIM2) --
//   BLAS list = { &bvh (BVH, the base tree), &bvh4 (BVH4_CPU), &bvh8 (BVH8_CPU), &soa (BVH_SoA) },
//   each built over the soup with useSIMDifavailable = false. Four instances, instance i uses
//   BLAS i with an identity transform translated by i * 1.1 * extent.x along x, plus a fifth
//   instance of BLAS 2 (BVH8_CPU) with the rotated/scaled transform refdump's third instance
//   uses (cells 0..11 = 0,0,0.5,0 / 0,0.5,0,0 / -0.5,0,0,1.1*extent.z), all masks 0xFFFF,
//   except instance 3 (the SoA one) which gets mask 0x0002. TLAS = BVH::Build( inst, 5, list, 4 ).
//   u32      instCount (5)
//            per instance: f32[16] transform, f32[16] invTransform, f32[3] aabbMin,
//            f32[3] aabbMax, u32 blasIdx, u32 mask     (after the build updated them)
//   u32      tlasUsedNodes; f32[3] aabbMin, f32[3] aabbMax
//   u32      tlasNodeCount; tlasNodeCount * 32 bytes; u32 tlasIdxCount; u32[]
//   u32      tlasRayCount (65536)
//            per ray: f32[3] O, f32[3] D, f32 t, f32 u, f32 v, u32 prim, u32 inst,
//            u32 occludedFull (48 bytes). Rays generated over the TLAS bounds as in refdump.

#define ENABLE_BVH_SOA
#define NO_THREADED_BUILDS
#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>

using namespace tinybvh;

#if !defined BVH_USEAVX2
#error "simddump must be compiled with /arch:AVX2 so BVH_USEAVX2 is defined"
#endif

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

struct Hit { float t, u, v; uint32_t prim, occludedFull, occludedHalf; };

template <class T> static void TraceRays( T& bvh, const std::vector<GenRay>& rays, std::vector<Hit>& out )
{
	out.resize( rays.size() );
	for (size_t i = 0; i < rays.size(); i++)
	{
		const GenRay& gr = rays[i];
		Ray ray( gr.O, gr.D );
		bvh.Intersect( ray );
		Hit& h = out[i];
		h.t = ray.hit.t, h.u = ray.hit.u, h.v = ray.hit.v, h.prim = ray.hit.prim;
		Ray occRay( gr.O, gr.D );
		h.occludedFull = bvh.IsOccluded( occRay ) ? 1u : 0u;
		float halfT = ray.hit.t < BVH_FAR ? 0.5f * ray.hit.t : BVH_FAR;
		Ray occHalfRay( gr.O, gr.D, halfT );
		h.occludedHalf = bvh.IsOccluded( occHalfRay ) ? 1u : 0u;
	}
}

static uint64_t Fnv1a64( const void* data, size_t bytes )
{
	const uint8_t* p = (const uint8_t*)data;
	uint64_t h = 14695981039346656037ull;
	for (size_t i = 0; i < bytes; i++) h = (h ^ p[i]) * 1099511628211ull;
	return h;
}

static void WriteU32( std::ofstream& f, uint32_t v ) { f.write( (const char*)&v, 4 ); }
static void WriteU64( std::ofstream& f, uint64_t v ) { f.write( (const char*)&v, 8 ); }
static void WriteF32( std::ofstream& f, float v ) { f.write( (const char*)&v, 4 ); }
static void WriteVec3( std::ofstream& f, const bvhvec3& v ) { WriteF32( f, v.x ); WriteF32( f, v.y ); WriteF32( f, v.z ); }

static void WriteHits( std::ofstream& f, const std::vector<Hit>& hits, const char* label )
{
	WriteU32( f, (uint32_t)hits.size() );
	uint32_t hitCount = 0;
	for (size_t i = 0; i < hits.size(); i++)
	{
		const Hit& h = hits[i];
		WriteF32( f, h.t ); WriteF32( f, h.u ); WriteF32( f, h.v );
		WriteU32( f, h.prim ); WriteU32( f, h.occludedFull ); WriteU32( f, h.occludedHalf );
		if (h.t < BVH_FAR) hitCount++;
	}
	printf( "  %s hit ratio: %.2f%% (%u/%u)\n", label, 100.0 * hitCount / hits.size(), hitCount, (uint32_t)hits.size() );
}

template <class T> static void Configure( T& bvh )
{
	bvh.settings.useSIMDifavailable = false;
	bvh.context.spawn = nullptr, bvh.context.barrier = nullptr, bvh.context.parallel_for = nullptr;
}

int main( int argc, char** argv )
{
	static_assert (sizeof( BVH::BVHNode ) == 32, "BVH::BVHNode must be 32 bytes");
	static_assert (sizeof( BVH_SoA::BVHNode ) == 64, "BVH_SoA::BVHNode must be 64 bytes");
	if (argc != 3)
	{
		fprintf( stderr, "usage: simddump <scene.bin> <out.simd.ref>\n" );
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
	f.write( "TBVHSIM2", 8 );
	WriteU32( f, (uint32_t)triCount );
	printf( "scene: %s (%d tris)\n", argv[1], triCount );

	// Base tree and rays.
	BVH binned;
	Configure( binned );
	binned.Build( verts, (uint32_t)triCount );
	std::vector<GenRay> rays = GenerateRays( binned.aabbMin, binned.aabbMax, N_RAYS );
	WriteU32( f, binned.usedNodes );
	f.write( (const char*)binned.bvhNode, (size_t)binned.usedNodes * sizeof( BVH::BVHNode ) );
	WriteU32( f, binned.idxCount );
	f.write( (const char*)binned.primIdx, (size_t)binned.idxCount * sizeof( uint32_t ) );
	WriteU32( f, N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++) { WriteVec3( f, rays[i].O ); WriteVec3( f, rays[i].D ); }
	printf( "  base: nodes %u\n", binned.usedNodes );

	// CWBVH on the CPU.
	BVH8_CWBVH cwbvh;
	Configure( cwbvh );
	cwbvh.Build( verts, (uint32_t)triCount );
	std::vector<Hit> cwbvhHits;
	TraceRays( cwbvh, rays, cwbvhHits );
	WriteU32( f, cwbvh.usedBlocks );
	WriteU64( f, Fnv1a64( cwbvh.bvh8Data, (size_t)cwbvh.usedBlocks * 16 ) );
	WriteU64( f, Fnv1a64( cwbvh.bvh8Tris, (size_t)cwbvh.bvh8.idxCount * 64 ) );
	printf( "  CWBVH: %u blocks, %u tri slots\n", cwbvh.usedBlocks, cwbvh.bvh8.idxCount );
	WriteHits( f, cwbvhHits, "CWBVH" );

	// BVH_SoA on the CPU.
	BVH_SoA soa;
	Configure( soa );
	soa.Build( verts, (uint32_t)triCount );
	std::vector<Hit> soaHits;
	TraceRays( soa, rays, soaHits );
	WriteU32( f, soa.usedNodes );
	f.write( (const char*)soa.bvhNode, (size_t)soa.usedNodes * sizeof( BVH_SoA::BVHNode ) );
	printf( "  SoA: %u nodes\n", soa.usedNodes );
	WriteHits( f, soaHits, "SoA" );

	// AVX binned builder.
	BVH avx;
	avx.context.spawn = nullptr, avx.context.barrier = nullptr, avx.context.parallel_for = nullptr;
	avx.Build( verts, (uint32_t)triCount ); // useSIMDifavailable defaults to true: BuildAVX
	std::vector<Hit> avxHits;
	TraceRays( avx, rays, avxHits );
	WriteU32( f, avx.usedNodes );
	WriteF32( f, avx.SAHCost() );
	WriteVec3( f, avx.aabbMin ); WriteVec3( f, avx.aabbMax );
	WriteU32( f, avx.usedNodes );
	f.write( (const char*)avx.bvhNode, (size_t)avx.usedNodes * sizeof( BVH::BVHNode ) );
	WriteU32( f, avx.idxCount );
	f.write( (const char*)avx.primIdx, (size_t)avx.idxCount * sizeof( uint32_t ) );
	printf( "  BuildAVX: nodes %u (scalar %u), SAH %f (scalar %f)\n", avx.usedNodes, binned.usedNodes, avx.SAHCost(), binned.SAHCost() );
	WriteHits( f, avxHits, "BuildAVX" );

	// Mixed-layout TLAS.
	BVH4_CPU bvh4;
	Configure( bvh4 );
	bvh4.Build( verts, (uint32_t)triCount );
	BVH8_CPU bvh8;
	Configure( bvh8 );
	bvh8.Build( verts, (uint32_t)triCount );
	const bvhvec3 ext = binned.aabbMax - binned.aabbMin;
	BLASInstance inst[5] = { BLASInstance( 0 ), BLASInstance( 1 ), BLASInstance( 2 ), BLASInstance( 3 ), BLASInstance( 2 ) };
	for (int i = 0; i < 4; i++) inst[i].transform.cell[3] = (float)i * 1.1f * ext.x, inst[i].mask = 0xFFFF;
	inst[3].mask = 0x0002;
	inst[4].transform.cell[0] = 0, inst[4].transform.cell[1] = 0, inst[4].transform.cell[2] = 0.5f, inst[4].transform.cell[3] = 0;
	inst[4].transform.cell[4] = 0, inst[4].transform.cell[5] = 0.5f, inst[4].transform.cell[6] = 0, inst[4].transform.cell[7] = 0;
	inst[4].transform.cell[8] = -0.5f, inst[4].transform.cell[9] = 0, inst[4].transform.cell[10] = 0, inst[4].transform.cell[11] = ext.z * 1.1f;
	inst[4].mask = 0xFFFF;
	BVH tlas;
	Configure( tlas );
	BVHBase* blasList[4] = { &binned, &bvh4, &bvh8, &soa };
	tlas.Build( inst, 5, blasList, 4 );
	std::vector<GenRay> tlasRays = GenerateRays( tlas.aabbMin, tlas.aabbMax, N_RAYS );
	WriteU32( f, 5 );
	for (int i = 0; i < 5; i++)
	{
		f.write( (const char*)inst[i].transform.cell, 16 * sizeof( float ) );
		f.write( (const char*)inst[i].invTransform.cell, 16 * sizeof( float ) );
		WriteVec3( f, inst[i].aabbMin ); WriteVec3( f, inst[i].aabbMax );
		WriteU32( f, inst[i].blasIdx ); WriteU32( f, inst[i].mask );
	}
	WriteU32( f, tlas.usedNodes );
	WriteVec3( f, tlas.aabbMin ); WriteVec3( f, tlas.aabbMax );
	WriteU32( f, tlas.usedNodes );
	f.write( (const char*)tlas.bvhNode, (size_t)tlas.usedNodes * sizeof( BVH::BVHNode ) );
	WriteU32( f, tlas.idxCount );
	f.write( (const char*)tlas.primIdx, (size_t)tlas.idxCount * sizeof( uint32_t ) );
	WriteU32( f, N_RAYS );
	uint32_t tlasHits = 0, perInst[5] = { 0, 0, 0, 0, 0 };
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		const GenRay& gr = tlasRays[i];
		Ray ray( gr.O, gr.D );
		tlas.Intersect( ray );
		Ray occRay( gr.O, gr.D );
		const uint32_t occluded = tlas.IsOccluded( occRay ) ? 1u : 0u;
		WriteVec3( f, gr.O ); WriteVec3( f, gr.D );
		WriteF32( f, ray.hit.t ); WriteF32( f, ray.hit.u ); WriteF32( f, ray.hit.v );
		WriteU32( f, ray.hit.prim ); WriteU32( f, ray.hit.inst ); WriteU32( f, occluded );
		if (ray.hit.t < BVH_FAR) { tlasHits++; if (ray.hit.inst < 5) perInst[ray.hit.inst]++; }
	}
	printf( "  mixed TLAS: nodes %u, %u/%u hit (per instance %u %u %u %u %u)\n", tlas.usedNodes, tlasHits, N_RAYS,
		perInst[0], perInst[1], perInst[2], perInst[3], perInst[4] );

	f.close();
	printf( "  wrote: %s\n", argv[2] );
	free64( verts );
	return 0;
}
