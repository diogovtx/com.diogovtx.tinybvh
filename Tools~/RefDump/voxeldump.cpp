// voxeldump.cpp - reference data for tinybvh's (v1.8.0) VoxelSet (ENABLE_VOXEL_SUPPORT): a
// 256^3 voxel object traversed with a three-level DDA, usable stand-alone and as a BLAS in a
// TLAS. Same conventions as refdump.cpp: scalar code, no threading, MSVC.
//
// Usage: voxeldump <scene.bin> <out.vox.ref>
//
// The voxel content is procedural so the consumer can rebuild it with the same Set() calls,
// in the same order (brick allocation order depends on it):
//   1. a spherical shell: for z, y, x in 0..255 (x innermost), with dx = x - 127.5 etc. and
//      d2 = dx*dx + dy*dy + dz*dz (float), set (x, y, z) to 1 + ( ( x * 3 + y * 5 + z * 7 ) & 255 )
//      when 90*90 <= d2 < 100*100;
//   2. 64 boxes: for b in 0..63, x0 = ( uint )( R() * 240 ), y0 = ( uint )( R() * 240 ),
//      z0 = ( uint )( R() * 240 ), sx = 4 + ( uint )( R() * 12 ), sy, sz likewise (six R() calls
//      per box, in that order), then for z in z0..z0+sz-1, y in y0..y0+sy-1, x in x0..x0+sx-1
//      (x innermost) set (x, y, z) to 300 + b.
//   The scene file is only used for the mixed TLAS at the end. R() is the refdump generator,
//   seeded 0x12345678 at start; the boxes consume it first, then the rays.
//
// Output file format (little-endian, no padding):
//   char[8]  magic = "TBVHVOX1"
//   u32      triCount
//   -- Voxel set contents (after UpdateTopGrid) --
//   u32      gridDim (32), u32 brickDim (8), u32 topGridDim (8)
//   u32      gridSize (32768);           gridSize * u32: grid[] (brick index per grid cell)
//   u32      freeBrickPtr;               freeBrickPtr * 512 * u32: brick[0 .. freeBrickPtr)
//   u32      topGridWords (16);          topGridWords * u32: topGrid bits
//   -- Rays in object space (the object is the unit cube) --
//   u32      rayCount (65536)
//            per ray: f32[3] O, f32[3] D, f32 t, u32 prim (the voxel value), u32 steps
//            (Intersect's return value), f32[3] normal (GetNormal after the hit; zero for a
//            miss), u32 occludedFull, u32 occludedHalf (60 bytes).
//            Rays: origin = ( 0.5, 0.5, 0.5 ) + RandomUnitVector() * 0.866 * 1.2 (outside the
//            cube), target = RandomPointInAABB( 0, 1 ), Ray( origin, target - origin ); the
//            second half of the rays (i >= 32768) instead start inside: origin =
//            RandomPointInAABB( 0, 1 ), direction = RandomUnitVector().
//   -- Mixed TLAS: the triangle BVH and the voxel set as BLASes --
//   BVHBase* blasList = { &bvh (BVH over the scene), &voxels }. Three instances:
//   [0] blas 0 (triangles), identity; [1] blas 1 (voxels) scaled to the scene extent and
//   translated to the scene minimum, then shifted by extent.x * 1.1 along x; [2] blas 1
//   scaled by 0.5 * extent, translated to the scene minimum and shifted by extent.z * 1.1 along
//   z. All masks 0xFFFF. TLAS = BVH::Build( inst, 3, blasList, 2 ).
//   u32      instCount (3)
//            per instance: f32[16] transform, f32[16] invTransform, f32[3] aabbMin,
//            f32[3] aabbMax, u32 blasIdx, u32 mask
//   u32      tlasUsedNodes; f32[3] aabbMin, f32[3] aabbMax
//   u32      tlasNodeCount; tlasNodeCount * 32 bytes raw BVH::BVHNode; u32 tlasIdxCount; u32[]
//   u32      tlasRayCount (65536)
//            per ray: f32[3] O, f32[3] D, f32 t, f32 u, f32 v, u32 prim, u32 inst,
//            u32 occludedFull (48 bytes). Rays generated over the TLAS bounds as in refdump.
//
// Upstream notes: VoxelSet's constructor does not set BVHBase::layout, so the TLAS traversal
// (which dispatches on it) never reaches a voxel BLAS unless the caller sets it, as this tool
// does; and BVH::IntersectTLAS asserts on a layout list that predates VoxelSet, so this tool is
// compiled with NDEBUG.

#define TINYBVH_NO_SIMD
#define NO_THREADED_BUILDS
#define ENABLE_VOXEL_SUPPORT
#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>

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

static void WriteU32( std::ofstream& f, uint32_t v ) { f.write( (const char*)&v, 4 ); }
static void WriteF32( std::ofstream& f, float v ) { f.write( (const char*)&v, 4 ); }
static void WriteVec3( std::ofstream& f, const bvhvec3& v ) { WriteF32( f, v.x ); WriteF32( f, v.y ); WriteF32( f, v.z ); }

// The private grid data is reached through a mirror of the class layout; the static_asserts
// on the public constants and the struct size guard the assumption.
struct VoxelSetMirror
{
	uint8_t base[sizeof( BVHBase )];
	uint32_t* grid;
	uint32_t* brick;
	uint32_t brickCount;
	uint32_t freeBrickPtr;
	uint32_t* topGrid;
};

int main( int argc, char** argv )
{
	static_assert (VoxelSet::objectDim == 256, "objectDim must be 256");
	static_assert (sizeof( VoxelSet ) == sizeof( VoxelSetMirror ), "VoxelSet layout mirror is out of date");
	if (argc != 3)
	{
		fprintf( stderr, "usage: voxeldump <scene.bin> <out.vox.ref>\n" );
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
	f.write( "TBVHVOX1", 8 );
	WriteU32( f, (uint32_t)triCount );
	printf( "scene: %s (%d tris)\n", argv[1], triCount );

	// Fill the voxel set. Upstream quirk: VoxelSet's constructor never sets BVHBase::layout, so
	// BVH::IntersectTLAS (which dispatches on it) would silently skip voxel BLASes; set it here.
	VoxelSet voxels;
	voxels.layout = BVHBase::LAYOUT_VOXELSET;
	uint32_t setCalls = 0;
	for (uint32_t z = 0; z < 256; z++) for (uint32_t y = 0; y < 256; y++) for (uint32_t x = 0; x < 256; x++)
	{
		const float dx = (float)x - 127.5f, dy = (float)y - 127.5f, dz = (float)z - 127.5f;
		const float d2 = dx * dx + dy * dy + dz * dz;
		if (d2 >= 90.0f * 90.0f && d2 < 100.0f * 100.0f) voxels.Set( x, y, z, 1 + ((x * 3 + y * 5 + z * 7) & 255) ), setCalls++;
	}
	for (uint32_t b = 0; b < 64; b++)
	{
		const uint32_t x0 = (uint32_t)(R() * 240.0f), y0 = (uint32_t)(R() * 240.0f), z0 = (uint32_t)(R() * 240.0f);
		const uint32_t sx = 4 + (uint32_t)(R() * 12.0f), sy = 4 + (uint32_t)(R() * 12.0f), sz = 4 + (uint32_t)(R() * 12.0f);
		for (uint32_t z = z0; z < z0 + sz; z++) for (uint32_t y = y0; y < y0 + sy; y++) for (uint32_t x = x0; x < x0 + sx; x++)
			voxels.Set( x, y, z, 300 + b ), setCalls++;
	}
	voxels.UpdateTopGrid();
	const VoxelSetMirror* m = (const VoxelSetMirror*)&voxels;
	const uint32_t gridDim = 32, brickDim = 8, topGridDim = 8, gridSize = gridDim * gridDim * gridDim, brickSize = brickDim * brickDim * brickDim;
	const uint32_t topGridWords = (topGridDim * topGridDim * topGridDim) / 32;
	WriteU32( f, gridDim ); WriteU32( f, brickDim ); WriteU32( f, topGridDim );
	WriteU32( f, gridSize );
	f.write( (const char*)m->grid, (size_t)gridSize * sizeof( uint32_t ) );
	WriteU32( f, m->freeBrickPtr );
	f.write( (const char*)m->brick, (size_t)m->freeBrickPtr * brickSize * sizeof( uint32_t ) );
	WriteU32( f, topGridWords );
	f.write( (const char*)m->topGrid, (size_t)topGridWords * sizeof( uint32_t ) );
	printf( "  voxels: %u Set calls, %u bricks used of %u\n", setCalls, m->freeBrickPtr, m->brickCount );

	// Object-space rays.
	const bvhvec3 unitMin( 0 ), unitMax( 1 ), unitCenter( 0.5f );
	WriteU32( f, N_RAYS );
	uint32_t voxHits = 0, voxOcc = 0;
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		Ray ray;
		if (i < N_RAYS / 2)
		{
			const bvhvec3 origin = unitCenter + RandomUnitVector() * 0.866f * 1.2f;
			const bvhvec3 target = RandomPointInAABB( unitMin, unitMax );
			ray = Ray( origin, target - origin );
		}
		else
		{
			const bvhvec3 origin = RandomPointInAABB( unitMin, unitMax );
			ray = Ray( origin, RandomUnitVector() );
		}
		const bvhvec3 O = ray.O, D = ray.D;
		const int32_t steps = voxels.Intersect( ray );
		bvhvec3 normal( 0 );
		if (ray.hit.t < BVH_FAR) normal = voxels.GetNormal( ray ), voxHits++;
		Ray occRay( O, D );
		const uint32_t occludedFull = voxels.IsOccluded( occRay ) ? 1u : 0u;
		const float halfT = ray.hit.t < BVH_FAR ? 0.5f * ray.hit.t : BVH_FAR;
		Ray occHalfRay( O, D, halfT );
		const uint32_t occludedHalf = voxels.IsOccluded( occHalfRay ) ? 1u : 0u;
		voxOcc += occludedFull;
		WriteVec3( f, O ); WriteVec3( f, D );
		WriteF32( f, ray.hit.t ); WriteU32( f, ray.hit.prim ); WriteU32( f, (uint32_t)steps );
		WriteVec3( f, normal ); WriteU32( f, occludedFull ); WriteU32( f, occludedHalf );
	}
	printf( "  voxel rays: %u/%u hit, %u occluded\n", voxHits, N_RAYS, voxOcc );

	// Mixed TLAS.
	BVH bvh;
	bvh.settings.useSIMDifavailable = false;
	bvh.context.spawn = nullptr, bvh.context.barrier = nullptr, bvh.context.parallel_for = nullptr;
	bvh.Build( verts, (uint32_t)triCount );
	const bvhvec3 ext = bvh.aabbMax - bvh.aabbMin;
	BLASInstance inst[3] = { BLASInstance( 0 ), BLASInstance( 1 ), BLASInstance( 1 ) };
	for (int i = 0; i < 3; i++) inst[i].mask = 0xFFFF;
	inst[1].transform.cell[0] = ext.x, inst[1].transform.cell[5] = ext.y, inst[1].transform.cell[10] = ext.z;
	inst[1].transform.cell[3] = bvh.aabbMin.x + ext.x * 1.1f, inst[1].transform.cell[7] = bvh.aabbMin.y, inst[1].transform.cell[11] = bvh.aabbMin.z;
	inst[2].transform.cell[0] = ext.x * 0.5f, inst[2].transform.cell[5] = ext.y * 0.5f, inst[2].transform.cell[10] = ext.z * 0.5f;
	inst[2].transform.cell[3] = bvh.aabbMin.x, inst[2].transform.cell[7] = bvh.aabbMin.y, inst[2].transform.cell[11] = bvh.aabbMin.z + ext.z * 1.1f;
	BVH tlas;
	tlas.settings.useSIMDifavailable = false;
	tlas.context.spawn = nullptr, tlas.context.barrier = nullptr, tlas.context.parallel_for = nullptr;
	BVHBase* blasList[2] = { &bvh, &voxels };
	tlas.Build( inst, 3, blasList, 2 );
	std::vector<GenRay> tlasRays = GenerateRays( tlas.aabbMin, tlas.aabbMax, N_RAYS );
	WriteU32( f, 3 );
	for (int i = 0; i < 3; i++)
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
	uint32_t tlasHits = 0, tlasVoxelHits = 0;
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
		if (ray.hit.t < BVH_FAR) { tlasHits++; if (ray.hit.inst > 0) tlasVoxelHits++; }
	}
	printf( "  mixed TLAS: nodes %u, %u/%u hit (%u on voxel instances)\n", tlas.usedNodes, tlasHits, N_RAYS, tlasVoxelHits );
	f.close();
	printf( "  wrote: %s\n", argv[2] );
	free64( verts );
	return 0;
}
