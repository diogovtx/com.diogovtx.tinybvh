// layoutdump.cpp - generates deterministic reference data for tinybvh's (v1.8.0)
// GPU-oriented BVH layout conversions, for validating the tinybvhcs-unity C# port's
// layout-conversion code bit for bit.
//
// Usage: layoutdump <scene.bin> <out.layouts.ref>
//
// Scene file format (input): same as refdump.cpp -
//   int32 triCount
//   triCount*3 vertices, each a 16-byte float4 (x,y,z,w); three consecutive
//   vertices form a triangle.
//
// Each conversion below follows the exact sequence the library's own layout
// Build() methods use (see the line numbers in tiny_bvh.h), rather than an
// ad hoc call order, so this reference data matches what a real caller of
// tinybvh would produce.
//
// Output file format (little-endian, packed, no padding). This comment is the
// authoritative description of the layout for the C# reader.
//
//   char[8]  magic = "TBVHLAY4"
//   u32      triCount
//
//   -- BVH_GPU (Aila & Laine layout) --
//   tinybvh::BVH_GPU gpu; gpu.ConvertFrom( bvh, false ); // compact=false, as
//   BVH_GPU::Build() does (tiny_bvh.h:4929-4938) - this only affects how much
//   spare capacity gpu.bvhNode is allocated with; the number of live nodes
//   written (usedNodes) is determined purely by walking the source tree, so it
//   is unaffected by the compact flag for a freshly built, hole-free source bvh.
//   u32      usedNodes
//            usedNodes * 64 bytes: raw tinybvh::BVH_GPU::BVHNode (sizeof == 64,
//            static_assert'ed below)
//   u32      rayCount (65536)
//            per ray: f32[3] O, f32[3] D, f32 t, f32 u, f32 v, u32 prim (40 bytes).
//            Rays are generated with the same fixed-seed LCG scheme as refdump.cpp,
//            over bvh.aabbMin/aabbMax, with the LCG state freshly reset to 0x12345678
//            before generation. O and D are read back from the tinybvh::Ray after
//            construction (D is therefore normalized). Rays are traced with
//            gpu.Intersect( ray ); t/u/v/prim are read from ray.hit afterward.
//
//   -- MBVH<4> --
//   tinybvh::MBVH<4> m4; m4.ConvertFrom( bvh, true );
//   u32      usedNodes
//            usedNodes * 52 bytes, written FIELD BY FIELD (not the raw struct,
//            which is padded with dummy alignment words) as:
//              f32[3] aabbMin, u32 firstTri, f32[3] aabbMax, u32 triCount,
//              u32 child[4], u32 childCount        (12+4+12+4+16+4 = 52 bytes)
//
//   -- BVH4_GPU --
//   tinybvh::BVH4_GPU g4; g4.ConvertFrom( m4, true ); // matches BVH4_GPU::Build,
//   tiny_bvh.h:5389-5398
//   u32      usedBlocks
//            usedBlocks * 16 bytes: raw data from g4.bvh4Data (bvhvec4 elements)
//   u32      rayCount (65536)
//            per ray, same 40-byte record as above; fresh LCG reset to 0x12345678,
//            traced with g4.Intersect( ray ).
//   u32      usedNodes
//            m4's nodes again, 52 bytes each, same field layout as above, but
//            read AFTER the BVH4_GPU conversion: BVH4_GPU::ConvertFrom copies the
//            MBVH<4> by value (bvh4 = original), which for a plain struct copy
//            aliases the SAME mbvhNode allocation as m4 (the copy is shallow -
//            only the pointer is copied) - so if a conversion reorders or rewrites
//            node contents in place, m4 would observe it. In practice, BVH4_GPU's
//            conversion does not reorder MBVH<4> node contents (see below for the
//            layout that does), so this second dump has been observed to be
//            byte-identical to the first - it is still written again, as specified,
//            for direct validation without assuming that.
//
//   -- Compacted + split BLAS (bvhC) --
//   A SECOND, independent tinybvh::BVH bvhC is built from the same vertex buffer
//   (bvh above is left untouched, so the sections before this one are unaffected),
//   then bvhC.Compact(); bvhC.SplitLeafs( 3 ); is applied - exactly the sequence
//   BVH8_CWBVH::Build() performs (tiny_bvh.h:6023-6036) before converting to
//   MBVH<8>. This section dumps bvhC AFTER both calls, so the C# port can
//   validate its own Compact()/SplitLeafs() against it directly:
//   u32      usedNodes
//            usedNodes * 32 bytes: raw tinybvh::BVH::BVHNode (sizeof == 32,
//            static_assert'ed below)
//   u32      idxCount
//            idxCount * 4 bytes: bvhC.primIdx[]
//
//   -- MBVH<8> --
//   tinybvh::MBVH<8> m8; m8.ConvertFrom( bvhC, true ); // built from the
//   compacted + split bvhC above, not the plain bvh, matching BVH8_CWBVH::Build.
//   u32      usedNodes
//            usedNodes * 68 bytes, written field by field:
//              f32[3] aabbMin, u32 firstTri, f32[3] aabbMax, u32 triCount,
//              u32 child[8], u32 childCount        (12+4+12+4+32+4 = 68 bytes)
//
//   -- CWBVH (BVH8_CWBVH) --
//   tinybvh::BVH8_CWBVH cw; cw.ConvertFrom( m8, true );
//   u32      usedBlocks
//            usedBlocks * 16 bytes: raw data from cw.bvh8Data (bvhvec4 elements)
//   u32      triBlocks
//            triBlocks * 16 bytes: raw data from cw.bvh8Tris (bvhvec4 elements).
//            Formula: triBlocks = 3 * cw.bvh8.idxCount.
//            Derivation: BVH8_CWBVH::ConvertFrom (tiny_bvh.h) walks every CWBVH
//            leaf and, for the uncompressed triangle layout (CWBVH_COMPRESSED_TRIS
//            is not defined here, matching refdump's plain build), writes exactly
//            3 bvhvec4s per triangle (v2-v0, v1-v0, v0-with-index-in-w) advancing
//            its local triDataPtr by 3 per triangle. Every primitive reference in
//            the tree ends up in exactly one leaf, and idxCount is the total count
//            of primitive references, so the total float4s actually written is
//            3 * idxCount (idxCount here is bvhC's, post-SplitLeafs - SplitLeafs
//            only subdivides existing leaves further, so it does not change the
//            number of primitive references, only how they're grouped). NOTE:
//            this is the count of float4s the conversion actually WRITES, not the
//            buffer's allocated capacity - ConvertFrom allocates idxCount * 4
//            float4s (one spare per triangle) as headroom, and BVH8_CWBVH::Save()
//            dumps that whole allocated capacity rather than the used portion;
//            layoutdump intentionally writes only the used 3 * idxCount float4s.
//   u32      usedNodes
//            m8's nodes again, 68 bytes each, same field layout as above, read
//            AFTER the CWBVH conversion. Unlike BVH4_GPU, BVH8_CWBVH::ConvertFrom
//            DOES mutate the MBVH<8> node data in place: it reorders each node's
//            child[] array in place (greedy octant assignment, "orig->child[
//            assignment[i]] = oldNode.child[i]") for the same aliased allocation
//            (cw.bvh8 = original is a shallow copy of m8, sharing mbvhNode). So
//            this second dump is generally NOT byte-identical to the first one.
//
//   -- BVH4_CPU (b4.Build) --
//   tinybvh::BVH4_CPU b4; b4.settings.useSIMDifavailable = false; (context hooks
//   nulled the same way as the other BVH objects above); b4.Build( verts, triCount );
//   BVH4_CPU's constructor sets c_int = 2 and l_quads = true (tiny_bvh.h:1475),
//   and BVH4_CPU::Build (tiny_bvh.h:5660) propagates settings/c_int/c_trav into
//   its own base BVH (b4.bvh4.bvh) and MBVH<4> (b4.bvh4) before calling
//   ConvertFrom( bvh4 ) (tiny_bvh.h:5723-5815), which runs
//   b4.bvh4.bvh.CombineLeafs( 4, firstIdx, 0 ), then b4.bvh4.bvh.SplitLeafs( 4 ),
//   then re-converts b4.bvh4 from that reshaped base (b4.bvh4.ConvertFrom(
//   b4.bvh4.bvh, true )) before building the final BVH4_CPU node/leaf data in
//   b4.bvh4Data. This section dumps all three intermediate/final states:
//   u32      baseUsedNodes
//            baseUsedNodes * 32 bytes: raw tinybvh::BVH::BVHNode from
//            b4.bvh4.bvh.bvhNode - the base scalar BVH AFTER CombineLeafs(4,..)
//            and SplitLeafs(4).
//   u32      baseIdxCount
//            baseIdxCount * 4 bytes: b4.bvh4.bvh.primIdx[]
//   u32      m4UsedNodes
//            m4UsedNodes * 52 bytes, field by field exactly like the MBVH<4>
//            section above, from b4.bvh4.mbvhNode (the MBVH<4> re-converted from
//            the reshaped base above).
//   u32      usedBlocks
//            usedBlocks * 64 bytes: raw data from b4.bvh4Data. NOTE: BVH4_CPU
//            groups its node/leaf data in 64-byte cache-line blocks (BVH4_CPU::
//            CacheLine, sizeof == 64), NOT the 16-byte bvhvec4 blocks used by
//            BVH4_GPU/CWBVH - usedBlocks therefore counts a different unit than
//            the "usedBlocks" fields earlier in this file.
//   BVH4_CPU::Intersect is SSE-only (not compiled/usable in this TINYBVH_NO_SIMD
//   scalar build in a meaningful form), so there is no ray section for this
//   layout - ConvertFrom itself, which this section validates, is plain scalar
//   code (SIMDVEC4/SIMDIVEC4 fall back to 16-byte plain-struct typedefs under
//   TINYBVH_NO_SIMD) and compiles and runs fine.
//
//   -- BVH8_CPU (b8.Build) - "TBVHLAY4" only --
//   tinybvh::BVH8_CPU b8; settings/context prepared like b4 above;
//   b8.Build( verts, triCount ). BVH8_CPU's constructor sets c_int = 2 and
//   l_quads = true (tiny_bvh.h:1584), and BVH8_CPU::Build (tiny_bvh.h:5839)
//   propagates settings/c_int/c_trav into its own base BVH (b8.bvh8.bvh) and
//   MBVH<8> (b8.bvh8), builds that base and - unlike BVH4_CPU::Build - calls
//   b8.bvh8.bvh.Compact() on it, then ConvertFrom( bvh8 ) (tiny_bvh.h:5899),
//   which runs b8.bvh8.bvh.CombineLeafs( 4, firstIdx, 0 ), then
//   b8.bvh8.bvh.SplitLeafs( 4 ), then re-converts b8.bvh8 from that reshaped
//   base (b8.bvh8.ConvertFrom( b8.bvh8.bvh, true )) before building the final
//   BVH8_CPU node/leaf data in b8.bvh8Data. This section dumps all three
//   intermediate/final states, mirroring the BVH4_CPU section:
//   u32      baseUsedNodes
//            baseUsedNodes * 32 bytes: raw tinybvh::BVH::BVHNode from
//            b8.bvh8.bvh.bvhNode - the base scalar BVH AFTER Compact(),
//            CombineLeafs(4,..) and SplitLeafs(4).
//   u32      baseIdxCount
//            baseIdxCount * 4 bytes: b8.bvh8.bvh.primIdx[]
//   u32      m8UsedNodes
//            m8UsedNodes * 68 bytes, field by field exactly like the MBVH<8>
//            section above, from b8.bvh8.mbvhNode. This is the MBVH<8> snapshot
//            right before the node/leaf block conversion: that conversion only
//            reads the MBVH<8>, it never mutates it (unlike BVH8_CWBVH's), so
//            reading it after Build() returns the same data a snapshot taken
//            inside ConvertFrom would.
//   u32      usedBlocks
//            usedBlocks * 64 bytes: raw data from b8.bvh8Data. Like BVH4_CPU,
//            this layout counts 64-byte cache-line blocks (BVH8_CPU::CacheLine,
//            sizeof == 64), not 16-byte bvhvec4 blocks. BVH8_CPU::BVHNode is
//            256 bytes (4 blocks) and BVHTri4Leaf 192 bytes (3 blocks); both
//            sizes are static_assert'ed below. Under TINYBVH_NO_SIMD, SIMDVEC8
//            and SIMDIVEC8 are plain 8-float/8-int structs, so the node has the
//            same byte layout it would have in an AVX2 build.
//   BVH8_CPU::Intersect/IsOccluded are AVX2+FMA only; in this TINYBVH_NO_SIMD
//   build they are fatal-error stubs (tiny_bvh.h:1756-1757), so they are never
//   called here and there is no ray section for this layout. Ray references for
//   the C# port's BVH8_CPU traversal come from the base-BVH rays in the .ref
//   files written by refdump.cpp.
//
// Determinism / build strategy: identical to refdump.cpp - TINYBVH_NO_SIMD and
// NO_THREADED_BUILDS are defined before including tiny_bvh.h, and
// bvh.settings.useSIMDifavailable / bvh.context hooks are set the same way (for
// both bvh and bvhC) so both scalar BVH builds are fully deterministic. The
// layout conversions (BVH_GPU, MBVH<4>, BVH4_GPU, MBVH<8>, BVH8_CWBVH) and
// Compact()/SplitLeafs() are plain, single-threaded passes with no dependency
// on SIMD or threading settings, so no extra settings are needed for them.
//
// BVH_GPU::Intersect and BVH4_GPU::Intersect were checked against tiny_bvh.h and
// are both fully implemented under TINYBVH_NO_SIMD (only IsOccluded() on these
// layouts falls back to FALLBACK_SHADOW_QUERY, which is not used here) - so both
// ray sections above are always populated; the "rayCount = 0" fallback described
// in the tool's spec was not needed.

#define TINYBVH_NO_SIMD
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

static const uint32_t N_RAYS = 65536;
static const float PI = 3.14159265358979323846f;

// Fixed LCG, as specified - do not use rand(). Reset() lets each layout's ray
// batch start from the same fresh state, matching the "fresh seed 0x12345678"
// requirement for both the BVH_GPU and BVH4_GPU ray sections.
static uint32_t s = 0x12345678;
static void ResetSeed() { s = 0x12345678; }
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

static void WriteU32( std::ofstream& f, uint32_t v ) { f.write( (const char*)&v, 4 ); }
static void WriteF32( std::ofstream& f, float v ) { f.write( (const char*)&v, 4 ); }

// Writes one MBVH<M> node pool, field by field (52 bytes for M=4, 68 for M=8) -
// NOT the raw struct, which carries extra alignment padding (see MBVHNode::dummy
// in tiny_bvh.h).
template <int M> static void WriteMBVHNode( std::ofstream& f, const typename MBVH<M>::MBVHNode& n )
{
	WriteVec3( f, n.aabbMin );
	WriteU32( f, n.firstTri );
	WriteVec3( f, n.aabbMax );
	WriteU32( f, n.triCount );
	for (int c = 0; c < M; c++) WriteU32( f, n.child[c] );
	WriteU32( f, n.childCount );
}

// Writes a snapshot of an MBVH node array taken before a downstream conversion mutated it.
template <int M> static void WriteMBVHSnapshot( std::ofstream& f, const std::vector<typename MBVH<M>::MBVHNode>& nodes )
{
	WriteU32( f, (uint32_t)nodes.size() );
	for (size_t i = 0; i < nodes.size(); i++) WriteMBVHNode<M>( f, nodes[i] );
}

template <int M> static void WriteMBVHNodes( std::ofstream& f, const MBVH<M>& m )
{
	WriteU32( f, m.usedNodes );
	for (uint32_t i = 0; i < m.usedNodes; i++)
	{
		const typename MBVH<M>::MBVHNode& n = m.mbvhNode[i];
		WriteVec3( f, n.aabbMin );
		WriteU32( f, n.firstTri );
		WriteVec3( f, n.aabbMax );
		WriteU32( f, n.triCount );
		for (int c = 0; c < M; c++) WriteU32( f, n.child[c] );
		WriteU32( f, n.childCount );
	}
}

// Writes the raw node pool and primitive-index array for a scalar BVH, same
// layout as refdump.cpp's WriteNodesAndIndices.
static void WriteBaseBVH( std::ofstream& f, const BVH& b )
{
	WriteU32( f, b.usedNodes );
	f.write( (const char*)b.bvhNode, (size_t)b.usedNodes * sizeof( BVH::BVHNode ) );
	WriteU32( f, b.idxCount );
	f.write( (const char*)b.primIdx, (size_t)b.idxCount * sizeof( uint32_t ) );
}

// Generates a fresh (reset-seed) ray batch over the given AABB, traces each ray
// against 'shape' (BVH_GPU or BVH4_GPU, both expose Intersect( Ray& )), and
// writes the 65536 40-byte records.
template <typename Shape> static void WriteRaySection( std::ofstream& f, Shape& shape, const bvhvec3& aabbMin, const bvhvec3& aabbMax )
{
	ResetSeed();
	std::vector<GenRay> rays = GenerateRays( aabbMin, aabbMax, N_RAYS );
	WriteU32( f, N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		Ray ray( rays[i].O, rays[i].D );
		shape.Intersect( ray );
		WriteVec3( f, ray.O ); WriteVec3( f, ray.D );
		WriteF32( f, ray.hit.t ); WriteF32( f, ray.hit.u ); WriteF32( f, ray.hit.v );
		WriteU32( f, ray.hit.prim );
	}
}

int main( int argc, char** argv )
{
	static_assert (sizeof( BVH::BVHNode ) == 32, "BVH::BVHNode must be 32 bytes");
	static_assert (sizeof( BVH_GPU::BVHNode ) == 64, "BVH_GPU::BVHNode must be 64 bytes");
	static_assert (sizeof( bvhvec4 ) == 16, "bvhvec4 must be 16 bytes");
	static_assert (sizeof( BVH4_CPU::BVHNode ) == 128, "BVH4_CPU::BVHNode must be 128 bytes");
	static_assert (sizeof( BVHTri4Leaf ) == 192, "BVHTri4Leaf must be 192 bytes");
	static_assert (sizeof( BVH4_CPU::CacheLine ) == 64, "BVH4_CPU::CacheLine must be 64 bytes");
	static_assert (sizeof( BVH8_CPU::BVHNode ) == 256, "BVH8_CPU::BVHNode must be 256 bytes");
	static_assert (sizeof( BVH8_CPU::CacheLine ) == 64, "BVH8_CPU::CacheLine must be 64 bytes");

	if (argc != 3)
	{
		fprintf( stderr, "usage: layoutdump <scene.bin> <out.layouts.ref>\n" );
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

	// --- Build the scalar reference BVH, exactly like refdump.cpp -------
	BVH bvh;
	bvh.settings.useSIMDifavailable = false;
	bvh.context.spawn = nullptr, bvh.context.barrier = nullptr, bvh.context.parallel_for = nullptr;
	bvh.Build( verts, (uint32_t)triCount );

	// --- BVH_GPU ------------------------------------------------------
	// compact=false, matching BVH_GPU::Build (tiny_bvh.h:4929-4938).
	BVH_GPU gpu;
	gpu.ConvertFrom( bvh, false );

	// --- MBVH<4> ------------------------------------------------------
	MBVH<4> m4;
	m4.ConvertFrom( bvh, true );
	// snapshot before BVH4_GPU::ConvertFrom, which may mutate the node array in place
	std::vector<MBVH<4>::MBVHNode> m4Before( m4.mbvhNode, m4.mbvhNode + m4.usedNodes );

	// --- BVH4_GPU -----------------------------------------------------
	BVH4_GPU g4;
	g4.ConvertFrom( m4, true );

	// --- Compacted + split BLAS (bvhC), matching BVH8_CWBVH::Build -----
	// (tiny_bvh.h:6023-6036). A second, independent BVH built from the same
	// vertices, so 'bvh' above (and everything derived from it) is untouched.
	BVH bvhC;
	bvhC.settings.useSIMDifavailable = false;
	bvhC.context.spawn = nullptr, bvhC.context.barrier = nullptr, bvhC.context.parallel_for = nullptr;
	bvhC.Build( verts, (uint32_t)triCount );
	bvhC.Compact();
	bvhC.SplitLeafs( 3 );

	// --- MBVH<8> ------------------------------------------------------
	// built from the compacted + split bvhC, not the plain bvh, matching
	// BVH8_CWBVH::Build.
	MBVH<8> m8;
	m8.ConvertFrom( bvhC, true );
	// snapshot before BVH8_CWBVH::ConvertFrom, which reorders children in place
	std::vector<MBVH<8>::MBVHNode> m8Before( m8.mbvhNode, m8.mbvhNode + m8.usedNodes );

	// --- CWBVH (BVH8_CWBVH) --------------------------------------------
	BVH8_CWBVH cw;
	cw.ConvertFrom( m8, true );
	const uint32_t triBlocks = 3 * cw.bvh8.idxCount;

	// --- BVH4_CPU -------------------------------------------------------
	// built exactly like the library does: BVH4_CPU::Build propagates settings
	// into its own base BVH and MBVH<4>, then CombineLeafs(4,..)+SplitLeafs(4)
	// reshape the base before a final MBVH<4> re-conversion and node/leaf build.
	BVH4_CPU b4;
	b4.settings.useSIMDifavailable = false;
	b4.context.spawn = nullptr, b4.context.barrier = nullptr, b4.context.parallel_for = nullptr;
	b4.Build( verts, (uint32_t)triCount );

	// --- BVH8_CPU -------------------------------------------------------
	// built exactly like the library does: BVH8_CPU::Build propagates settings
	// into its own base BVH and MBVH<8>, builds and Compact()s that base, then
	// CombineLeafs(4,..)+SplitLeafs(4) reshape it before a final MBVH<8>
	// re-conversion and node/leaf block build.
	BVH8_CPU b8;
	b8.settings.useSIMDifavailable = false;
	b8.context.spawn = nullptr, b8.context.barrier = nullptr, b8.context.parallel_for = nullptr;
	b8.Build( verts, (uint32_t)triCount );

	// --- Write output file ----------------------------------------------
	std::ofstream f( outFile, std::ios::binary );
	if (!f)
	{
		fprintf( stderr, "cannot open output file: %s\n", outFile );
		return 1;
	}
	f.write( "TBVHLAY4", 8 );
	WriteU32( f, (uint32_t)triCount );

	// BVH_GPU section
	WriteU32( f, gpu.usedNodes );
	f.write( (const char*)gpu.bvhNode, (size_t)gpu.usedNodes * sizeof( BVH_GPU::BVHNode ) );
	WriteRaySection( f, gpu, bvh.aabbMin, bvh.aabbMax );

	// MBVH<4> section
	WriteMBVHSnapshot<4>( f, m4Before );

	// BVH4_GPU section
	WriteU32( f, g4.usedBlocks );
	f.write( (const char*)g4.bvh4Data, (size_t)g4.usedBlocks * sizeof( bvhvec4 ) );
	WriteRaySection( f, g4, bvh.aabbMin, bvh.aabbMax );
	WriteMBVHNodes( f, m4 ); // m4's nodes again, after BVH4_GPU::ConvertFrom

	// Compacted + split BLAS section (bvhC, after Compact()+SplitLeafs(3))
	WriteBaseBVH( f, bvhC );

	// MBVH<8> section (snapshot taken before the CWBVH conversion)
	WriteMBVHSnapshot<8>( f, m8Before );

	// CWBVH section
	WriteU32( f, cw.usedBlocks );
	f.write( (const char*)cw.bvh8Data, (size_t)cw.usedBlocks * sizeof( bvhvec4 ) );
	WriteU32( f, triBlocks );
	f.write( (const char*)cw.bvh8Tris, (size_t)triBlocks * sizeof( bvhvec4 ) );
	WriteMBVHNodes( f, m8 ); // m8's nodes again, after BVH8_CWBVH::ConvertFrom (mutated)

	// BVH4_CPU section
	WriteBaseBVH( f, b4.bvh4.bvh ); // baseUsedNodes/nodes + baseIdxCount/primIdx, after CombineLeafs(4,..)+SplitLeafs(4)
	WriteMBVHNodes( f, b4.bvh4 );   // m4UsedNodes, 52 bytes each, re-converted MBVH<4>
	WriteU32( f, b4.usedBlocks );
	f.write( (const char*)b4.bvh4Data, (size_t)b4.usedBlocks * sizeof( BVH4_CPU::CacheLine ) );

	// BVH8_CPU section
	WriteBaseBVH( f, b8.bvh8.bvh ); // baseUsedNodes/nodes + baseIdxCount/primIdx, after Compact()+CombineLeafs(4,..)+SplitLeafs(4)
	WriteMBVHNodes( f, b8.bvh8 );   // m8UsedNodes, 68 bytes each, re-converted MBVH<8>
	WriteU32( f, b8.usedBlocks );
	f.write( (const char*)b8.bvh8Data, (size_t)b8.usedBlocks * sizeof( BVH8_CPU::CacheLine ) );

	const std::streamoff writtenSize = f.tellp();
	f.close();

	// --- Self-check: header fields must sum to the exact file size --------
	uint64_t expected = 8 + 4; // magic + triCount
	expected += 4 + (uint64_t)gpu.usedNodes * sizeof( BVH_GPU::BVHNode );
	expected += 4 + (uint64_t)N_RAYS * 40;
	expected += 4 + (uint64_t)m4.usedNodes * 52;
	expected += 4 + (uint64_t)g4.usedBlocks * 16;
	expected += 4 + (uint64_t)N_RAYS * 40;
	expected += 4 + (uint64_t)m4.usedNodes * 52;
	expected += 4 + (uint64_t)bvhC.usedNodes * sizeof( BVH::BVHNode );
	expected += 4 + (uint64_t)bvhC.idxCount * sizeof( uint32_t );
	expected += 4 + (uint64_t)m8.usedNodes * 68;
	expected += 4 + (uint64_t)cw.usedBlocks * 16;
	expected += 4 + (uint64_t)triBlocks * 16;
	expected += 4 + (uint64_t)m8.usedNodes * 68;
	expected += 4 + (uint64_t)b4.bvh4.bvh.usedNodes * sizeof( BVH::BVHNode );
	expected += 4 + (uint64_t)b4.bvh4.bvh.idxCount * sizeof( uint32_t );
	expected += 4 + (uint64_t)b4.bvh4.usedNodes * 52;
	expected += 4 + (uint64_t)b4.usedBlocks * sizeof( BVH4_CPU::CacheLine );
	expected += 4 + (uint64_t)b8.bvh8.bvh.usedNodes * sizeof( BVH::BVHNode );
	expected += 4 + (uint64_t)b8.bvh8.bvh.idxCount * sizeof( uint32_t );
	expected += 4 + (uint64_t)b8.bvh8.usedNodes * 68;
	expected += 4 + (uint64_t)b8.usedBlocks * sizeof( BVH8_CPU::CacheLine );
	const bool sizeOk = expected == (uint64_t)writtenSize;

	// --- Ray hit ratios, for the summary (re-trace, cheap relative to build) ---
	ResetSeed();
	std::vector<GenRay> gpuRays = GenerateRays( bvh.aabbMin, bvh.aabbMax, N_RAYS );
	uint32_t gpuHits = 0;
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		Ray ray( gpuRays[i].O, gpuRays[i].D );
		gpu.Intersect( ray );
		if (ray.hit.t < BVH_FAR) gpuHits++;
	}
	ResetSeed();
	std::vector<GenRay> g4Rays = GenerateRays( bvh.aabbMin, bvh.aabbMax, N_RAYS );
	uint32_t g4Hits = 0;
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		Ray ray( g4Rays[i].O, g4Rays[i].D );
		g4.Intersect( ray );
		if (ray.hit.t < BVH_FAR) g4Hits++;
	}

	// --- Summary ----------------------------------------------------------
	printf( "scene: %s\n", sceneFile );
	printf( "  triCount:            %d\n", triCount );
	printf( "  BVH_GPU usedNodes:   %u\n", gpu.usedNodes );
	printf( "  BVH_GPU hit ratio:   %.2f%% (%u/%u)\n", 100.0 * gpuHits / N_RAYS, gpuHits, N_RAYS );
	printf( "  MBVH<4> usedNodes:   %u\n", m4.usedNodes );
	printf( "  BVH4_GPU usedBlocks: %u\n", g4.usedBlocks );
	printf( "  BVH4_GPU hit ratio:  %.2f%% (%u/%u)\n", 100.0 * g4Hits / N_RAYS, g4Hits, N_RAYS );
	printf( "  bvhC usedNodes:      %u (after Compact+SplitLeafs(3))\n", bvhC.usedNodes );
	printf( "  bvhC idxCount:       %u\n", bvhC.idxCount );
	printf( "  MBVH<8> usedNodes:   %u\n", m8.usedNodes );
	printf( "  CWBVH usedBlocks:    %u\n", cw.usedBlocks );
	printf( "  CWBVH triBlocks:     %u (== 3 * idxCount, idxCount=%u)\n", triBlocks, cw.bvh8.idxCount );
	printf( "  BVH4_CPU baseUsedNodes: %u\n", b4.bvh4.bvh.usedNodes );
	printf( "  BVH4_CPU baseIdxCount:  %u\n", b4.bvh4.bvh.idxCount );
	printf( "  BVH4_CPU m4UsedNodes:   %u\n", b4.bvh4.usedNodes );
	printf( "  BVH4_CPU usedBlocks:    %u (64-byte cache lines)\n", b4.usedBlocks );
	printf( "  BVH8_CPU baseUsedNodes: %u\n", b8.bvh8.bvh.usedNodes );
	printf( "  BVH8_CPU baseIdxCount:  %u\n", b8.bvh8.bvh.idxCount );
	printf( "  BVH8_CPU m8UsedNodes:   %u\n", b8.bvh8.usedNodes );
	printf( "  BVH8_CPU usedBlocks:    %u (64-byte cache lines)\n", b8.usedBlocks );
	printf( "  file size:           %lld bytes (expected %llu) - %s\n",
		(long long)writtenSize, (unsigned long long)expected, sizeOk ? "PASS" : "MISMATCH" );
	printf( "  wrote: %s\n", outFile );

	free64( verts );
	return sizeOk ? 0 : 1;
}
