using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Tests for the CPU traversal of the CWBVH layout against the reference dump of
	/// Tools/RefDump/simddump.cpp, the only dump tool compiled with SIMD enabled - the C++
	/// BVH8_CWBVH::Intersect only exists under BVH_USEAVX, because it needs __lzcnt / __popcnt.
	/// The builders in that tool still run the scalar binned path, so the dump also carries the
	/// base tree; <see cref="BaseTree_MatchesReference"/> checks it first, so a drift between the
	/// SIMD and scalar reference builds cannot be mistaken for a traversal bug.
	/// Tests are ignored when their reference data is missing.
	/// </summary>
	public class BvhCwbvhCpuTests
	{
		/// <summary>Barycentric distance to a triangle edge below which a hit counts as grazing.</summary>
		const float GrazingEps = 1e-3f;

		/// <summary>Number of differing entries listed in an assertion message before it is truncated.</summary>
		const int MaxReportedDiffs = 4;

		/// <summary>True when the hit lies within GrazingEps of a triangle edge or vertex, where last-bit rounding decides hit or miss.</summary>
		static bool IsGrazing( float u, float v )
		{
			return u < GrazingEps || v < GrazingEps || ( 1f - u - v ) < GrazingEps;
		}

		static bool BitsEqual( float a, float b )
		{
			return math.asuint( a ) == math.asuint( b );
		}

		static bool BitsEqual( float3 a, float3 b )
		{
			return BitsEqual( a.x, b.x ) && BitsEqual( a.y, b.y ) && BitsEqual( a.z, b.z );
		}

		static bool BvhNodesEqual( BvhNode a, BvhNode b )
		{
			return BitsEqual( a.AabbMin, b.AabbMin ) && a.LeftFirst == b.LeftFirst
				&& BitsEqual( a.AabbMax, b.AabbMax ) && a.TriCount == b.TriCount;
		}

		static string Describe( List<int> diffs )
		{
			if ( diffs.Count == 0 )
			{
				return "none";
			}
			string list = string.Join( ", ", diffs.GetRange( 0, math.min( diffs.Count, MaxReportedDiffs ) ) );
			return $"{diffs.Count} differing, first at [{list}]";
		}

		/// <summary>FNV-1a 64, byte-wise, exactly as Tools/RefDump/simddump.cpp computes it.</summary>
		static unsafe ulong Fnv1a64( void* data, long bytes )
		{
			byte* p = ( byte* )data;
			ulong hash = 14695981039346656037ul;
			for ( long i = 0; i < bytes; i++ )
			{
				hash = ( hash ^ p[ i ] ) * 1099511628211ul;
			}
			return hash;
		}

		/// <summary>
		/// Traces the reference rays through the CWBVH inside Burst, where float expressions are
		/// not widened to double. Mirrors the tracing loop of simddump.cpp: the closest hit, an
		/// occlusion query over the full ray, and a second one shortened to half the hit distance.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		private struct IntersectJob : IJobParallelFor
		{
			public BvhCwbvh Cwbvh;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> OccludedFull;
			public NativeArray<int> OccludedHalf;

			public void Execute( int i )
			{
				Ray ray = new Ray( Origins[ i ], Directions[ i ] );
				Cwbvh.Intersect( ref ray );
				Hits[ i ] = ray.Hit;
				OccludedFull[ i ] = Cwbvh.IsOccluded( new Ray( Origins[ i ], Directions[ i ] ) ) ? 1 : 0;
				float halfT = ray.Hit.T < BvhConstants.Far ? 0.5f * ray.Hit.T : BvhConstants.Far;
				OccludedHalf[ i ] = Cwbvh.IsOccluded( new Ray( Origins[ i ], Directions[ i ], halfT ) ) ? 1 : 0;
			}
		}

		static bool TryGetPaths( string sceneName, out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( sceneName + ".simd.ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		/// <summary>
		/// The dump tool is the only one compiled with /arch:AVX2, so its binned builder could in
		/// principle produce a different tree from the scalar tools. This asserts it does not,
		/// which is what makes the traversal comparison below meaningful.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void BaseTree_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			SimdDumpFile refFile = SimdDumpFile.Load( refPath );
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );

				Assert.AreEqual( refFile.TriCount, triCount, "triCount" );
				Assert.AreEqual( refFile.BaseNodes.Length, ( int )bvh.UsedNodes, "UsedNodes" );
				List<int> nodeDiffs = new List<int>();
				for ( int i = 0; i < refFile.BaseNodes.Length; i++ )
				{
					if ( !BvhNodesEqual( bvh.Nodes[ i ], refFile.BaseNodes[ i ] ) )
					{
						nodeDiffs.Add( i );
					}
				}
				Assert.AreEqual( refFile.BasePrimIdx.Length, ( int )bvh.IdxCount, "IdxCount" );
				List<int> idxDiffs = new List<int>();
				for ( int i = 0; i < refFile.BasePrimIdx.Length; i++ )
				{
					if ( bvh.PrimIdx[ i ] != refFile.BasePrimIdx[ i ] )
					{
						idxDiffs.Add( i );
					}
				}
				TestContext.WriteLine( $"{sceneName} simd base: node mismatches {nodeDiffs.Count}/{refFile.BaseNodes.Length}, primIdx mismatches {idxDiffs.Count}/{refFile.BasePrimIdx.Length}" );
				Assert.AreEqual( 0, nodeDiffs.Count, $"base node mismatch: {Describe( nodeDiffs )}" );
				Assert.AreEqual( 0, idxDiffs.Count, $"base primIdx mismatch: {Describe( idxDiffs )}" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>
		/// The block data the traversal reads, against the two FNV-1a hashes in the dump. The
		/// layout itself is compared block by block by BvhLayoutTests.Cwbvh_MatchesReference; this
		/// is the check that the SIMD reference build produced that same layout.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void Convert_MatchesReferenceHashes( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			SimdDumpFile refFile = SimdDumpFile.Load( refPath );
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh8 = Mbvh.Create( 8, Allocator.Persistent );
			BvhCwbvh cwbvh = BvhCwbvh.Create( Allocator.Persistent );
			try
			{
				Build( verts, triCount, ref bvh, ref mbvh8, ref cwbvh );

				Assert.AreEqual( ( int )refFile.CwbvhUsedBlocks, ( int )cwbvh.UsedBlocks, "UsedBlocks" );
				ulong nodeHash = Fnv1a64( cwbvh.Data, ( long )cwbvh.UsedBlocks * 16 );
				ulong triHash = Fnv1a64( cwbvh.Tris, ( long )mbvh8.IdxCount * 64 );
				TestContext.WriteLine( $"{sceneName} Cwbvh: blocks {cwbvh.UsedBlocks}, tri slots {mbvh8.IdxCount}, nodeHash 0x{nodeHash:x16}, triHash 0x{triHash:x16}" );
				Assert.AreEqual( refFile.CwbvhNodeHash, nodeHash, $"node hash: got 0x{nodeHash:x16}, expected 0x{refFile.CwbvhNodeHash:x16}" );
				Assert.AreEqual( refFile.CwbvhTriHash, triHash, $"tri hash: got 0x{triHash:x16}, expected 0x{refFile.CwbvhTriHash:x16}" );
			}
			finally
			{
				cwbvh.Dispose();
				mbvh8.Dispose();
				bvh.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>
		/// The ported BVH8_CWBVH::Intersect and the two FALLBACK_SHADOW_QUERY occlusion tests,
		/// over all 65536 reference rays.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void IntersectJob_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			SimdDumpFile refFile = SimdDumpFile.Load( refPath );
			int rayCount = refFile.Rays.Length;
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> hits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			NativeArray<int> occludedFull = new NativeArray<int>( rayCount, Allocator.Persistent );
			NativeArray<int> occludedHalf = new NativeArray<int>( rayCount, Allocator.Persistent );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh8 = Mbvh.Create( 8, Allocator.Persistent );
			BvhCwbvh cwbvh = BvhCwbvh.Create( Allocator.Persistent );
			try
			{
				Build( verts, triCount, ref bvh, ref mbvh8, ref cwbvh );
				for ( int i = 0; i < rayCount; i++ )
				{
					origins[ i ] = refFile.Rays[ i ].O;
					directions[ i ] = refFile.Rays[ i ].D;
				}

				IntersectJob job = new IntersectJob
				{
					Cwbvh = cwbvh,
					Origins = origins,
					Directions = directions,
					Hits = hits,
					OccludedFull = occludedFull,
					OccludedHalf = occludedHalf
				};
				job.Schedule( rayCount, 64 ).Complete();

				Compare( $"{sceneName} Cwbvh cpu", refFile, hits, occludedFull, occludedHalf );
			}
			finally
			{
				cwbvh.Dispose();
				mbvh8.Dispose();
				bvh.Dispose();
				verts.Dispose();
				origins.Dispose();
				directions.Dispose();
				hits.Dispose();
				occludedFull.Dispose();
				occludedHalf.Dispose();
			}
		}

		/// <summary>
		/// Port of BVH8_CWBVH::Build: the base BVH is compacted and its leaves split to at most
		/// three primitives before the MBVH&lt;8&gt; is derived, because the CWBVH leaf encoding
		/// cannot express more. Same sequence as BvhLayoutTests.Cwbvh_MatchesReference.
		/// </summary>
		static void Build( NativeArray<float4> verts, uint triCount, ref Bvh bvh, ref Mbvh mbvh8, ref BvhCwbvh cwbvh )
		{
			bvh.Build( verts, triCount );
			bvh.Compact();
			bvh.SplitLeafs( 3 );
			mbvh8.ConvertFrom( ref bvh, true );
			cwbvh.ConvertFrom( ref mbvh8, true );
		}

		/// <summary>
		/// Compares against the CWBVH section of the dump. The traversal is the same algorithm on
		/// both sides, so the hits are expected to agree bit for bit; a hit within a hair of a
		/// triangle edge is still allowed to differ, since the reference is a separately compiled
		/// build and last-bit rounding decides hit or miss there.
		/// </summary>
		static void Compare( string label, SimdDumpFile refFile, NativeArray<Intersection> hits, NativeArray<int> occludedFull, NativeArray<int> occludedHalf )
		{
			int rayCount = refFile.Rays.Length;
			int mismatches = 0, grazing = 0, fullMismatches = 0, halfMismatches = 0;
			for ( int i = 0; i < rayCount; i++ )
			{
				SimdDumpFile.Hit rh = refFile.CwbvhHits[ i ];
				Intersection hit = hits[ i ];
				bool same = BitsEqual( hit.T, rh.T ) && BitsEqual( hit.U, rh.U ) && BitsEqual( hit.V, rh.V ) && hit.Prim == rh.Prim;
				if ( !same )
				{
					bool refHit = rh.T < BvhConstants.Far;
					bool gotHit = hit.T < BvhConstants.Far;
					if ( ( refHit && IsGrazing( rh.U, rh.V ) ) || ( gotHit && IsGrazing( hit.U, hit.V ) ) )
					{
						grazing++;
					}
					else
					{
						if ( mismatches < 3 )
						{
							TestContext.WriteLine( $"  mismatch ray {i}: O {refFile.Rays[ i ].O} D {refFile.Rays[ i ].D} ref t {rh.T} u {rh.U} v {rh.V} prim {rh.Prim} | got t {hit.T} u {hit.U} v {hit.V} prim {hit.Prim}" );
						}
						mismatches++;
					}
				}
				if ( ( occludedFull[ i ] != 0 ) != ( rh.OccludedFull != 0 ) )
				{
					fullMismatches++;
				}
				if ( ( occludedHalf[ i ] != 0 ) != ( rh.OccludedHalf != 0 ) )
				{
					halfMismatches++;
				}
			}
			TestContext.WriteLine( $"{label}: mismatches {mismatches}/{rayCount}, grazing {grazing}, occlusion mismatches full {fullMismatches} half {halfMismatches}" );
			Assert.AreEqual( 0, mismatches, label + " intersect mismatches" );
			Assert.AreEqual( 0, fullMismatches, label + " full-length occlusion mismatches" );
			Assert.AreEqual( 0, halfMismatches, label + " half-distance occlusion mismatches" );
		}
	}
}
