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
	/// Tests for the AVX binned-SAH builder (Bvh.BuildAvx, the port of BVH::BuildAVX) against the
	/// AVX-build section of the reference dump produced by Tools~/RefDump/simddump.cpp, the only
	/// dump tool compiled with /arch:AVX2 and therefore the only one whose BVH::Build routes to
	/// BuildAVX. That builder bins differently from the scalar one, so its tree is not the tree the
	/// rest of the suite checks; the dump carries it separately.
	///
	/// The builder runs under Burst - its entry points carry BurstCompile( CompileSynchronously =
	/// true ), so a direct call is compiled - which is what makes the node comparison bit-exact:
	/// Mono evaluates scalar float math in double and does not have the intrinsics. Tests are
	/// ignored when their reference data is missing.
	/// The timeout is raised well above the NUnit default: building and then tracing 65536 rays
	/// through the largest scene takes minutes.
	/// </summary>
	[Timeout( 1800000 )]
	public class BvhAvxBuildTests
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

		static string Describe( BvhNode n )
		{
			return $"min {n.AabbMin} max {n.AabbMax} leftFirst {n.LeftFirst} triCount {n.TriCount}";
		}

		/// <summary>
		/// Traces the reference rays through the AVX-built tree inside Burst, where float
		/// expressions are not widened to double. Mirrors the tracing loop of simddump.cpp: the
		/// closest hit, an occlusion query over the full ray, and a second one shortened to half
		/// the hit distance.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		private struct IntersectJob : IJobParallelFor
		{
			public Bvh Bvh;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> OccludedFull;
			public NativeArray<int> OccludedHalf;

			public void Execute( int i )
			{
				Ray ray = new Ray( Origins[ i ], Directions[ i ] );
				Bvh.Intersect( ref ray );
				Hits[ i ] = ray.Hit;
				OccludedFull[ i ] = Bvh.IsOccluded( new Ray( Origins[ i ], Directions[ i ] ) ) ? 1 : 0;
				float halfT = ray.Hit.T < BvhConstants.Far ? 0.5f * ray.Hit.T : BvhConstants.Far;
				OccludedHalf[ i ] = Bvh.IsOccluded( new Ray( Origins[ i ], Directions[ i ], halfT ) ) ? 1 : 0;
			}
		}

		static bool TryGetPaths( string sceneName, out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( sceneName + ".simd.ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		static void RequireBurstAndAvx()
		{
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );
			Assert.IsTrue( Bvh.AvxBuilderSupported, "the AVX builder is not available; Burst did not compile the AVX intrinsics for this machine" );
		}

		/// <summary>
		/// The tree of BVH::BuildAVX: node pool, index array, root bounds and SAH cost, all
		/// bit-exact. The AVX builder is a separate builder, not a faster spelling of the scalar
		/// one, so none of this is expected to match the ".ref" binned tree.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void BuildAvx_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 and Tools~/RefDump/run_all.bat" );
			}
			RequireBurstAndAvx();

			SimdDumpFile refFile = SimdDumpFile.Load( refPath );
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.BuildAvx( verts, triCount );

				Assert.AreEqual( refFile.TriCount, triCount, "triCount" );
				Assert.AreEqual( refFile.AvxUsedNodes, bvh.UsedNodes, "UsedNodes" );
				Assert.AreEqual( refFile.AvxNodes.Length, ( int )bvh.UsedNodes, "node count" );
				List<int> nodeDiffs = new List<int>();
				for ( int i = 0; i < refFile.AvxNodes.Length; i++ )
				{
					if ( !BvhNodesEqual( bvh.Nodes[ i ], refFile.AvxNodes[ i ] ) )
					{
						if ( nodeDiffs.Count == 0 )
						{
							TestContext.WriteLine( $"  first differing node {i}:" );
							TestContext.WriteLine( $"    ref: {Describe( refFile.AvxNodes[ i ] )}" );
							TestContext.WriteLine( $"    got: {Describe( bvh.Nodes[ i ] )}" );
						}
						nodeDiffs.Add( i );
					}
				}
				Assert.AreEqual( refFile.AvxPrimIdx.Length, ( int )bvh.IdxCount, "IdxCount" );
				List<int> idxDiffs = new List<int>();
				for ( int i = 0; i < refFile.AvxPrimIdx.Length; i++ )
				{
					if ( bvh.PrimIdx[ i ] != refFile.AvxPrimIdx[ i ] )
					{
						idxDiffs.Add( i );
					}
				}
				float sahCost = bvh.SahCost();
				TestContext.WriteLine( $"{sceneName} BuildAvx: nodes {bvh.UsedNodes}, node mismatches {nodeDiffs.Count}/{refFile.AvxNodes.Length}, primIdx mismatches {idxDiffs.Count}/{refFile.AvxPrimIdx.Length}, SAH {sahCost} (ref {refFile.AvxSahCost})" );
				Assert.AreEqual( 0, nodeDiffs.Count, $"node mismatch: {Describe( nodeDiffs )}" );
				Assert.AreEqual( 0, idxDiffs.Count, $"primIdx mismatch: {Describe( idxDiffs )}" );
				Assert.IsTrue( BitsEqual( bvh.AabbMin, refFile.AvxAabbMin ), $"AabbMin: ref {refFile.AvxAabbMin}, got {bvh.AabbMin}" );
				Assert.IsTrue( BitsEqual( bvh.AabbMax, refFile.AvxAabbMax ), $"AabbMax: ref {refFile.AvxAabbMax}, got {bvh.AabbMax}" );
				Assert.IsTrue( BitsEqual( sahCost, refFile.AvxSahCost ), $"SAH cost: ref {refFile.AvxSahCost}, got {sahCost}" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>
		/// BVH::Intersect and the two IsOccluded queries over the AVX-built tree, for all 65536
		/// reference rays. The traversal is the same code the scalar tests exercise; this checks
		/// that the tree it walks is the reference one all the way down.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void BuildAvxRays_MatchReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 and Tools~/RefDump/run_all.bat" );
			}
			RequireBurstAndAvx();

			SimdDumpFile refFile = SimdDumpFile.Load( refPath );
			int rayCount = refFile.Rays.Length;
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> hits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			NativeArray<int> occludedFull = new NativeArray<int>( rayCount, Allocator.Persistent );
			NativeArray<int> occludedHalf = new NativeArray<int>( rayCount, Allocator.Persistent );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.BuildAvx( verts, triCount );
				for ( int i = 0; i < rayCount; i++ )
				{
					origins[ i ] = refFile.Rays[ i ].O;
					directions[ i ] = refFile.Rays[ i ].D;
				}

				IntersectJob job = new IntersectJob
				{
					Bvh = bvh,
					Origins = origins,
					Directions = directions,
					Hits = hits,
					OccludedFull = occludedFull,
					OccludedHalf = occludedHalf
				};
				job.Schedule( rayCount, 64 ).Complete();

				Compare( $"{sceneName} BuildAvx", refFile, hits, occludedFull, occludedHalf );
			}
			finally
			{
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
		/// Compares against the AVX-build section of the dump. The traversal is the same algorithm
		/// on both sides, so the hits are expected to agree bit for bit; a hit within a hair of a
		/// triangle edge is still allowed to differ, since the reference is a separately compiled
		/// build and last-bit rounding decides hit or miss there.
		/// </summary>
		static void Compare( string label, SimdDumpFile refFile, NativeArray<Intersection> hits, NativeArray<int> occludedFull, NativeArray<int> occludedHalf )
		{
			int rayCount = refFile.Rays.Length;
			int mismatches = 0, grazing = 0, fullMismatches = 0, halfMismatches = 0;
			for ( int i = 0; i < rayCount; i++ )
			{
				SimdDumpFile.Hit rh = refFile.AvxHits[ i ];
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

		/// <summary>
		/// Bvh.Build routes to the AVX builder when UseSimdIfAvailable is set, as BVH::Build does
		/// for settings.useSIMDifavailable. Only the two trees are compared; the reference tests
		/// above already pin BuildAvx itself to the dump.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void Build_WithUseSimdIfAvailable_MatchesBuildAvx( string sceneName )
		{
			string binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			if ( !File.Exists( binPath ) )
			{
				Assert.Ignore( $"missing {binPath}; run Tools~/fetch.ps1" );
			}
			RequireBurstAndAvx();

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh direct = Bvh.Create( Allocator.Persistent );
			Bvh dispatched = Bvh.Create( Allocator.Persistent );
			try
			{
				direct.BuildAvx( verts, triCount );
				dispatched.UseSimdIfAvailable = true;
				dispatched.Build( verts, triCount );

				Assert.AreEqual( direct.UsedNodes, dispatched.UsedNodes, "UsedNodes" );
				List<int> nodeDiffs = new List<int>();
				for ( uint i = 0; i < direct.UsedNodes; i++ )
				{
					if ( !BvhNodesEqual( direct.Nodes[ i ], dispatched.Nodes[ i ] ) )
					{
						nodeDiffs.Add( ( int )i );
					}
				}
				Assert.AreEqual( 0, nodeDiffs.Count, $"node mismatch: {Describe( nodeDiffs )}" );
				Assert.AreEqual( direct.IdxCount, dispatched.IdxCount, "IdxCount" );
				List<int> idxDiffs = new List<int>();
				for ( uint i = 0; i < direct.IdxCount; i++ )
				{
					if ( direct.PrimIdx[ i ] != dispatched.PrimIdx[ i ] )
					{
						idxDiffs.Add( ( int )i );
					}
				}
				Assert.AreEqual( 0, idxDiffs.Count, $"primIdx mismatch: {Describe( idxDiffs )}" );
			}
			finally
			{
				dispatched.Dispose();
				direct.Dispose();
				verts.Dispose();
			}
		}
	}
}
