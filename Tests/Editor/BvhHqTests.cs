using System;
using System.IO;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Tests for the SBVH builder (Bvh.UseSpatialSplits) against the SBVH section of the reference
	/// dump produced by Tools/RefDump/refdump.cpp. The builder runs under Burst (the entry points
	/// carry BurstCompile( CompileSynchronously = true ), so a direct call is compiled), which is
	/// what makes the node comparison bit-exact: Mono evaluates scalar float math in double.
	/// Tests are ignored when their reference data is missing.
	/// </summary>
	public class BvhHqTests
	{
		/// <summary>Barycentric distance to a triangle edge below which a hit counts as grazing.</summary>
		const float GrazingEps = 1e-3f;

		/// <summary>True when the hit lies within GrazingEps of a triangle edge or vertex, where last-bit rounding decides hit or miss.</summary>
		static bool IsGrazing( float u, float v )
		{
			return u < GrazingEps || v < GrazingEps || ( 1f - u - v ) < GrazingEps;
		}

		static bool BitsEqual( float a, float b )
		{
			return math.asuint( a ) == math.asuint( b );
		}

		static bool NodesEqualExact( BvhNode a, BvhNode b )
		{
			return BitsEqual( a.AabbMin.x, b.AabbMin.x )
				&& BitsEqual( a.AabbMin.y, b.AabbMin.y )
				&& BitsEqual( a.AabbMin.z, b.AabbMin.z )
				&& a.LeftFirst == b.LeftFirst
				&& BitsEqual( a.AabbMax.x, b.AabbMax.x )
				&& BitsEqual( a.AabbMax.y, b.AabbMax.y )
				&& BitsEqual( a.AabbMax.z, b.AabbMax.z )
				&& a.TriCount == b.TriCount;
		}

		static string Describe( BvhNode n )
		{
			return $"min {n.AabbMin} max {n.AabbMax} leftFirst {n.LeftFirst} triCount {n.TriCount}";
		}

		static bool TryGetPaths( string sceneName, out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( sceneName + ".ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		static Bvh BuildSbvh( NativeArray<float4> verts, uint triCount )
		{
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			bvh.UseSpatialSplits = true;
			bvh.Build( verts, triCount );
			return bvh;
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void BuildHq_MatchesReferenceStructure( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 to fetch scenes and Tools/RefDump/run_all.bat to generate reference data" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = BuildSbvh( verts, triCount );
			try
			{
				RefDumpFile refFile = RefDumpFile.Load( refPath );

				Assert.AreEqual( refFile.SbvhUsedNodes, bvh.UsedNodes, "SBVH UsedNodes" );
				Assert.IsFalse( bvh.Refittable, "an SBVH must not be marked refittable" );
				Assert.IsFalse( bvh.MayHaveHoles, "the SBVH node list must be hole-free after Compact" );

				int nodeMismatches = 0;
				for ( uint i = 0; i < bvh.UsedNodes; i++ )
				{
					if ( !NodesEqualExact( bvh.Nodes[ i ], refFile.SbvhNodes[ i ] ) )
					{
						if ( nodeMismatches == 0 )
						{
							TestContext.WriteLine( $"  first differing node {i}:" );
							TestContext.WriteLine( $"    ref: {Describe( refFile.SbvhNodes[ i ] )}" );
							TestContext.WriteLine( $"    got: {Describe( bvh.Nodes[ i ] )}" );
						}
						nodeMismatches++;
					}
				}

				// The reference dumps PrimCount() entries, not idxCount: BuildHQ ends with Compact,
				// which leaves the tail of the freshly allocated index array uninitialized.
				int primCount = bvh.PrimCount();
				TestContext.WriteLine( $"{sceneName} sbvh: nodes {bvh.UsedNodes}, prims {primCount} ({( float )primCount / triCount:0.000}x), node mismatches {nodeMismatches}/{bvh.UsedNodes}" );
				Assert.AreEqual( 0, nodeMismatches, "SBVH node structure mismatch count" );

				Assert.AreEqual( refFile.SbvhPrimIdx.Length, primCount, "SBVH prim count" );
				int idxMismatches = 0;
				for ( int i = 0; i < refFile.SbvhPrimIdx.Length; i++ )
				{
					if ( refFile.SbvhPrimIdx[ i ] != bvh.PrimIdx[ i ] )
					{
						if ( idxMismatches == 0 )
						{
							TestContext.WriteLine( $"  first differing primIdx {i}: ref {refFile.SbvhPrimIdx[ i ]}, got {bvh.PrimIdx[ i ]}" );
						}
						idxMismatches++;
					}
				}
				Assert.AreEqual( 0, idxMismatches, "SBVH primIdx mismatch count" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void IntersectHq_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = BuildSbvh( verts, triCount );
			try
			{
				RefDumpFile refFile = RefDumpFile.Load( refPath );

				int mismatches = 0, ties = 0, grazing = 0, occlusionMismatches = 0, hitCount = 0;
				double tSum = 0.0, refTSum = 0.0;

				for ( int i = 0; i < refFile.SbvhRays.Length; i++ )
				{
					RefDumpFile.RayHit rh = refFile.SbvhRays[ i ];
					Ray ray = new Ray( rh.O, rh.D );
					bvh.Intersect( ref ray );

					bool refHit = rh.T < BvhConstants.Far;
					bool gotHit = ray.Hit.T < BvhConstants.Far;
					bool sameT = refHit && gotHit && math.abs( ray.Hit.T - rh.T ) <= 1e-4f * math.max( math.abs( rh.T ), 1e-12f );
					if ( refHit != gotHit || ( refHit && !sameT ) )
					{
						// A hit within a hair of a triangle edge or vertex is decided by last-bit
						// rounding; the two sides can legitimately disagree and then hit different
						// geometry. With spatial splits the same triangle also sits in several
						// leaves, so the traversal order around such a hit differs more often.
						if ( ( refHit && IsGrazing( rh.U, rh.V ) ) || ( gotHit && IsGrazing( ray.Hit.U, ray.Hit.V ) ) )
						{
							grazing++;
						}
						else
						{
							if ( mismatches < 3 )
							{
								TestContext.WriteLine( $"  mismatch ray {i}: O {rh.O} D {rh.D} ref t {rh.T} u {rh.U} v {rh.V} prim {rh.Prim} | got t {ray.Hit.T} u {ray.Hit.U} v {ray.Hit.V} prim {ray.Hit.Prim}" );
							}
							mismatches++;
						}
					}
					else if ( refHit && ray.Hit.Prim != rh.Prim )
					{
						ties++;
					}

					if ( refHit && gotHit )
					{
						hitCount++;
						refTSum += rh.T;
						tSum += ray.Hit.T;
					}

					bool occludedFull = bvh.IsOccluded( new Ray( rh.O, rh.D ) );
					if ( occludedFull != ( rh.OccludedFull != 0 ) )
					{
						occlusionMismatches++;
					}
					float halfT = rh.T < BvhConstants.Far ? 0.5f * rh.T : BvhConstants.Far;
					bool occludedHalf = bvh.IsOccluded( new Ray( rh.O, rh.D, halfT ) );
					if ( occludedHalf != ( rh.OccludedHalf != 0 ) )
					{
						occlusionMismatches++;
					}
				}

				float hitRatio = ( float )hitCount / refFile.SbvhRays.Length;
				TestContext.WriteLine( $"{sceneName} sbvh rays: hit ratio {hitRatio:P2}, mismatches {mismatches}/{refFile.SbvhRays.Length}, same-distance ties {ties}, grazing {grazing}, occlusion mismatches {occlusionMismatches}/{refFile.SbvhRays.Length * 2}" );
				Assert.AreEqual( 0, mismatches, "SBVH intersect mismatches" );
				Assert.AreEqual( 0, occlusionMismatches, "SBVH occlusion mismatches" );

				double scale = Math.Max( Math.Abs( refTSum ), 1e-12 );
				Assert.That( tSum, Is.EqualTo( refTSum ).Within( 1e-3 * scale ), "SBVH sum of T over hit rays" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		[Test]
		public void BuildHq_SahCostIsNotWorseThanBinnedBuild()
		{
			const string sceneName = "cryteksponza";
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh standard = Bvh.Create( Allocator.Persistent );
			Bvh sbvh = default;
			try
			{
				standard.Build( verts, triCount );
				float standardCost = standard.SahCost();

				sbvh = BuildSbvh( verts, triCount );
				float sbvhCost = sbvh.SahCost();

				RefDumpFile refFile = RefDumpFile.Load( refPath );
				TestContext.WriteLine( $"{sceneName}: binned SAH cost {standardCost} (ref {refFile.SahCost}), SBVH SAH cost {sbvhCost} (ref {refFile.SbvhSahCost})" );
				TestContext.WriteLine( $"{sceneName}: binned nodes {standard.UsedNodes}, SBVH nodes {sbvh.UsedNodes}, SBVH prims {sbvh.PrimCount()} for {triCount} triangles" );

				Assert.LessOrEqual( sbvhCost, standardCost, "the SBVH should not have a worse SAH cost than the binned build" );
			}
			finally
			{
				if ( sbvh.IsCreated )
				{
					sbvh.Dispose();
				}
				standard.Dispose();
				verts.Dispose();
			}
		}

		[Test]
		public void Refit_RejectsAnSbvh()
		{
			const string sceneName = "suzanne";
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = BuildSbvh( verts, triCount );
			try
			{
				bool threw = false;
				try
				{
					bvh.Refit();
				}
				catch ( InvalidOperationException )
				{
					threw = true;
				}
				Assert.IsTrue( threw, "Refit must reject an SBVH, matching the C++ refittable flag" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>
		/// Regression test for the sample crash when spatial splits were switched on: an SBVH has more
		/// index entries than the binned tree of the same mesh, and the CWBVH converter used to size
		/// its triangle buffer only on the first conversion, so reusing the object overran it.
		/// </summary>
		[Test]
		public void Cwbvh_ReconvertsAfterSpatialSplits()
		{
			const string sceneName = "suzanne";
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh = Mbvh.Create( 8, Allocator.Persistent );
			BvhCwbvh cwbvh = BvhCwbvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				bvh.Compact();
				bvh.SplitLeafs( 3 );
				mbvh.ConvertFrom( ref bvh );
				cwbvh.ConvertFrom( ref mbvh );
				uint binnedIdxCount = cwbvh.IdxCount;

				bvh.UseSpatialSplits = true;
				bvh.Build( verts, triCount );
				bvh.Compact();
				bvh.SplitLeafs( 3 );
				mbvh.ConvertFrom( ref bvh );
				cwbvh.ConvertFrom( ref mbvh );

				TestContext.WriteLine( $"{sceneName}: index entries binned {binnedIdxCount}, sbvh {cwbvh.IdxCount}; tri blocks {cwbvh.TriBlocks}/{cwbvh.AllocatedTriBlocks}, node blocks {cwbvh.UsedBlocks}/{cwbvh.AllocatedBlocks}" );
				Assert.Greater( cwbvh.IdxCount, binnedIdxCount, "the SBVH must have more index entries for this test to exercise buffer growth" );
				Assert.LessOrEqual( cwbvh.TriBlocks, cwbvh.AllocatedTriBlocks, "triangle blocks exceed the triangle buffer" );
				Assert.LessOrEqual( cwbvh.UsedBlocks, cwbvh.AllocatedBlocks, "node blocks exceed the node buffer" );
			}
			finally
			{
				cwbvh.Dispose();
				mbvh.Dispose();
				bvh.Dispose();
				verts.Dispose();
			}
		}
	}
}
