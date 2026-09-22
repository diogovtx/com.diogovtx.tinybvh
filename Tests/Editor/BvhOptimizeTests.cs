using System;
using System.IO;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Tests for the tree-rotation optimizer (Bvh.Optimize and BvhVerbose) against the two
	/// optimizer sections of the reference dump produced by Tools~/RefDump/refdump.cpp. The
	/// optimizer runs under Burst (its entry points carry BurstCompile( CompileSynchronously =
	/// true ), so a direct call is compiled), which is what makes the node comparison bit-exact:
	/// Mono evaluates scalar float math in double. Tests are ignored when reference data is missing.
	/// </summary>
	public class BvhOptimizeTests
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

		/// <summary>Port of the C++ pipeline of reference section 2: SplitLeafs( 1 ), Optimize, MergeLeafs.</summary>
		static void OptimizeSplitMerge( ref Bvh bvh, uint iterations )
		{
			BvhVerbose verbose = BvhVerbose.Create( Allocator.Persistent );
			try
			{
				verbose.ConvertFrom( ref bvh );
				verbose.SplitLeafs( 1 );
				verbose.Optimize( iterations, false );
				verbose.MergeLeafs();
				bvh.ConvertFrom( ref verbose );
			}
			finally
			{
				verbose.Dispose();
			}
		}

		/// <summary>Compares a freshly optimized tree against one of the two reference sections.</summary>
		unsafe void CheckStructure( string sceneName, bool splitMerge )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 to fetch scenes and Tools~/RefDump/run_all.bat to generate reference data" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				RefDumpFile refFile = RefDumpFile.Load( refPath );
				RefDumpFile.OptimizeSection section = splitMerge ? refFile.OptMerged : refFile.OptPlain;
				string label = splitMerge ? "split/opt/merge" : "optimize";

				bvh.Build( verts, triCount );
				float sahBefore = bvh.SahCost();
				if ( splitMerge )
				{
					OptimizeSplitMerge( ref bvh, section.Iterations );
				}
				else
				{
					bvh.Optimize( section.Iterations, false );
				}
				float sahAfter = bvh.SahCost();

				Assert.AreEqual( section.UsedNodes, bvh.UsedNodes, $"{label} UsedNodes ({section.Iterations} iterations)" );

				int nodeMismatches = 0;
				for ( uint i = 0; i < bvh.UsedNodes; i++ )
				{
					if ( !NodesEqualExact( bvh.Nodes[ i ], section.Nodes[ i ] ) )
					{
						if ( nodeMismatches == 0 )
						{
							TestContext.WriteLine( $"  first differing node {i} ({section.Iterations} iterations):" );
							TestContext.WriteLine( $"    ref: {Describe( section.Nodes[ i ] )}" );
							TestContext.WriteLine( $"    got: {Describe( bvh.Nodes[ i ] )}" );
						}
						nodeMismatches++;
					}
				}

				// The reference dumps PrimCount() entries, not idxCount: MergeLeafs installs a
				// fresh index array of which only the first PrimCount() entries are written.
				int primCount = bvh.PrimCount();
				TestContext.WriteLine( $"{sceneName} {label}: iterations {section.Iterations}, nodes {bvh.UsedNodes}, prims {primCount}, node mismatches {nodeMismatches}/{bvh.UsedNodes}" );
				TestContext.WriteLine( $"{sceneName} {label}: SAH before {sahBefore}, after {sahAfter} (ref {section.SahCostBefore} -> {section.SahCost})" );
				Assert.AreEqual( 0, nodeMismatches, $"{label} node structure mismatch count" );

				Assert.AreEqual( section.PrimIdx.Length, primCount, $"{label} prim count" );
				int idxMismatches = 0;
				for ( int i = 0; i < section.PrimIdx.Length; i++ )
				{
					if ( section.PrimIdx[ i ] != bvh.PrimIdx[ i ] )
					{
						if ( idxMismatches == 0 )
						{
							TestContext.WriteLine( $"  first differing primIdx {i}: ref {section.PrimIdx[ i ]}, got {bvh.PrimIdx[ i ]}" );
						}
						idxMismatches++;
					}
				}
				Assert.AreEqual( 0, idxMismatches, $"{label} primIdx mismatch count" );
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
		public void Optimize_MatchesReferenceStructure( string sceneName )
		{
			CheckStructure( sceneName, false );
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void OptimizeSplitMerge_MatchesReferenceStructure( string sceneName )
		{
			CheckStructure( sceneName, true );
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void IntersectOptimized_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 and Tools~/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				RefDumpFile refFile = RefDumpFile.Load( refPath );
				RefDumpFile.OptimizeSection section = refFile.OptPlain;

				bvh.Build( verts, triCount );
				bvh.Optimize( section.Iterations, false );

				int mismatches = 0, ties = 0, grazing = 0, occlusionMismatches = 0, hitCount = 0;
				double tSum = 0.0, refTSum = 0.0;

				for ( int i = 0; i < section.Rays.Length; i++ )
				{
					RefDumpFile.RayHit rh = section.Rays[ i ];
					Ray ray = new Ray( rh.O, rh.D );
					bvh.Intersect( ref ray );

					bool refHit = rh.T < BvhConstants.Far;
					bool gotHit = ray.Hit.T < BvhConstants.Far;
					bool sameT = refHit && gotHit && math.abs( ray.Hit.T - rh.T ) <= 1e-4f * math.max( math.abs( rh.T ), 1e-12f );
					if ( refHit != gotHit || ( refHit && !sameT ) )
					{
						// A hit within a hair of a triangle edge or vertex is decided by last-bit
						// rounding; the two sides can legitimately disagree and then hit different
						// geometry.
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

				float hitRatio = ( float )hitCount / section.Rays.Length;
				TestContext.WriteLine( $"{sceneName} optimized rays: hit ratio {hitRatio:P2}, mismatches {mismatches}/{section.Rays.Length}, same-distance ties {ties}, grazing {grazing}, occlusion mismatches {occlusionMismatches}/{section.Rays.Length * 2}" );
				Assert.AreEqual( 0, mismatches, "optimized intersect mismatches" );
				Assert.AreEqual( 0, occlusionMismatches, "optimized occlusion mismatches" );

				double scale = Math.Max( Math.Abs( refTSum ), 1e-12 );
				Assert.That( tSum, Is.EqualTo( refTSum ).Within( 1e-3 * scale ), "optimized sum of T over hit rays" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>
		/// The derived layouts have no reference data of their own - the C++ forwarders just
		/// optimize the base BVH and re-convert - so check that their trees still find the same
		/// geometry after Optimize. Distances are compared, not primitive indices: coincident
		/// triangles at the same distance can swap places when the traversal order changes.
		/// </summary>
		[Test]
		public void Optimize_DerivedLayoutsStillTraceTheSameGeometry()
		{
			const string sceneName = "suzanne";
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 and Tools~/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh4Cpu bvh4 = Bvh4Cpu.Create( Allocator.Persistent );
			Bvh8Cpu bvh8 = Bvh8Cpu.Create( Allocator.Persistent );
			try
			{
				RefDumpFile refFile = RefDumpFile.Load( refPath );
				uint iterations = refFile.OptPlain.Iterations;

				bvh4.Build( verts, triCount );
				bvh8.Build( verts, triCount );
				float sah4Before = bvh4.SahCost(), sah8Before = bvh8.SahCost();

				float[] t4 = new float[ refFile.Rays.Length ];
				float[] t8 = new float[ refFile.Rays.Length ];
				for ( int i = 0; i < refFile.Rays.Length; i++ )
				{
					Ray r4 = new Ray( refFile.Rays[ i ].O, refFile.Rays[ i ].D );
					bvh4.Intersect( ref r4 );
					t4[ i ] = r4.Hit.T;
					Ray r8 = new Ray( refFile.Rays[ i ].O, refFile.Rays[ i ].D );
					bvh8.IntersectScalarPath( ref r8 );
					t8[ i ] = r8.Hit.T;
				}

				bvh4.Optimize( iterations, false );
				bvh8.Optimize( iterations, false );

				int mismatches4 = 0, mismatches8 = 0;
				for ( int i = 0; i < refFile.Rays.Length; i++ )
				{
					Ray r4 = new Ray( refFile.Rays[ i ].O, refFile.Rays[ i ].D );
					bvh4.Intersect( ref r4 );
					if ( math.abs( r4.Hit.T - t4[ i ] ) > 1e-4f * math.max( math.abs( t4[ i ] ), 1e-12f ) )
					{
						mismatches4++;
					}
					Ray r8 = new Ray( refFile.Rays[ i ].O, refFile.Rays[ i ].D );
					bvh8.IntersectScalarPath( ref r8 );
					if ( math.abs( r8.Hit.T - t8[ i ] ) > 1e-4f * math.max( math.abs( t8[ i ] ), 1e-12f ) )
					{
						mismatches8++;
					}
				}

				TestContext.WriteLine( $"{sceneName} Bvh4Cpu: SAH {sah4Before} -> {bvh4.SahCost()}, distance mismatches {mismatches4}/{refFile.Rays.Length}" );
				TestContext.WriteLine( $"{sceneName} Bvh8Cpu: SAH {sah8Before} -> {bvh8.SahCost()}, distance mismatches {mismatches8}/{refFile.Rays.Length}" );
				Assert.AreEqual( 0, mismatches4, "Bvh4Cpu hit distances changed after Optimize" );
				Assert.AreEqual( 0, mismatches8, "Bvh8Cpu hit distances changed after Optimize" );
			}
			finally
			{
				bvh8.Dispose();
				bvh4.Dispose();
				verts.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Optimize_SahCostTracksReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 and Tools~/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				RefDumpFile refFile = RefDumpFile.Load( refPath );
				RefDumpFile.OptimizeSection section = refFile.OptPlain;

				bvh.Build( verts, triCount );
				float before = bvh.SahCost();
				bvh.Optimize( section.Iterations, false );
				float after = bvh.SahCost();

				float refRatio = section.SahCost / section.SahCostBefore;
				float gotRatio = after / before;
				TestContext.WriteLine( $"{sceneName}: SAH before {before}, after {after} ({100f * ( gotRatio - 1f ):0.00}%) over {section.Iterations} iterations; reference {section.SahCostBefore} -> {section.SahCost} ({100f * ( refRatio - 1f ):0.00}%)" );

				// tinybvh steers each reinsertion with SAHCostUp, a local sum of surface areas
				// along two root paths, not with the tree's SAH cost, so a kept rotation is not
				// guaranteed to lower the global cost. v1.8.0 gains 12% on Crytek Sponza but
				// loses 0.3% on bunny, so assert that the port tracks the C++ rather than an
				// absolute improvement, and require the improvement only where the C++ has one.
				Assert.That( gotRatio, Is.EqualTo( refRatio ).Within( 1e-4f ), "SAH before/after ratio versus the C++ reference" );
				if ( refRatio <= 1f )
				{
					Assert.LessOrEqual( after, before, "the optimized tree should not have a worse SAH cost than the binned build" );
				}

				// Optimize does not reshape leafs, so the flags survive the round trip through
				// BVH_Verbose exactly as CopyBasePropertiesFrom leaves them in the C++.
				Assert.IsTrue( bvh.Refittable, "an optimized binned tree stays refittable" );
				Assert.IsFalse( bvh.MayHaveHoles, "Optimize does not introduce holes" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}
	}
}
