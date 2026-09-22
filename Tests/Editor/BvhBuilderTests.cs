using System;
using System.IO;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Tests for the alternative builders (BuildQuick, full-sweep, presplitting, SBVH bin
	/// settings) and for BVH::Build's postOptimize path, against the tree blocks of the feature
	/// dump produced by Tools/RefDump/featdump.cpp and, for postOptimize, the OptPlain section of
	/// the reference dump produced by Tools/RefDump/refdump.cpp. The builders run under Burst
	/// (their entry points carry BurstCompile( CompileSynchronously = true ), so a direct call is
	/// compiled), which is what makes the node comparison bit-exact: Mono evaluates scalar float
	/// math in double. Tests are ignored when their reference data is missing.
	/// The timeout is raised well above the NUnit default: BuildQuick produces a mid-point tree with
	/// no SAH at all, and tracing the block's 65536 rays through it takes minutes on the largest
	/// scene.
	/// </summary>
	[Timeout( 1800000 )]
	public class BvhBuilderTests
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

		static bool TryGetPaths( string sceneName, out string binPath, out string featPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			featPath = BvhSceneFile.TestDataPath( sceneName + ".feat.ref" );
			return File.Exists( binPath ) && File.Exists( featPath );
		}

		static bool TryGetRefPaths( string sceneName, out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( sceneName + ".ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		/// <summary>Compares a freshly built tree against one tree block of the feature dump.</summary>
		static unsafe void CheckTree( string sceneName, string label, ref Bvh bvh, FeatDumpFile.TreeBlock block, int primIdxCount )
		{
			Assert.AreEqual( block.UsedNodes, bvh.UsedNodes, $"{label} UsedNodes" );

			int nodeMismatches = 0;
			for ( uint i = 0; i < bvh.UsedNodes; i++ )
			{
				if ( !NodesEqualExact( bvh.Nodes[ i ], block.Nodes[ i ] ) )
				{
					if ( nodeMismatches == 0 )
					{
						TestContext.WriteLine( $"  first differing node {i}:" );
						TestContext.WriteLine( $"    ref: {Describe( block.Nodes[ i ] )}" );
						TestContext.WriteLine( $"    got: {Describe( bvh.Nodes[ i ] )}" );
					}
					nodeMismatches++;
				}
			}
			TestContext.WriteLine( $"{sceneName} {label}: nodes {bvh.UsedNodes}, node mismatches {nodeMismatches}/{bvh.UsedNodes}" );
			Assert.AreEqual( 0, nodeMismatches, $"{label} node structure mismatch count" );

			Assert.AreEqual( block.PrimIdx.Length, primIdxCount, $"{label} prim count" );
			int idxMismatches = 0;
			for ( int i = 0; i < block.PrimIdx.Length; i++ )
			{
				if ( block.PrimIdx[ i ] != bvh.PrimIdx[ i ] )
				{
					if ( idxMismatches == 0 )
					{
						TestContext.WriteLine( $"  first differing primIdx {i}: ref {block.PrimIdx[ i ]}, got {bvh.PrimIdx[ i ]}" );
					}
					idxMismatches++;
				}
			}
			Assert.AreEqual( 0, idxMismatches, $"{label} primIdx mismatch count" );

			Assert.IsTrue( BitsEqual( bvh.AabbMin.x, block.AabbMin.x ), $"{label} AabbMin.x" );
			Assert.IsTrue( BitsEqual( bvh.AabbMin.y, block.AabbMin.y ), $"{label} AabbMin.y" );
			Assert.IsTrue( BitsEqual( bvh.AabbMin.z, block.AabbMin.z ), $"{label} AabbMin.z" );
			Assert.IsTrue( BitsEqual( bvh.AabbMax.x, block.AabbMax.x ), $"{label} AabbMax.x" );
			Assert.IsTrue( BitsEqual( bvh.AabbMax.y, block.AabbMax.y ), $"{label} AabbMax.y" );
			Assert.IsTrue( BitsEqual( bvh.AabbMax.z, block.AabbMax.z ), $"{label} AabbMax.z" );

			Assert.IsTrue( BitsEqual( bvh.SahCost(), block.SahCost ), $"{label} SAH cost: ref {block.SahCost}, got {bvh.SahCost()}" );
		}

		/// <summary>Traces the block's rays and compares hits and occlusion, as BvhHqTests does.</summary>
		static void CheckRays( string sceneName, string label, ref Bvh bvh, FeatDumpFile.TreeBlock block )
		{
			int mismatches = 0, ties = 0, grazing = 0, occlusionMismatches = 0, hitCount = 0;
			double tSum = 0.0, refTSum = 0.0;

			for ( int i = 0; i < block.Rays.Length; i++ )
			{
				RefDumpFile.RayHit rh = block.Rays[ i ];
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

			float hitRatio = ( float )hitCount / block.Rays.Length;
			TestContext.WriteLine( $"{sceneName} {label} rays: hit ratio {hitRatio:P2}, mismatches {mismatches}/{block.Rays.Length}, same-distance ties {ties}, grazing {grazing}, occlusion mismatches {occlusionMismatches}/{block.Rays.Length * 2}" );
			Assert.AreEqual( 0, mismatches, $"{label} intersect mismatches" );
			Assert.AreEqual( 0, occlusionMismatches, $"{label} occlusion mismatches" );

			double scale = Math.Max( Math.Abs( refTSum ), 1e-12 );
			Assert.That( tSum, Is.EqualTo( refTSum ).Within( 1e-3 * scale ), $"{label} sum of T over hit rays" );
		}

		/// <summary>Covers the "quick" block: BVH::BuildQuick, mid-point splits with no SAH.</summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void BuildQuick_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string featPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {featPath}; run TestData/fetch.ps1 to fetch scenes and Tools/RefDump/run_all.bat to generate reference data" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			bvh.BuildQuick( verts, triCount );
			try
			{
				FeatDumpFile featFile = FeatDumpFile.Load( featPath );
				CheckTree( sceneName, "quick", ref bvh, featFile.Quick, ( int )bvh.IdxCount );
				CheckRays( sceneName, "quick", ref bvh, featFile.Quick );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>Covers the "fullsweep" block: settings.useFullSweep with the binned builder.</summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void BuildFullSweep_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string featPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {featPath}; run TestData/fetch.ps1 to fetch scenes and Tools/RefDump/run_all.bat to generate reference data" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			bvh.UseFullSweep = true;
			bvh.Build( verts, triCount );
			try
			{
				FeatDumpFile featFile = FeatDumpFile.Load( featPath );
				CheckTree( sceneName, "fullsweep", ref bvh, featFile.FullSweep, ( int )bvh.IdxCount );
				CheckRays( sceneName, "fullsweep", ref bvh, featFile.FullSweep );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>Covers the "presplit" block: settings.usePresplitting, binned, post pass on.</summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Presplit_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string featPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {featPath}; run TestData/fetch.ps1 to fetch scenes and Tools/RefDump/run_all.bat to generate reference data" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			bvh.UsePresplitting = true;
			bvh.Build( verts, triCount );
			try
			{
				FeatDumpFile featFile = FeatDumpFile.Load( featPath );
				CheckTree( sceneName, "presplit", ref bvh, featFile.Presplit, ( int )bvh.IdxCount );
				Assert.IsFalse( bvh.Refittable, "a presplit tree must not be marked refittable" );
				Assert.AreEqual( featFile.Presplit.PrimIdx.Length, ( int )bvh.TriCount, "presplitting makes TriCount the fragment count" );
				CheckRays( sceneName, "presplit", ref bvh, featFile.Presplit );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>Covers the "presplit+fullsweep" block: usePresplitting with useFullSweep.</summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void PresplitFullSweep_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string featPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {featPath}; run TestData/fetch.ps1 to fetch scenes and Tools/RefDump/run_all.bat to generate reference data" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			bvh.UsePresplitting = true;
			bvh.UseFullSweep = true;
			bvh.Build( verts, triCount );
			try
			{
				FeatDumpFile featFile = FeatDumpFile.Load( featPath );
				CheckTree( sceneName, "presplit+fullsweep", ref bvh, featFile.PresplitFullSweep, ( int )bvh.IdxCount );
				CheckRays( sceneName, "presplit+fullsweep", ref bvh, featFile.PresplitFullSweep );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>Covers the "presplit-nopostpass" block: usePresplitting with presplitPostPass = false, binned.</summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void PresplitNoPostPass_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string featPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {featPath}; run TestData/fetch.ps1 to fetch scenes and Tools/RefDump/run_all.bat to generate reference data" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			bvh.UsePresplitting = true;
			bvh.PresplitPostPass = false;
			bvh.Build( verts, triCount );
			try
			{
				FeatDumpFile featFile = FeatDumpFile.Load( featPath );
				CheckTree( sceneName, "presplit-nopostpass", ref bvh, featFile.PresplitNoPostPass, ( int )bvh.IdxCount );
				CheckRays( sceneName, "presplit-nopostpass", ref bvh, featFile.PresplitNoPostPass );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>Covers the "hqbins" block: SBVH with hqbvhbins = 32 and hqbvhoddeven = true.</summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void HqBins_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string featPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {featPath}; run TestData/fetch.ps1 to fetch scenes and Tools/RefDump/run_all.bat to generate reference data" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			bvh.UseSpatialSplits = true;
			bvh.HqBvhBins = 32;
			bvh.HqBvhOddEven = true;
			bvh.Build( verts, triCount );
			try
			{
				FeatDumpFile featFile = FeatDumpFile.Load( featPath );
				CheckTree( sceneName, "hqbins", ref bvh, featFile.HqBins, bvh.PrimCount() );
				CheckRays( sceneName, "hqbins", ref bvh, featFile.HqBins );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>Covers BVH::Build's postOptimize path against the OptPlain section of the ".ref" dump.</summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void PostOptimize_MatchesOptPlain( string sceneName )
		{
			if ( !TryGetRefPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				RefDumpFile refFile = RefDumpFile.Load( refPath );
				RefDumpFile.OptimizeSection section = refFile.OptPlain;

				bvh.PostOptimize = true;
				bvh.OptimizeIterations = section.Iterations;
				bvh.Build( verts, triCount );

				Assert.AreEqual( section.UsedNodes, bvh.UsedNodes, $"postoptimize UsedNodes ({section.Iterations} iterations)" );

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

				int primCount = bvh.PrimCount();
				TestContext.WriteLine( $"{sceneName} postoptimize: iterations {section.Iterations}, nodes {bvh.UsedNodes}, prims {primCount}, node mismatches {nodeMismatches}/{bvh.UsedNodes}" );
				TestContext.WriteLine( $"{sceneName} postoptimize: SAH before {section.SahCostBefore}, after {section.SahCost} (got {bvh.SahCost()})" );
				Assert.AreEqual( 0, nodeMismatches, "postoptimize node structure mismatch count" );

				Assert.AreEqual( section.PrimIdx.Length, primCount, "postoptimize prim count" );
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
				Assert.AreEqual( 0, idxMismatches, "postoptimize primIdx mismatch count" );

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
				TestContext.WriteLine( $"{sceneName} postoptimize rays: hit ratio {hitRatio:P2}, mismatches {mismatches}/{section.Rays.Length}, same-distance ties {ties}, grazing {grazing}, occlusion mismatches {occlusionMismatches}/{section.Rays.Length * 2}" );
				Assert.AreEqual( 0, mismatches, "postoptimize intersect mismatches" );
				Assert.AreEqual( 0, occlusionMismatches, "postoptimize occlusion mismatches" );

				double scale = Math.Max( Math.Abs( refTSum ), 1e-12 );
				Assert.That( tSum, Is.EqualTo( refTSum ).Within( 1e-3 * scale ), "postoptimize sum of T over hit rays" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}
	}
}
