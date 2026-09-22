using System;
using System.IO;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Tests for the stochastic variant of the tree-rotation optimizer, BVH::Optimize( n, false,
	/// true ), against the "stochastic" block of the ".feat.ref" dump. That block is the only
	/// tinybvh code path that draws from the C runtime's rand(), so it also pins down the MSVC
	/// generator the port reproduces in BvhVerbose.RandomState: the reference tool never seeds,
	/// so the sequence starts from state 1. The deterministic path is re-checked here as well,
	/// because both now share one loop. Tests are ignored when reference data is missing.
	/// </summary>
	public class BvhStochasticOptimizeTests
	{
		/// <summary>Barycentric distance to a triangle edge below which a hit counts as grazing.</summary>
		const float GrazingEps = 1e-3f;

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

		static void RequireData( string sceneName, string refExtension, out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( sceneName + refExtension );
			if ( !File.Exists( binPath ) || !File.Exists( refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 to fetch scenes and Tools~/RefDump/run_all.bat to generate reference data" );
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void OptimizeStochastic_MatchesReferenceStructure( string sceneName )
		{
			RequireData( sceneName, ".feat.ref", out string binPath, out string featPath );

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				FeatDumpFile featFile = FeatDumpFile.Load( featPath );
				FeatDumpFile.TreeBlock block = featFile.Stochastic;

				bvh.Build( verts, triCount );
				float sahBefore = bvh.SahCost();
				Assert.That( sahBefore, Is.EqualTo( featFile.StochasticSahBefore ).Within( 1e-4f * math.abs( featFile.StochasticSahBefore ) ),
					"SAH cost of the binned build the stochastic block starts from" );

				bvh.Optimize( featFile.StochasticIterations, false, true );
				float sahAfter = bvh.SahCost();

				Assert.AreEqual( block.UsedNodes, bvh.UsedNodes, $"stochastic UsedNodes ({featFile.StochasticIterations} iterations)" );

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

				// The reference dumps PrimCount() entries, for the reason given in refdump.cpp's
				// optimizer note.
				int primCount = bvh.PrimCount();
				TestContext.WriteLine( $"{sceneName} stochastic: iterations {featFile.StochasticIterations}, nodes {bvh.UsedNodes}, prims {primCount}, node mismatches {nodeMismatches}/{bvh.UsedNodes}" );
				TestContext.WriteLine( $"{sceneName} stochastic: SAH before {sahBefore}, after {sahAfter} (ref {featFile.StochasticSahBefore} -> {block.SahCost})" );
				Assert.AreEqual( 0, nodeMismatches, "stochastic node structure mismatch count" );

				Assert.AreEqual( block.PrimIdx.Length, primCount, "stochastic prim count" );
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
				Assert.AreEqual( 0, idxMismatches, "stochastic primIdx mismatch count" );

				Assert.That( sahAfter, Is.EqualTo( block.SahCost ).Within( 1e-4f * math.abs( block.SahCost ) ), "stochastic SAH cost" );

				// Trace the block's rays: the tree is bit-exact by now, so this checks that the
				// converted-back layout is traversable and finds the same geometry.
				int mismatches = 0, grazing = 0, ties = 0, hitCount = 0;
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
					}
				}

				TestContext.WriteLine( $"{sceneName} stochastic rays: hit ratio {( float )hitCount / block.Rays.Length:P2}, mismatches {mismatches}/{block.Rays.Length}, grazing {grazing}, same-distance ties {ties}" );
				Assert.AreEqual( 0, mismatches, "stochastic intersect mismatches" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>
		/// Regression for the deterministic path: adding the stochastic branches to the shared
		/// loop must not change what Optimize( n, false, false ) produces.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void OptimizeDeterministic_IsUnchanged( string sceneName )
		{
			RequireData( sceneName, ".ref", out string binPath, out string refPath );

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				RefDumpFile refFile = RefDumpFile.Load( refPath );
				RefDumpFile.OptimizeSection section = refFile.OptPlain;

				bvh.Build( verts, triCount );
				bvh.Optimize( section.Iterations, false, false );

				Assert.AreEqual( section.UsedNodes, bvh.UsedNodes, "deterministic UsedNodes" );
				int nodeMismatches = 0;
				for ( uint i = 0; i < bvh.UsedNodes; i++ )
				{
					if ( !NodesEqualExact( bvh.Nodes[ i ], section.Nodes[ i ] ) )
					{
						nodeMismatches++;
					}
				}
				int idxMismatches = 0;
				for ( int i = 0; i < section.PrimIdx.Length; i++ )
				{
					if ( section.PrimIdx[ i ] != bvh.PrimIdx[ i ] )
					{
						idxMismatches++;
					}
				}
				TestContext.WriteLine( $"{sceneName} deterministic: nodes {bvh.UsedNodes}, node mismatches {nodeMismatches}, primIdx mismatches {idxMismatches}" );
				Assert.AreEqual( 0, nodeMismatches, "deterministic node structure mismatch count" );
				Assert.AreEqual( section.PrimIdx.Length, bvh.PrimCount(), "deterministic prim count" );
				Assert.AreEqual( 0, idxMismatches, "deterministic primIdx mismatch count" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>
		/// The generator is part of the contract: the reference data only matches when the port
		/// reproduces MSVC's rand() from its initial state.
		/// </summary>
		[Test]
		public void RandomState_ReproducesTheMsvcGenerator()
		{
			// The first ten values of rand() in a freshly started MSVC process.
			int[] expected = { 41, 18467, 6334, 26500, 19169, 15724, 11478, 29358, 26962, 24464 };
			uint state = 1;
			for ( int i = 0; i < expected.Length; i++ )
			{
				state = ( state * 214013u ) + 2531011u;
				int value = ( int )( ( state >> 16 ) & 0x7fffu );
				Assert.AreEqual( expected[ i ], value, $"rand() call {i}" );
			}

			BvhVerbose verbose = BvhVerbose.Create( Allocator.Persistent );
			try
			{
				Assert.AreEqual( 1u, verbose.RandomState, "BvhVerbose.Create seeds the generator to 1" );
			}
			finally
			{
				verbose.Dispose();
			}
		}
	}
}
