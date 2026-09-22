using System;
using System.Diagnostics;
using System.IO;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Tests for Bvh.UseThreadedBuild. A threaded build hands out node indices from an atomic counter,
	/// so its node numbering is not reproducible and cannot be compared against the C++ dump; what is
	/// reproducible is the tree shape, so the threaded trees are compared against the serial ones by
	/// walking both in the same depth-first order. The reference ray sets are traced through the
	/// threaded trees as well, and a build below BvhConstants.MtBuildThreshold is checked to be the
	/// serial build, byte for byte.
	/// </summary>
	public class BvhThreadedTests
	{
		/// <summary>cryteksponza has 262267 primitives, well above the threading threshold.</summary>
		const string LargeScene = "cryteksponza";
		/// <summary>suzanne has 15488 primitives, below the threading threshold.</summary>
		const string SmallScene = "suzanne";
		const float MaxMismatchFraction = 0.0005f;
		/// <summary>Barycentric distance to a triangle edge below which a hit counts as grazing.</summary>
		const float GrazingEps = 1e-3f;

		static bool TryGetPaths( string sceneName, out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( sceneName + ".ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		static bool BitsEqual( float a, float b )
		{
			return math.asuint( a ) == math.asuint( b );
		}

		static bool BoundsEqual( BvhNode a, BvhNode b )
		{
			return BitsEqual( a.AabbMin.x, b.AabbMin.x )
				&& BitsEqual( a.AabbMin.y, b.AabbMin.y )
				&& BitsEqual( a.AabbMin.z, b.AabbMin.z )
				&& BitsEqual( a.AabbMax.x, b.AabbMax.x )
				&& BitsEqual( a.AabbMax.y, b.AabbMax.y )
				&& BitsEqual( a.AabbMax.z, b.AabbMax.z );
		}

		static bool IsGrazing( float u, float v )
		{
			return u < GrazingEps || v < GrazingEps || ( 1f - u - v ) < GrazingEps;
		}

		static Bvh BuildScene( NativeArray<float4> verts, uint triCount, bool spatialSplits, bool threaded )
		{
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			bvh.UseSpatialSplits = spatialSplits;
			bvh.UseThreadedBuild = threaded;
			bvh.Build( verts, triCount );
			return bvh;
		}

		/// <summary>Compares the primitive index sets of two matching leaves, order-independently.</summary>
		static unsafe void AssertSameLeafPrims( ref Bvh serial, BvhNode a, ref Bvh threaded, BvhNode b, uint nodeA, string label )
		{
			int count = ( int )a.TriCount;
			uint[] primsA = new uint[ count ];
			uint[] primsB = new uint[ count ];
			for ( int i = 0; i < count; i++ )
			{
				primsA[ i ] = serial.PrimIdx[ a.LeftFirst + i ];
				primsB[ i ] = threaded.PrimIdx[ b.LeftFirst + i ];
			}
			Array.Sort( primsA );
			Array.Sort( primsB );
			for ( int i = 0; i < count; i++ )
			{
				Assert.AreEqual( primsA[ i ], primsB[ i ], $"{label}: leaf primitive sets differ at serial node {nodeA}, entry {i}" );
			}
		}

		/// <summary>
		/// Walks both trees from the root in the same depth-first order and asserts that every node
		/// pair has bit-identical bounds, the same leaf/interior kind and, for leaves, the same
		/// multiset of primitive indices.
		/// </summary>
		static unsafe void AssertSameTree( ref Bvh serial, ref Bvh threaded, string label )
		{
			const int MaxDepth = 256;
			uint* stackA = stackalloc uint[ MaxDepth ];
			uint* stackB = stackalloc uint[ MaxDepth ];
			uint stackPtr = 0, a = 0, b = 0;
			int nodes = 0, leaves = 0, prims = 0;
			while ( true )
			{
				BvhNode na = serial.Nodes[ a ];
				BvhNode nb = threaded.Nodes[ b ];
				nodes++;
				Assert.IsTrue( BoundsEqual( na, nb ), $"{label}: bounds differ at serial node {a} / threaded node {b}" );
				Assert.AreEqual( na.IsLeaf, nb.IsLeaf, $"{label}: leaf/interior differs at serial node {a} / threaded node {b}" );
				if ( na.IsLeaf )
				{
					Assert.AreEqual( na.TriCount, nb.TriCount, $"{label}: leaf size differs at serial node {a} / threaded node {b}" );
					AssertSameLeafPrims( ref serial, na, ref threaded, nb, a, label );
					leaves++;
					prims += ( int )na.TriCount;
					if ( stackPtr == 0 )
					{
						break;
					}
					stackPtr--;
					a = stackA[ stackPtr ];
					b = stackB[ stackPtr ];
				}
				else
				{
					Assert.Less( stackPtr, MaxDepth, $"{label}: tree deeper than the walk stack" );
					stackA[ stackPtr ] = na.LeftFirst + 1;
					stackB[ stackPtr ] = nb.LeftFirst + 1;
					stackPtr++;
					a = na.LeftFirst;
					b = nb.LeftFirst;
				}
			}
			TestContext.WriteLine( $"{label}: {nodes} nodes walked, {leaves} leaves, {prims} leaf prims, bounds bit-identical, leaf prim sets equal" );
		}

		[TestCase( false )]
		[TestCase( true )]
		public unsafe void ThreadedBuild_MatchesSerialTree( bool spatialSplits )
		{
			if ( !TryGetPaths( LargeScene, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 to fetch scenes and Tools~/RefDump/run_all.bat to generate reference data" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh serial = default;
			Bvh threaded = default;
			try
			{
				serial = BuildScene( verts, triCount, spatialSplits, false );
				threaded = BuildScene( verts, triCount, spatialSplits, true );

				string label = $"{LargeScene} {( spatialSplits ? "sbvh" : "binned" )}";
				TestContext.WriteLine( $"{label}: serial nodes {serial.UsedNodes}, threaded nodes {threaded.UsedNodes}, subtree jobs {threaded.ThreadedSubtrees}" );
				Assert.AreEqual( 0u, serial.ThreadedSubtrees, "the serial build must not report subtree jobs" );
				Assert.Greater( threaded.ThreadedSubtrees, 1u, "the threaded build must have run more than one subtree job" );
				Assert.AreEqual( serial.UsedNodes, threaded.UsedNodes, "UsedNodes" );
				AssertSameTree( ref serial, ref threaded, label );
			}
			finally
			{
				if ( threaded.IsCreated )
				{
					threaded.Dispose();
				}
				if ( serial.IsCreated )
				{
					serial.Dispose();
				}
				verts.Dispose();
			}
		}

		[Test]
		public void ThreadedBuild_BinnedIntersectMatchesReference()
		{
			if ( !TryGetPaths( LargeScene, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 and Tools~/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = BuildScene( verts, triCount, false, true );
			try
			{
				RefDumpFile refFile = RefDumpFile.Load( refPath );
				int mismatches = 0, ties = 0, hitCount = 0;

				for ( int i = 0; i < refFile.Rays.Length; i++ )
				{
					RefDumpFile.RayHit rh = refFile.Rays[ i ];
					Ray ray = new Ray( rh.O, rh.D );
					bvh.Intersect( ref ray );

					bool refHit = rh.T < BvhConstants.Far;
					bool gotHit = ray.Hit.T < BvhConstants.Far;
					if ( refHit != gotHit )
					{
						mismatches++;
						continue;
					}
					if ( refHit )
					{
						hitCount++;
						bool sameT = math.abs( ray.Hit.T - rh.T ) <= 1e-4f * math.max( math.abs( rh.T ), 1e-12f );
						if ( !sameT )
						{
							mismatches++;
						}
						else if ( ray.Hit.Prim != rh.Prim )
						{
							// A different primitive at the same distance is a legitimate tie between
							// overlapping triangles, decided by traversal order and last-bit rounding.
							ties++;
						}
					}
				}

				TestContext.WriteLine( $"{LargeScene} binned threaded rays: hits {hitCount}/{refFile.Rays.Length}, mismatches {mismatches}, same-distance ties {ties}" );
				Assert.LessOrEqual( mismatches, ( int )( refFile.Rays.Length * MaxMismatchFraction ), "mismatch rate too high" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		[Test]
		public void ThreadedBuild_SbvhIntersectMatchesReference()
		{
			if ( !TryGetPaths( LargeScene, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 and Tools~/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = BuildScene( verts, triCount, true, true );
			try
			{
				RefDumpFile refFile = RefDumpFile.Load( refPath );
				int mismatches = 0, ties = 0, grazing = 0, hitCount = 0;

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
						// rounding; see BvhHqTests for the same classification.
						if ( ( refHit && IsGrazing( rh.U, rh.V ) ) || ( gotHit && IsGrazing( ray.Hit.U, ray.Hit.V ) ) )
						{
							grazing++;
						}
						else
						{
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

				TestContext.WriteLine( $"{LargeScene} sbvh threaded rays: hits {hitCount}/{refFile.SbvhRays.Length}, mismatches {mismatches}, same-distance ties {ties}, grazing {grazing}" );
				Assert.AreEqual( 0, mismatches, "SBVH intersect mismatches" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>
		/// suzanne is below BvhConstants.MtBuildThreshold, so setting the flag must change nothing:
		/// the build stays serial and produces the very same node and index arrays.
		/// </summary>
		[TestCase( false )]
		[TestCase( true )]
		public unsafe void BelowThreshold_ThreadedFlagBuildsTheSerialTree( bool spatialSplits )
		{
			if ( !TryGetPaths( SmallScene, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 and Tools~/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh serial = default;
			Bvh flagged = default;
			try
			{
				Assert.Less( triCount, BvhConstants.MtBuildThreshold, "this test needs a scene below the threading threshold" );
				serial = BuildScene( verts, triCount, spatialSplits, false );
				flagged = BuildScene( verts, triCount, spatialSplits, true );

				Assert.AreEqual( 0u, flagged.ThreadedSubtrees, "a build below the threshold must not run subtree jobs" );
				Assert.AreEqual( serial.UsedNodes, flagged.UsedNodes, "UsedNodes" );
				int nodeMismatches = 0;
				for ( uint i = 0; i < serial.UsedNodes; i++ )
				{
					BvhNode a = serial.Nodes[ i ], b = flagged.Nodes[ i ];
					if ( !BoundsEqual( a, b ) || a.LeftFirst != b.LeftFirst || a.TriCount != b.TriCount )
					{
						nodeMismatches++;
					}
				}
				int idxMismatches = 0;
				int idxCount = spatialSplits ? serial.PrimCount() : ( int )serial.IdxCount;
				for ( int i = 0; i < idxCount; i++ )
				{
					if ( serial.PrimIdx[ i ] != flagged.PrimIdx[ i ] )
					{
						idxMismatches++;
					}
				}
				TestContext.WriteLine( $"{SmallScene} {( spatialSplits ? "sbvh" : "binned" )} below threshold: {serial.UsedNodes} nodes, node mismatches {nodeMismatches}, primIdx mismatches {idxMismatches}/{idxCount}" );
				Assert.AreEqual( 0, nodeMismatches, "the threaded flag must not change a build below the threshold" );
				Assert.AreEqual( 0, idxMismatches, "the threaded flag must not change the index array below the threshold" );
			}
			finally
			{
				if ( flagged.IsCreated )
				{
					flagged.Dispose();
				}
				if ( serial.IsCreated )
				{
					serial.Dispose();
				}
				verts.Dispose();
			}
		}

		/// <summary>Informational: serial versus threaded build time. No assertion.</summary>
		[Test]
		public void BuildTime_SerialVersusThreaded()
		{
			if ( !TryGetPaths( LargeScene, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 and Tools~/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				for ( int pass = 0; pass < 2; pass++ )
				{
					bool spatialSplits = pass == 1;
					bvh.UseSpatialSplits = spatialSplits;
					bvh.UseThreadedBuild = false;
					Stopwatch timer = Stopwatch.StartNew();
					bvh.Build( verts, triCount );
					double serialMs = timer.Elapsed.TotalMilliseconds;
					bvh.UseThreadedBuild = true;
					timer.Restart();
					bvh.Build( verts, triCount );
					double threadedMs = timer.Elapsed.TotalMilliseconds;
					TestContext.WriteLine( $"{LargeScene} {( spatialSplits ? "sbvh" : "binned" )} build: serial {serialMs:0.0} ms, threaded {threadedMs:0.0} ms over {bvh.ThreadedSubtrees} subtrees ({serialMs / threadedMs:0.00}x)" );
				}
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}
	}
}
