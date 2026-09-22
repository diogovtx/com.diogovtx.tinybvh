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
	/// Validates the indexed-geometry build path against the indexed sections of the reference dump
	/// (Tools/RefDump/refdump.cpp): a BVH built over a welded vertex array plus a separate index
	/// array, where every layout and traversal addresses vertices through Bvh.VertIdx instead of
	/// prim * 3. The welded mesh comes out of the dump rather than being re-welded here, so the two
	/// sides cannot drift apart over the welding rule.
	/// </summary>
	public class BvhIndexedTests
	{
		/// <summary>Barycentric distance to a triangle edge below which a hit counts as grazing.</summary>
		const float GrazingEps = 1e-3f;

		const float RelativeTolerance = 1e-4f;

		/// <summary>Number of differing entries listed in an assertion message before it is truncated.</summary>
		const int MaxReportedDiffs = 4;

		[BurstCompile( CompileSynchronously = true )]
		private struct BvhJob : IJobParallelFor
		{
			public Bvh Bvh;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> Occluded;

			public void Execute( int i )
			{
				Ray ray = new Ray( Origins[ i ], Directions[ i ] );
				Bvh.Intersect( ref ray );
				Hits[ i ] = ray.Hit;
				Occluded[ i ] = Bvh.IsOccluded( new Ray( Origins[ i ], Directions[ i ] ) ) ? 1 : 0;
			}
		}

		[BurstCompile( CompileSynchronously = true )]
		private struct BvhGpuJob : IJobParallelFor
		{
			public BvhGpu Bvh;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> Occluded;

			public void Execute( int i )
			{
				Ray ray = new Ray( Origins[ i ], Directions[ i ] );
				Bvh.Intersect( ref ray );
				Hits[ i ] = ray.Hit;
				Occluded[ i ] = Bvh.IsOccluded( new Ray( Origins[ i ], Directions[ i ] ) ) ? 1 : 0;
			}
		}

		[BurstCompile( CompileSynchronously = true )]
		private struct Bvh4GpuJob : IJobParallelFor
		{
			public Bvh4Gpu Bvh;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> Occluded;

			public void Execute( int i )
			{
				Ray ray = new Ray( Origins[ i ], Directions[ i ] );
				Bvh.Intersect( ref ray );
				Hits[ i ] = ray.Hit;
				Occluded[ i ] = Bvh.IsOccluded( new Ray( Origins[ i ], Directions[ i ] ) ) ? 1 : 0;
			}
		}

		[BurstCompile( CompileSynchronously = true )]
		private struct Bvh4CpuJob : IJobParallelFor
		{
			public Bvh4Cpu Bvh;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> Occluded;

			public void Execute( int i )
			{
				Ray ray = new Ray( Origins[ i ], Directions[ i ] );
				Bvh.Intersect( ref ray );
				Hits[ i ] = ray.Hit;
				Occluded[ i ] = Bvh.IsOccluded( new Ray( Origins[ i ], Directions[ i ] ) ) ? 1 : 0;
			}
		}

		[BurstCompile( CompileSynchronously = true )]
		private struct Bvh8CpuJob : IJobParallelFor
		{
			public Bvh8Cpu Bvh;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> Occluded;

			public void Execute( int i )
			{
				Ray ray = new Ray( Origins[ i ], Directions[ i ] );
				Bvh.Intersect( ref ray );
				Hits[ i ] = ray.Hit;
				Occluded[ i ] = Bvh.IsOccluded( new Ray( Origins[ i ], Directions[ i ] ) ) ? 1 : 0;
			}
		}

		/// <summary>
		/// The welded mesh and the reference sections that go with it, plus the ray set in the form
		/// the Burst jobs want. Owns its NativeArrays; Dispose frees them.
		/// </summary>
		private struct IndexedScene
		{
			public RefDumpFile Ref;
			public uint TriCount;
			public NativeArray<float4> Vertices;
			public NativeArray<uint> Indices;
			public NativeArray<float3> Origins;
			public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> Occluded;

			public int RayCount
			{
				get { return Origins.Length; }
			}

			public void Dispose()
			{
				if ( Occluded.IsCreated )
				{
					Occluded.Dispose();
				}
				if ( Hits.IsCreated )
				{
					Hits.Dispose();
				}
				if ( Directions.IsCreated )
				{
					Directions.Dispose();
				}
				if ( Origins.IsCreated )
				{
					Origins.Dispose();
				}
				if ( Indices.IsCreated )
				{
					Indices.Dispose();
				}
				if ( Vertices.IsCreated )
				{
					Vertices.Dispose();
				}
			}
		}

		/// <summary>
		/// Loads the indexed sections of a scene's reference dump. Ignores the test when the dump is
		/// missing or predates them.
		/// </summary>
		static IndexedScene LoadScene( string sceneName )
		{
			string refPath = BvhSceneFile.TestDataPath( sceneName + ".ref" );
			if ( !File.Exists( refPath ) )
			{
				Assert.Ignore( $"missing {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}
			RefDumpFile refFile = RefDumpFile.Load( refPath );
			if ( refFile.WeldedVertices == null || refFile.WeldedVertices.Length == 0 )
			{
				Assert.Ignore( $"{refPath} predates the indexed-geometry sections; re-run Tools/RefDump/run_all.bat" );
			}

			IndexedScene scene = new IndexedScene();
			scene.Ref = refFile;
			scene.TriCount = refFile.TriCount;
			scene.Vertices = new NativeArray<float4>( refFile.WeldedVertices, Allocator.Persistent );
			scene.Indices = new NativeArray<uint>( refFile.Indices, Allocator.Persistent );
			int rayCount = refFile.IndexedRays.Length;
			scene.Origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
			scene.Directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
			scene.Hits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			scene.Occluded = new NativeArray<int>( rayCount, Allocator.Persistent );
			for ( int i = 0; i < rayCount; i++ )
			{
				scene.Origins[ i ] = refFile.IndexedRays[ i ].O;
				scene.Directions[ i ] = refFile.IndexedRays[ i ].D;
			}
			return scene;
		}

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

		static bool NodesEqual( BvhNode a, BvhNode b )
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

		static unsafe void CompareNodes( string label, BvhNode* nodes, uint usedNodes, uint expectedUsedNodes, BvhNode[] expected )
		{
			Assert.AreEqual( expectedUsedNodes, usedNodes, label + " UsedNodes" );
			List<int> diffs = new List<int>();
			for ( int i = 0; i < expected.Length; i++ )
			{
				if ( !NodesEqual( nodes[ i ], expected[ i ] ) )
				{
					diffs.Add( i );
				}
			}
			TestContext.WriteLine( $"{label}: node mismatches {diffs.Count}/{expected.Length}" );
			Assert.AreEqual( 0, diffs.Count, $"{label} node mismatch: {Describe( diffs )}" );
		}

		static unsafe void ComparePrimIdx( string label, uint* primIdx, int count, uint[] expected )
		{
			Assert.AreEqual( expected.Length, count, label + " prim count" );
			List<int> diffs = new List<int>();
			for ( int i = 0; i < expected.Length; i++ )
			{
				if ( primIdx[ i ] != expected[ i ] )
				{
					diffs.Add( i );
				}
			}
			Assert.AreEqual( 0, diffs.Count, $"{label} primIdx mismatch: {Describe( diffs )}" );
		}

		/// <summary>
		/// Compares a traced ray set against one of the indexed reference ray sections. Same-distance
		/// hits on a different primitive are legitimate ties between overlapping triangles, and hits
		/// within a hair of a triangle edge are decided by last-bit rounding; both are counted
		/// separately and do not fail the test, exactly as in the other reference suites.
		/// </summary>
		static void CompareRays( string label, RefDumpFile.RayHit[] expected, NativeArray<Intersection> hits, NativeArray<int> occluded )
		{
			int rayCount = expected.Length;
			int mismatches = 0, ties = 0, grazing = 0, hitCount = 0, occlusionMismatches = 0;
			for ( int i = 0; i < rayCount; i++ )
			{
				RefDumpFile.RayHit rh = expected[ i ];
				Intersection hit = hits[ i ];
				bool refHit = rh.T < BvhConstants.Far;
				bool gotHit = hit.T < BvhConstants.Far;
				bool sameT = refHit && gotHit
					&& math.abs( hit.T - rh.T ) <= RelativeTolerance * math.max( math.abs( rh.T ), 1e-12f );
				if ( refHit != gotHit || ( refHit && !sameT ) )
				{
					if ( ( refHit && IsGrazing( rh.U, rh.V ) ) || ( gotHit && IsGrazing( hit.U, hit.V ) ) )
					{
						grazing++;
					}
					else
					{
						if ( mismatches < 3 )
						{
							TestContext.WriteLine( $"  mismatch ray {i}: O {rh.O} D {rh.D} ref t {rh.T} u {rh.U} v {rh.V} prim {rh.Prim} | got t {hit.T} u {hit.U} v {hit.V} prim {hit.Prim}" );
						}
						mismatches++;
					}
				}
				else if ( refHit )
				{
					hitCount++;
					if ( hit.Prim != rh.Prim )
					{
						ties++;
					}
				}
				if ( ( occluded[ i ] != 0 ) != ( rh.OccludedFull != 0 ) )
				{
					occlusionMismatches++;
				}
			}
			float hitRatio = ( float )hitCount / rayCount;
			TestContext.WriteLine( $"{label}: hit ratio {hitRatio:P2}, mismatches {mismatches}/{rayCount}, same-distance ties {ties}, grazing {grazing}, occlusion mismatches {occlusionMismatches}" );
			Assert.AreEqual( 0, mismatches, label + " intersect mismatches" );
			Assert.AreEqual( 0, occlusionMismatches, label + " occlusion mismatches" );
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void Build_MatchesIndexedReference( string sceneName )
		{
			IndexedScene scene = LoadScene( sceneName );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( scene.Vertices, scene.Indices, scene.TriCount );

				Assert.IsTrue( bvh.BvhOverIndices, "BvhOverIndices" );
				CompareNodes( $"{sceneName} indexed", bvh.Nodes, bvh.UsedNodes, scene.Ref.IndexedUsedNodes, scene.Ref.IndexedNodes );
				ComparePrimIdx( $"{sceneName} indexed", bvh.PrimIdx, ( int )bvh.IdxCount, scene.Ref.IndexedPrimIdx );

				new BvhJob
				{
					Bvh = bvh,
					Origins = scene.Origins,
					Directions = scene.Directions,
					Hits = scene.Hits,
					Occluded = scene.Occluded
				}.Schedule( scene.RayCount, 64 ).Complete();
				CompareRays( $"{sceneName} indexed Bvh", scene.Ref.IndexedRays, scene.Hits, scene.Occluded );
			}
			finally
			{
				bvh.Dispose();
				scene.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void BuildHq_MatchesIndexedReference( string sceneName )
		{
			IndexedScene scene = LoadScene( sceneName );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.UseSpatialSplits = true;
				bvh.Build( scene.Vertices, scene.Indices, scene.TriCount );

				Assert.IsTrue( bvh.BvhOverIndices, "BvhOverIndices" );
				CompareNodes( $"{sceneName} indexed SBVH", bvh.Nodes, bvh.UsedNodes, scene.Ref.IndexedSbvhUsedNodes, scene.Ref.IndexedSbvhNodes );
				// The reference dumps PrimCount() entries, not idxCount; BuildHQ ends with Compact,
				// which leaves the tail of the index array uninitialized. See the refdump.cpp note.
				ComparePrimIdx( $"{sceneName} indexed SBVH", bvh.PrimIdx, bvh.PrimCount(), scene.Ref.IndexedSbvhPrimIdx );

				new BvhJob
				{
					Bvh = bvh,
					Origins = scene.Origins,
					Directions = scene.Directions,
					Hits = scene.Hits,
					Occluded = scene.Occluded
				}.Schedule( scene.RayCount, 64 ).Complete();
				CompareRays( $"{sceneName} indexed SBVH", scene.Ref.IndexedSbvhRays, scene.Hits, scene.Occluded );
			}
			finally
			{
				bvh.Dispose();
				scene.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void BvhGpuLayout_MatchesIndexedReference( string sceneName )
		{
			IndexedScene scene = LoadScene( sceneName );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			BvhGpu gpu = BvhGpu.Create( Allocator.Persistent );
			try
			{
				bvh.Build( scene.Vertices, scene.Indices, scene.TriCount );
				gpu.ConvertFrom( ref bvh );
				Assert.IsTrue( gpu.BvhOverIndices, "BvhOverIndices" );

				new BvhGpuJob
				{
					Bvh = gpu,
					Origins = scene.Origins,
					Directions = scene.Directions,
					Hits = scene.Hits,
					Occluded = scene.Occluded
				}.Schedule( scene.RayCount, 64 ).Complete();
				CompareRays( $"{sceneName} indexed BvhGpu", scene.Ref.IndexedRays, scene.Hits, scene.Occluded );
			}
			finally
			{
				gpu.Dispose();
				bvh.Dispose();
				scene.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Bvh4GpuLayout_MatchesIndexedReference( string sceneName )
		{
			IndexedScene scene = LoadScene( sceneName );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh4 = Mbvh.Create( 4, Allocator.Persistent );
			Bvh4Gpu gpu4 = Bvh4Gpu.Create( Allocator.Persistent );
			try
			{
				bvh.Build( scene.Vertices, scene.Indices, scene.TriCount );
				mbvh4.ConvertFrom( ref bvh );
				Assert.IsTrue( mbvh4.BvhOverIndices, "Mbvh4 BvhOverIndices" );
				gpu4.ConvertFrom( ref mbvh4 );
				Assert.IsTrue( gpu4.BvhOverIndices, "Bvh4Gpu BvhOverIndices" );

				new Bvh4GpuJob
				{
					Bvh = gpu4,
					Origins = scene.Origins,
					Directions = scene.Directions,
					Hits = scene.Hits,
					Occluded = scene.Occluded
				}.Schedule( scene.RayCount, 64 ).Complete();
				CompareRays( $"{sceneName} indexed Bvh4Gpu", scene.Ref.IndexedRays, scene.Hits, scene.Occluded );
			}
			finally
			{
				gpu4.Dispose();
				mbvh4.Dispose();
				bvh.Dispose();
				scene.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Bvh4CpuLayout_MatchesIndexedReference( string sceneName )
		{
			IndexedScene scene = LoadScene( sceneName );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh4 = Mbvh.Create( 4, Allocator.Persistent );
			Bvh4Cpu bvh4Cpu = Bvh4Cpu.Create( Allocator.Persistent );
			try
			{
				bvh.Build( scene.Vertices, scene.Indices, scene.TriCount );
				mbvh4.ConvertFrom( ref bvh );
				bvh4Cpu.ConvertFrom( ref mbvh4 );
				Assert.IsTrue( bvh4Cpu.BvhOverIndices, "BvhOverIndices" );
				// ConvertFrom takes a value copy of the Mbvh, reshapes the base BVH it holds with
				// CombineLeafs( 4 ) / SplitLeafs( 4 ) and re-converts, which reallocates both node
				// pools. The locals above are stale from here on, so hand ownership of the two
				// up-to-date copies - bvh4Cpu.Mbvh4 and bvh4Cpu.Base - to bvh4Cpu.Dispose.
				bvh4Cpu.OwnsSource = true;
				mbvh4 = default;
				bvh = default;

				new Bvh4CpuJob
				{
					Bvh = bvh4Cpu,
					Origins = scene.Origins,
					Directions = scene.Directions,
					Hits = scene.Hits,
					Occluded = scene.Occluded
				}.Schedule( scene.RayCount, 64 ).Complete();
				CompareRays( $"{sceneName} indexed Bvh4Cpu", scene.Ref.IndexedRays, scene.Hits, scene.Occluded );
			}
			finally
			{
				bvh4Cpu.Dispose();
				if ( mbvh4.IsCreated )
				{
					mbvh4.Dispose();
				}
				if ( bvh.IsCreated )
				{
					bvh.Dispose();
				}
				scene.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Bvh8CpuLayout_MatchesIndexedReference( string sceneName )
		{
			IndexedScene scene = LoadScene( sceneName );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh8 = Mbvh.Create( 8, Allocator.Persistent );
			Bvh8Cpu bvh8Cpu = Bvh8Cpu.Create( Allocator.Persistent );
			try
			{
				bvh.Build( scene.Vertices, scene.Indices, scene.TriCount );
				// BVH8_CPU::Build compacts the base before converting; do the same by hand here.
				bvh.Compact();
				mbvh8.ConvertFrom( ref bvh );
				Assert.IsTrue( mbvh8.BvhOverIndices, "Mbvh8 BvhOverIndices" );
				bvh8Cpu.ConvertFrom( ref mbvh8 );
				Assert.IsTrue( bvh8Cpu.BvhOverIndices, "BvhOverIndices" );
				// As in the 4-wide case, ConvertFrom reallocates the node pools of its own copies
				// and leaves the locals stale; hand them to bvh8Cpu.Dispose instead.
				bvh8Cpu.OwnsSource = true;
				mbvh8 = default;
				bvh = default;

				new Bvh8CpuJob
				{
					Bvh = bvh8Cpu,
					Origins = scene.Origins,
					Directions = scene.Directions,
					Hits = scene.Hits,
					Occluded = scene.Occluded
				}.Schedule( scene.RayCount, 64 ).Complete();
				CompareRays( $"{sceneName} indexed Bvh8Cpu", scene.Ref.IndexedRays, scene.Hits, scene.Occluded );
			}
			finally
			{
				bvh8Cpu.Dispose();
				if ( mbvh8.IsCreated )
				{
					mbvh8.Dispose();
				}
				if ( bvh.IsCreated )
				{
					bvh.Dispose();
				}
				scene.Dispose();
			}
		}

		/// <summary>
		/// CWBVH has no CPU traversal in this port, so its indexed conversion is checked against the
		/// same conversion over the un-welded triangle soup instead: welding only merges vertices
		/// whose x, y and z bits are identical, so the two trees and the positions they bake into the
		/// triangle blob must agree exactly. Only xyz is compared, because the blob stores whole
		/// float4 differences and the scenes carry per-vertex payload in w, of which welding keeps
		/// the first occurrence's copy.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void CwbvhLayout_MatchesSoupConversion( string sceneName )
		{
			string binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			if ( !File.Exists( binPath ) )
			{
				Assert.Ignore( $"missing {binPath}; run TestData/fetch.ps1" );
			}
			IndexedScene scene = LoadScene( sceneName );
			NativeArray<float4> soup = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint soupTriCount );
			Bvh indexedBvh = Bvh.Create( Allocator.Persistent );
			Bvh soupBvh = Bvh.Create( Allocator.Persistent );
			Mbvh indexedMbvh = Mbvh.Create( 8, Allocator.Persistent );
			Mbvh soupMbvh = Mbvh.Create( 8, Allocator.Persistent );
			BvhCwbvh indexedCwbvh = BvhCwbvh.Create( Allocator.Persistent );
			BvhCwbvh soupCwbvh = BvhCwbvh.Create( Allocator.Persistent );
			try
			{
				indexedBvh.Build( scene.Vertices, scene.Indices, scene.TriCount );
				indexedBvh.Compact();
				indexedBvh.SplitLeafs( 3 );
				indexedMbvh.ConvertFrom( ref indexedBvh );
				indexedCwbvh.ConvertFrom( ref indexedMbvh );
				Assert.IsTrue( indexedCwbvh.BvhOverIndices, "BvhOverIndices" );

				soupBvh.Build( soup, soupTriCount );
				soupBvh.Compact();
				soupBvh.SplitLeafs( 3 );
				soupMbvh.ConvertFrom( ref soupBvh );
				soupCwbvh.ConvertFrom( ref soupMbvh );

				Assert.AreEqual( soupCwbvh.UsedBlocks, indexedCwbvh.UsedBlocks, "UsedBlocks" );
				Assert.AreEqual( soupCwbvh.TriBlocks, indexedCwbvh.TriBlocks, "TriBlocks" );

				List<int> nodeDiffs = new List<int>();
				for ( uint i = 0; i < soupCwbvh.UsedBlocks; i++ )
				{
					if ( !BitsEqual( indexedCwbvh.Data[ i ].x, soupCwbvh.Data[ i ].x )
						|| !BitsEqual( indexedCwbvh.Data[ i ].y, soupCwbvh.Data[ i ].y )
						|| !BitsEqual( indexedCwbvh.Data[ i ].z, soupCwbvh.Data[ i ].z )
						|| !BitsEqual( indexedCwbvh.Data[ i ].w, soupCwbvh.Data[ i ].w ) )
					{
						nodeDiffs.Add( ( int )i );
					}
				}
				Assert.AreEqual( 0, nodeDiffs.Count, $"cwbvh node blob mismatch: {Describe( nodeDiffs )}" );

				List<int> triDiffs = new List<int>();
				for ( uint i = 0; i < soupCwbvh.TriBlocks; i++ )
				{
					if ( !BitsEqual( indexedCwbvh.Tris[ i ].xyz, soupCwbvh.Tris[ i ].xyz ) )
					{
						triDiffs.Add( ( int )i );
					}
				}
				TestContext.WriteLine( $"{sceneName} indexed CWBVH: {indexedCwbvh.UsedBlocks} node blocks, {indexedCwbvh.TriBlocks} tri blocks, node diffs {nodeDiffs.Count}, tri diffs {triDiffs.Count}" );
				Assert.AreEqual( 0, triDiffs.Count, $"cwbvh triangle blob mismatch: {Describe( triDiffs )}" );
			}
			finally
			{
				soupCwbvh.Dispose();
				indexedCwbvh.Dispose();
				soupMbvh.Dispose();
				indexedMbvh.Dispose();
				soupBvh.Dispose();
				indexedBvh.Dispose();
				soup.Dispose();
				scene.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void SaveLoad_RoundTripsIndexedBvh( string sceneName )
		{
			IndexedScene scene = LoadScene( sceneName );
			string path = Path.Combine( Path.GetTempPath(), $"tinybvh-indexed-{sceneName}.bvh" );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Bvh loaded = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( scene.Vertices, scene.Indices, scene.TriCount );
				bvh.Save( path );
				Assert.IsTrue( loaded.Load( path, scene.Vertices, scene.Indices, scene.TriCount ), "Load" );
				Assert.IsTrue( loaded.BvhOverIndices, "BvhOverIndices" );
				Assert.IsTrue( loaded.VertIdx == bvh.VertIdx, "VertIdx" );

				CompareNodes( $"{sceneName} indexed reloaded", loaded.Nodes, loaded.UsedNodes, scene.Ref.IndexedUsedNodes, scene.Ref.IndexedNodes );
				ComparePrimIdx( $"{sceneName} indexed reloaded", loaded.PrimIdx, ( int )loaded.IdxCount, scene.Ref.IndexedPrimIdx );

				new BvhJob
				{
					Bvh = loaded,
					Origins = scene.Origins,
					Directions = scene.Directions,
					Hits = scene.Hits,
					Occluded = scene.Occluded
				}.Schedule( scene.RayCount, 64 ).Complete();
				CompareRays( $"{sceneName} indexed reloaded", scene.Ref.IndexedRays, scene.Hits, scene.Occluded );

				// A soup Load over the same file must be refused: the flags say the tree is indexed.
				Assert.IsFalse( loaded.Load( path, scene.Vertices, scene.TriCount ), "Load without indices must be refused" );
			}
			finally
			{
				loaded.Dispose();
				bvh.Dispose();
				scene.Dispose();
				if ( File.Exists( path ) )
				{
					File.Delete( path );
				}
			}
		}

		/// <summary>
		/// Traces the indexed reference rays on the GPU. Only the single-BVH Bvh2 and BvhGpu kernels
		/// read the index buffer; the blob layouts bake their triangles at conversion time and the
		/// TLAS kernels reject indexed BLASses outright.
		/// </summary>
		[TestCase( "suzanne", GpuLayout.Bvh2 )]
		[TestCase( "suzanne", GpuLayout.BvhGpu )]
		[TestCase( "bunny", GpuLayout.Bvh2 )]
		[TestCase( "bunny", GpuLayout.BvhGpu )]
		[TestCase( "cryteksponza", GpuLayout.Bvh2 )]
		[TestCase( "cryteksponza", GpuLayout.BvhGpu )]
		public void GpuTraceBatch_MatchesIndexedReference( string sceneName, GpuLayout layout )
		{
			if ( !GpuTracer.IsSupported )
			{
				Assert.Ignore( "compute shaders are unavailable; run the test runner with graphics enabled" );
			}
			IndexedScene scene = LoadScene( sceneName );
			int rayCount = scene.RayCount;
			NativeArray<float4> origins = new NativeArray<float4>( rayCount, Allocator.Persistent );
			NativeArray<float4> directions = new NativeArray<float4>( rayCount, Allocator.Persistent );
			NativeArray<float4> hits = new NativeArray<float4>( rayCount, Allocator.Persistent );
			NativeArray<uint> occluded = new NativeArray<uint>( rayCount, Allocator.Persistent );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			BvhGpu gpu = default;
			GpuTracer tracer = null;
			try
			{
				bvh.Build( scene.Vertices, scene.Indices, scene.TriCount );
				tracer = new GpuTracer();
				if ( layout == GpuLayout.BvhGpu )
				{
					gpu = BvhGpu.Create( Allocator.Persistent );
					gpu.ConvertFrom( ref bvh );
					tracer.Upload( ref gpu );
				}
				else
				{
					tracer.Upload( ref bvh );
				}

				for ( int i = 0; i < rayCount; i++ )
				{
					RefDumpFile.RayHit rh = scene.Ref.IndexedRays[ i ];
					origins[ i ] = new float4( rh.O, BvhConstants.Far );
					directions[ i ] = new float4( math.normalize( rh.D ), 0f );
				}
				tracer.TraceBatch( origins, directions, hits, occluded );

				for ( int i = 0; i < rayCount; i++ )
				{
					float4 got = hits[ i ];
					Intersection hit = default;
					hit.T = got.x;
					hit.U = got.y;
					hit.V = got.z;
					hit.Prim = math.asuint( got.w );
					scene.Hits[ i ] = hit;
					scene.Occluded[ i ] = occluded[ i ] != 0 ? 1 : 0;
				}
				CompareRays( $"{sceneName} indexed {layout}", scene.Ref.IndexedRays, scene.Hits, scene.Occluded );
			}
			finally
			{
				tracer?.Dispose();
				if ( gpu.IsCreated )
				{
					gpu.Dispose();
				}
				bvh.Dispose();
				occluded.Dispose();
				hits.Dispose();
				directions.Dispose();
				origins.Dispose();
				scene.Dispose();
			}
		}
	}
}
