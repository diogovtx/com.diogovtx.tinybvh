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
	/// Data-driven tests comparing the layout conversions (BVH_GPU, MBVH&lt;4&gt;, BVH4_GPU, the
	/// Compact + SplitLeafs( 3 ) preparation and CWBVH) against reference data dumped by
	/// Tools/RefDump/layoutdump.cpp from the original tinybvh library. If TestData/&lt;name&gt;.bin
	/// or TestData/&lt;name&gt;.layouts.ref is missing, the test is ignored.
	/// </summary>
	public class BvhLayoutTests
	{
		/// <summary>Number of differing entries listed in an assertion message before it is truncated.</summary>
		const int MaxReportedDiffs = 4;

		/// <summary>Barycentric distance to a triangle edge below which a hit counts as grazing.</summary>
		const float GrazingEps = 1e-3f;

		/// <summary>True when the hit lies within GrazingEps of a triangle edge or vertex, where last-bit rounding decides hit or miss.</summary>
		static bool IsGrazing( float u, float v )
		{
			return u < GrazingEps || v < GrazingEps || ( 1f - u - v ) < GrazingEps;
		}

		/// <summary>Traces the BVH4_GPU layout from a Burst job, so the scalar traversal runs in single precision.</summary>
		[BurstCompile( CompileSynchronously = true )]
		struct Bvh4GpuTraceJob : IJobParallelFor
		{
			public Bvh4Gpu Bvh4Gpu;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;

			public void Execute( int i )
			{
				Ray ray = new Ray( Origins[ i ], Directions[ i ] );
				Bvh4Gpu.Intersect( ref ray );
				Hits[ i ] = ray.Hit;
			}
		}

		static bool TryGetPaths( string sceneName, out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( sceneName + ".layouts.ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		static bool BitsEqual( float a, float b )
		{
			return math.asuint( a ) == math.asuint( b );
		}

		static bool BitsEqual( float3 a, float3 b )
		{
			return BitsEqual( a.x, b.x ) && BitsEqual( a.y, b.y ) && BitsEqual( a.z, b.z );
		}

		static bool GpuNodesEqual( BvhGpuNode a, BvhGpuNode b )
		{
			return BitsEqual( a.LMin, b.LMin ) && a.Left == b.Left
				&& BitsEqual( a.LMax, b.LMax ) && a.Right == b.Right
				&& BitsEqual( a.RMin, b.RMin ) && a.TriCount == b.TriCount
				&& BitsEqual( a.RMax, b.RMax ) && a.FirstTri == b.FirstTri;
		}

		static bool BvhNodesEqual( BvhNode a, BvhNode b )
		{
			return BitsEqual( a.AabbMin, b.AabbMin ) && a.LeftFirst == b.LeftFirst
				&& BitsEqual( a.AabbMax, b.AabbMax ) && a.TriCount == b.TriCount;
		}

		/// <summary>Only the first M children are meaningful; the C# node always has eight slots.</summary>
		static unsafe bool MbvhNodesEqual( MbvhNode* a, LayoutDumpFile.MbvhNodeRecord b, int m )
		{
			if ( !BitsEqual( a->AabbMin, b.AabbMin ) || a->FirstTri != b.FirstTri
				|| !BitsEqual( a->AabbMax, b.AabbMax ) || a->TriCount != b.TriCount
				|| a->ChildCount != b.ChildCount )
			{
				return false;
			}
			for ( int i = 0; i < m; i++ )
			{
				if ( a->Child[ i ] != b.Child[ i ] )
				{
					return false;
				}
			}
			return true;
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

		static unsafe void CompareMbvhNodes( string label, ref Mbvh mbvh, LayoutDumpFile.MbvhNodeRecord[] expected, int m )
		{
			Assert.AreEqual( expected.Length, ( int )mbvh.UsedNodes, label + " UsedNodes" );
			List<int> diffs = new List<int>();
			for ( int i = 0; i < expected.Length; i++ )
			{
				if ( !MbvhNodesEqual( mbvh.Nodes + i, expected[ i ], m ) )
				{
					diffs.Add( i );
				}
			}
			TestContext.WriteLine( $"{label}: node mismatches {diffs.Count}/{expected.Length}" );
			if ( diffs.Count > 0 )
			{
				MbvhNode* a = mbvh.Nodes + diffs[ 0 ];
				LayoutDumpFile.MbvhNodeRecord b = expected[ diffs[ 0 ] ];
				TestContext.WriteLine( $"  first diff node {diffs[ 0 ]}: ours min {a->AabbMin} max {a->AabbMax} firstTri {a->FirstTri} triCount {a->TriCount} childCount {a->ChildCount} children {a->Child[ 0 ]},{a->Child[ 1 ]},{a->Child[ 2 ]},{a->Child[ 3 ]},{a->Child[ 4 ]},{a->Child[ 5 ]},{a->Child[ 6 ]},{a->Child[ 7 ]}" );
				TestContext.WriteLine( $"  first diff node {diffs[ 0 ]}:  ref min {b.AabbMin} max {b.AabbMax} firstTri {b.FirstTri} triCount {b.TriCount} childCount {b.ChildCount} children {string.Join( ",", b.Child )}" );
			}
			Assert.AreEqual( 0, diffs.Count, $"{label} node mismatch: {Describe( diffs )}" );
		}

		static unsafe void CompareBlocks( string label, uint4* actual, uint actualCount, uint4[] expected )
		{
			Assert.AreEqual( expected.Length, ( int )actualCount, label + " block count" );
			List<int> diffs = new List<int>();
			for ( int i = 0; i < expected.Length; i++ )
			{
				if ( !math.all( actual[ i ] == expected[ i ] ) )
				{
					diffs.Add( i );
				}
			}
			TestContext.WriteLine( $"{label}: block mismatches {diffs.Count}/{expected.Length}" );
			Assert.AreEqual( 0, diffs.Count, $"{label} block mismatch: {Describe( diffs )}" );
		}

		static void CompareRayHits( string label, LayoutDumpFile.RayHit[] expected, Intersection[] actual, bool assert = true )
		{
			int mismatches = 0, ties = 0, grazing = 0, hits = 0;
			for ( int i = 0; i < expected.Length; i++ )
			{
				LayoutDumpFile.RayHit rh = expected[ i ];
				Intersection hit = actual[ i ];
				bool refHit = rh.T < BvhConstants.Far;
				bool gotHit = hit.T < BvhConstants.Far;
				if ( refHit )
				{
					hits++;
				}
				bool sameT = refHit && gotHit && math.abs( hit.T - rh.T ) <= 1e-4f * math.max( math.abs( rh.T ), 1e-12f );
				if ( refHit != gotHit || ( refHit && !sameT ) )
				{
					// A hit within a hair of a triangle edge or vertex is decided by last-bit rounding;
					// the two sides can legitimately disagree on it and then hit different geometry.
					if ( ( refHit && IsGrazing( rh.U, rh.V ) ) || ( gotHit && IsGrazing( hit.U, hit.V ) ) )
					{
						grazing++;
						continue;
					}
					if ( mismatches < 3 )
					{
						TestContext.WriteLine( $"  mismatch ray {i}: O {rh.O} D {rh.D} ref t {rh.T} u {rh.U} v {rh.V} prim {rh.Prim} | got t {hit.T} u {hit.U} v {hit.V} prim {hit.Prim}" );
					}
					mismatches++;
					continue;
				}
				// A different primitive at the same distance is a legitimate tie between overlapping
				// triangles, decided by traversal order and last-bit rounding.
				if ( refHit && ( hit.Prim != rh.Prim
					|| math.abs( hit.U - rh.U ) > 1e-4f
					|| math.abs( hit.V - rh.V ) > 1e-4f ) )
				{
					ties++;
				}
			}
			TestContext.WriteLine( $"{label}: rays {expected.Length}, hits {hits}, mismatches {mismatches}, same-distance ties {ties}, grazing {grazing}" );
			if ( assert )
			{
				Assert.AreEqual( 0, mismatches, label + " ray mismatches" );
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void BvhGpu_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			LayoutDumpFile refFile = LayoutDumpFile.Load( refPath );
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			BvhGpu gpu = BvhGpu.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				gpu.ConvertFrom( ref bvh, false );

				Assert.AreEqual( refFile.GpuNodes.Length, ( int )gpu.UsedNodes, "BvhGpu UsedNodes" );
				List<int> diffs = new List<int>();
				for ( int i = 0; i < refFile.GpuNodes.Length; i++ )
				{
					if ( !GpuNodesEqual( gpu.Nodes[ i ], refFile.GpuNodes[ i ] ) )
					{
						diffs.Add( i );
					}
				}
				TestContext.WriteLine( $"{sceneName} BvhGpu: node mismatches {diffs.Count}/{refFile.GpuNodes.Length}" );
				Assert.AreEqual( 0, diffs.Count, $"BvhGpu node mismatch: {Describe( diffs )}" );

				Intersection[] hits = new Intersection[ refFile.GpuRays.Length ];
				for ( int i = 0; i < refFile.GpuRays.Length; i++ )
				{
					Ray ray = new Ray( refFile.GpuRays[ i ].O, refFile.GpuRays[ i ].D );
					gpu.Intersect( ref ray );
					hits[ i ] = ray.Hit;
				}
				CompareRayHits( $"{sceneName} BvhGpu", refFile.GpuRays, hits );
			}
			finally
			{
				gpu.Dispose();
				bvh.Dispose();
				verts.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void Mbvh4_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			LayoutDumpFile refFile = LayoutDumpFile.Load( refPath );
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh4 = Mbvh.Create( 4, Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				mbvh4.ConvertFrom( ref bvh, true );
				CompareMbvhNodes( $"{sceneName} Mbvh4", ref mbvh4, refFile.Mbvh4Nodes, 4 );
			}
			finally
			{
				mbvh4.Dispose();
				bvh.Dispose();
				verts.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void Bvh4Gpu_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			LayoutDumpFile refFile = LayoutDumpFile.Load( refPath );
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh4 = Mbvh.Create( 4, Allocator.Persistent );
			Bvh4Gpu gpu4 = Bvh4Gpu.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				mbvh4.ConvertFrom( ref bvh, true );
				gpu4.ConvertFrom( ref mbvh4, true );

				CompareBlocks( $"{sceneName} Bvh4Gpu", ( uint4* )gpu4.Data, gpu4.UsedBlocks, refFile.Bvh4GpuBlocks );

				// The scalar BVH4_GPU traversal decodes quantized child bounds; under Mono that runs in
				// double precision and occasionally culls a box the single-precision C++ keeps, so the
				// Mono result is reported only and the assertion runs on the Burst-compiled trace.
				int rayCount = refFile.Bvh4GpuRays.Length;
				Intersection[] hits = new Intersection[ rayCount ];
				for ( int i = 0; i < rayCount; i++ )
				{
					Ray ray = new Ray( refFile.Bvh4GpuRays[ i ].O, refFile.Bvh4GpuRays[ i ].D );
					gpu4.Intersect( ref ray );
					hits[ i ] = ray.Hit;
				}
				CompareRayHits( $"{sceneName} Bvh4Gpu (Mono)", refFile.Bvh4GpuRays, hits, false );

				NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.TempJob );
				NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.TempJob );
				NativeArray<Intersection> burstHits = new NativeArray<Intersection>( rayCount, Allocator.TempJob );
				try
				{
					for ( int i = 0; i < rayCount; i++ )
					{
						origins[ i ] = refFile.Bvh4GpuRays[ i ].O;
						directions[ i ] = refFile.Bvh4GpuRays[ i ].D;
					}
					new Bvh4GpuTraceJob { Bvh4Gpu = gpu4, Origins = origins, Directions = directions, Hits = burstHits }.Schedule( rayCount, 64 ).Complete();
					burstHits.CopyTo( hits );
				}
				finally
				{
					origins.Dispose();
					directions.Dispose();
					burstHits.Dispose();
				}
				CompareRayHits( $"{sceneName} Bvh4Gpu (Burst)", refFile.Bvh4GpuRays, hits );

				CompareMbvhNodes( $"{sceneName} Mbvh4 after Bvh4Gpu", ref mbvh4, refFile.Mbvh4NodesAfter, 4 );
			}
			finally
			{
				gpu4.Dispose();
				mbvh4.Dispose();
				bvh.Dispose();
				verts.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void PreparedBase_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			LayoutDumpFile refFile = LayoutDumpFile.Load( refPath );
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				bvh.Compact();
				bvh.SplitLeafs( 3 );

				Assert.AreEqual( refFile.PreparedNodes.Length, ( int )bvh.UsedNodes, "prepared base UsedNodes" );
				List<int> nodeDiffs = new List<int>();
				for ( int i = 0; i < refFile.PreparedNodes.Length; i++ )
				{
					if ( !BvhNodesEqual( bvh.Nodes[ i ], refFile.PreparedNodes[ i ] ) )
					{
						nodeDiffs.Add( i );
					}
				}
				TestContext.WriteLine( $"{sceneName} prepared base: node mismatches {nodeDiffs.Count}/{refFile.PreparedNodes.Length}" );
				Assert.AreEqual( 0, nodeDiffs.Count, $"prepared base node mismatch: {Describe( nodeDiffs )}" );

				Assert.AreEqual( refFile.PreparedPrimIdx.Length, ( int )bvh.IdxCount, "prepared base IdxCount" );
				List<int> idxDiffs = new List<int>();
				for ( int i = 0; i < refFile.PreparedPrimIdx.Length; i++ )
				{
					if ( bvh.PrimIdx[ i ] != refFile.PreparedPrimIdx[ i ] )
					{
						idxDiffs.Add( i );
					}
				}
				TestContext.WriteLine( $"{sceneName} prepared base: primIdx mismatches {idxDiffs.Count}/{refFile.PreparedPrimIdx.Length}" );
				Assert.AreEqual( 0, idxDiffs.Count, $"prepared base primIdx mismatch: {Describe( idxDiffs )}" );
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
		public unsafe void Mbvh8_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			LayoutDumpFile refFile = LayoutDumpFile.Load( refPath );
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh8 = Mbvh.Create( 8, Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				bvh.Compact();
				bvh.SplitLeafs( 3 );
				mbvh8.ConvertFrom( ref bvh, true );
				CompareMbvhNodes( $"{sceneName} Mbvh8", ref mbvh8, refFile.Mbvh8Nodes, 8 );
			}
			finally
			{
				mbvh8.Dispose();
				bvh.Dispose();
				verts.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void Cwbvh_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			LayoutDumpFile refFile = LayoutDumpFile.Load( refPath );
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh8 = Mbvh.Create( 8, Allocator.Persistent );
			BvhCwbvh cwbvh = BvhCwbvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				bvh.Compact();
				bvh.SplitLeafs( 3 );
				mbvh8.ConvertFrom( ref bvh, true );
				cwbvh.ConvertFrom( ref mbvh8, true );

				CompareBlocks( $"{sceneName} Cwbvh nodes", ( uint4* )cwbvh.Data, cwbvh.UsedBlocks, refFile.CwbvhBlocks );
				CompareBlocks( $"{sceneName} Cwbvh tris", ( uint4* )cwbvh.Tris, cwbvh.TriBlocks, refFile.CwbvhTriBlocks );
				CompareMbvhNodes( $"{sceneName} Mbvh8 after Cwbvh", ref mbvh8, refFile.Mbvh8NodesAfter, 8 );
			}
			finally
			{
				cwbvh.Dispose();
				mbvh8.Dispose();
				bvh.Dispose();
				verts.Dispose();
			}
		}
	}
}
