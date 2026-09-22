using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Tests for the BVH_SoA layout against the reference dump of Tools~/RefDump/simddump.cpp, the
	/// only dump tool compiled with SIMD enabled - the C++ BVH_SoA traversal only exists under
	/// BVH_USEAVX. The builders in that tool still run the scalar binned path, so the dump also
	/// carries the base tree; <see cref="BaseTree_MatchesReference"/> checks it first, so a drift
	/// between the SIMD and scalar reference builds cannot be mistaken for a conversion or
	/// traversal bug. Tests are ignored when their reference data is missing.
	/// </summary>
	public class BvhSoaTests
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

		static bool BitsEqual( float4 a, float4 b )
		{
			return BitsEqual( a.x, b.x ) && BitsEqual( a.y, b.y ) && BitsEqual( a.z, b.z ) && BitsEqual( a.w, b.w );
		}

		static bool BvhNodesEqual( BvhNode a, BvhNode b )
		{
			return BitsEqual( a.AabbMin, b.AabbMin ) && a.LeftFirst == b.LeftFirst
				&& BitsEqual( a.AabbMax, b.AabbMax ) && a.TriCount == b.TriCount;
		}

		static bool SoaNodesEqual( BvhSoaNode a, SimdDumpFile.SoaNodeRecord b )
		{
			return BitsEqual( a.Xxxx, b.X ) && BitsEqual( a.Yyyy, b.Y ) && BitsEqual( a.Zzzz, b.Z )
				&& a.Left == b.Left && a.Right == b.Right && a.TriCount == b.TriCount && a.FirstTri == b.FirstTri;
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

		/// <summary>
		/// Traces the reference rays through the SoA layout inside Burst, where float expressions
		/// are not widened to double. Mirrors the tracing loop of simddump.cpp: the closest hit, an
		/// occlusion query over the full ray, and a second one shortened to half the hit distance.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		private struct IntersectJob : IJobParallelFor
		{
			public BvhSoa Soa;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> OccludedFull;
			public NativeArray<int> OccludedHalf;

			public void Execute( int i )
			{
				Ray ray = new Ray( Origins[ i ], Directions[ i ] );
				Soa.Intersect( ref ray );
				Hits[ i ] = ray.Hit;
				OccludedFull[ i ] = Soa.IsOccluded( new Ray( Origins[ i ], Directions[ i ] ) ) ? 1 : 0;
				float halfT = ray.Hit.T < BvhConstants.Far ? 0.5f * ray.Hit.T : BvhConstants.Far;
				OccludedHalf[ i ] = Soa.IsOccluded( new Ray( Origins[ i ], Directions[ i ], halfT ) ) ? 1 : 0;
			}
		}

		/// <summary>Traces the same rays through two layouts, for the save/load round trip.</summary>
		[BurstCompile( CompileSynchronously = true )]
		private struct CompareJob : IJobParallelFor
		{
			public BvhSoa Original;
			public BvhSoa Loaded;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> OriginalHits;
			public NativeArray<Intersection> LoadedHits;

			public void Execute( int i )
			{
				Ray rayA = new Ray( Origins[ i ], Directions[ i ] );
				Original.Intersect( ref rayA );
				OriginalHits[ i ] = rayA.Hit;
				Ray rayB = new Ray( Origins[ i ], Directions[ i ] );
				Loaded.Intersect( ref rayB );
				LoadedHits[ i ] = rayB.Hit;
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
		/// which is what makes the conversion and traversal comparisons below meaningful.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void BaseTree_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 and Tools~/RefDump/run_all.bat" );
			}
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			SimdDumpFile refFile = SimdDumpFile.Load( refPath );
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			BvhSoa soa = BvhSoa.Create( Allocator.Persistent );
			try
			{
				soa.Build( verts, triCount );

				Assert.AreEqual( refFile.TriCount, triCount, "triCount" );
				Assert.AreEqual( refFile.BaseNodes.Length, ( int )soa.Source.UsedNodes, "UsedNodes" );
				List<int> nodeDiffs = new List<int>();
				for ( int i = 0; i < refFile.BaseNodes.Length; i++ )
				{
					if ( !BvhNodesEqual( soa.Source.Nodes[ i ], refFile.BaseNodes[ i ] ) )
					{
						nodeDiffs.Add( i );
					}
				}
				Assert.AreEqual( refFile.BasePrimIdx.Length, ( int )soa.Source.IdxCount, "IdxCount" );
				List<int> idxDiffs = new List<int>();
				for ( int i = 0; i < refFile.BasePrimIdx.Length; i++ )
				{
					if ( soa.Source.PrimIdx[ i ] != refFile.BasePrimIdx[ i ] )
					{
						idxDiffs.Add( i );
					}
				}
				TestContext.WriteLine( $"{sceneName} soa base: node mismatches {nodeDiffs.Count}/{refFile.BaseNodes.Length}, primIdx mismatches {idxDiffs.Count}/{refFile.BasePrimIdx.Length}" );
				Assert.AreEqual( 0, nodeDiffs.Count, $"base node mismatch: {Describe( nodeDiffs )}" );
				Assert.AreEqual( 0, idxDiffs.Count, $"base primIdx mismatch: {Describe( idxDiffs )}" );
			}
			finally
			{
				soa.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>The converted 64-byte nodes, against the raw node dump.</summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void Convert_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 and Tools~/RefDump/run_all.bat" );
			}
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			SimdDumpFile refFile = SimdDumpFile.Load( refPath );
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			BvhSoa soa = BvhSoa.Create( Allocator.Persistent );
			try
			{
				soa.Build( verts, triCount );

				Assert.AreEqual( refFile.SoaUsedNodes, soa.UsedNodes, "UsedNodes" );
				List<int> nodeDiffs = new List<int>();
				for ( int i = 0; i < refFile.SoaNodes.Length; i++ )
				{
					if ( !SoaNodesEqual( soa.Nodes[ i ], refFile.SoaNodes[ i ] ) )
					{
						nodeDiffs.Add( i );
					}
				}
				TestContext.WriteLine( $"{sceneName} soa convert: nodes {soa.UsedNodes}, mismatches {nodeDiffs.Count}/{refFile.SoaNodes.Length}" );
				Assert.AreEqual( 0, nodeDiffs.Count, $"soa node mismatch: {Describe( nodeDiffs )}" );
			}
			finally
			{
				soa.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>
		/// The ported BVH_SoA::Intersect and BVH_SoA::IsOccluded, over all 65536 reference rays.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void IntersectJob_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 and Tools~/RefDump/run_all.bat" );
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
			BvhSoa soa = BvhSoa.Create( Allocator.Persistent );
			try
			{
				soa.Build( verts, triCount );
				for ( int i = 0; i < rayCount; i++ )
				{
					origins[ i ] = refFile.Rays[ i ].O;
					directions[ i ] = refFile.Rays[ i ].D;
				}

				IntersectJob job = new IntersectJob
				{
					Soa = soa,
					Origins = origins,
					Directions = directions,
					Hits = hits,
					OccludedFull = occludedFull,
					OccludedHalf = occludedHalf
				};
				job.Schedule( rayCount, 64 ).Complete();

				Compare( $"{sceneName} soa cpu", refFile, hits, occludedFull, occludedHalf );
			}
			finally
			{
				soa.Dispose();
				verts.Dispose();
				origins.Dispose();
				directions.Dispose();
				hits.Dispose();
				occludedFull.Dispose();
				occludedHalf.Dispose();
			}
		}

		/// <summary>
		/// BvhSoa.Save writes the underlying BVH, exactly as the C++ does; a Load re-converts it,
		/// so the round trip has to reproduce the SoA nodes byte for byte and trace identically.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void SaveLoad_RoundTrip( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run Tools~/fetch.ps1 and Tools~/RefDump/run_all.bat" );
			}

			string savePath = Path.GetTempFileName();
			SimdDumpFile refFile = SimdDumpFile.Load( refPath );
			int rayCount = refFile.Rays.Length;
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> originalHits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> loadedHits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			BvhSoa original = BvhSoa.Create( Allocator.Persistent );
			BvhSoa loaded = BvhSoa.Create( Allocator.Persistent );
			try
			{
				original.Build( verts, triCount );
				original.Save( savePath );

				bool ok = loaded.Load( savePath, verts, triCount );

				Assert.IsTrue( ok, "Load" );
				Assert.AreEqual( original.UsedNodes, loaded.UsedNodes, "UsedNodes" );
				Assert.AreEqual( original.TriCount, loaded.TriCount, "TriCount" );
				Assert.AreEqual( original.IdxCount, loaded.IdxCount, "IdxCount" );
				Assert.AreEqual( original.Refittable, loaded.Refittable, "Refittable" );
				Assert.AreEqual( original.MayHaveHoles, loaded.MayHaveHoles, "MayHaveHoles" );
				Assert.AreEqual( original.BvhOverAabbs, loaded.BvhOverAabbs, "BvhOverAabbs" );
				Assert.AreEqual( original.BvhOverIndices, loaded.BvhOverIndices, "BvhOverIndices" );
				Assert.AreEqual( original.AabbMin, loaded.AabbMin, "AabbMin" );
				Assert.AreEqual( original.AabbMax, loaded.AabbMax, "AabbMax" );

				long nodeBytes = ( long )original.UsedNodes * sizeof( BvhSoaNode );
				Assert.AreEqual( 0, UnsafeUtility.MemCmp( original.Nodes, loaded.Nodes, nodeBytes ), "node bytes differ" );

				for ( int i = 0; i < rayCount; i++ )
				{
					origins[ i ] = refFile.Rays[ i ].O;
					directions[ i ] = refFile.Rays[ i ].D;
				}
				CompareJob job = new CompareJob
				{
					Original = original,
					Loaded = loaded,
					Origins = origins,
					Directions = directions,
					OriginalHits = originalHits,
					LoadedHits = loadedHits
				};
				job.Schedule( rayCount, 64 ).Complete();

				int mismatches = 0;
				for ( int i = 0; i < rayCount; i++ )
				{
					Intersection a = originalHits[ i ], b = loadedHits[ i ];
					if ( !BitsEqual( a.T, b.T ) || !BitsEqual( a.U, b.U ) || !BitsEqual( a.V, b.V ) || a.Prim != b.Prim )
					{
						mismatches++;
					}
				}
				Assert.AreEqual( 0, mismatches, "round-trip hit mismatches" );
			}
			finally
			{
				original.Dispose();
				loaded.Dispose();
				verts.Dispose();
				origins.Dispose();
				directions.Dispose();
				originalHits.Dispose();
				loadedHits.Dispose();
				if ( File.Exists( savePath ) )
				{
					File.Delete( savePath );
				}
			}
		}

		/// <summary>
		/// Compares against the SoA section of the dump. The traversal is the same algorithm on
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
				SimdDumpFile.Hit rh = refFile.SoaHits[ i ];
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
