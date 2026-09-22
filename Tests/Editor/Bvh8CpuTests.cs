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
	/// Tests for the BVH8_CPU port: the conversion against the reference dump of
	/// Tools/RefDump/layoutdump.cpp, and both traversal paths - the AVX2 path and the scalar
	/// fallback - against the ray dump of Tools/RefDump/refdump.cpp. The C++ BVH8_CPU traversal
	/// needs AVX2 and is a fatal-error stub in the scalar reference build, so the rays are the
	/// base-BVH ones, exactly as Bvh4CpuTests uses them. Tests are ignored when their reference
	/// data is missing.
	/// </summary>
	public class Bvh8CpuTests
	{
		/// <summary>Barycentric distance to a triangle edge below which a hit counts as grazing.</summary>
		const float GrazingEps = 1e-3f;

		/// <summary>True when the hit lies within GrazingEps of a triangle edge or vertex, where last-bit rounding decides hit or miss.</summary>
		static bool IsGrazing( float u, float v )
		{
			return u < GrazingEps || v < GrazingEps || ( 1f - u - v ) < GrazingEps;
		}

		/// <summary>Number of differing entries listed in an assertion message before it is truncated.</summary>
		const int MaxReportedDiffs = 4;

		[BurstCompile( CompileSynchronously = true )]
		private struct IntersectJob : IJobParallelFor
		{
			public Bvh8Cpu Bvh8;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> Occluded;
			/// <summary>Forces the scalar fallback instead of the AVX2 path.</summary>
			public bool Scalar;

			public void Execute( int i )
			{
				Ray ray = new Ray( Origins[ i ], Directions[ i ] );
				Ray shadowRay = new Ray( Origins[ i ], Directions[ i ] );
				if ( Scalar )
				{
					Bvh8.IntersectScalarPath( ref ray );
					Occluded[ i ] = Bvh8.IsOccludedScalarPath( shadowRay ) ? 1 : 0;
				}
				else
				{
					Bvh8.Intersect( ref ray );
					Occluded[ i ] = Bvh8.IsOccluded( shadowRay ) ? 1 : 0;
				}
				Hits[ i ] = ray.Hit;
			}
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

		static unsafe bool MbvhNodesEqual( MbvhNode* a, LayoutDumpFile.MbvhNodeRecord b )
		{
			if ( !BitsEqual( a->AabbMin, b.AabbMin ) || a->FirstTri != b.FirstTri
				|| !BitsEqual( a->AabbMax, b.AabbMax ) || a->TriCount != b.TriCount
				|| a->ChildCount != b.ChildCount )
			{
				return false;
			}
			for ( int i = 0; i < 8; i++ )
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

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void Build_MatchesReference( string sceneName )
		{
			string binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			string refPath = BvhSceneFile.TestDataPath( sceneName + ".layouts.ref" );
			if ( !File.Exists( binPath ) || !File.Exists( refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			LayoutDumpFile refFile = LayoutDumpFile.Load( refPath );
			if ( !refFile.HasBvh8Cpu )
			{
				Assert.Ignore( $"{refPath} predates the BVH8_CPU section (needs TBVHLAY4); re-run Tools/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh8Cpu bvh8Cpu = Bvh8Cpu.Create( Allocator.Persistent );
			try
			{
				bvh8Cpu.Build( verts, triCount );

				// the base BVH, after Compact(), CombineLeafs( 4, .. ) and SplitLeafs( 4 )
				Assert.AreEqual( refFile.Bvh8CpuBaseNodes.Length, ( int )bvh8Cpu.Base.UsedNodes, "base UsedNodes" );
				List<int> nodeDiffs = new List<int>();
				for ( int i = 0; i < refFile.Bvh8CpuBaseNodes.Length; i++ )
				{
					if ( !BvhNodesEqual( bvh8Cpu.Base.Nodes[ i ], refFile.Bvh8CpuBaseNodes[ i ] ) )
					{
						nodeDiffs.Add( i );
					}
				}
				TestContext.WriteLine( $"{sceneName} Bvh8Cpu base: node mismatches {nodeDiffs.Count}/{refFile.Bvh8CpuBaseNodes.Length}" );
				Assert.AreEqual( 0, nodeDiffs.Count, $"base node mismatch: {Describe( nodeDiffs )}" );

				Assert.AreEqual( refFile.Bvh8CpuBasePrimIdx.Length, ( int )bvh8Cpu.Base.IdxCount, "base IdxCount" );
				List<int> idxDiffs = new List<int>();
				for ( int i = 0; i < refFile.Bvh8CpuBasePrimIdx.Length; i++ )
				{
					if ( bvh8Cpu.Base.PrimIdx[ i ] != refFile.Bvh8CpuBasePrimIdx[ i ] )
					{
						idxDiffs.Add( i );
					}
				}
				TestContext.WriteLine( $"{sceneName} Bvh8Cpu base: primIdx mismatches {idxDiffs.Count}/{refFile.Bvh8CpuBasePrimIdx.Length}" );
				Assert.AreEqual( 0, idxDiffs.Count, $"base primIdx mismatch: {Describe( idxDiffs )}" );

				// the re-converted MBVH<8>, i.e. the snapshot the dump takes right before the
				// node/leaf block conversion reads it
				Assert.AreEqual( refFile.Bvh8CpuMbvhNodes.Length, ( int )bvh8Cpu.Mbvh8.UsedNodes, "Mbvh8 UsedNodes" );
				List<int> mbvhDiffs = new List<int>();
				for ( int i = 0; i < refFile.Bvh8CpuMbvhNodes.Length; i++ )
				{
					if ( !MbvhNodesEqual( bvh8Cpu.Mbvh8.Nodes + i, refFile.Bvh8CpuMbvhNodes[ i ] ) )
					{
						mbvhDiffs.Add( i );
					}
				}
				TestContext.WriteLine( $"{sceneName} Bvh8Cpu mbvh8: node mismatches {mbvhDiffs.Count}/{refFile.Bvh8CpuMbvhNodes.Length}" );

				// the 64-byte node/leaf blocks
				Assert.AreEqual( ( int )refFile.Bvh8CpuUsedBlocks, ( int )bvh8Cpu.UsedBlocks, "UsedBlocks" );
				uint* words = ( uint* )bvh8Cpu.Data;
				uint4[] refBlocks = refFile.Bvh8CpuBlocks;
				List<int> blockDiffs = new List<int>();
				int firstWord = -1;
				for ( int b = 0; b < ( int )refFile.Bvh8CpuUsedBlocks; b++ )
				{
					for ( int w = 0; w < 16; w++ )
					{
						int idx = ( b * 16 ) + w;
						if ( words[ idx ] != refBlocks[ idx >> 2 ][ idx & 3 ] )
						{
							if ( firstWord < 0 )
							{
								firstWord = idx;
							}
							blockDiffs.Add( b );
							break;
						}
					}
				}
				TestContext.WriteLine( $"{sceneName} Bvh8Cpu: block mismatches {blockDiffs.Count}/{refFile.Bvh8CpuUsedBlocks}" );
				if ( blockDiffs.Count > 0 )
				{
					int b = blockDiffs[ 0 ];
					int w = firstWord - ( b * 16 );
					uint expected = refBlocks[ firstWord >> 2 ][ firstWord & 3 ];
					TestContext.WriteLine( $"  first difference in block {b}, byte offset {w * 4}: got 0x{words[ firstWord ]:x8}, expected 0x{expected:x8}" );
					TestContext.WriteLine( mbvhDiffs.Count > 0
						? $"  the MBVH<8> snapshot already differs ({Describe( mbvhDiffs )}), so the block conversion is not necessarily at fault"
						: "  the MBVH<8> snapshot matches, so the difference comes from the block conversion itself" );
				}
				Assert.AreEqual( 0, mbvhDiffs.Count, $"mbvh8 node mismatch: {Describe( mbvhDiffs )}" );
				Assert.AreEqual( 0, blockDiffs.Count, $"block mismatch: {Describe( blockDiffs )}" );
			}
			finally
			{
				bvh8Cpu.Dispose();
				verts.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void IntersectJob_MatchesReference( string sceneName )
		{
			if ( !Bvh8Cpu.IsSimdSupported )
			{
				Assert.Ignore( "this machine has no AVX2/FMA support, so the SIMD traversal cannot run; the scalar fallback is covered by IntersectScalarJob_MatchesReference" );
			}
			RunJob( sceneName, false, $"{sceneName} Bvh8Cpu burst avx2", true );
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void IntersectScalarJob_MatchesReference( string sceneName )
		{
			RunJob( sceneName, true, $"{sceneName} Bvh8Cpu burst scalar", true );
		}

		/// <summary>
		/// The scalar fallback under Mono, where float arithmetic is evaluated in double: reported
		/// only, since that widening can legitimately move a hit across a triangle edge.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Intersect_MatchesReference( string sceneName )
		{
			string binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			string refPath = BvhSceneFile.TestDataPath( sceneName + ".ref" );
			if ( !File.Exists( binPath ) || !File.Exists( refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			RefDumpFile refFile = RefDumpFile.Load( refPath );
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh8Cpu bvh8Cpu = Bvh8Cpu.Create( Allocator.Persistent );
			try
			{
				bvh8Cpu.Build( verts, triCount );

				int rayCount = refFile.Rays.Length;
				Intersection[] hits = new Intersection[ rayCount ];
				int[] occluded = new int[ rayCount ];
				for ( int i = 0; i < rayCount; i++ )
				{
					Ray ray = new Ray( refFile.Rays[ i ].O, refFile.Rays[ i ].D );
					bvh8Cpu.IntersectScalarPath( ref ray );
					hits[ i ] = ray.Hit;
					occluded[ i ] = bvh8Cpu.IsOccludedScalarPath( new Ray( refFile.Rays[ i ].O, refFile.Rays[ i ].D ) ) ? 1 : 0;
				}
				Compare( $"{sceneName} Bvh8Cpu mono scalar", refFile, hits, occluded, false );
			}
			finally
			{
				bvh8Cpu.Dispose();
				verts.Dispose();
			}
		}

		static void RunJob( string sceneName, bool scalar, string label, bool assert )
		{
			string binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			string refPath = BvhSceneFile.TestDataPath( sceneName + ".ref" );
			if ( !File.Exists( binPath ) || !File.Exists( refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			RefDumpFile refFile = RefDumpFile.Load( refPath );
			int rayCount = refFile.Rays.Length;
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> hits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			NativeArray<int> occluded = new NativeArray<int>( rayCount, Allocator.Persistent );
			Bvh8Cpu bvh8Cpu = Bvh8Cpu.Create( Allocator.Persistent );
			try
			{
				bvh8Cpu.Build( verts, triCount );
				for ( int i = 0; i < rayCount; i++ )
				{
					origins[ i ] = refFile.Rays[ i ].O;
					directions[ i ] = refFile.Rays[ i ].D;
				}

				IntersectJob job = new IntersectJob
				{
					Bvh8 = bvh8Cpu,
					Origins = origins,
					Directions = directions,
					Hits = hits,
					Occluded = occluded,
					Scalar = scalar
				};
				job.Schedule( rayCount, 64 ).Complete();

				Intersection[] hitArray = new Intersection[ rayCount ];
				int[] occludedArray = new int[ rayCount ];
				for ( int i = 0; i < rayCount; i++ )
				{
					hitArray[ i ] = hits[ i ];
					occludedArray[ i ] = occluded[ i ];
				}
				Compare( label, refFile, hitArray, occludedArray, assert );
			}
			finally
			{
				bvh8Cpu.Dispose();
				verts.Dispose();
				origins.Dispose();
				directions.Dispose();
				hits.Dispose();
				occluded.Dispose();
			}
		}

		/// <summary>
		/// Builds over the un-perturbed vertices, applies the same deterministic perturbation the
		/// reference applies, refits the layout and traces the reference rays through it under a
		/// Burst job, against the refit section of the ray dump. Takes the AVX2 path where the
		/// machine supports it and the scalar fallback otherwise.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Refit_MatchesReference( string sceneName )
		{
			string binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			string refPath = BvhSceneFile.TestDataPath( sceneName + ".ref" );
			if ( !File.Exists( binPath ) || !File.Exists( refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			bool scalar = !Bvh8Cpu.IsSimdSupported;
			RefDumpFile refFile = RefDumpFile.Load( refPath );
			int rayCount = refFile.Rays.Length;
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> hits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			NativeArray<int> occluded = new NativeArray<int>( rayCount, Allocator.Persistent );
			Bvh8Cpu bvh8Cpu = Bvh8Cpu.Create( Allocator.Persistent );
			try
			{
				bvh8Cpu.Build( verts, triCount );

				// Deterministic per-vertex perturbation, identical to refdump.cpp; ext is measured
				// against the root bounds before perturbation.
				float3 ext = bvh8Cpu.AabbMax - bvh8Cpu.AabbMin;
				uint vertCount = triCount * 3;
				for ( uint i = 0; i < vertCount; i++ )
				{
					float4 v = verts[ ( int )i ];
					v.x = v.x + ( ( ( int )( i % 7 ) - 3 ) * 0.005f * ext.x );
					v.y = v.y + ( ( ( int )( i % 5 ) - 2 ) * 0.005f * ext.y );
					verts[ ( int )i ] = v;
				}

				bvh8Cpu.Refit();

				// The sample refits every frame, so the reshaping ConvertFrom does must be idempotent.
				uint usedNodes = bvh8Cpu.Base.UsedNodes, usedBlocks = bvh8Cpu.UsedBlocks;
				bvh8Cpu.Refit();
				Assert.AreEqual( usedNodes, bvh8Cpu.Base.UsedNodes, "a repeated refit grew the base node pool" );
				Assert.AreEqual( usedBlocks, bvh8Cpu.UsedBlocks, "a repeated refit grew the block blob" );

				Assert.IsTrue( BoundsMatch( bvh8Cpu.AabbMin, refFile.RefitAabbMin ), $"refit AabbMin {bvh8Cpu.AabbMin} != {refFile.RefitAabbMin}" );
				Assert.IsTrue( BoundsMatch( bvh8Cpu.AabbMax, refFile.RefitAabbMax ), $"refit AabbMax {bvh8Cpu.AabbMax} != {refFile.RefitAabbMax}" );

				for ( int i = 0; i < rayCount; i++ )
				{
					origins[ i ] = refFile.Rays[ i ].O;
					directions[ i ] = refFile.Rays[ i ].D;
				}

				IntersectJob job = new IntersectJob
				{
					Bvh8 = bvh8Cpu,
					Origins = origins,
					Directions = directions,
					Hits = hits,
					Occluded = occluded,
					Scalar = scalar
				};
				job.Schedule( rayCount, 64 ).Complete();

				Intersection[] hitArray = new Intersection[ rayCount ];
				for ( int i = 0; i < rayCount; i++ )
				{
					hitArray[ i ] = hits[ i ];
				}
				CompareRefit( $"{sceneName} Bvh8Cpu refit {( scalar ? "scalar" : "avx2" )}", refFile, hitArray );
			}
			finally
			{
				bvh8Cpu.Dispose();
				verts.Dispose();
				origins.Dispose();
				directions.Dispose();
				hits.Dispose();
				occluded.Dispose();
			}
		}

		/// <summary>A spatial-split build is not refittable, so Refit must reject it like Bvh.Refit does.</summary>
		[Test]
		public void Refit_AfterSpatialSplitBuild_Throws()
		{
			string binPath = BvhSceneFile.TestDataPath( "suzanne.bin" );
			if ( !File.Exists( binPath ) )
			{
				Assert.Ignore( $"missing {binPath}; run TestData/fetch.ps1" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh8Cpu bvh8Cpu = Bvh8Cpu.Create( Allocator.Persistent );
			try
			{
				bvh8Cpu.UseSpatialSplits = true;
				bvh8Cpu.Build( verts, triCount );
				Assert.IsFalse( bvh8Cpu.Refittable, "an SBVH must not be marked refittable" );

				System.InvalidOperationException caught = null;
				try
				{
					bvh8Cpu.Refit();
				}
				catch ( System.InvalidOperationException e )
				{
					caught = e;
				}
				Assert.IsNotNull( caught, "Refit on a spatial-split build must throw" );
				Assert.AreEqual( "Bvh.Refit( .. ), refitting an SBVH or pre-splitted BVH.", caught.Message );
			}
			finally
			{
				bvh8Cpu.Dispose();
				verts.Dispose();
			}
		}

		/// <summary>Relative comparison of two bounds vectors, with an absolute floor for components near zero.</summary>
		static bool BoundsMatch( float3 actual, float3 expected )
		{
			return math.all( math.abs( actual - expected ) <= ( 1e-5f * math.max( math.abs( expected ), 1f ) ) );
		}

		/// <summary>
		/// Compares against the refit section of the reference dump: one hit per BLAS ray, traced
		/// through the perturbed geometry. Same tie and grazing classification as <see cref="Compare"/>;
		/// the refit section carries no occlusion results, so only the closest hits are compared.
		/// </summary>
		static void CompareRefit( string label, RefDumpFile refFile, Intersection[] hits )
		{
			int rayCount = refFile.Rays.Length;
			int mismatches = 0, ties = 0, grazing = 0;
			for ( int i = 0; i < rayCount; i++ )
			{
				RefDumpFile.RefitHit rh = refFile.RefitHits[ i ];
				Intersection hit = hits[ i ];
				bool refHit = rh.T < BvhConstants.Far;
				bool gotHit = hit.T < BvhConstants.Far;
				bool sameT = refHit && gotHit && math.abs( hit.T - rh.T ) <= 1e-4f * math.max( math.abs( rh.T ), 1e-12f );
				if ( refHit != gotHit || ( refHit && !sameT ) )
				{
					// A hit within a hair of a triangle edge or vertex is decided by last-bit rounding;
					// the two sides can legitimately disagree on it and then hit different geometry.
					if ( ( refHit && IsGrazing( rh.U, rh.V ) ) || ( gotHit && IsGrazing( hit.U, hit.V ) ) )
					{
						grazing++;
					}
					else
					{
						if ( mismatches < 3 )
						{
							TestContext.WriteLine( $"  mismatch ray {i}: ref t {rh.T} u {rh.U} v {rh.V} prim {rh.Prim} | got t {hit.T} u {hit.U} v {hit.V} prim {hit.Prim}" );
						}
						mismatches++;
					}
				}
				else if ( refHit && hit.Prim != rh.Prim )
				{
					ties++;
				}
			}
			TestContext.WriteLine( $"{label}: mismatches {mismatches}/{rayCount}, same-distance ties {ties}, grazing {grazing}" );
			Assert.AreEqual( 0, mismatches, label + " intersect mismatches" );
		}

		/// <summary>
		/// Compares against the scalar BVH reference. A different primitive at the same distance is
		/// a legitimate tie between overlapping triangles, decided by traversal order and last-bit
		/// rounding, so it is counted separately and does not fail the test.
		/// </summary>
		static void Compare( string label, RefDumpFile refFile, Intersection[] hits, int[] occluded, bool assert )
		{
			int rayCount = refFile.Rays.Length;
			int mismatches = 0, ties = 0, grazing = 0, occlusionMismatches = 0;
			for ( int i = 0; i < rayCount; i++ )
			{
				RefDumpFile.RayHit rh = refFile.Rays[ i ];
				Intersection hit = hits[ i ];
				bool refHit = rh.T < BvhConstants.Far;
				bool gotHit = hit.T < BvhConstants.Far;
				bool sameT = refHit && gotHit && math.abs( hit.T - rh.T ) <= 1e-4f * math.max( math.abs( rh.T ), 1e-12f );
				if ( refHit != gotHit || ( refHit && !sameT ) )
				{
					// A hit within a hair of a triangle edge or vertex is decided by last-bit rounding;
					// the two sides can legitimately disagree on it and then hit different geometry.
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
				else if ( refHit && hit.Prim != rh.Prim )
				{
					ties++;
				}
				if ( ( occluded[ i ] != 0 ) != ( rh.OccludedFull != 0 ) )
				{
					occlusionMismatches++;
				}
			}
			TestContext.WriteLine( $"{label}: mismatches {mismatches}/{rayCount}, same-distance ties {ties}, grazing {grazing}, occlusion mismatches {occlusionMismatches}" );
			if ( assert )
			{
				Assert.AreEqual( 0, mismatches, label + " intersect mismatches" );
				Assert.AreEqual( 0, occlusionMismatches, label + " occlusion mismatches" );
			}
		}
	}
}
