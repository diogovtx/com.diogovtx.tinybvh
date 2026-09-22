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
	/// Tests for a TLAS over BLASses of mixed layouts - the C++ 'BVHBase** blasList' - against the
	/// mixed-TLAS section of the reference dump of Tools/RefDump/simddump.cpp. The BLAS list is
	/// { BVH, BVH4_CPU, BVH8_CPU, BVH_SoA }, each built over the same soup with the scalar binned
	/// builder, with five instances; see simddump.cpp for the transforms and masks, which are read
	/// straight from the dump so the setup cannot drift from the reference.
	/// Tests are ignored when their reference data is missing.
	/// </summary>
	public class BvhMixedTlasTests
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

		/// <summary>
		/// Traces the reference rays through the TLAS inside Burst, where float expressions are not
		/// widened to double. Mirrors the tracing loop of simddump.cpp: the closest hit and one
		/// full-length occlusion query.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		private struct TlasIntersectJob : IJobParallelFor
		{
			public Bvh Tlas;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> OccludedFull;

			public void Execute( int i )
			{
				Ray ray = new Ray( Origins[ i ], Directions[ i ] );
				Tlas.Intersect( ref ray );
				Hits[ i ] = ray.Hit;
				OccludedFull[ i ] = Tlas.IsOccluded( new Ray( Origins[ i ], Directions[ i ] ) ) ? 1 : 0;
			}
		}

		/// <summary>Traces the same rays through two TLASses, for the Bvh*-versus-BlasRef comparison.</summary>
		[BurstCompile( CompileSynchronously = true )]
		private struct TlasCompareJob : IJobParallelFor
		{
			public Bvh TlasA;
			public Bvh TlasB;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> HitsA;
			public NativeArray<Intersection> HitsB;

			public void Execute( int i )
			{
				Ray rayA = new Ray( Origins[ i ], Directions[ i ] );
				TlasA.Intersect( ref rayA );
				HitsA[ i ] = rayA.Hit;
				Ray rayB = new Ray( Origins[ i ], Directions[ i ] );
				TlasB.Intersect( ref rayB );
				HitsB[ i ] = rayB.Hit;
			}
		}

		static bool TryGetPaths( string sceneName, out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( sceneName + ".simd.ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		/// <summary>
		/// Builds the four BLASses the dump uses and fills the instance array from the dump's own
		/// transforms, so no float arithmetic in managed code can move the setup off the reference.
		/// </summary>
		static void Prepare( SimdDumpFile refFile, NativeArray<float4> verts, uint triCount,
			ref Bvh binned, ref Bvh4Cpu bvh4, ref Bvh8Cpu bvh8, ref BvhSoa soa,
			NativeArray<BlasInstance> instances )
		{
			binned.Build( verts, triCount );
			bvh4.Build( verts, triCount );
			bvh8.Build( verts, triCount );
			soa.Build( verts, triCount );
			for ( int i = 0; i < instances.Length; i++ )
			{
				SimdDumpFile.InstanceRecord expected = refFile.TlasInstances[ i ];
				BlasInstance inst = BlasInstance.Create( expected.BlasIdx );
				for ( int c = 0; c < 16; c++ )
				{
					inst.Transform[ c ] = expected.Transform[ c ];
				}
				inst.Mask = expected.Mask;
				instances[ i ] = inst;
			}
		}

		/// <summary>
		/// The whole mixed TLAS: the instance records the build leaves behind, the tree itself and
		/// all 65536 reference rays, each bit for bit.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void MixedTlas_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			SimdDumpFile refFile = SimdDumpFile.Load( refPath );
			int instCount = refFile.TlasInstances.Length;
			int rayCount = refFile.TlasRays.Length;
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<Bvh> bvhArray = new NativeArray<Bvh>( 1, Allocator.Persistent );
			NativeArray<Bvh4Cpu> bvh4Array = new NativeArray<Bvh4Cpu>( 1, Allocator.Persistent );
			NativeArray<Bvh8Cpu> bvh8Array = new NativeArray<Bvh8Cpu>( 1, Allocator.Persistent );
			NativeArray<BvhSoa> soaArray = new NativeArray<BvhSoa>( 1, Allocator.Persistent );
			NativeArray<BlasRef> blasList = new NativeArray<BlasRef>( 4, Allocator.Persistent );
			NativeArray<BlasInstance> instances = new NativeArray<BlasInstance>( instCount, Allocator.Persistent );
			NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> hits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			NativeArray<int> occludedFull = new NativeArray<int>( rayCount, Allocator.Persistent );
			Bvh binned = Bvh.Create( Allocator.Persistent );
			Bvh4Cpu bvh4 = Bvh4Cpu.Create( Allocator.Persistent );
			Bvh8Cpu bvh8 = Bvh8Cpu.Create( Allocator.Persistent );
			BvhSoa soa = BvhSoa.Create( Allocator.Persistent );
			Bvh tlas = Bvh.Create( Allocator.Persistent );
			try
			{
				Prepare( refFile, verts, triCount, ref binned, ref bvh4, ref bvh8, ref soa, instances );
				bvhArray[ 0 ] = binned;
				bvh4Array[ 0 ] = bvh4;
				bvh8Array[ 0 ] = bvh8;
				soaArray[ 0 ] = soa;
				blasList[ 0 ] = BlasRef.From( ( Bvh* )bvhArray.GetUnsafePtr() );
				blasList[ 1 ] = BlasRef.From( ( Bvh4Cpu* )bvh4Array.GetUnsafePtr() );
				blasList[ 2 ] = BlasRef.From( ( Bvh8Cpu* )bvh8Array.GetUnsafePtr() );
				blasList[ 3 ] = BlasRef.From( ( BvhSoa* )soaArray.GetUnsafePtr() );

				tlas.BuildTlas( ( BlasInstance* )instances.GetUnsafePtr(), ( uint )instCount,
					( BlasRef* )blasList.GetUnsafePtr(), ( uint )blasList.Length );

				// instance records, as the build left them.
				for ( int i = 0; i < instCount; i++ )
				{
					BlasInstance built = instances[ i ];
					SimdDumpFile.InstanceRecord expected = refFile.TlasInstances[ i ];
					Assert.AreEqual( expected.BlasIdx, built.BlasIdx, $"instance {i} blasIdx" );
					Assert.AreEqual( expected.Mask, built.Mask, $"instance {i} mask" );
					for ( int c = 0; c < 16; c++ )
					{
						Assert.IsTrue( BitsEqual( built.Transform[ c ], expected.Transform[ c ] ), $"instance {i} transform[{c}]" );
						Assert.IsTrue( BitsEqual( built.InvTransform[ c ], expected.InvTransform[ c ] ), $"instance {i} invTransform[{c}]: got {built.InvTransform[ c ]}, expected {expected.InvTransform[ c ]}" );
					}
					Assert.IsTrue( BitsEqual( built.AabbMin, expected.AabbMin ), $"instance {i} aabbMin: got {built.AabbMin}, expected {expected.AabbMin}" );
					Assert.IsTrue( BitsEqual( built.AabbMax, expected.AabbMax ), $"instance {i} aabbMax: got {built.AabbMax}, expected {expected.AabbMax}" );
				}

				// the TLAS itself.
				Assert.AreEqual( refFile.TlasUsedNodes, tlas.UsedNodes, "TLAS UsedNodes" );
				Assert.IsTrue( BitsEqual( tlas.AabbMin, refFile.TlasAabbMin ), $"TLAS aabbMin: got {tlas.AabbMin}, expected {refFile.TlasAabbMin}" );
				Assert.IsTrue( BitsEqual( tlas.AabbMax, refFile.TlasAabbMax ), $"TLAS aabbMax: got {tlas.AabbMax}, expected {refFile.TlasAabbMax}" );
				List<int> nodeDiffs = new List<int>();
				for ( int i = 0; i < refFile.TlasNodes.Length; i++ )
				{
					if ( !BvhNodesEqual( tlas.Nodes[ i ], refFile.TlasNodes[ i ] ) )
					{
						nodeDiffs.Add( i );
					}
				}
				Assert.AreEqual( 0, nodeDiffs.Count, $"TLAS node mismatch: {Describe( nodeDiffs )}" );
				Assert.AreEqual( refFile.TlasPrimIdx.Length, ( int )tlas.IdxCount, "TLAS IdxCount" );
				for ( int i = 0; i < refFile.TlasPrimIdx.Length; i++ )
				{
					Assert.AreEqual( refFile.TlasPrimIdx[ i ], tlas.PrimIdx[ i ], $"TLAS primIdx[{i}]" );
				}

				// traversal.
				for ( int i = 0; i < rayCount; i++ )
				{
					origins[ i ] = refFile.TlasRays[ i ].O;
					directions[ i ] = refFile.TlasRays[ i ].D;
				}
				TlasIntersectJob job = new TlasIntersectJob
				{
					Tlas = tlas,
					Origins = origins,
					Directions = directions,
					Hits = hits,
					OccludedFull = occludedFull
				};
				job.Schedule( rayCount, 64 ).Complete();

				int mismatches = 0, grazing = 0, occMismatches = 0;
				int[] perInstance = new int[ instCount ];
				for ( int i = 0; i < rayCount; i++ )
				{
					RefDumpFile.TlasRayHit rh = refFile.TlasRays[ i ];
					Intersection hit = hits[ i ];
					bool same = BitsEqual( hit.T, rh.T ) && BitsEqual( hit.U, rh.U ) && BitsEqual( hit.V, rh.V )
						&& hit.Prim == rh.Prim && hit.Inst == rh.Inst;
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
								TestContext.WriteLine( $"  mismatch ray {i}: O {rh.O} D {rh.D} ref t {rh.T} u {rh.U} v {rh.V} prim {rh.Prim} inst {rh.Inst} | got t {hit.T} u {hit.U} v {hit.V} prim {hit.Prim} inst {hit.Inst}" );
							}
							mismatches++;
						}
					}
					if ( ( occludedFull[ i ] != 0 ) != ( rh.OccludedFull != 0 ) )
					{
						occMismatches++;
					}
					if ( hit.T < BvhConstants.Far && hit.Inst < instCount )
					{
						perInstance[ hit.Inst ]++;
					}
				}
				TestContext.WriteLine( $"{sceneName} mixed tlas: nodes {tlas.UsedNodes}, mismatches {mismatches}/{rayCount}, grazing {grazing}, occlusion mismatches {occMismatches}, per instance [{string.Join( ", ", perInstance )}]" );
				Assert.AreEqual( 0, mismatches, "mixed tlas intersect mismatches" );
				Assert.AreEqual( 0, occMismatches, "mixed tlas occlusion mismatches" );
			}
			finally
			{
				tlas.Dispose();
				soa.Dispose();
				bvh8.Dispose();
				bvh4.Dispose();
				binned.Dispose();
				verts.Dispose();
				bvhArray.Dispose();
				bvh4Array.Dispose();
				bvh8Array.Dispose();
				soaArray.Dispose();
				blasList.Dispose();
				instances.Dispose();
				origins.Dispose();
				directions.Dispose();
				hits.Dispose();
				occludedFull.Dispose();
			}
		}

		/// <summary>
		/// A BlasRef list holding only Bvh entries has to produce exactly the TLAS the older Bvh*
		/// overload produces, and trace identically: the two differ only in how the traversal
		/// reaches the BLAS.
		/// </summary>
		[TestCase( "bunny" )]
		public unsafe void BlasRefList_MatchesBvhOverload( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			SimdDumpFile refFile = SimdDumpFile.Load( refPath );
			int instCount = refFile.TlasInstances.Length;
			int rayCount = refFile.TlasRays.Length;
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<Bvh> bvhArray = new NativeArray<Bvh>( 1, Allocator.Persistent );
			NativeArray<BlasRef> blasList = new NativeArray<BlasRef>( 1, Allocator.Persistent );
			NativeArray<BlasInstance> instancesA = new NativeArray<BlasInstance>( instCount, Allocator.Persistent );
			NativeArray<BlasInstance> instancesB = new NativeArray<BlasInstance>( instCount, Allocator.Persistent );
			NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> hitsA = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> hitsB = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			Bvh blas = Bvh.Create( Allocator.Persistent );
			Bvh tlasA = Bvh.Create( Allocator.Persistent );
			Bvh tlasB = Bvh.Create( Allocator.Persistent );
			try
			{
				blas.Build( verts, triCount );
				bvhArray[ 0 ] = blas;
				blasList[ 0 ] = BlasRef.From( ( Bvh* )bvhArray.GetUnsafePtr() );
				for ( int i = 0; i < instCount; i++ )
				{
					BlasInstance inst = BlasInstance.Create( 0 );
					for ( int c = 0; c < 16; c++ )
					{
						inst.Transform[ c ] = refFile.TlasInstances[ i ].Transform[ c ];
					}
					inst.Mask = refFile.TlasInstances[ i ].Mask;
					instancesA[ i ] = inst;
					instancesB[ i ] = inst;
				}

				tlasA.BuildTlas( ( BlasInstance* )instancesA.GetUnsafePtr(), ( uint )instCount, ( Bvh* )bvhArray.GetUnsafePtr(), 1 );
				tlasB.BuildTlas( ( BlasInstance* )instancesB.GetUnsafePtr(), ( uint )instCount, ( BlasRef* )blasList.GetUnsafePtr(), 1 );

				Assert.AreEqual( tlasA.UsedNodes, tlasB.UsedNodes, "UsedNodes" );
				Assert.AreEqual( 0, UnsafeUtility.MemCmp( tlasA.Nodes, tlasB.Nodes, ( long )tlasA.UsedNodes * sizeof( BvhNode ) ), "TLAS node bytes differ" );
				Assert.AreEqual( tlasA.IdxCount, tlasB.IdxCount, "IdxCount" );
				Assert.AreEqual( 0, UnsafeUtility.MemCmp( tlasA.PrimIdx, tlasB.PrimIdx, ( long )tlasA.IdxCount * sizeof( uint ) ), "TLAS primIdx bytes differ" );
				Assert.AreEqual( 0, UnsafeUtility.MemCmp( instancesA.GetUnsafePtr(), instancesB.GetUnsafePtr(), ( long )instCount * sizeof( BlasInstance ) ), "instance records differ" );

				for ( int i = 0; i < rayCount; i++ )
				{
					origins[ i ] = refFile.TlasRays[ i ].O;
					directions[ i ] = refFile.TlasRays[ i ].D;
				}
				TlasCompareJob job = new TlasCompareJob
				{
					TlasA = tlasA,
					TlasB = tlasB,
					Origins = origins,
					Directions = directions,
					HitsA = hitsA,
					HitsB = hitsB
				};
				job.Schedule( rayCount, 64 ).Complete();

				int mismatches = 0;
				for ( int i = 0; i < rayCount; i++ )
				{
					Intersection a = hitsA[ i ], b = hitsB[ i ];
					if ( !BitsEqual( a.T, b.T ) || !BitsEqual( a.U, b.U ) || !BitsEqual( a.V, b.V ) || a.Prim != b.Prim || a.Inst != b.Inst )
					{
						mismatches++;
					}
				}
				Assert.AreEqual( 0, mismatches, "BlasRef list traced differently from the Bvh* list" );
			}
			finally
			{
				tlasA.Dispose();
				tlasB.Dispose();
				blas.Dispose();
				verts.Dispose();
				bvhArray.Dispose();
				blasList.Dispose();
				instancesA.Dispose();
				instancesB.Dispose();
				origins.Dispose();
				directions.Dispose();
				hitsA.Dispose();
				hitsB.Dispose();
			}
		}
	}
}
