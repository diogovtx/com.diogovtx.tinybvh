using System.IO;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace TinyBVH.Tests
{
	/// <summary>Runs traversal inside a Burst-compiled job and compares against the C++ reference dumps.</summary>
	public class BvhBurstTests
	{
		/// <summary>
		/// Guards the whole suite: when any Burst direct call in the TinyBVH assembly fails to compile,
		/// every builder falls back to Mono and the bit-exact reference tests drift by a few nodes.
		/// This test turns that into one clear failure.
		/// </summary>
		[Test]
		public void DirectCalls_RunUnderBurst()
		{
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );
		}

		[BurstCompile( CompileSynchronously = true )]
		private struct IntersectJob : IJobParallelFor
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

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void IntersectJob_MatchesReference( string sceneName )
		{
			string binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			string refPath = BvhSceneFile.TestDataPath( sceneName + ".ref" );
			if ( !File.Exists( binPath ) || !File.Exists( refPath ) )
			{
				Assert.Ignore( $"Missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat." );
			}

			RefDumpFile refFile = RefDumpFile.Load( refPath );
			int rayCount = refFile.Rays.Length;
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> hits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			NativeArray<int> occluded = new NativeArray<int>( rayCount, Allocator.Persistent );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				for ( int i = 0; i < rayCount; i++ )
				{
					origins[ i ] = refFile.Rays[ i ].O;
					directions[ i ] = refFile.Rays[ i ].D;
				}

				IntersectJob job = new IntersectJob
				{
					Bvh = bvh,
					Origins = origins,
					Directions = directions,
					Hits = hits,
					Occluded = occluded
				};
				job.Schedule( rayCount, 64 ).Complete();

				int mismatches = 0, ties = 0, occlusionMismatches = 0;
				for ( int i = 0; i < rayCount; i++ )
				{
					RefDumpFile.RayHit rh = refFile.Rays[ i ];
					Intersection hit = hits[ i ];
					bool refHit = rh.T < BvhConstants.Far;
					bool gotHit = hit.T < BvhConstants.Far;
					if ( refHit != gotHit )
					{
						mismatches++;
					}
					else if ( refHit )
					{
						bool sameT = math.abs( hit.T - rh.T ) <= 1e-4f * math.abs( rh.T );
						if ( !sameT )
						{
							mismatches++;
						}
						else if ( hit.Prim != rh.Prim )
						{
							ties++;
						}
					}
					if ( ( occluded[ i ] != 0 ) != ( rh.OccludedFull != 0 ) )
					{
						occlusionMismatches++;
					}
				}
				TestContext.WriteLine( $"{sceneName} burst: mismatches {mismatches}/{rayCount}, same-distance ties {ties}, occlusion mismatches {occlusionMismatches}" );
				Assert.AreEqual( 0, mismatches, "intersect mismatches" );
				Assert.AreEqual( 0, occlusionMismatches, "occlusion mismatches" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
				origins.Dispose();
				directions.Dispose();
				hits.Dispose();
				occluded.Dispose();
			}
		}
	}
}
