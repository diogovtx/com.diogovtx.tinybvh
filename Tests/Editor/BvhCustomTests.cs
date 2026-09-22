using AOT;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

namespace TinyBVH.Tests
{
	/// <summary>
	/// BVHs over custom geometry: a scene of procedurally generated spheres, built from an AABB
	/// array and from a getAabb callback, traversed through Burst-compiled intersection callbacks
	/// and checked against a brute-force loop over all spheres.
	/// </summary>
	public unsafe class BvhCustomTests
	{
		private const uint SphereCount = 4096;
		private const int RayCount = 4096;
		/// <summary>Ray length used for the occlusion queries, short enough to leave rays unblocked.</summary>
		private const float OccludeDist = 8f;

		private struct Sphere
		{
			public float3 Center;
			public float Radius;
		}

		/// <summary>
		/// The custom geometry. The spheres are derived from the primitive index alone, so the
		/// callbacks need no scene data and the same code can generate the reference AABBs.
		/// </summary>
		[BurstCompile]
		private static class SphereGeometry
		{
			/// <summary>
			/// Center and radius of one sphere. Everything is derived with integer arithmetic and
			/// scaled by a power of two, so no rounding takes place at all and Mono - which
			/// evaluates float expressions in double - and Burst agree bit for bit.
			/// </summary>
			public static Sphere SphereAt( uint prim )
			{
				Random rng = Random.CreateFromIndex( prim );
				uint3 c = rng.NextUInt3( new uint3( 8192u ) );
				uint r = rng.NextUInt( 205u, 1229u );
				Sphere s;
				s.Center = ( new float3( ( int )c.x, ( int )c.y, ( int )c.z ) - 4096f ) * ( 1f / 1024f );
				s.Radius = r * ( 1f / 4096f );
				return s;
			}

			/// <summary>
			/// Analytic ray/sphere intersection: nearest root in front of the origin, or false when
			/// there is none. The direction is expected to be normalised.
			/// </summary>
			public static bool SphereHit( float3 o, float3 d, Sphere s, out float t )
			{
				float3 oc = o - s.Center;
				float b = math.dot( oc, d );
				float c = math.dot( oc, oc ) - ( s.Radius * s.Radius );
				float disc = ( b * b ) - c;
				t = 0f;
				if ( disc < 0f )
				{
					return false;
				}
				float sq = math.sqrt( disc );
				t = -b - sq;
				if ( t < 0f )
				{
					t = -b + sq;
				}
				return t > 0f;
			}

			[BurstCompile( CompileSynchronously = true )]
			[MonoPInvokeCallback( typeof( GetAabbDelegate ) )]
			public static void GetAabb( uint prim, float3* aabbMin, float3* aabbMax )
			{
				Sphere s = SphereAt( prim );
				*aabbMin = s.Center - s.Radius;
				*aabbMax = s.Center + s.Radius;
			}

			[BurstCompile( CompileSynchronously = true )]
			[MonoPInvokeCallback( typeof( CustomIntersectDelegate ) )]
			public static byte Intersect( Ray* ray, uint prim )
			{
				if ( !SphereHit( ray->O, ray->D, SphereAt( prim ), out float t ) || t >= ray->Hit.T )
				{
					return 0;
				}
				ray->Hit.T = t;
				ray->Hit.U = 0f;
				ray->Hit.V = 0f;
				ray->Hit.Prim = prim;
				return 1;
			}

			[BurstCompile( CompileSynchronously = true )]
			[MonoPInvokeCallback( typeof( CustomOccludedDelegate ) )]
			public static byte IsOccluded( Ray* ray, uint prim )
			{
				bool hit = SphereHit( ray->O, ray->D, SphereAt( prim ), out float t ) && t < ray->Hit.T;
				return hit ? ( byte )1 : ( byte )0;
			}
		}

		/// <summary>
		/// Builds a ray without the Ray constructor, which normalises the direction: math.normalize
		/// gives Mono and Burst directions that differ in the last bit, and a sphere grazed by the
		/// ray then lands at a very different distance. The directions here are already unit length.
		/// </summary>
		private static Ray MakeRay( float3 o, float3 d, float t )
		{
			Ray ray = default;
			ray.O = o;
			ray.D = d;
			ray.RD = BvhMath.Rcp( d );
			ray.Mask = BvhConstants.RayMaskIntersectAll;
			ray.Hit.T = t;
			return ray;
		}

		/// <summary>Traverses the BVH, which calls the sphere callbacks for the primitives in a leaf.</summary>
		[BurstCompile( CompileSynchronously = true )]
		private struct SphereTraceJob : IJobParallelFor
		{
			public Bvh Bvh;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> Occluded;

			public void Execute( int i )
			{
				Ray ray = MakeRay( Origins[ i ], Directions[ i ], BvhConstants.Far );
				Bvh.Intersect( ref ray );
				Hits[ i ] = ray.Hit;
				Occluded[ i ] = Bvh.IsOccluded( MakeRay( Origins[ i ], Directions[ i ], OccludeDist ) ) ? 1 : 0;
			}
		}

		/// <summary>
		/// The same queries without a BVH: every sphere is handed to the same callbacks, in order.
		/// Using the callbacks themselves keeps the arithmetic identical to the traversal, so the
		/// results can be compared exactly.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		private struct BruteForceJob : IJobParallelFor
		{
			public FunctionPointer<CustomIntersectDelegate> Intersect;
			public FunctionPointer<CustomOccludedDelegate> IsOccluded;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> Occluded;

			public void Execute( int i )
			{
				Ray ray = MakeRay( Origins[ i ], Directions[ i ], BvhConstants.Far );
				Ray shadowRay = MakeRay( Origins[ i ], Directions[ i ], OccludeDist );
				int occluded = 0;
				for ( uint p = 0; p < SphereCount; p++ )
				{
					Intersect.Invoke( &ray, p );
					if ( occluded == 0 && IsOccluded.Invoke( &shadowRay, p ) != 0 )
					{
						occluded = 1;
					}
				}
				Hits[ i ] = ray.Hit;
				Occluded[ i ] = occluded;
			}
		}

		private static NativeArray<float4> BuildAabbArray()
		{
			NativeArray<float4> aabbs = new NativeArray<float4>( ( int )SphereCount * 2, Allocator.Persistent, NativeArrayOptions.UninitializedMemory );
			for ( uint i = 0; i < SphereCount; i++ )
			{
				Sphere s = SphereGeometry.SphereAt( i );
				aabbs[ ( int )i * 2 ] = new float4( s.Center - s.Radius, 0f );
				aabbs[ ( ( int )i * 2 ) + 1 ] = new float4( s.Center + s.Radius, 0f );
			}
			return aabbs;
		}

		private static void MakeRays( NativeArray<float3> origins, NativeArray<float3> directions )
		{
			Random rng = new Random( 0x5EED1234u );
			for ( int i = 0; i < origins.Length; i++ )
			{
				float3 o = rng.NextFloat3Direction() * 12f;
				origins[ i ] = o;
				directions[ i ] = math.normalize( rng.NextFloat3( new float3( -4f ), new float3( 4f ) ) - o );
			}
		}

		/// <summary>Every fragment referenced by a leaf must fit inside the bounds of that leaf.</summary>
		private static void AssertLeafContainment( ref Bvh bvh )
		{
			uint* stack = stackalloc uint[ 64 ];
			uint stackPtr = 0, nodeIdx = 0, leaves = 0, prims = 0;
			while ( true )
			{
				BvhNode* node = bvh.Nodes + nodeIdx;
				if ( node->IsLeaf )
				{
					leaves++;
					for ( uint i = 0; i < node->TriCount; i++ )
					{
						uint prim = bvh.PrimIdx[ node->LeftFirst + i ];
						Sphere s = SphereGeometry.SphereAt( prim );
						float3 bmin = s.Center - s.Radius, bmax = s.Center + s.Radius;
						Assert.IsTrue( math.all( bmin >= node->AabbMin ) && math.all( bmax <= node->AabbMax ),
							$"fragment {prim} is outside the bounds of its leaf" );
						prims++;
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
				}
				else
				{
					nodeIdx = node->LeftFirst;
					stack[ stackPtr++ ] = node->LeftFirst + 1;
				}
			}
			Assert.AreEqual( SphereCount, prims, "every primitive should end up in exactly one leaf" );
			TestContext.WriteLine( $"custom bvh: {bvh.UsedNodes} nodes, {leaves} leaves" );
		}

		[Test]
		public void BuildAabbs_AndCallbackBuild_ProduceTheSameTree()
		{
			NativeArray<float4> aabbs = BuildAabbArray();
			Bvh fromArray = Bvh.Create( Allocator.Persistent );
			Bvh fromCallback = Bvh.Create( Allocator.Persistent );
			try
			{
				fromArray.BuildAabbs( aabbs, SphereCount );
				fromCallback.Build( BurstCompiler.CompileFunctionPointer<GetAabbDelegate>( SphereGeometry.GetAabb ), SphereCount );

				Assert.IsTrue( fromArray.BvhOverAabbs, "a BVH over AABBs has no vertices" );
				Assert.IsTrue( fromArray.Refittable );
				Assert.IsFalse( fromArray.BvhOverIndices );
				Assert.IsTrue( fromArray.Verts == null );
				Assert.AreEqual( SphereCount, fromArray.TriCount );
				Assert.AreEqual( SphereCount, fromArray.IdxCount );

				Assert.AreEqual( fromArray.UsedNodes, fromCallback.UsedNodes, "node count" );
				Assert.AreEqual( 0, UnsafeUtility.MemCmp( fromArray.Nodes, fromCallback.Nodes, ( long )fromArray.UsedNodes * sizeof( BvhNode ) ), "nodes" );
				Assert.AreEqual( 0, UnsafeUtility.MemCmp( fromArray.PrimIdx, fromCallback.PrimIdx, ( long )fromArray.IdxCount * sizeof( uint ) ), "primIdx" );

				AssertLeafContainment( ref fromArray );
			}
			finally
			{
				fromArray.Dispose();
				fromCallback.Dispose();
				aabbs.Dispose();
			}
		}

		[Test]
		public void CustomCallbacks_MatchBruteForce()
		{
			NativeArray<float4> aabbs = BuildAabbArray();
			NativeArray<float3> origins = new NativeArray<float3>( RayCount, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( RayCount, Allocator.Persistent );
			NativeArray<Intersection> hits = new NativeArray<Intersection>( RayCount, Allocator.Persistent );
			NativeArray<int> occluded = new NativeArray<int>( RayCount, Allocator.Persistent );
			NativeArray<Intersection> bruteHits = new NativeArray<Intersection>( RayCount, Allocator.Persistent );
			NativeArray<int> bruteOccluded = new NativeArray<int>( RayCount, Allocator.Persistent );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.BuildAabbs( aabbs, SphereCount );
				bvh.CustomIntersect = BurstCompiler.CompileFunctionPointer<CustomIntersectDelegate>( SphereGeometry.Intersect );
				bvh.CustomIsOccluded = BurstCompiler.CompileFunctionPointer<CustomOccludedDelegate>( SphereGeometry.IsOccluded );
				MakeRays( origins, directions );

				SphereTraceJob traceJob = new SphereTraceJob
				{
					Bvh = bvh,
					Origins = origins,
					Directions = directions,
					Hits = hits,
					Occluded = occluded
				};
				traceJob.Schedule( RayCount, 64 ).Complete();

				BruteForceJob bruteJob = new BruteForceJob
				{
					Intersect = bvh.CustomIntersect,
					IsOccluded = bvh.CustomIsOccluded,
					Origins = origins,
					Directions = directions,
					Hits = bruteHits,
					Occluded = bruteOccluded
				};
				bruteJob.Schedule( RayCount, 64 ).Complete();

				int mismatches = 0, ties = 0, occlusionMismatches = 0, monoMismatches = 0, monoTies = 0, hitCount = 0;
				float monoMaxRelDelta = 0f;
				for ( int i = 0; i < RayCount; i++ )
				{
					Intersection hit = hits[ i ], brute = bruteHits[ i ];
					if ( hit.T != brute.T )
					{
						mismatches++;
					}
					else if ( hit.T < BvhConstants.Far )
					{
						hitCount++;
						if ( hit.Prim != brute.Prim )
						{
							ties++;
						}
					}
					if ( occluded[ i ] != bruteOccluded[ i ] )
					{
						occlusionMismatches++;
					}

					// the same queries from Mono, which reaches the Burst-compiled callbacks through
					// the function pointers; only the traversal itself is not Burst compiled here.
					// Both see the same ray and the same callbacks, so the distances match exactly;
					// the tolerance only covers the slab test, which Mono evaluates in double.
					Ray monoRay = MakeRay( origins[ i ], directions[ i ], BvhConstants.Far );
					bvh.Intersect( ref monoRay );
					bool monoOccluded = bvh.IsOccluded( MakeRay( origins[ i ], directions[ i ], OccludeDist ) );
					float delta = math.abs( monoRay.Hit.T - hit.T );
					bool sameT = delta <= 1e-5f * math.abs( hit.T );
					monoMaxRelDelta = math.max( monoMaxRelDelta, delta / math.abs( hit.T ) );
					if ( !sameT || monoOccluded != ( occluded[ i ] != 0 ) )
					{
						monoMismatches++;
					}
					else if ( monoRay.Hit.Prim != hit.Prim )
					{
						monoTies++;
					}
				}
				TestContext.WriteLine( $"custom spheres: {hitCount}/{RayCount} hits, mismatches {mismatches}, same-distance ties {ties}, occlusion mismatches {occlusionMismatches}, mono vs job {monoMismatches} (ties {monoTies}, max relative delta {monoMaxRelDelta})" );
				Assert.AreEqual( 0, mismatches, "intersect mismatches" );
				Assert.AreEqual( 0, occlusionMismatches, "occlusion mismatches" );
				Assert.AreEqual( 0, monoMismatches, "mono vs job mismatches" );
				Assert.Greater( hitCount, RayCount / 10, "the rays should mostly hit the sphere cloud" );
			}
			finally
			{
				bvh.Dispose();
				aabbs.Dispose();
				origins.Dispose();
				directions.Dispose();
				hits.Dispose();
				occluded.Dispose();
				bruteHits.Dispose();
				bruteOccluded.Dispose();
			}
		}
	}
}
