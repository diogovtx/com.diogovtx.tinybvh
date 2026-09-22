using NUnit.Framework;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>Self-contained BVH tests that need no external data files.</summary>
	public class BvhBasicTests
	{
		const int GridSize = 12;

		/// <summary>Deterministic LCG, same recurrence as tinybvh's reference generators (no UnityEngine.Random).</summary>
		struct Lcg
		{
			uint state;

			public Lcg( uint seed )
			{
				state = seed;
			}

			public float NextFloat()
			{
				state = ( state * 1664525u ) + 1013904223u;
				return ( state >> 8 ) * ( 1f / 16777216f );
			}
		}

		/// <summary>Port of Möller-Trumbore, used as a brute-force cross-check against BVH traversal.</summary>
		static bool RayTriangle( float3 O, float3 D, float3 v0, float3 v1, float3 v2, out float t, out float u, out float v )
		{
			t = 0f;
			u = 0f;
			v = 0f;

			float3 e1 = v1 - v0;
			float3 e2 = v2 - v0;
			float3 h = math.cross( D, e2 );
			float a = math.dot( e1, h );
			if ( math.abs( a ) < 1e-12f )
			{
				return false;
			}

			float f = 1f / a;
			float3 s = O - v0;
			u = f * math.dot( s, h );
			if ( u < 0f || u > 1f )
			{
				return false;
			}

			float3 q = math.cross( s, e1 );
			v = f * math.dot( D, q );
			if ( v < 0f || u + v > 1f )
			{
				return false;
			}

			t = f * math.dot( e2, q );
			return t > 1e-6f;
		}

		/// <summary>Port of TestIndexedGeometry's grid mesh (tiny_bvh_double_test.cpp), in single precision.</summary>
		static void BuildGrid( out float3[] gridVerts, out uint[] indices, out float4[] soup, out int triCount )
		{
			int quads = ( GridSize - 1 ) * ( GridSize - 1 );
			triCount = quads * 2;

			gridVerts = new float3[ GridSize * GridSize ];
			for ( int j = 0; j < GridSize; j++ )
			{
				for ( int i = 0; i < GridSize; i++ )
				{
					float z = 0.05f * ( ( i * 7 + j * 13 ) % 5 );
					gridVerts[ ( j * GridSize ) + i ] = new float3( i, j, z );
				}
			}

			indices = new uint[ triCount * 3 ];
			soup = new float4[ triCount * 3 ];
			int t = 0;
			for ( int j = 0; j < GridSize - 1; j++ )
			{
				for ( int i = 0; i < GridSize - 1; i++ )
				{
					uint a = ( uint )( ( j * GridSize ) + i );
					uint b = ( uint )( ( j * GridSize ) + i + 1 );
					uint c = ( uint )( ( ( j + 1 ) * GridSize ) + i );
					uint d = ( uint )( ( ( j + 1 ) * GridSize ) + i + 1 );
					uint[][] tris = new uint[][] { new uint[] { a, b, c }, new uint[] { b, d, c } };
					for ( int k = 0; k < 2; k++, t++ )
					{
						for ( int v = 0; v < 3; v++ )
						{
							indices[ ( t * 3 ) + v ] = tris[ k ][ v ];
							soup[ ( t * 3 ) + v ] = new float4( gridVerts[ tris[ k ][ v ] ], 0f );
						}
					}
				}
			}
		}

		[Test]
		public void RandomTriangles_BuildAndIntersect()
		{
			const uint triCount = 8192;
			NativeArray<float4> verts = new NativeArray<float4>( ( int )triCount * 3, Allocator.Persistent, NativeArrayOptions.UninitializedMemory );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				Lcg rng = new Lcg( 0x12345678u );
				for ( int i = 0; i < triCount; i++ )
				{
					float3 basePos = new float3( rng.NextFloat(), rng.NextFloat(), rng.NextFloat() );
					for ( int v = 0; v < 3; v++ )
					{
						float3 jitter = new float3( rng.NextFloat(), rng.NextFloat(), rng.NextFloat() ) * 0.1f;
						verts[ ( i * 3 ) + v ] = new float4( basePos + jitter, 0f );
					}
				}

				bvh.Build( verts, triCount );

				Ray ray = new Ray( new float3( 0.5f, 0.5f, -1f ), new float3( 0.1f, 0f, 2f ) );
				bvh.Intersect( ref ray );

				Assert.Less( ray.Hit.T, BvhConstants.Far );

				float bestT = BvhConstants.Far;
				uint bestPrim = uint.MaxValue;
				for ( uint i = 0; i < triCount; i++ )
				{
					float3 v0 = verts[ ( int )( i * 3 ) ].xyz;
					float3 v1 = verts[ ( int )( ( i * 3 ) + 1 ) ].xyz;
					float3 v2 = verts[ ( int )( ( i * 3 ) + 2 ) ].xyz;
					if ( RayTriangle( ray.O, ray.D, v0, v1, v2, out float t, out _, out _ ) && t < bestT )
					{
						bestT = t;
						bestPrim = i;
					}
				}

				Assert.AreEqual( bestPrim, ray.Hit.Prim );
				Assert.That( ray.Hit.T, Is.EqualTo( bestT ).Within( 1e-5f * bestT ) );

				Assert.Greater( bvh.NodeCount(), 1 );
				Assert.LessOrEqual( bvh.UsedNodes, 2 * triCount );

				const float eps = 1e-4f;
				unsafe
				{
					for ( uint n = 0; n < bvh.UsedNodes; n++ )
					{
						BvhNode node = bvh.Nodes[ n ];
						if ( node.TriCount == 0 )
						{
							continue;
						}
						for ( uint k = 0; k < node.TriCount; k++ )
						{
							uint prim = bvh.PrimIdx[ node.LeftFirst + k ];
							for ( int v = 0; v < 3; v++ )
							{
								float3 p = verts[ ( int )( ( prim * 3 ) + v ) ].xyz;
								Assert.GreaterOrEqual( p.x, node.AabbMin.x - eps );
								Assert.GreaterOrEqual( p.y, node.AabbMin.y - eps );
								Assert.GreaterOrEqual( p.z, node.AabbMin.z - eps );
								Assert.LessOrEqual( p.x, node.AabbMax.x + eps );
								Assert.LessOrEqual( p.y, node.AabbMax.y + eps );
								Assert.LessOrEqual( p.z, node.AabbMax.z + eps );
							}
						}
					}
				}
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		[Test]
		public void IndexedVsSoup_GridMesh()
		{
			BuildGrid( out float3[] gridVerts, out uint[] indices, out float4[] soup, out int triCount );

			float4[] vertsAsFloat4 = new float4[ gridVerts.Length ];
			for ( int i = 0; i < gridVerts.Length; i++ )
			{
				vertsAsFloat4[ i ] = new float4( gridVerts[ i ], 0f );
			}

			NativeArray<float4> indexedVerts = new NativeArray<float4>( vertsAsFloat4, Allocator.Persistent );
			NativeArray<uint> indicesNative = new NativeArray<uint>( indices, Allocator.Persistent );
			NativeArray<float4> soupNative = new NativeArray<float4>( soup, Allocator.Persistent );
			Bvh indexedBvh = Bvh.Create( Allocator.Persistent );
			Bvh plainBvh = Bvh.Create( Allocator.Persistent );
			try
			{
				indexedBvh.Build( indexedVerts, indicesNative, ( uint )triCount );
				plainBvh.Build( soupNative, ( uint )triCount );

				int hits = 0;
				for ( int sy = 0; sy < 40; sy++ )
				{
					for ( int sx = 0; sx < 40; sx++ )
					{
						float px = -1f + ( ( float )sx / 39f * ( GridSize + 1 ) );
						float py = -1f + ( ( float )sy / 39f * ( GridSize + 1 ) );
						float3 O = new float3( px, py, -5f );
						float3 D = new float3( 0f, 0f, 1f );

						Ray ri = new Ray( O, D );
						Ray rp = new Ray( O, D );
						indexedBvh.Intersect( ref ri );
						plainBvh.Intersect( ref rp );

						bool hitI = ri.Hit.T < BvhConstants.Far;
						bool hitP = rp.Hit.T < BvhConstants.Far;
						Assert.AreEqual( hitP, hitI );
						if ( hitI )
						{
							hits++;
							Assert.AreEqual( rp.Hit.Prim, ri.Hit.Prim );
							Assert.That( ri.Hit.T, Is.EqualTo( rp.Hit.T ).Within( 1e-6f ) );
							Assert.That( ri.Hit.U, Is.EqualTo( rp.Hit.U ).Within( 1e-6f ) );
							Assert.That( ri.Hit.V, Is.EqualTo( rp.Hit.V ).Within( 1e-6f ) );
						}

						Ray si = new Ray( O, D );
						Ray sp = new Ray( O, D );
						bool oi = indexedBvh.IsOccluded( si );
						bool op = plainBvh.IsOccluded( sp );
						Assert.AreEqual( op, oi );
						Assert.AreEqual( hitI, oi );
					}
				}

				Assert.Greater( hits, 0 );
			}
			finally
			{
				indexedBvh.Dispose();
				plainBvh.Dispose();
				indexedVerts.Dispose();
				indicesNative.Dispose();
				soupNative.Dispose();
			}
		}

		[Test]
		public void Refit_MovesBounds()
		{
			BuildGrid( out _, out _, out float4[] soup, out int triCount );

			NativeArray<float4> verts = new NativeArray<float4>( soup, Allocator.Persistent );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, ( uint )triCount );

				float3 originalMin = bvh.AabbMin;
				float3 originalMax = bvh.AabbMax;

				float3 O = new float3( 5f, 5f, -5f );
				float3 D = new float3( 0f, 0f, 1f );
				Ray rayBefore = new Ray( O, D );
				bvh.Intersect( ref rayBefore );
				Assert.Less( rayBefore.Hit.T, BvhConstants.Far );

				for ( int i = 0; i < verts.Length; i++ )
				{
					float4 v = verts[ i ];
					v.y += 20f;
					verts[ i ] = v;
				}
				bvh.Refit();

				Assert.That( bvh.AabbMin.y, Is.EqualTo( originalMin.y + 20f ).Within( 1e-4f ) );
				Assert.That( bvh.AabbMax.y, Is.EqualTo( originalMax.y + 20f ).Within( 1e-4f ) );
				Assert.That( bvh.AabbMin.x, Is.EqualTo( originalMin.x ).Within( 1e-4f ) );
				Assert.That( bvh.AabbMax.x, Is.EqualTo( originalMax.x ).Within( 1e-4f ) );

				Ray rayAfterSame = new Ray( O, D );
				bvh.Intersect( ref rayAfterSame );
				Assert.AreEqual( BvhConstants.Far, rayAfterSame.Hit.T );

				Ray rayShifted = new Ray( new float3( 5f, 25f, -5f ), D );
				bvh.Intersect( ref rayShifted );
				Assert.Less( rayShifted.Hit.T, BvhConstants.Far );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		[Test]
		public unsafe void Tlas_TwoInstances()
		{
			BuildGrid( out _, out _, out float4[] soup, out int triCount );

			NativeArray<float4> verts = new NativeArray<float4>( soup, Allocator.Persistent );
			NativeArray<Bvh> blasArray = new NativeArray<Bvh>( 1, Allocator.Persistent );
			NativeArray<BlasInstance> instArray = new NativeArray<BlasInstance>( 2, Allocator.Persistent );
			Bvh tlas = Bvh.Create( Allocator.Persistent );
			try
			{
				Bvh blas = Bvh.Create( Allocator.Persistent );
				blas.Build( verts, ( uint )triCount );
				blasArray[ 0 ] = blas;

				BlasInstance inst0 = BlasInstance.Create( 0 );
				BlasInstance inst1 = BlasInstance.Create( 0 );
				inst1.Transform[ 3 ] = 10f; // row-major cell 3 = Row0.w: translate +10 along x.
				inst1.Mask = 0x2;
				instArray[ 0 ] = inst0;
				instArray[ 1 ] = inst1;

				tlas.BuildTlas( ( BlasInstance* )instArray.GetUnsafePtr(), 2, ( Bvh* )blasArray.GetUnsafePtr(), 1 );

				float3 O = new float3( 5f, 5f, -5f );
				float3 D = new float3( 0f, 0f, 1f );

				Ray localRay = new Ray( O, D );
				Bvh builtBlas = blasArray[ 0 ];
				builtBlas.Intersect( ref localRay );
				Assert.Less( localRay.Hit.T, BvhConstants.Far );

				Ray ray0 = new Ray( O, D );
				tlas.Intersect( ref ray0 );
				Assert.Less( ray0.Hit.T, BvhConstants.Far );
				Assert.AreEqual( 0u, ray0.Hit.Inst );
				Assert.AreEqual( localRay.Hit.Prim, ray0.Hit.Prim );

				float3 O1 = new float3( 15f, 5f, -5f );
				Ray ray1 = new Ray( O1, D );
				tlas.Intersect( ref ray1 );
				Assert.Less( ray1.Hit.T, BvhConstants.Far );
				Assert.AreEqual( 1u, ray1.Hit.Inst );
				Assert.AreEqual( localRay.Hit.Prim, ray1.Hit.Prim );

				// Instance 1's mask is 0x2; a ray restricted to 0x1 must miss it entirely.
				Ray maskedRay = new Ray( O1, D, BvhConstants.Far, 0x1 );
				tlas.Intersect( ref maskedRay );
				Assert.AreEqual( BvhConstants.Far, maskedRay.Hit.T );
			}
			finally
			{
				tlas.Dispose();
				for ( int i = 0; i < blasArray.Length; i++ )
				{
					Bvh b = blasArray[ i ];
					b.Dispose();
				}
				blasArray.Dispose();
				instArray.Dispose();
				verts.Dispose();
			}
		}

		[Test]
		public void IsOccluded_RespectsMaxDistance()
		{
			BuildGrid( out _, out _, out float4[] soup, out int triCount );

			NativeArray<float4> verts = new NativeArray<float4>( soup, Allocator.Persistent );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, ( uint )triCount );

				float3 O = new float3( 5f, 5f, -5f );
				float3 D = new float3( 0f, 0f, 1f );
				Ray probe = new Ray( O, D );
				bvh.Intersect( ref probe );
				Assert.Less( probe.Hit.T, BvhConstants.Far );
				float t = probe.Hit.T;

				Ray shortRay = new Ray( O, D, 0.5f * t );
				Assert.IsFalse( bvh.IsOccluded( shortRay ) );

				Ray fullRay = new Ray( O, D, BvhConstants.Far );
				Assert.IsTrue( bvh.IsOccluded( fullRay ) );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}
	}
}
