using System;
using System.IO;
using NUnit.Framework;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Data-driven tests comparing the C# port against reference data dumped by Tools/RefDump/refdump.cpp
	/// from the original tinybvh library. If TestData/&lt;name&gt;.bin or .ref is missing, the test is ignored
	/// with a pointer to TestData/fetch.ps1 and Tools/RefDump/run_all.bat.
	/// </summary>
	public class BvhReferenceTests
	{
		const float MaxMismatchFraction = 0.0005f;

		static class Compare
		{
			public static bool BitsEqual( float a, float b )
			{
				return math.asuint( a ) == math.asuint( b );
			}

			public static bool NodesEqualExact( BvhNode a, BvhNode b )
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

			public static bool WithinRelative( float actual, float expected, float relTol )
			{
				float scale = math.max( math.abs( expected ), 1e-12f );
				return math.abs( actual - expected ) <= relTol * scale;
			}

			public static bool WithinAbsolute( float actual, float expected, float absTol )
			{
				return math.abs( actual - expected ) <= absTol;
			}

			public static void AssertRelative( float actual, float expected, float relTol, string message )
			{
				Assert.IsTrue( WithinRelative( actual, expected, relTol ), $"{message}: expected {expected}, got {actual}" );
			}

			public static void AssertAbsolute( float actual, float expected, float absTol, string message )
			{
				Assert.IsTrue( WithinAbsolute( actual, expected, absTol ), $"{message}: expected {expected}, got {actual}" );
			}

			public static void AssertFloat3Relative( float3 actual, float3 expected, float relTol, string message )
			{
				AssertRelative( actual.x, expected.x, relTol, message + ".x" );
				AssertRelative( actual.y, expected.y, relTol, message + ".y" );
				AssertRelative( actual.z, expected.z, relTol, message + ".z" );
			}

			public static void AssertFloat3Absolute( float3 actual, float3 expected, float absTol, string message )
			{
				AssertAbsolute( actual.x, expected.x, absTol, message + ".x" );
				AssertAbsolute( actual.y, expected.y, absTol, message + ".y" );
				AssertAbsolute( actual.z, expected.z, absTol, message + ".z" );
			}
		}

		static bool TryGetPaths( string sceneName, out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( sceneName + ".ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void Build_MatchesReferenceStructure( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 to fetch scenes and Tools/RefDump/run_all.bat to generate reference data" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				RefDumpFile refFile = RefDumpFile.Load( refPath );

				Assert.AreEqual( refFile.UsedNodes, bvh.UsedNodes, "UsedNodes" );
				Compare.AssertRelative( bvh.SahCost(), refFile.SahCost, 1e-4f, "SahCost" );
				Compare.AssertFloat3Absolute( bvh.AabbMin, refFile.AabbMin, 1e-6f, "AabbMin" );
				Compare.AssertFloat3Absolute( bvh.AabbMax, refFile.AabbMax, 1e-6f, "AabbMax" );

				int nodeMismatches = 0;
				for ( uint i = 0; i < bvh.UsedNodes; i++ )
				{
					if ( !Compare.NodesEqualExact( bvh.Nodes[ i ], refFile.Nodes[ i ] ) )
					{
						nodeMismatches++;
					}
				}
				TestContext.WriteLine( $"{sceneName}: node mismatches {nodeMismatches}/{bvh.UsedNodes}" );
				Assert.AreEqual( 0, nodeMismatches, "node structure mismatch count" );

				Assert.AreEqual( refFile.PrimIdx.Length, ( int )bvh.IdxCount, "IdxCount" );
				for ( int i = 0; i < refFile.PrimIdx.Length; i++ )
				{
					Assert.AreEqual( refFile.PrimIdx[ i ], bvh.PrimIdx[ i ], $"primIdx[{i}]" );
				}
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
		public void Intersect_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				RefDumpFile refFile = RefDumpFile.Load( refPath );

				int mismatchCount = 0;
				int tieCount = 0;
				int hitCount = 0;
				double tSum = 0.0;
				double refTSum = 0.0;

				for ( int i = 0; i < refFile.Rays.Length; i++ )
				{
					RefDumpFile.RayHit rh = refFile.Rays[ i ];
					Ray ray = new Ray( rh.O, rh.D );
					bvh.Intersect( ref ray );

					bool refHit = rh.T < BvhConstants.Far;
					bool gotHit = ray.Hit.T < BvhConstants.Far;

					if ( refHit != gotHit )
					{
						mismatchCount++;
						continue;
					}

					if ( refHit )
					{
						hitCount++;
						refTSum += rh.T;
						tSum += ray.Hit.T;

						bool sameT = Compare.WithinRelative( ray.Hit.T, rh.T, 1e-4f );
						bool ok = ray.Hit.Prim == rh.Prim
							&& sameT
							&& Compare.WithinAbsolute( ray.Hit.U, rh.U, 1e-4f )
							&& Compare.WithinAbsolute( ray.Hit.V, rh.V, 1e-4f );
						if ( !ok )
						{
							// A different primitive at the same distance is a legitimate tie between
							// overlapping triangles, decided by traversal order and last-bit rounding.
							if ( sameT )
							{
								tieCount++;
							}
							else
							{
								mismatchCount++;
							}
						}
					}
				}

				float hitRatio = ( float )hitCount / refFile.Rays.Length;
				TestContext.WriteLine( $"{sceneName}: hit ratio {hitRatio:P2}, mismatches {mismatchCount}/{refFile.Rays.Length}, same-distance ties {tieCount}" );

				Assert.LessOrEqual( mismatchCount, ( int )( refFile.Rays.Length * MaxMismatchFraction ), "mismatch rate too high" );

				double scale = Math.Max( Math.Abs( refTSum ), 1e-12 );
				Assert.That( tSum, Is.EqualTo( refTSum ).Within( 1e-3 * scale ), "sum of T over hit rays" );
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
		public void IsOccluded_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				RefDumpFile refFile = RefDumpFile.Load( refPath );

				int mismatches = 0;
				int total = refFile.Rays.Length * 2;

				for ( int i = 0; i < refFile.Rays.Length; i++ )
				{
					RefDumpFile.RayHit rh = refFile.Rays[ i ];

					Ray fullRay = new Ray( rh.O, rh.D );
					bool occludedFull = bvh.IsOccluded( fullRay );
					if ( occludedFull != ( rh.OccludedFull != 0 ) )
					{
						mismatches++;
					}

					float halfT = rh.T < BvhConstants.Far ? 0.5f * rh.T : BvhConstants.Far;
					Ray halfRay = new Ray( rh.O, rh.D, halfT );
					bool occludedHalf = bvh.IsOccluded( halfRay );
					if ( occludedHalf != ( rh.OccludedHalf != 0 ) )
					{
						mismatches++;
					}
				}

				TestContext.WriteLine( $"{sceneName}: occlusion mismatches {mismatches}/{total}" );
				Assert.LessOrEqual( mismatches, ( int )( total * MaxMismatchFraction ), "occlusion mismatch rate too high" );
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
		public void Refit_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				RefDumpFile refFile = RefDumpFile.Load( refPath );

				// Deterministic per-vertex perturbation, identical to refdump.cpp; ext is measured
				// against the BLAS root bounds before perturbation.
				float3 ext = bvh.AabbMax - bvh.AabbMin;
				uint vertCount = triCount * 3;
				for ( uint i = 0; i < vertCount; i++ )
				{
					float4 v = verts[ ( int )i ];
					v.x = v.x + ( ( int )( i % 7 ) - 3 ) * 0.005f * ext.x;
					v.y = v.y + ( ( int )( i % 5 ) - 2 ) * 0.005f * ext.y;
					verts[ ( int )i ] = v;
				}

				bvh.Refit();

				Compare.AssertFloat3Relative( bvh.AabbMin, refFile.RefitAabbMin, 1e-5f, "refit AabbMin" );
				Compare.AssertFloat3Relative( bvh.AabbMax, refFile.RefitAabbMax, 1e-5f, "refit AabbMax" );

				int mismatchCount = 0;
				int tieCount = 0;
				int hitCount = 0;
				double tSum = 0.0;
				double refTSum = 0.0;

				for ( int i = 0; i < refFile.Rays.Length; i++ )
				{
					RefDumpFile.RayHit blasRay = refFile.Rays[ i ];
					RefDumpFile.RefitHit rh = refFile.RefitHits[ i ];

					Ray ray = new Ray( blasRay.O, blasRay.D );
					bvh.Intersect( ref ray );

					bool refHit = rh.T < BvhConstants.Far;
					bool gotHit = ray.Hit.T < BvhConstants.Far;
					if ( refHit != gotHit )
					{
						mismatchCount++;
						continue;
					}

					if ( refHit )
					{
						hitCount++;
						refTSum += rh.T;
						tSum += ray.Hit.T;

						bool sameT = Compare.WithinRelative( ray.Hit.T, rh.T, 1e-4f );
						bool ok = ray.Hit.Prim == rh.Prim
							&& sameT
							&& Compare.WithinAbsolute( ray.Hit.U, rh.U, 1e-4f )
							&& Compare.WithinAbsolute( ray.Hit.V, rh.V, 1e-4f );
						if ( !ok )
						{
							// A different primitive at the same distance is a legitimate tie between
							// overlapping triangles, decided by traversal order and last-bit rounding.
							if ( sameT )
							{
								tieCount++;
							}
							else
							{
								mismatchCount++;
							}
						}
					}
				}

				float hitRatio = ( float )hitCount / refFile.Rays.Length;
				TestContext.WriteLine( $"{sceneName} refit: hit ratio {hitRatio:P2}, mismatches {mismatchCount}/{refFile.Rays.Length}, same-distance ties {tieCount}" );
				Assert.LessOrEqual( mismatchCount, ( int )( refFile.Rays.Length * MaxMismatchFraction ), "refit mismatch rate too high" );

				double scale = Math.Max( Math.Abs( refTSum ), 1e-12 );
				Assert.That( tSum, Is.EqualTo( refTSum ).Within( 1e-3 * scale ), "refit sum of T over hit rays" );
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
		public unsafe void Tlas_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = default;
			NativeArray<Bvh> blasArray = default;
			NativeArray<BlasInstance> instArray = default;
			Bvh blas = Bvh.Create( Allocator.Persistent );
			Bvh tlas = Bvh.Create( Allocator.Persistent );
			try
			{
				verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
				blas.Build( verts, triCount );

				blasArray = new NativeArray<Bvh>( 1, Allocator.Persistent );
				blasArray[ 0 ] = blas;

				RefDumpFile refFile = RefDumpFile.Load( refPath );
				int instCount = refFile.Instances.Length;
				instArray = new NativeArray<BlasInstance>( instCount, Allocator.Persistent );
				for ( int i = 0; i < instCount; i++ )
				{
					BlasInstance inst = BlasInstance.Create( 0 );
					for ( int c = 0; c < 16; c++ )
					{
						inst.Transform[ c ] = refFile.Instances[ i ].Transform[ c ];
					}
					inst.Mask = refFile.Instances[ i ].Mask;
					instArray[ i ] = inst;
				}

				tlas.BuildTlas( ( BlasInstance* )instArray.GetUnsafePtr(), ( uint )instCount, ( Bvh* )blasArray.GetUnsafePtr(), 1 );

				for ( int i = 0; i < instCount; i++ )
				{
					BlasInstance built = instArray[ i ];
					RefDumpFile.InstanceRecord expected = refFile.Instances[ i ];
					for ( int c = 0; c < 16; c++ )
					{
						Compare.AssertRelative( built.InvTransform[ c ], expected.InvTransform[ c ], 1e-5f, $"instance {i} invTransform[{c}]" );
					}
					Compare.AssertFloat3Relative( built.AabbMin, expected.AabbMin, 1e-5f, $"instance {i} aabbMin" );
					Compare.AssertFloat3Relative( built.AabbMax, expected.AabbMax, 1e-5f, $"instance {i} aabbMax" );
				}

				Assert.AreEqual( refFile.TlasUsedNodes, tlas.UsedNodes, "TLAS UsedNodes" );

				int nodeMismatches = 0;
				for ( uint i = 0; i < tlas.UsedNodes; i++ )
				{
					if ( !Compare.NodesEqualExact( tlas.Nodes[ i ], refFile.TlasNodes[ i ] ) )
					{
						nodeMismatches++;
					}
				}
				TestContext.WriteLine( $"{sceneName} tlas: node mismatches {nodeMismatches}/{tlas.UsedNodes}" );
				Assert.AreEqual( 0, nodeMismatches, "tlas node structure mismatch count" );

				Assert.AreEqual( refFile.TlasPrimIdx.Length, ( int )tlas.IdxCount, "tlas IdxCount" );
				for ( int i = 0; i < refFile.TlasPrimIdx.Length; i++ )
				{
					Assert.AreEqual( refFile.TlasPrimIdx[ i ], tlas.PrimIdx[ i ], $"tlas primIdx[{i}]" );
				}

				int mismatchCount = 0;
				int occMismatch = 0;

				for ( int i = 0; i < refFile.TlasRays.Length; i++ )
				{
					RefDumpFile.TlasRayHit rh = refFile.TlasRays[ i ];
					Ray ray = new Ray( rh.O, rh.D );
					tlas.Intersect( ref ray );

					bool refHit = rh.T < BvhConstants.Far;
					bool gotHit = ray.Hit.T < BvhConstants.Far;
					if ( refHit != gotHit )
					{
						mismatchCount++;
					}
					else if ( refHit )
					{
						bool ok = ray.Hit.Inst == rh.Inst
							&& ray.Hit.Prim == rh.Prim
							&& Compare.WithinRelative( ray.Hit.T, rh.T, 1e-4f );
						if ( !ok )
						{
							mismatchCount++;
						}
					}

					Ray occRay = new Ray( rh.O, rh.D );
					bool occluded = tlas.IsOccluded( occRay );
					if ( occluded != ( rh.OccludedFull != 0 ) )
					{
						occMismatch++;
					}
				}

				TestContext.WriteLine( $"{sceneName} tlas: hit mismatches {mismatchCount}/{refFile.TlasRays.Length}, occlusion mismatches {occMismatch}/{refFile.TlasRays.Length}" );
				Assert.LessOrEqual( mismatchCount, ( int )( refFile.TlasRays.Length * MaxMismatchFraction ), "tlas intersect mismatch rate too high" );
				Assert.LessOrEqual( occMismatch, ( int )( refFile.TlasRays.Length * MaxMismatchFraction ), "tlas occlusion mismatch rate too high" );
			}
			finally
			{
				tlas.Dispose();
				if ( blasArray.IsCreated )
				{
					for ( int i = 0; i < blasArray.Length; i++ )
					{
						Bvh b = blasArray[ i ];
						b.Dispose();
					}
					blasArray.Dispose();
				}
				else
				{
					blas.Dispose();
				}
				if ( instArray.IsCreated )
				{
					instArray.Dispose();
				}
				if ( verts.IsCreated )
				{
					verts.Dispose();
				}
			}
		}
	}
}
