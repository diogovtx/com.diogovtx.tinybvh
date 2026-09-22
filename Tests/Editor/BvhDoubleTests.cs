using System.Collections.Generic;
using System.IO;
using AOT;
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
	/// Validates the double-precision port (BvhDouble, RayDouble, BlasInstanceDouble) against the
	/// reference data dumped by Tools/RefDump/dbldump.cpp. Everything is compared bit for bit:
	/// double arithmetic is exact on both sides, so unlike the single-precision suites there is no
	/// relative tolerance here. Hits within a hair of a triangle edge are still classified and
	/// reported separately, in case the two sides ever disagree there.
	/// </summary>
	public unsafe class BvhDoubleTests
	{
		/// <summary>Barycentric distance to a triangle edge below which a hit counts as grazing.</summary>
		const double GrazingEps = 1e-3;

		/// <summary>Number of differing entries listed in an assertion message before it is truncated.</summary>
		const int MaxReportedDiffs = 4;

		/// <summary>Traverses a double-precision BVH: the closest hit plus the two occlusion queries of the dump.</summary>
		[BurstCompile( CompileSynchronously = true )]
		private struct TraceJob : IJobParallelFor
		{
			public BvhDouble Bvh;
			[ReadOnly] public NativeArray<double3> Origins;
			[ReadOnly] public NativeArray<double3> Directions;
			public NativeArray<IntersectionDouble> Hits;
			public NativeArray<int> OccludedFull;
			public NativeArray<int> OccludedHalf;

			public void Execute( int i )
			{
				RayDouble ray = new RayDouble( Origins[ i ], Directions[ i ] );
				Bvh.Intersect( ref ray );
				Hits[ i ] = ray.Hit;
				OccludedFull[ i ] = Bvh.IsOccluded( new RayDouble( Origins[ i ], Directions[ i ] ) ) ? 1 : 0;
				double halfT = ray.Hit.T < BvhDoubleConstants.Far ? 0.5 * ray.Hit.T : BvhDoubleConstants.Far;
				OccludedHalf[ i ] = Bvh.IsOccluded( new RayDouble( Origins[ i ], Directions[ i ], halfT ) ) ? 1 : 0;
			}
		}

		/// <summary>The custom geometry of the self-consistency test: the scene triangles behind the callbacks.</summary>
		[BurstCompile]
		private static class TriGeometry
		{
			private struct VertsContext
			{
			}

			private struct VertsKey
			{
			}

			/// <summary>
			/// Address of the double3 vertex array. Burst function pointers cannot capture state,
			/// and a plain static field is not visible from compiled code, so it travels through
			/// a SharedStatic.
			/// </summary>
			public static readonly SharedStatic<long> VertsAddress = SharedStatic<long>.GetOrCreate<VertsContext, VertsKey>();

			[BurstCompile( CompileSynchronously = true )]
			[MonoPInvokeCallback( typeof( GetAabbDoubleDelegate ) )]
			public static void GetAabb( ulong prim, double3* aabbMin, double3* aabbMax )
			{
				double3* verts = ( double3* )VertsAddress.Data;
				double3 v0 = verts[ prim * 3 ], v1 = verts[ ( prim * 3 ) + 1 ], v2 = verts[ ( prim * 3 ) + 2 ];
				*aabbMin = BvhDoubleMath.Min( BvhDoubleMath.Min( v0, v1 ), v2 );
				*aabbMax = BvhDoubleMath.Max( BvhDoubleMath.Max( v0, v1 ), v2 );
			}

			[BurstCompile( CompileSynchronously = true )]
			[MonoPInvokeCallback( typeof( CustomIntersectDoubleDelegate ) )]
			public static byte Intersect( RayDouble* ray, ulong prim )
			{
				double3* verts = ( double3* )VertsAddress.Data;
				double3 v0 = verts[ prim * 3 ], v1 = verts[ ( prim * 3 ) + 1 ], v2 = verts[ ( prim * 3 ) + 2 ];
				double3 e1 = v1 - v0;
				double3 e2 = v2 - v0;
				double3 h = BvhDoubleMath.Cross( ray->D, e2 );
				double a = BvhDoubleMath.Dot( e1, h );
				if ( math.abs( a ) < 0.0000001 )
				{
					return 0;
				}
				double f = 1 / a;
				double3 s = ray->O - v0;
				double u = f * BvhDoubleMath.Dot( s, h );
				double3 q = BvhDoubleMath.Cross( s, e1 );
				double v = f * BvhDoubleMath.Dot( ray->D, q );
				if ( u < 0 || v < 0 || u + v > 1 )
				{
					return 0;
				}
				double t = f * BvhDoubleMath.Dot( e2, q );
				if ( t > 0 && t < ray->Hit.T )
				{
					ray->Hit.T = t;
					ray->Hit.U = u;
					ray->Hit.V = v;
					ray->Hit.Prim = prim;
					return 1;
				}
				return 0;
			}

			[BurstCompile( CompileSynchronously = true )]
			[MonoPInvokeCallback( typeof( CustomOccludedDoubleDelegate ) )]
			public static byte IsOccluded( RayDouble* ray, ulong prim )
			{
				double3* verts = ( double3* )VertsAddress.Data;
				double3 v0 = verts[ prim * 3 ], v1 = verts[ ( prim * 3 ) + 1 ], v2 = verts[ ( prim * 3 ) + 2 ];
				double3 e1 = v1 - v0;
				double3 e2 = v2 - v0;
				double3 h = BvhDoubleMath.Cross( ray->D, e2 );
				double a = BvhDoubleMath.Dot( e1, h );
				if ( math.abs( a ) < 0.0000001 )
				{
					return 0;
				}
				double f = 1 / a;
				double3 s = ray->O - v0;
				double u = f * BvhDoubleMath.Dot( s, h );
				double3 q = BvhDoubleMath.Cross( s, e1 );
				double v = f * BvhDoubleMath.Dot( ray->D, q );
				if ( u < 0 || v < 0 || u + v > 1 )
				{
					return 0;
				}
				double t = f * BvhDoubleMath.Dot( e2, q );
				return ( t > 0 && t < ray->Hit.T ) ? ( byte )1 : ( byte )0;
			}
		}

		static bool TryGetPaths( string sceneName, out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( sceneName + ".dbl.ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		/// <summary>Widens the float4 vertices of a ".bin" scene to double3, exactly as dbldump.cpp does.</summary>
		static NativeArray<double3> WidenVertices( NativeArray<float4> verts )
		{
			NativeArray<double3> wide = new NativeArray<double3>( verts.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory );
			for ( int i = 0; i < verts.Length; i++ )
			{
				float4 v = verts[ i ];
				wide[ i ] = new double3( v.x, v.y, v.z );
			}
			return wide;
		}

		static bool BitsEqual( double a, double b )
		{
			return math.asulong( a ) == math.asulong( b );
		}

		static bool BitsEqual( double3 a, double3 b )
		{
			return BitsEqual( a.x, b.x ) && BitsEqual( a.y, b.y ) && BitsEqual( a.z, b.z );
		}

		static bool NodesEqual( BvhDoubleNode a, BvhDoubleNode b )
		{
			return BitsEqual( a.AabbMin, b.AabbMin ) && BitsEqual( a.AabbMax, b.AabbMax )
				&& a.LeftFirst == b.LeftFirst && a.TriCount == b.TriCount;
		}

		/// <summary>True when the hit lies within GrazingEps of a triangle edge or vertex.</summary>
		static bool IsGrazing( double u, double v )
		{
			return u < GrazingEps || v < GrazingEps || ( 1.0 - u - v ) < GrazingEps;
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

		static void AssertBitsEqual( double actual, double expected, string message )
		{
			Assert.IsTrue( BitsEqual( actual, expected ), $"{message}: expected {expected:R} ({math.asulong( expected ):X16}), got {actual:R} ({math.asulong( actual ):X16})" );
		}

		static void AssertDouble3BitsEqual( double3 actual, double3 expected, string message )
		{
			AssertBitsEqual( actual.x, expected.x, message + ".x" );
			AssertBitsEqual( actual.y, expected.y, message + ".y" );
			AssertBitsEqual( actual.z, expected.z, message + ".z" );
		}

		static void CompareNodes( string label, BvhDoubleNode* nodes, ulong usedNodes, ulong expectedUsedNodes, BvhDoubleNode[] expected )
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

		static void ComparePrimIdx( string label, ulong* primIdx, ulong count, ulong[] expected )
		{
			Assert.AreEqual( ( ulong )expected.Length, count, label + " prim count" );
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

		/// <summary>Runs the trace job over one ray set and returns the results; the caller disposes them.</summary>
		static void Trace( ref BvhDouble bvh, DblDumpFile.RayHit[] rays,
			out NativeArray<IntersectionDouble> hits, out NativeArray<int> occludedFull, out NativeArray<int> occludedHalf )
		{
			int rayCount = rays.Length;
			NativeArray<double3> origins = new NativeArray<double3>( rayCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory );
			NativeArray<double3> directions = new NativeArray<double3>( rayCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory );
			for ( int i = 0; i < rayCount; i++ )
			{
				origins[ i ] = rays[ i ].O;
				directions[ i ] = rays[ i ].D;
			}
			hits = new NativeArray<IntersectionDouble>( rayCount, Allocator.Persistent );
			occludedFull = new NativeArray<int>( rayCount, Allocator.Persistent );
			occludedHalf = new NativeArray<int>( rayCount, Allocator.Persistent );
			new TraceJob
			{
				Bvh = bvh,
				Origins = origins,
				Directions = directions,
				Hits = hits,
				OccludedFull = occludedFull,
				OccludedHalf = occludedHalf
			}.Schedule( rayCount, 256 ).Complete();
			origins.Dispose();
			directions.Dispose();
		}

		/// <summary>Compares a traced ray set against one of the BLAS ray sections, bit for bit.</summary>
		static void CompareRays( string label, DblDumpFile.RayHit[] expected,
			NativeArray<IntersectionDouble> hits, NativeArray<int> occludedFull, NativeArray<int> occludedHalf )
		{
			int rayCount = expected.Length;
			int mismatches = 0, grazing = 0, hitCount = 0, occlusionMismatches = 0;
			for ( int i = 0; i < rayCount; i++ )
			{
				DblDumpFile.RayHit rh = expected[ i ];
				IntersectionDouble hit = hits[ i ];
				bool refHit = rh.T < BvhDoubleConstants.Far;
				bool gotHit = hit.T < BvhDoubleConstants.Far;
				if ( refHit )
				{
					hitCount++;
				}
				bool same = BitsEqual( hit.T, rh.T ) && BitsEqual( hit.U, rh.U ) && BitsEqual( hit.V, rh.V ) && hit.Prim == rh.Prim;
				if ( !same )
				{
					if ( ( refHit && IsGrazing( rh.U, rh.V ) ) || ( gotHit && IsGrazing( hit.U, hit.V ) ) )
					{
						grazing++;
					}
					else
					{
						if ( mismatches < 3 )
						{
							TestContext.WriteLine( $"  mismatch ray {i}: ref t {rh.T:R} u {rh.U:R} v {rh.V:R} prim {rh.Prim} | got t {hit.T:R} u {hit.U:R} v {hit.V:R} prim {hit.Prim}" );
						}
						mismatches++;
					}
				}
				if ( ( occludedFull[ i ] != 0 ) != ( rh.OccludedFull != 0 ) || ( occludedHalf[ i ] != 0 ) != ( rh.OccludedHalf != 0 ) )
				{
					occlusionMismatches++;
				}
			}
			double hitRatio = ( double )hitCount / rayCount;
			TestContext.WriteLine( $"{label}: hit ratio {hitRatio:P2}, mismatches {mismatches}/{rayCount}, grazing {grazing}, occlusion mismatches {occlusionMismatches}" );
			Assert.AreEqual( 0, mismatches, label + " intersect mismatches" );
			Assert.AreEqual( 0, occlusionMismatches, label + " occlusion mismatches" );
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Build_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 to fetch scenes and Tools/RefDump/run_all.bat to generate reference data" );
			}
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<double3> wide = WidenVertices( verts );
			verts.Dispose();
			BvhDouble bvh = BvhDouble.Create( Allocator.Persistent );
			try
			{
				bvh.Build( wide, triCount );
				DblDumpFile refFile = DblDumpFile.Load( refPath );

				Assert.AreEqual( refFile.TriCount, triCount, "triCount" );
				Assert.IsFalse( bvh.IsIndexed, "IsIndexed" );
				Assert.IsFalse( bvh.BvhOverIndices, "BvhOverIndices" );
				Assert.IsFalse( bvh.BvhOverAabbs, "BvhOverAabbs" );
				Assert.IsTrue( bvh.Refittable, "Refittable" );
				CompareNodes( $"{sceneName} blas", bvh.Nodes, bvh.UsedNodes, refFile.UsedNodes, refFile.Nodes );
				ComparePrimIdx( $"{sceneName} blas", bvh.PrimIdx, bvh.IdxCount, refFile.PrimIdx );
				AssertDouble3BitsEqual( bvh.AabbMin, refFile.AabbMin, "AabbMin" );
				AssertDouble3BitsEqual( bvh.AabbMax, refFile.AabbMax, "AabbMax" );
				AssertBitsEqual( bvh.SahCost(), refFile.SahCost, "SahCost" );

				Trace( ref bvh, refFile.Rays, out NativeArray<IntersectionDouble> hits,
					out NativeArray<int> occludedFull, out NativeArray<int> occludedHalf );
				try
				{
					CompareRays( $"{sceneName} blas", refFile.Rays, hits, occludedFull, occludedHalf );
				}
				finally
				{
					hits.Dispose();
					occludedFull.Dispose();
					occludedHalf.Dispose();
				}
			}
			finally
			{
				bvh.Dispose();
				wide.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Build_MatchesIndexedReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			DblDumpFile refFile = DblDumpFile.Load( refPath );
			NativeArray<double3> welded = new NativeArray<double3>( refFile.WeldedVertices, Allocator.Persistent );
			NativeArray<uint> indices = new NativeArray<uint>( refFile.Indices, Allocator.Persistent );
			BvhDouble bvh = BvhDouble.Create( Allocator.Persistent );
			try
			{
				bvh.Build( welded, indices, refFile.TriCount );

				Assert.IsTrue( bvh.IsIndexed, "IsIndexed" );
				Assert.IsTrue( bvh.BvhOverIndices, "BvhOverIndices" );
				CompareNodes( $"{sceneName} indexed", bvh.Nodes, bvh.UsedNodes, refFile.IndexedUsedNodes, refFile.IndexedNodes );
				ComparePrimIdx( $"{sceneName} indexed", bvh.PrimIdx, bvh.IdxCount, refFile.IndexedPrimIdx );

				Trace( ref bvh, refFile.IndexedRays, out NativeArray<IntersectionDouble> hits,
					out NativeArray<int> occludedFull, out NativeArray<int> occludedHalf );
				try
				{
					CompareRays( $"{sceneName} indexed", refFile.IndexedRays, hits, occludedFull, occludedHalf );
				}
				finally
				{
					hits.Dispose();
					occludedFull.Dispose();
					occludedHalf.Dispose();
				}
			}
			finally
			{
				bvh.Dispose();
				indices.Dispose();
				welded.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Tlas_MatchesReference( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<double3> wide = WidenVertices( verts );
			verts.Dispose();
			DblDumpFile refFile = DblDumpFile.Load( refPath );
			int instCount = refFile.Instances.Length;
			NativeArray<BvhDouble> blasArray = default;
			NativeArray<BlasInstanceDouble> instArray = default;
			BvhDouble blas = BvhDouble.Create( Allocator.Persistent );
			BvhDouble tlas = BvhDouble.Create( Allocator.Persistent );
			try
			{
				blas.Build( wide, triCount );
				blasArray = new NativeArray<BvhDouble>( 1, Allocator.Persistent );
				blasArray[ 0 ] = blas;

				// only the transform and the mask come from the file; the build produces the rest.
				instArray = new NativeArray<BlasInstanceDouble>( instCount, Allocator.Persistent );
				for ( int i = 0; i < instCount; i++ )
				{
					BlasInstanceDouble inst = BlasInstanceDouble.Create( 0 );
					for ( int c = 0; c < 16; c++ )
					{
						inst.Transform[ c ] = refFile.Instances[ i ].Transform[ c ];
					}
					inst.Mask = refFile.Instances[ i ].Mask;
					instArray[ i ] = inst;
				}

				tlas.BuildTlas( ( BlasInstanceDouble* )instArray.GetUnsafePtr(), ( ulong )instCount,
					( BvhDouble* )blasArray.GetUnsafePtr(), 1 );

				Assert.IsTrue( tlas.IsTlas, "IsTlas" );
				Assert.IsTrue( tlas.BvhOverAabbs, "BvhOverAabbs" );
				for ( int i = 0; i < instCount; i++ )
				{
					BlasInstanceDouble inst = instArray[ i ];
					DblDumpFile.InstanceRecord record = refFile.Instances[ i ];
					for ( int c = 0; c < 16; c++ )
					{
						AssertBitsEqual( inst.Transform[ c ], record.Transform[ c ], $"instance {i} transform[{c}]" );
						AssertBitsEqual( inst.InvTransform[ c ], record.InvTransform[ c ], $"instance {i} invTransform[{c}]" );
					}
					AssertDouble3BitsEqual( inst.AabbMin, record.AabbMin, $"instance {i} AabbMin" );
					AssertDouble3BitsEqual( inst.AabbMax, record.AabbMax, $"instance {i} AabbMax" );
					Assert.AreEqual( record.BlasIdx, inst.BlasIdx, $"instance {i} BlasIdx" );
					Assert.AreEqual( record.Mask, inst.Mask, $"instance {i} Mask" );
				}

				CompareNodes( $"{sceneName} tlas", tlas.Nodes, tlas.UsedNodes, refFile.TlasUsedNodes, refFile.TlasNodes );
				ComparePrimIdx( $"{sceneName} tlas", tlas.PrimIdx, tlas.IdxCount, refFile.TlasPrimIdx );
				AssertDouble3BitsEqual( tlas.AabbMin, refFile.TlasAabbMin, "TLAS AabbMin" );
				AssertDouble3BitsEqual( tlas.AabbMax, refFile.TlasAabbMax, "TLAS AabbMax" );

				CompareTlasRays( sceneName, ref tlas, refFile.TlasRays );
			}
			finally
			{
				tlas.Dispose();
				blas.Dispose();
				if ( instArray.IsCreated )
				{
					instArray.Dispose();
				}
				if ( blasArray.IsCreated )
				{
					blasArray.Dispose();
				}
				wide.Dispose();
			}
		}

		/// <summary>
		/// Traces the TLAS ray set and compares it against the dump. The occlusion flags are part
		/// of the comparison: they are zero throughout the reference, because BVH_Double::
		/// IsOccludedTLAS never initialises the ray it hands to the BLAS; see the comment on
		/// BvhDouble.IsOccludedTlas.
		/// </summary>
		static void CompareTlasRays( string sceneName, ref BvhDouble tlas, DblDumpFile.TlasRayHit[] expected )
		{
			int rayCount = expected.Length;
			NativeArray<double3> origins = new NativeArray<double3>( rayCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory );
			NativeArray<double3> directions = new NativeArray<double3>( rayCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory );
			for ( int i = 0; i < rayCount; i++ )
			{
				origins[ i ] = expected[ i ].O;
				directions[ i ] = expected[ i ].D;
			}
			NativeArray<IntersectionDouble> hits = new NativeArray<IntersectionDouble>( rayCount, Allocator.Persistent );
			NativeArray<int> occludedFull = new NativeArray<int>( rayCount, Allocator.Persistent );
			NativeArray<int> occludedHalf = new NativeArray<int>( rayCount, Allocator.Persistent );
			try
			{
				new TraceJob
				{
					Bvh = tlas,
					Origins = origins,
					Directions = directions,
					Hits = hits,
					OccludedFull = occludedFull,
					OccludedHalf = occludedHalf
				}.Schedule( rayCount, 256 ).Complete();

				int mismatches = 0, grazing = 0, hitCount = 0, occlusionMismatches = 0;
				for ( int i = 0; i < rayCount; i++ )
				{
					DblDumpFile.TlasRayHit rh = expected[ i ];
					IntersectionDouble hit = hits[ i ];
					bool refHit = rh.T < BvhDoubleConstants.Far;
					bool gotHit = hit.T < BvhDoubleConstants.Far;
					if ( refHit )
					{
						hitCount++;
					}
					bool same = BitsEqual( hit.T, rh.T ) && BitsEqual( hit.U, rh.U ) && BitsEqual( hit.V, rh.V )
						&& hit.Prim == rh.Prim && hit.Inst == rh.Inst;
					if ( !same )
					{
						if ( ( refHit && IsGrazing( rh.U, rh.V ) ) || ( gotHit && IsGrazing( hit.U, hit.V ) ) )
						{
							grazing++;
						}
						else
						{
							if ( mismatches < 3 )
							{
								TestContext.WriteLine( $"  mismatch ray {i}: ref t {rh.T:R} u {rh.U:R} v {rh.V:R} prim {rh.Prim} inst {rh.Inst} | got t {hit.T:R} u {hit.U:R} v {hit.V:R} prim {hit.Prim} inst {hit.Inst}" );
							}
							mismatches++;
						}
					}
					if ( ( occludedFull[ i ] != 0 ) != ( rh.OccludedFull != 0 ) )
					{
						occlusionMismatches++;
					}
				}
				double hitRatio = ( double )hitCount / rayCount;
				TestContext.WriteLine( $"{sceneName} tlas: hit ratio {hitRatio:P2}, mismatches {mismatches}/{rayCount}, grazing {grazing}, occlusion mismatches {occlusionMismatches}" );
				Assert.AreEqual( 0, mismatches, sceneName + " tlas intersect mismatches" );
				Assert.AreEqual( 0, occlusionMismatches, sceneName + " tlas occlusion mismatches" );
			}
			finally
			{
				hits.Dispose();
				occludedFull.Dispose();
				occludedHalf.Dispose();
				origins.Dispose();
				directions.Dispose();
			}
		}

		/// <summary>
		/// Self-consistency of the custom-geometry path: the same triangles, this time handed to
		/// the builder as AABBs from a callback and intersected by callbacks, must produce exactly
		/// the tree and the hits of the triangle path.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void CustomGeometry_MatchesTrianglePath( string sceneName )
		{
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<double3> wide = WidenVertices( verts );
			verts.Dispose();
			DblDumpFile refFile = DblDumpFile.Load( refPath );
			BvhDouble custom = BvhDouble.Create( Allocator.Persistent );
			try
			{
				TriGeometry.VertsAddress.Data = ( long )( double3* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( wide );
				custom.Build( BurstCompiler.CompileFunctionPointer<GetAabbDoubleDelegate>( TriGeometry.GetAabb ), triCount );
				custom.CustomIntersect = BurstCompiler.CompileFunctionPointer<CustomIntersectDoubleDelegate>( TriGeometry.Intersect );
				custom.CustomIsOccluded = BurstCompiler.CompileFunctionPointer<CustomOccludedDoubleDelegate>( TriGeometry.IsOccluded );

				Assert.IsTrue( custom.BvhOverAabbs, "a BVH over AABBs has no vertices" );
				Assert.IsTrue( custom.Verts == null, "Verts" );
				// the fragments are the same boxes as the triangle build's, so the tree is too.
				CompareNodes( $"{sceneName} custom", custom.Nodes, custom.UsedNodes, refFile.UsedNodes, refFile.Nodes );
				ComparePrimIdx( $"{sceneName} custom", custom.PrimIdx, custom.IdxCount, refFile.PrimIdx );

				Trace( ref custom, refFile.Rays, out NativeArray<IntersectionDouble> hits,
					out NativeArray<int> occludedFull, out NativeArray<int> occludedHalf );
				try
				{
					CompareRays( $"{sceneName} custom", refFile.Rays, hits, occludedFull, occludedHalf );
				}
				finally
				{
					hits.Dispose();
					occludedFull.Dispose();
					occludedHalf.Dispose();
				}
			}
			finally
			{
				custom.Dispose();
				wide.Dispose();
			}
		}
	}
}
