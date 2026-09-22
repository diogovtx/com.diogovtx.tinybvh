using System;
using System.IO;
using System.Runtime.InteropServices;
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
	/// Tests for the opacity micro map port. Every layout that supports maps - the base BVH, the
	/// BVH4_CPU and BVH8_CPU SIMD layouts and the Bvh2, BvhGpu and TLAS-over-BvhGpu GPU kernels -
	/// is given the procedural map Tools/RefDump/refdump.cpp attaches and traced with the same
	/// 65,536 rays, then compared against that file's opacity section. BVH4_GPU and CWBVH have no
	/// opacity support here because the OpenCL kernels they are ported from have none either.
	///
	/// A last test works on one triangle with a two-by-two map and checks, micro-triangle by
	/// micro-triangle, that exactly the opaque one is hit.
	/// </summary>
	public class BvhOpacityTests
	{
		/// <summary>Subdivision the reference uses; RefDumpFile.OpMapN is asserted against it.</summary>
		const uint MapN = 8;

		/// <summary>Barycentric distance to a micro-triangle boundary below which a hit counts as grazing.</summary>
		const float GrazingEps = 1e-3f;

		const float RelativeTolerance = 1e-4f;

		/// <summary>
		/// One traced ray, reduced to the fields the reference stores, so the CPU layouts and the
		/// GPU batch kernels can go through the same comparison.
		/// </summary>
		struct HitRecord
		{
			public float T;
			public float U;
			public float V;
			public uint Prim;
			public bool Occluded;
		}

		[BurstCompile( CompileSynchronously = true )]
		private struct BvhIntersectJob : IJobParallelFor
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
		private struct Bvh4IntersectJob : IJobParallelFor
		{
			public Bvh4Cpu Bvh4;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> Occluded;

			/// <summary>Forces the scalar fallback instead of the SSE path.</summary>
			[MarshalAs( UnmanagedType.U1 )] public bool Scalar;

			public void Execute( int i )
			{
				Ray ray = new Ray( Origins[ i ], Directions[ i ] );
				Ray shadowRay = new Ray( Origins[ i ], Directions[ i ] );
				if ( Scalar )
				{
					Bvh4.IntersectScalarPath( ref ray );
					Occluded[ i ] = Bvh4.IsOccludedScalarPath( shadowRay ) ? 1 : 0;
				}
				else
				{
					Bvh4.Intersect( ref ray );
					Occluded[ i ] = Bvh4.IsOccluded( shadowRay ) ? 1 : 0;
				}
				Hits[ i ] = ray.Hit;
			}
		}

		[BurstCompile( CompileSynchronously = true )]
		private struct Bvh8IntersectJob : IJobParallelFor
		{
			public Bvh8Cpu Bvh8;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> Occluded;

			/// <summary>Forces the scalar fallback instead of the AVX2 path.</summary>
			[MarshalAs( UnmanagedType.U1 )] public bool Scalar;

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

		// -------------------------------------------------------------------
		// The procedural map, and the grazing rule that goes with it
		// -------------------------------------------------------------------

		/// <summary>
		/// The procedural opacity map refdump.cpp attaches, reproduced exactly: with N = n each
		/// primitive owns n * n bits, one per micro-triangle, packed low bit first into
		/// ( ( n * n ) + 31 ) / 32 words. Micro-triangle b of primitive t is opaque when
		/// ( ( t * 7 + b * 13 ) &amp; 3 ) != 0, so one micro-triangle in four is a hole.
		///
		/// Two zeroed slack words follow the last primitive, as in refdump.cpp: tinybvh derives
		/// the bit index from u and v without clamping, so a hit with u + v == 1 exactly reads one
		/// word past the last primitive's own, and the slack makes both sides read the same zero.
		/// </summary>
		static NativeArray<uint> BuildProceduralMap( uint triCount, uint n, Allocator allocator )
		{
			uint words = ( ( n * n ) + 31 ) >> 5;
			NativeArray<uint> map = new NativeArray<uint>( ( int )( ( triCount * words ) + 2 ), allocator );
			for ( uint t = 0; t < triCount; t++ )
			{
				for ( uint b = 0; b < n * n; b++ )
				{
					if ( ( ( ( t * 7u ) + ( b * 13u ) ) & 3u ) != 0u )
					{
						int w = ( int )( ( t * words ) + ( b >> 5 ) );
						map[ w ] = map[ w ] | ( 1u << ( int )( b & 31 ) );
					}
				}
			}
			return map;
		}

		/// <summary>
		/// True when x is within GrazingEps of a multiple of 1 / MapN. The three micro-triangle
		/// boundary families are u + v, v and 1 - u at those multiples, and the multiples 0 and 1
		/// are the triangle's own edges, so this one rule covers both the edge grazing the other
		/// suites classify and the new micro-triangle grazing.
		/// </summary>
		static bool NearGridLine( float x )
		{
			float scaled = x * MapN;
			return math.abs( scaled - math.round( scaled ) ) < GrazingEps * MapN;
		}

		/// <summary>
		/// True when the hit sits close enough to a micro-triangle boundary that last-bit rounding
		/// decides which micro-triangle it lands in, and therefore whether it is a hit at all.
		/// </summary>
		static bool IsGrazing( float u, float v )
		{
			return NearGridLine( u + v ) || NearGridLine( v ) || NearGridLine( 1f - u );
		}

		static bool WithinRelative( float actual, float expected, float relTol )
		{
			float scale = math.max( math.abs( expected ), 1e-12f );
			return math.abs( actual - expected ) <= relTol * scale;
		}

		// -------------------------------------------------------------------
		// Shared scaffolding
		// -------------------------------------------------------------------

		static bool TryGetPaths( string sceneName, out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( sceneName + ".ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		/// <summary>Loads the reference file and skips the test when it predates the opacity section.</summary>
		static RefDumpFile LoadReference( string sceneName, out string binPath )
		{
			if ( !TryGetPaths( sceneName, out binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}
			RefDumpFile refFile = RefDumpFile.Load( refPath );
			if ( refFile.OpacityRays == null || refFile.OpacityRays.Length == 0 )
			{
				Assert.Ignore( $"{refPath} has no opacity section; re-run Tools/RefDump/run_all.bat" );
			}
			Assert.AreEqual( MapN, refFile.OpMapN, "reference opacity subdivision" );
			Assert.AreEqual( ( ( MapN * MapN ) + 31 ) >> 5, refFile.OpMapWords, "reference opacity words per triangle" );
			return refFile;
		}

		static HitRecord[] ToRecords( Intersection[] hits, int[] occluded )
		{
			HitRecord[] records = new HitRecord[ hits.Length ];
			for ( int i = 0; i < hits.Length; i++ )
			{
				records[ i ] = new HitRecord
				{
					T = hits[ i ].T,
					U = hits[ i ].U,
					V = hits[ i ].V,
					Prim = hits[ i ].Prim,
					Occluded = occluded[ i ] != 0
				};
			}
			return records;
		}

		static HitRecord[] ToRecords( NativeArray<float4> hits, NativeArray<uint> occluded )
		{
			HitRecord[] records = new HitRecord[ hits.Length ];
			for ( int i = 0; i < hits.Length; i++ )
			{
				float4 got = hits[ i ];
				records[ i ] = new HitRecord
				{
					T = got.x,
					U = got.y,
					V = got.z,
					Prim = math.asuint( got.w ),
					Occluded = occluded[ i ] != 0
				};
			}
			return records;
		}

		/// <summary>
		/// Compares one traced ray set against a reference ray set. A different primitive at the
		/// same distance is a legitimate tie, and a hit within GrazingEps of a micro-triangle
		/// boundary is decided by last-bit rounding, so both are counted apart from the real
		/// mismatches. Pass assert = false for the paths that only get reported: under Mono float
		/// arithmetic is evaluated in double, which can move a barycentric across a boundary.
		/// </summary>
		static void Compare( string label, RefDumpFile.RayHit[] expected, HitRecord[] got, bool assert )
		{
			int rayCount = expected.Length;
			int mismatches = 0, ties = 0, grazing = 0, hitCount = 0;
			int occlusionMismatches = 0, occlusionGrazing = 0;
			for ( int i = 0; i < rayCount; i++ )
			{
				RefDumpFile.RayHit rh = expected[ i ];
				HitRecord hit = got[ i ];
				bool refHit = rh.T < BvhConstants.Far;
				bool gotHit = hit.T < BvhConstants.Far;
				bool nearBoundary = ( refHit && IsGrazing( rh.U, rh.V ) ) || ( gotHit && IsGrazing( hit.U, hit.V ) );
				if ( refHit != gotHit )
				{
					if ( nearBoundary )
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
					bool sameT = WithinRelative( hit.T, rh.T, RelativeTolerance );
					bool ok = sameT
						&& hit.Prim == rh.Prim
						&& math.abs( hit.U - rh.U ) <= RelativeTolerance
						&& math.abs( hit.V - rh.V ) <= RelativeTolerance;
					if ( !ok )
					{
						if ( sameT )
						{
							ties++;
						}
						else if ( nearBoundary )
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
				}

				if ( hit.Occluded != ( rh.OccludedFull != 0 ) )
				{
					// The any-hit query walks the same micro-triangles, so it grazes the same way;
					// the closest hit's barycentrics are the only ones the reference records.
					if ( nearBoundary )
					{
						occlusionGrazing++;
					}
					else
					{
						occlusionMismatches++;
					}
				}
			}

			float hitRatio = ( float )hitCount / rayCount;
			TestContext.WriteLine( $"{label}: hit ratio {hitRatio:P2}, mismatches {mismatches}/{rayCount}, same-distance ties {ties}, " +
				$"grazing {grazing}, occlusion mismatches {occlusionMismatches}, occlusion grazing {occlusionGrazing}" );
			if ( assert )
			{
				Assert.AreEqual( 0, mismatches, label + " intersect mismatches" );
				Assert.AreEqual( 0, occlusionMismatches, label + " occlusion mismatches" );
			}
		}

		// -------------------------------------------------------------------
		// (a) the base BVH
		// -------------------------------------------------------------------

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Bvh_MatchesReference( string sceneName )
		{
			RefDumpFile refFile = LoadReference( sceneName, out string binPath );
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<uint> map = BuildProceduralMap( triCount, MapN, Allocator.Persistent );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				bvh.SetOpacityMicroMaps( map, MapN );

				int rayCount = refFile.OpacityRays.Length;
				Intersection[] hits = new Intersection[ rayCount ];
				int[] occluded = new int[ rayCount ];
				for ( int i = 0; i < rayCount; i++ )
				{
					RefDumpFile.RayHit rh = refFile.OpacityRays[ i ];
					Ray ray = new Ray( rh.O, rh.D );
					bvh.Intersect( ref ray );
					hits[ i ] = ray.Hit;
					occluded[ i ] = bvh.IsOccluded( new Ray( rh.O, rh.D ) ) ? 1 : 0;
				}
				Compare( $"{sceneName} Bvh opacity mono", refFile.OpacityRays, ToRecords( hits, occluded ), true );
			}
			finally
			{
				bvh.Dispose();
				map.Dispose();
				verts.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void BvhJob_MatchesReference( string sceneName )
		{
			RefDumpFile refFile = LoadReference( sceneName, out string binPath );
			int rayCount = refFile.OpacityRays.Length;
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<uint> map = BuildProceduralMap( triCount, MapN, Allocator.Persistent );
			NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> hits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			NativeArray<int> occluded = new NativeArray<int>( rayCount, Allocator.Persistent );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				bvh.SetOpacityMicroMaps( map, MapN );
				FillRays( refFile.OpacityRays, origins, directions );

				BvhIntersectJob job = new BvhIntersectJob
				{
					Bvh = bvh,
					Origins = origins,
					Directions = directions,
					Hits = hits,
					Occluded = occluded
				};
				job.Schedule( rayCount, 64 ).Complete();

				Compare( $"{sceneName} Bvh opacity burst", refFile.OpacityRays, ToRecords( ToArray( hits ), ToArray( occluded ) ), true );
			}
			finally
			{
				bvh.Dispose();
				map.Dispose();
				verts.Dispose();
				origins.Dispose();
				directions.Dispose();
				hits.Dispose();
				occluded.Dispose();
			}
		}

		// -------------------------------------------------------------------
		// (b) the wide CPU layouts
		// -------------------------------------------------------------------

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Bvh4Cpu_MatchesReference( string sceneName )
		{
			RefDumpFile refFile = LoadReference( sceneName, out string binPath );
			int rayCount = refFile.OpacityRays.Length;
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<uint> map = BuildProceduralMap( triCount, MapN, Allocator.Persistent );
			NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> hits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			NativeArray<int> occluded = new NativeArray<int>( rayCount, Allocator.Persistent );
			Bvh4Cpu bvh4Cpu = Bvh4Cpu.Create( Allocator.Persistent );
			try
			{
				bvh4Cpu.Build( verts, triCount );
				bvh4Cpu.SetOpacityMicroMaps( map, MapN );
				FillRays( refFile.OpacityRays, origins, directions );

				Bvh4IntersectJob job = new Bvh4IntersectJob
				{
					Bvh4 = bvh4Cpu,
					Origins = origins,
					Directions = directions,
					Hits = hits,
					Occluded = occluded,
					Scalar = false
				};
				job.Schedule( rayCount, 64 ).Complete();
				Compare( $"{sceneName} Bvh4Cpu opacity burst simd", refFile.OpacityRays, ToRecords( ToArray( hits ), ToArray( occluded ) ), true );

				job.Scalar = true;
				job.Schedule( rayCount, 64 ).Complete();
				Compare( $"{sceneName} Bvh4Cpu opacity burst scalar", refFile.OpacityRays, ToRecords( ToArray( hits ), ToArray( occluded ) ), true );

				// The same scalar fallback under Mono, where float arithmetic is evaluated in
				// double: reported rather than asserted, as Bvh8CpuTests does.
				Intersection[] monoHits = new Intersection[ rayCount ];
				int[] monoOccluded = new int[ rayCount ];
				for ( int i = 0; i < rayCount; i++ )
				{
					RefDumpFile.RayHit rh = refFile.OpacityRays[ i ];
					Ray ray = new Ray( rh.O, rh.D );
					bvh4Cpu.IntersectScalarPath( ref ray );
					monoHits[ i ] = ray.Hit;
					monoOccluded[ i ] = bvh4Cpu.IsOccludedScalarPath( new Ray( rh.O, rh.D ) ) ? 1 : 0;
				}
				Compare( $"{sceneName} Bvh4Cpu opacity mono scalar", refFile.OpacityRays, ToRecords( monoHits, monoOccluded ), false );
			}
			finally
			{
				bvh4Cpu.Dispose();
				map.Dispose();
				verts.Dispose();
				origins.Dispose();
				directions.Dispose();
				hits.Dispose();
				occluded.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Bvh8Cpu_MatchesReference( string sceneName )
		{
			RefDumpFile refFile = LoadReference( sceneName, out string binPath );
			int rayCount = refFile.OpacityRays.Length;
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<uint> map = BuildProceduralMap( triCount, MapN, Allocator.Persistent );
			NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> hits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			NativeArray<int> occluded = new NativeArray<int>( rayCount, Allocator.Persistent );
			Bvh8Cpu bvh8Cpu = Bvh8Cpu.Create( Allocator.Persistent );
			try
			{
				bvh8Cpu.Build( verts, triCount );
				bvh8Cpu.SetOpacityMicroMaps( map, MapN );
				FillRays( refFile.OpacityRays, origins, directions );

				Bvh8IntersectJob job = new Bvh8IntersectJob
				{
					Bvh8 = bvh8Cpu,
					Origins = origins,
					Directions = directions,
					Hits = hits,
					Occluded = occluded,
					Scalar = false
				};
				if ( Bvh8Cpu.IsSimdSupported )
				{
					job.Schedule( rayCount, 64 ).Complete();
					Compare( $"{sceneName} Bvh8Cpu opacity burst avx2", refFile.OpacityRays, ToRecords( ToArray( hits ), ToArray( occluded ) ), true );
				}
				else
				{
					TestContext.WriteLine( "no AVX2/FMA on this machine; the SIMD path is skipped" );
				}

				job.Scalar = true;
				job.Schedule( rayCount, 64 ).Complete();
				Compare( $"{sceneName} Bvh8Cpu opacity burst scalar", refFile.OpacityRays, ToRecords( ToArray( hits ), ToArray( occluded ) ), true );

				Intersection[] monoHits = new Intersection[ rayCount ];
				int[] monoOccluded = new int[ rayCount ];
				for ( int i = 0; i < rayCount; i++ )
				{
					RefDumpFile.RayHit rh = refFile.OpacityRays[ i ];
					Ray ray = new Ray( rh.O, rh.D );
					bvh8Cpu.IntersectScalarPath( ref ray );
					monoHits[ i ] = ray.Hit;
					monoOccluded[ i ] = bvh8Cpu.IsOccludedScalarPath( new Ray( rh.O, rh.D ) ) ? 1 : 0;
				}
				Compare( $"{sceneName} Bvh8Cpu opacity mono scalar", refFile.OpacityRays, ToRecords( monoHits, monoOccluded ), false );
			}
			finally
			{
				bvh8Cpu.Dispose();
				map.Dispose();
				verts.Dispose();
				origins.Dispose();
				directions.Dispose();
				hits.Dispose();
				occluded.Dispose();
			}
		}

		// -------------------------------------------------------------------
		// (c) the GPU kernels
		// -------------------------------------------------------------------

		[TestCase( "suzanne", GpuLayout.Bvh2 )]
		[TestCase( "suzanne", GpuLayout.BvhGpu )]
		[TestCase( "bunny", GpuLayout.Bvh2 )]
		[TestCase( "bunny", GpuLayout.BvhGpu )]
		[TestCase( "cryteksponza", GpuLayout.Bvh2 )]
		[TestCase( "cryteksponza", GpuLayout.BvhGpu )]
		public void GpuTraceBatch_MatchesReference( string sceneName, GpuLayout layout )
		{
			if ( !GpuTracer.IsSupported )
			{
				Assert.Ignore( "compute shaders are unavailable; run the test runner with graphics enabled" );
			}
			RefDumpFile refFile = LoadReference( sceneName, out string binPath );
			int rayCount = refFile.OpacityRays.Length;
			NativeArray<float4> verts = default;
			NativeArray<uint> map = default;
			NativeArray<float4> origins = default;
			NativeArray<float4> directions = default;
			NativeArray<float4> hits = default;
			NativeArray<uint> occluded = default;
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			BvhGpu bvhGpu = default;
			GpuTracer tracer = null;
			try
			{
				verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
				map = BuildProceduralMap( triCount, MapN, Allocator.Persistent );
				bvh.Build( verts, triCount );
				bvh.SetOpacityMicroMaps( map, MapN );

				tracer = new GpuTracer();
				if ( layout == GpuLayout.BvhGpu )
				{
					// The map has to be on the base BVH before the conversion: BvhGpu.Source is a
					// value copy of it, and that copy is what GpuTracer uploads the map from.
					bvhGpu = BvhGpu.Create( Allocator.Persistent );
					bvhGpu.ConvertFrom( ref bvh );
					tracer.Upload( ref bvhGpu );
				}
				else
				{
					tracer.Upload( ref bvh );
				}

				origins = new NativeArray<float4>( rayCount, Allocator.Persistent );
				directions = new NativeArray<float4>( rayCount, Allocator.Persistent );
				hits = new NativeArray<float4>( rayCount, Allocator.Persistent );
				occluded = new NativeArray<uint>( rayCount, Allocator.Persistent );
				FillRays( refFile.OpacityRays, origins, directions );

				tracer.TraceBatch( origins, directions, hits, occluded );
				Compare( $"{sceneName} / {layout} opacity", refFile.OpacityRays, ToRecords( hits, occluded ), true );
			}
			finally
			{
				tracer?.Dispose();
				if ( bvhGpu.IsCreated )
				{
					bvhGpu.Dispose();
				}
				bvh.Dispose();
				DisposeIfCreated( ref occluded );
				DisposeIfCreated( ref hits );
				DisposeIfCreated( ref directions );
				DisposeIfCreated( ref origins );
				DisposeIfCreated( ref map );
				DisposeIfCreated( ref verts );
			}
		}

		/// <summary>
		/// The TLAS path over Aila-Laine BLASses. Three BLASses are uploaded over the same scene:
		/// two carrying the map and one without, so the run covers a map at offset zero, a map at a
		/// non-zero offset and the "no map" sentinel in the BLAS descriptor. The TLAS itself holds
		/// one identity instance, pointed at one BLAS at a time, so the traced result has to be the
		/// BLAS result: the opacity reference for the mapped BLASses and the plain BLAS reference
		/// for the unmapped one.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void GpuTlasTraceBatch_MatchesReference( string sceneName )
		{
			if ( !GpuTracer.IsSupported )
			{
				Assert.Ignore( "compute shaders are unavailable; run the test runner with graphics enabled" );
			}
			RefDumpFile refFile = LoadReference( sceneName, out string binPath );
			int rayCount = refFile.OpacityRays.Length;
			NativeArray<float4> verts = default;
			NativeArray<uint> map = default;
			NativeArray<float4> origins = default;
			NativeArray<float4> directions = default;
			NativeArray<float4> hits = default;
			NativeArray<uint> occluded = default;
			NativeArray<uint> hitInstances = default;
			NativeArray<Bvh> blasArray = default;
			NativeArray<BlasInstance> instArray = default;
			Bvh blas = Bvh.Create( Allocator.Persistent );
			Bvh tlas = Bvh.Create( Allocator.Persistent );
			BvhGpu mappedA = default;
			BvhGpu mappedB = default;
			BvhGpu plain = default;
			GpuTracer tracer = null;
			try
			{
				verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
				map = BuildProceduralMap( triCount, MapN, Allocator.Persistent );
				blas.Build( verts, triCount );

				// Three shallow copies of the one built BVH; only blas itself owns the memory.
				Bvh plainSource = blas;
				Bvh mappedSource = blas;
				mappedSource.SetOpacityMicroMaps( map, MapN );

				mappedA = BvhGpu.Create( Allocator.Persistent );
				mappedA.ConvertFrom( ref mappedSource );
				mappedB = BvhGpu.Create( Allocator.Persistent );
				mappedB.ConvertFrom( ref mappedSource );
				plain = BvhGpu.Create( Allocator.Persistent );
				plain.ConvertFrom( ref plainSource );

				blasArray = new NativeArray<Bvh>( 3, Allocator.Persistent );
				blasArray[ 0 ] = mappedSource;
				blasArray[ 1 ] = mappedSource;
				blasArray[ 2 ] = plainSource;
				instArray = new NativeArray<BlasInstance>( 1, Allocator.Persistent );

				origins = new NativeArray<float4>( rayCount, Allocator.Persistent );
				directions = new NativeArray<float4>( rayCount, Allocator.Persistent );
				hits = new NativeArray<float4>( rayCount, Allocator.Persistent );
				occluded = new NativeArray<uint>( rayCount, Allocator.Persistent );
				hitInstances = new NativeArray<uint>( rayCount, Allocator.Persistent );
				FillRays( refFile.OpacityRays, origins, directions );

				tracer = new GpuTracer();
				BvhGpu* blasses = stackalloc BvhGpu[ 3 ];
				blasses[ 0 ] = mappedA;
				blasses[ 1 ] = mappedB;
				blasses[ 2 ] = plain;

				string[] labels = { "map at offset 0", "map at a non-zero offset", "no map (sentinel)" };
				for ( uint blasIdx = 0; blasIdx < 3; blasIdx++ )
				{
					BlasInstance inst = BlasInstance.Create( blasIdx );
					inst.Mask = 0xFFFF;
					instArray[ 0 ] = inst;
					tlas.BuildTlas( ( BlasInstance* )instArray.GetUnsafePtr(), 1, ( Bvh* )blasArray.GetUnsafePtr(), 3 );
					tracer.UploadTlas( ref tlas, new ReadOnlySpan<BvhGpu>( blasses, 3 ) );
					tracer.TraceBatch( origins, directions, hits, occluded, hitInstances );

					RefDumpFile.RayHit[] expected = blasIdx == 2 ? refFile.Rays : refFile.OpacityRays;
					Compare( $"{sceneName} / tlas over BvhGpu, {labels[ blasIdx ]}", expected, ToRecords( hits, occluded ), true );
					for ( int i = 0; i < rayCount; i++ )
					{
						Assert.AreEqual( 0u, hitInstances[ i ], $"hit instance of ray {i}" );
					}
				}
			}
			finally
			{
				tracer?.Dispose();
				if ( plain.IsCreated )
				{
					plain.Dispose();
				}
				if ( mappedB.IsCreated )
				{
					mappedB.Dispose();
				}
				if ( mappedA.IsCreated )
				{
					mappedA.Dispose();
				}
				tlas.Dispose();
				blas.Dispose();
				DisposeIfCreated( ref instArray );
				DisposeIfCreated( ref blasArray );
				DisposeIfCreated( ref hitInstances );
				DisposeIfCreated( ref occluded );
				DisposeIfCreated( ref hits );
				DisposeIfCreated( ref directions );
				DisposeIfCreated( ref origins );
				DisposeIfCreated( ref map );
				DisposeIfCreated( ref verts );
			}
		}

		// -------------------------------------------------------------------
		// (d) one triangle, four micro-triangles
		// -------------------------------------------------------------------

		/// <summary>
		/// The four micro-triangles of a two-by-two map, as barycentric probe points well inside
		/// each of them. With N = 2 the bit index tinybvh computes - row * row + int( v * N ) plus
		/// the diagonal correction - is 0 for the corner cell at the origin, then 1, 2 and 3 across
		/// the second row: the cell at u = 1, the flipped middle cell and the cell at v = 1.
		/// </summary>
		static readonly float2[] microProbes =
		{
			new float2( 0.1f, 0.1f ),
			new float2( 0.8f, 0.1f ),
			new float2( 0.3f, 0.3f ),
			new float2( 0.1f, 0.8f )
		};

		[Test]
		public void MicroTriangles_OnlyTheOpaqueOneIsHit()
		{
			NativeArray<float4> verts = UnitTriangle( Allocator.Persistent );
			NativeArray<uint> map = new NativeArray<uint>( 3, Allocator.Persistent );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, 1 );
				bvh.SetOpacityMicroMaps( map, 2 );
				for ( int opaque = 0; opaque < 4; opaque++ )
				{
					map[ 0 ] = 1u << opaque;
					for ( int probe = 0; probe < 4; probe++ )
					{
						float2 uv = microProbes[ probe ];
						Ray ray = new Ray( new float3( uv.x, uv.y, 1f ), new float3( 0f, 0f, -1f ) );
						bvh.Intersect( ref ray );
						bool hit = ray.Hit.T < BvhConstants.Far;
						bool occluded = bvh.IsOccluded( new Ray( new float3( uv.x, uv.y, 1f ), new float3( 0f, 0f, -1f ) ) );
						Assert.AreEqual( probe == opaque, hit, $"opaque micro-triangle {opaque}, probe {probe}: intersect" );
						Assert.AreEqual( probe == opaque, occluded, $"opaque micro-triangle {opaque}, probe {probe}: occlusion" );
						if ( hit )
						{
							Assert.AreEqual( 1f, ray.Hit.T, 1e-5f, "hit distance" );
							Assert.AreEqual( uv.x, ray.Hit.U, 1e-5f, "hit u" );
							Assert.AreEqual( uv.y, ray.Hit.V, 1e-5f, "hit v" );
						}
					}
				}
			}
			finally
			{
				bvh.Dispose();
				map.Dispose();
				verts.Dispose();
			}
		}

		[Test]
		public void MicroTriangles_OnlyTheOpaqueOneIsHit_Burst()
		{
			NativeArray<float4> verts = UnitTriangle( Allocator.Persistent );
			NativeArray<uint> map = new NativeArray<uint>( 3, Allocator.Persistent );
			NativeArray<float3> origins = new NativeArray<float3>( 4, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( 4, Allocator.Persistent );
			NativeArray<Intersection> hits = new NativeArray<Intersection>( 4, Allocator.Persistent );
			NativeArray<int> occluded = new NativeArray<int>( 4, Allocator.Persistent );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, 1 );
				bvh.SetOpacityMicroMaps( map, 2 );
				for ( int probe = 0; probe < 4; probe++ )
				{
					origins[ probe ] = new float3( microProbes[ probe ].x, microProbes[ probe ].y, 1f );
					directions[ probe ] = new float3( 0f, 0f, -1f );
				}
				for ( int opaque = 0; opaque < 4; opaque++ )
				{
					map[ 0 ] = 1u << opaque;
					BvhIntersectJob job = new BvhIntersectJob
					{
						Bvh = bvh,
						Origins = origins,
						Directions = directions,
						Hits = hits,
						Occluded = occluded
					};
					job.Schedule( 4, 1 ).Complete();
					for ( int probe = 0; probe < 4; probe++ )
					{
						bool hit = hits[ probe ].T < BvhConstants.Far;
						Assert.AreEqual( probe == opaque, hit, $"burst: opaque micro-triangle {opaque}, probe {probe}: intersect" );
						Assert.AreEqual( probe == opaque, occluded[ probe ] != 0, $"burst: opaque micro-triangle {opaque}, probe {probe}: occlusion" );
					}
				}
			}
			finally
			{
				bvh.Dispose();
				map.Dispose();
				verts.Dispose();
				origins.Dispose();
				directions.Dispose();
				hits.Dispose();
				occluded.Dispose();
			}
		}

		/// <summary>
		/// One triangle in the z = 0 plane whose barycentrics are its own x and y, so a ray from
		/// ( u, v, 1 ) straight down the negative z axis hits it at exactly ( u, v ) with t = 1.
		/// </summary>
		static NativeArray<float4> UnitTriangle( Allocator allocator )
		{
			NativeArray<float4> verts = new NativeArray<float4>( 3, allocator );
			verts[ 0 ] = new float4( 0f, 0f, 0f, 0f );
			verts[ 1 ] = new float4( 1f, 0f, 0f, 0f );
			verts[ 2 ] = new float4( 0f, 1f, 0f, 0f );
			return verts;
		}

		// -------------------------------------------------------------------
		// Small helpers
		// -------------------------------------------------------------------

		static void FillRays( RefDumpFile.RayHit[] rays, NativeArray<float3> origins, NativeArray<float3> directions )
		{
			for ( int i = 0; i < rays.Length; i++ )
			{
				origins[ i ] = rays[ i ].O;
				directions[ i ] = rays[ i ].D;
			}
		}

		static void FillRays( RefDumpFile.RayHit[] rays, NativeArray<float4> origins, NativeArray<float4> directions )
		{
			for ( int i = 0; i < rays.Length; i++ )
			{
				origins[ i ] = new float4( rays[ i ].O, BvhConstants.Far );
				directions[ i ] = new float4( math.normalize( rays[ i ].D ), 0f );
			}
		}

		static T[] ToArray<T>( NativeArray<T> source ) where T : struct
		{
			T[] result = new T[ source.Length ];
			for ( int i = 0; i < source.Length; i++ )
			{
				result[ i ] = source[ i ];
			}
			return result;
		}

		static void DisposeIfCreated<T>( ref NativeArray<T> array ) where T : struct
		{
			if ( array.IsCreated )
			{
				array.Dispose();
			}
		}
	}
}
