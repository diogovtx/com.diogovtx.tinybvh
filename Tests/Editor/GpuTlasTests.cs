using System.IO;
using NUnit.Framework;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Traces the reference TLAS ray set with the GPU TLAS kernels and compares the results
	/// against the data dumped by Tools/RefDump/refdump.cpp. The scene is reconstructed exactly
	/// as BvhReferenceTests.Tlas_MatchesReference does: one BLAS over the scene triangles and the
	/// three instances of the .ref TLAS section, with only the transform and the mask taken from
	/// the file so the rest is produced by the C# code under test.
	///
	/// The test runner defaults to -nographics, where compute shaders are unavailable and every
	/// case is ignored; run it with graphics enabled.
	/// </summary>
	public class GpuTlasTests
	{
		/// <summary>Barycentric distance to a triangle edge below which a hit counts as grazing.</summary>
		const float GrazingEps = 1e-3f;

		const float RelativeTolerance = 1e-4f;

		/// <summary>True when the hit lies within GrazingEps of a triangle edge or vertex, where last-bit rounding decides hit or miss.</summary>
		static bool IsGrazing( float u, float v )
		{
			return u < GrazingEps || v < GrazingEps || ( 1f - u - v ) < GrazingEps;
		}

		static bool WithinRelative( float actual, float expected, float relTol )
		{
			float scale = math.max( math.abs( expected ), 1e-12f );
			return math.abs( actual - expected ) <= relTol * scale;
		}

		static bool TryGetPaths( string sceneName, out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( sceneName + ".ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		[TestCase( "suzanne", GpuBlasType.BvhGpu )]
		[TestCase( "suzanne", GpuBlasType.Cwbvh )]
		[TestCase( "bunny", GpuBlasType.BvhGpu )]
		[TestCase( "bunny", GpuBlasType.Cwbvh )]
		[TestCase( "cryteksponza", GpuBlasType.BvhGpu )]
		[TestCase( "cryteksponza", GpuBlasType.Cwbvh )]
		public unsafe void TraceTlasBatch_MatchesReference( string sceneName, GpuBlasType blasType )
		{
			if ( !GpuTracer.IsSupported )
			{
				Assert.Ignore( "compute shaders are unavailable; run the test runner with graphics enabled" );
			}
			if ( !TryGetPaths( sceneName, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			NativeArray<float4> verts = default;
			NativeArray<Bvh> blasArray = default;
			NativeArray<BlasInstance> instArray = default;
			NativeArray<float4> origins = default;
			NativeArray<float4> directions = default;
			NativeArray<float4> hits = default;
			NativeArray<uint> occluded = default;
			NativeArray<uint> hitInstances = default;
			Bvh blas = Bvh.Create( Allocator.Persistent );
			Bvh tlas = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh = default;
			BvhGpu bvhGpu = default;
			BvhCwbvh cwbvh = default;
			GpuTracer tracer = null;
			try
			{
				verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
				blas.Build( verts, triCount );

				// Convert the BLAS before the TLAS is built: the CWBVH preparation reshapes the
				// leaves of the base BVH in place, and blasArray holds a value copy of it.
				tracer = new GpuTracer();
				if ( blasType == GpuBlasType.Cwbvh )
				{
					// BVH8_CWBVH::Build prepares the base BVH this way: leaves of at most 3 triangles.
					blas.Compact();
					blas.SplitLeafs( 3 );
					mbvh = Mbvh.Create( 8, Allocator.Persistent );
					mbvh.ConvertFrom( ref blas );
					cwbvh = BvhCwbvh.Create( Allocator.Persistent );
					cwbvh.ConvertFrom( ref mbvh );
				}
				else
				{
					bvhGpu = BvhGpu.Create( Allocator.Persistent );
					bvhGpu.ConvertFrom( ref blas );
				}

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

				if ( blasType == GpuBlasType.Cwbvh )
				{
					tracer.UploadTlas( ref tlas, new System.ReadOnlySpan<BvhCwbvh>( &cwbvh, 1 ) );
				}
				else
				{
					tracer.UploadTlas( ref tlas, new System.ReadOnlySpan<BvhGpu>( &bvhGpu, 1 ) );
				}

				int rayCount = refFile.TlasRays.Length;
				origins = new NativeArray<float4>( rayCount, Allocator.Persistent );
				directions = new NativeArray<float4>( rayCount, Allocator.Persistent );
				hits = new NativeArray<float4>( rayCount, Allocator.Persistent );
				occluded = new NativeArray<uint>( rayCount, Allocator.Persistent );
				hitInstances = new NativeArray<uint>( rayCount, Allocator.Persistent );
				for ( int i = 0; i < rayCount; i++ )
				{
					RefDumpFile.TlasRayHit rh = refFile.TlasRays[ i ];
					origins[ i ] = new float4( rh.O, BvhConstants.Far );
					directions[ i ] = new float4( math.normalize( rh.D ), 0f );
				}

				tracer.TraceBatch( origins, directions, hits, occluded, hitInstances );

				int mismatches = 0;
				int ties = 0;
				int grazing = 0;
				int hitCount = 0;
				int occlusionMismatches = 0;
				for ( int i = 0; i < rayCount; i++ )
				{
					RefDumpFile.TlasRayHit rh = refFile.TlasRays[ i ];
					float4 got = hits[ i ];
					uint gotInst = hitInstances[ i ];

					bool refHit = rh.T < BvhConstants.Far;
					bool gotHit = got.x < BvhConstants.Far;
					if ( refHit != gotHit )
					{
						// A hit within a hair of a triangle edge or vertex is decided by last-bit
						// rounding; the two sides can legitimately disagree on it.
						if ( ( refHit && IsGrazing( rh.U, rh.V ) ) || ( gotHit && IsGrazing( got.y, got.z ) ) )
						{
							grazing++;
							continue;
						}
						if ( mismatches < 3 )
						{
							TestContext.WriteLine( $"  mismatch ray {i}: O {rh.O} D {rh.D} ref t {rh.T} u {rh.U} v {rh.V} prim {rh.Prim} inst {rh.Inst} | got t {got.x} u {got.y} v {got.z} prim {math.asuint( got.w )} inst {gotInst}" );
						}
						mismatches++;
					}
					else if ( refHit )
					{
						hitCount++;
						bool sameT = WithinRelative( got.x, rh.T, RelativeTolerance );
						uint prim = math.asuint( got.w );
						bool ok = sameT
							&& prim == rh.Prim
							&& gotInst == rh.Inst
							&& math.abs( got.y - rh.U ) <= RelativeTolerance
							&& math.abs( got.z - rh.V ) <= RelativeTolerance;
						if ( !ok )
						{
							// A different primitive or instance at the same distance is a legitimate
							// tie between overlapping triangles, decided by traversal order and
							// last-bit rounding.
							if ( sameT )
							{
								ties++;
							}
							else if ( IsGrazing( rh.U, rh.V ) || IsGrazing( got.y, got.z ) )
							{
								grazing++;
							}
							else
							{
								if ( mismatches < 3 )
								{
									TestContext.WriteLine( $"  mismatch ray {i}: O {rh.O} D {rh.D} ref t {rh.T} u {rh.U} v {rh.V} prim {rh.Prim} inst {rh.Inst} | got t {got.x} u {got.y} v {got.z} prim {prim} inst {gotInst}" );
								}
								mismatches++;
							}
						}
					}

					if ( ( occluded[ i ] != 0 ) != ( rh.OccludedFull != 0 ) )
					{
						occlusionMismatches++;
					}
				}

				float hitRatio = ( float )hitCount / rayCount;
				TestContext.WriteLine( $"{sceneName} / tlas over {blasType}: instances {instCount}, nodes {tracer.NodeCount}, blocks {tracer.BlockCount}, " +
					$"hit ratio {hitRatio:P2}, mismatches {mismatches}/{rayCount}, same-distance ties {ties}, grazing {grazing}, " +
					$"occlusion mismatches {occlusionMismatches}/{rayCount}" );
				Assert.AreEqual( 0, mismatches, "intersection mismatches" );
				Assert.AreEqual( 0, occlusionMismatches, "occlusion mismatches" );
			}
			finally
			{
				tracer?.Dispose();
				if ( cwbvh.IsCreated )
				{
					cwbvh.Dispose();
				}
				if ( bvhGpu.IsCreated )
				{
					bvhGpu.Dispose();
				}
				if ( mbvh.IsCreated )
				{
					mbvh.Dispose();
				}
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
				if ( hitInstances.IsCreated )
				{
					hitInstances.Dispose();
				}
				if ( occluded.IsCreated )
				{
					occluded.Dispose();
				}
				if ( hits.IsCreated )
				{
					hits.Dispose();
				}
				if ( directions.IsCreated )
				{
					directions.Dispose();
				}
				if ( origins.IsCreated )
				{
					origins.Dispose();
				}
				if ( verts.IsCreated )
				{
					verts.Dispose();
				}
			}
		}
	}
}
