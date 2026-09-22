using System.IO;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Traces the reference ray set with each GPU layout and compares the results against the
	/// data dumped by Tools/RefDump/refdump.cpp. The test runner defaults to -nographics, where
	/// compute shaders are unavailable and every case is ignored; run it with graphics enabled.
	/// </summary>
	public class GpuTraceTests
	{
		/// <summary>Barycentric distance to a triangle edge below which a hit counts as grazing.</summary>
		const float GrazingEps = 1e-3f;

		/// <summary>True when the hit lies within GrazingEps of a triangle edge or vertex, where last-bit rounding decides hit or miss.</summary>
		static bool IsGrazing( float u, float v )
		{
			return u < GrazingEps || v < GrazingEps || ( 1f - u - v ) < GrazingEps;
		}

		const float RelativeTolerance = 1e-4f;

		static bool TryGetPaths( string sceneName, out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( sceneName + ".ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		static bool WithinRelative( float actual, float expected, float relTol )
		{
			float scale = math.max( math.abs( expected ), 1e-12f );
			return math.abs( actual - expected ) <= relTol * scale;
		}

		[TestCase( "suzanne", GpuLayout.Bvh2 )]
		[TestCase( "suzanne", GpuLayout.BvhGpu )]
		[TestCase( "suzanne", GpuLayout.Bvh4Gpu )]
		[TestCase( "suzanne", GpuLayout.Cwbvh )]
		[TestCase( "bunny", GpuLayout.Bvh2 )]
		[TestCase( "bunny", GpuLayout.BvhGpu )]
		[TestCase( "bunny", GpuLayout.Bvh4Gpu )]
		[TestCase( "bunny", GpuLayout.Cwbvh )]
		[TestCase( "cryteksponza", GpuLayout.Bvh2 )]
		[TestCase( "cryteksponza", GpuLayout.BvhGpu )]
		[TestCase( "cryteksponza", GpuLayout.Bvh4Gpu )]
		[TestCase( "cryteksponza", GpuLayout.Cwbvh )]
		public void TraceBatch_MatchesReference( string sceneName, GpuLayout layout )
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
			NativeArray<float4> origins = default;
			NativeArray<float4> directions = default;
			NativeArray<float4> hits = default;
			NativeArray<uint> occluded = default;
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh = default;
			BvhGpu bvhGpu = default;
			Bvh4Gpu bvh4Gpu = default;
			BvhCwbvh cwbvh = default;
			GpuTracer tracer = null;
			try
			{
				verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
				bvh.Build( verts, triCount );

				tracer = new GpuTracer();
				switch ( layout )
				{
					case GpuLayout.BvhGpu:
						bvhGpu = BvhGpu.Create( Allocator.Persistent );
						bvhGpu.ConvertFrom( ref bvh );
						tracer.Upload( ref bvhGpu );
						break;
					case GpuLayout.Bvh4Gpu:
						mbvh = Mbvh.Create( 4, Allocator.Persistent );
						mbvh.ConvertFrom( ref bvh );
						bvh4Gpu = Bvh4Gpu.Create( Allocator.Persistent );
						bvh4Gpu.ConvertFrom( ref mbvh );
						tracer.Upload( ref bvh4Gpu );
						break;
					case GpuLayout.Cwbvh:
						mbvh = Mbvh.Create( 8, Allocator.Persistent );
						bvh.Compact();
						bvh.SplitLeafs( 3 );
						mbvh.ConvertFrom( ref bvh );
						cwbvh = BvhCwbvh.Create( Allocator.Persistent );
						cwbvh.ConvertFrom( ref mbvh );
						tracer.Upload( ref cwbvh );
						break;
					default:
						tracer.Upload( ref bvh );
						break;
				}

				RefDumpFile refFile = RefDumpFile.Load( refPath );
				int rayCount = refFile.Rays.Length;
				origins = new NativeArray<float4>( rayCount, Allocator.Persistent );
				directions = new NativeArray<float4>( rayCount, Allocator.Persistent );
				hits = new NativeArray<float4>( rayCount, Allocator.Persistent );
				occluded = new NativeArray<uint>( rayCount, Allocator.Persistent );
				for ( int i = 0; i < rayCount; i++ )
				{
					RefDumpFile.RayHit rh = refFile.Rays[ i ];
					origins[ i ] = new float4( rh.O, BvhConstants.Far );
					directions[ i ] = new float4( math.normalize( rh.D ), 0f );
				}

				tracer.TraceBatch( origins, directions, hits, occluded );

				int mismatches = 0;
				int ties = 0;
				int grazing = 0;
				int hitCount = 0;
				int occlusionMismatches = 0;
				for ( int i = 0; i < rayCount; i++ )
				{
					RefDumpFile.RayHit rh = refFile.Rays[ i ];
					float4 got = hits[ i ];

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
							TestContext.WriteLine( $"  mismatch ray {i}: O {rh.O} D {rh.D} ref t {rh.T} u {rh.U} v {rh.V} prim {rh.Prim} | got t {got.x} u {got.y} v {got.z} prim {math.asuint( got.w )}" );
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
							&& math.abs( got.y - rh.U ) <= RelativeTolerance
							&& math.abs( got.z - rh.V ) <= RelativeTolerance;
						if ( !ok )
						{
							// A different primitive at the same distance is a legitimate tie between
							// overlapping triangles, decided by traversal order and last-bit rounding.
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
									TestContext.WriteLine( $"  mismatch ray {i}: O {rh.O} D {rh.D} ref t {rh.T} u {rh.U} v {rh.V} prim {rh.Prim} | got t {got.x} u {got.y} v {got.z} prim {prim}" );
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
				TestContext.WriteLine( $"{sceneName} / {layout}: nodes {tracer.NodeCount}, blocks {tracer.BlockCount}, " +
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
				if ( bvh4Gpu.IsCreated )
				{
					bvh4Gpu.Dispose();
				}
				if ( bvhGpu.IsCreated )
				{
					bvhGpu.Dispose();
				}
				if ( mbvh.IsCreated )
				{
					mbvh.Dispose();
				}
				bvh.Dispose();
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
