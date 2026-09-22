using Stopwatch = System.Diagnostics.Stopwatch;
using TinyBVH;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace TinyBVH.Samples
{
	/// <summary>CPU backend that traces the scene with a TinyBVH.Bvh8Cpu from a Burst job.</summary>
	public sealed class CpuBvh8Backend : IRefittableBackend
	{
		public string Name => Bvh8Cpu.IsSimdSupported ? "CPU BVH8 (AVX2)" : "CPU BVH8 (scalar)";
		public bool IsGpu => false;
		public BvhBuilder Builder { get; set; }
		public bool Presplit { get; set; }
		public bool ThreadedBuild { get; set; }
		public bool Optimize { get; set; }
		public NativeArray<uint> OpacityMap { get; set; }
		public uint OpacityMapN { get; set; }
		public bool SupportsOpacityMap => true;
		public float BuildMs { get; private set; }
		public float RefitMs { get; private set; }
		public long NodeCount => bvh8.IsCreated ? bvh8.Mbvh8.UsedNodes : 0;

		private Bvh8Cpu bvh8;
		private NativeArray<float4> vertices;

		private Texture2D texture;
		private NativeArray<Color32> pixelBuffer;
		private int texWidth;
		private int texHeight;

		public void Build( NativeArray<float4> vertices, uint triCount )
		{
			this.vertices = vertices;
			if ( !bvh8.IsCreated )
			{
				bvh8 = Bvh8Cpu.Create( Allocator.Persistent );
			}
			Stopwatch stopwatch = Stopwatch.StartNew();
			BvhBuilderOptions.Build( ref bvh8, Builder, Presplit, ThreadedBuild, vertices, triCount );
			if ( Optimize )
			{
				bvh8.Optimize();
			}
			if ( OpacityMap.IsCreated && OpacityMapN > 0 && SupportsOpacityMap )
			{
				bvh8.SetOpacityMicroMaps( OpacityMap, OpacityMapN );
			}
			else
			{
				bvh8.ClearOpacityMicroMaps();
			}
			stopwatch.Stop();
			BuildMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
		}

		public void Refit()
		{
			Stopwatch stopwatch = Stopwatch.StartNew();
			bvh8.Refit();
			stopwatch.Stop();
			RefitMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
		}

		public Texture Render( in RenderSettings settings, bool waitForGpu )
		{
			EnsureTexture( settings.Width, settings.Height );
			RenderJob job = new RenderJob
			{
				Vertices = vertices,
				Bvh8 = bvh8,
				Pixels = pixelBuffer,
				Width = settings.Width,
				Height = settings.Height,
				Origin = settings.Origin,
				TopLeftDir = settings.TopLeftDir,
				Horizontal = settings.Horizontal,
				Vertical = settings.Vertical,
				Mode = settings.Mode,
				Shadows = settings.Shadows,
				SunDir = settings.SunDir,
				SceneDiagonal = settings.SceneDiagonal
			};
			JobHandle handle = job.Schedule( settings.Height, 4 );
			handle.Complete();
			texture.Apply( false );
			return texture;
		}

		public void Dispose()
		{
			if ( bvh8.IsCreated )
			{
				bvh8.Dispose();
			}
			if ( texture != null )
			{
				Object.Destroy( texture );
				texture = null;
			}
		}

		private void EnsureTexture( int width, int height )
		{
			if ( texture != null && texWidth == width && texHeight == height )
			{
				return;
			}
			if ( texture != null )
			{
				Object.Destroy( texture );
			}
			texture = new Texture2D( width, height, TextureFormat.RGBA32, false );
			texture.filterMode = FilterMode.Point;
			texWidth = width;
			texHeight = height;
			pixelBuffer = texture.GetRawTextureData<Color32>();
		}

		/// <summary>Traces one ray per pixel; one job index covers a full scanline of the output texture.</summary>
		[BurstCompile]
		private struct RenderJob : IJobParallelFor
		{
			[ReadOnly] public NativeArray<float4> Vertices;
			public Bvh8Cpu Bvh8;
			[NativeDisableParallelForRestriction] public NativeArray<Color32> Pixels;
			public int Width;
			public int Height;
			public float3 Origin;
			public float3 TopLeftDir;
			public float3 Horizontal;
			public float3 Vertical;
			public DisplayMode Mode;
			public bool Shadows;
			public float3 SunDir;
			public float SceneDiagonal;

			public void Execute( int row )
			{
				// Row 0 of the raw texture buffer is the bottom scanline; flip v to match.
				float v = 1f - ( ( row + 0.5f ) / Height );
				int rowOffset = row * Width;
				for ( int x = 0; x < Width; x++ )
				{
					float u = ( x + 0.5f ) / Width;
					float3 dir = TopLeftDir + ( Horizontal * u ) + ( Vertical * v );
					Ray ray = new Ray( Origin, dir );
					int steps = Bvh8.Intersect( ref ray );
					Pixels[ rowOffset + x ] = Shade( ray, steps );
				}
			}

			private Color32 Shade( in Ray ray, int steps )
			{
				if ( ray.Hit.T >= BvhConstants.Far )
				{
					return new Color32( 20, 20, 28, 255 );
				}
				switch ( Mode )
				{
					case DisplayMode.Normals:
						return ToColor32( ( ComputeNormal( ray ) * 0.5f ) + 0.5f );
					case DisplayMode.Depth:
						float depth = math.saturate( ray.Hit.T / SceneDiagonal );
						return ToColor32( new float3( depth, depth, depth ) );
					case DisplayMode.TraversalSteps:
						return HeatMap( steps );
					case DisplayMode.Barycentrics:
						return ToColor32( new float3( ray.Hit.U, ray.Hit.V, 1f - ray.Hit.U - ray.Hit.V ) );
					default:
						return ShadeLit( ray );
				}
			}

			private Color32 ShadeLit( in Ray ray )
			{
				float3 n = ComputeNormal( ray );
				float lit = ( math.saturate( math.dot( n, -SunDir ) ) * 0.8f ) + 0.2f;
				if ( Shadows )
				{
					float3 hitPos = ray.O + ( ray.D * ray.Hit.T );
					Ray shadowRay = new Ray( hitPos + ( n * ( 1e-3f * SceneDiagonal ) ), -SunDir );
					if ( Bvh8.IsOccluded( shadowRay ) )
					{
						lit = 0.2f;
					}
				}
				float3 albedo = new float3( 0.65f, 0.65f, 0.65f );
				return ToColor32( albedo * lit );
			}

			private float3 ComputeNormal( in Ray ray )
			{
				int prim = ( int )ray.Hit.Prim * 3;
				float3 a = Vertices[ prim ].xyz;
				float3 b = Vertices[ prim + 1 ].xyz;
				float3 c = Vertices[ prim + 2 ].xyz;
				float3 n = math.normalize( math.cross( b - a, c - a ) );
				return math.dot( n, ray.D ) > 0f ? -n : n;
			}

			private static Color32 HeatMap( int steps )
			{
				float t = math.saturate( steps / 64f );
				float3 col = t < 0.5f
					? math.lerp( new float3( 0f, 0f, 1f ), new float3( 0f, 1f, 0f ), t * 2f )
					: math.lerp( new float3( 0f, 1f, 0f ), new float3( 1f, 0f, 0f ), ( t - 0.5f ) * 2f );
				return ToColor32( col );
			}

			private static Color32 ToColor32( float3 c )
			{
				c = math.saturate( c );
				return new Color32( ( byte )( c.x * 255f ), ( byte )( c.y * 255f ), ( byte )( c.z * 255f ), 255 );
			}
		}
	}
}
