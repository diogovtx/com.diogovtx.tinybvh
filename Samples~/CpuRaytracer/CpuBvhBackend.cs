using Stopwatch = System.Diagnostics.Stopwatch;
using TinyBVH;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace TinyBVH.Samples
{
	/// <summary>CPU backend that traces the scene with a TinyBVH.Bvh from a Burst job.</summary>
	public sealed unsafe class CpuBvhBackend : IRefittableBackend
	{
		/// <summary>Side of the pixel block one 256-ray packet covers.</summary>
		private const int PacketBlockSide = 16;

		public string Name => "CPU BVH2 (Burst)";
		public bool IsGpu => false;
		public BvhBuilder Builder { get; set; }
		public bool Presplit { get; set; }
		public bool ThreadedBuild { get; set; }
		public bool Optimize { get; set; }
		public NativeArray<uint> OpacityMap { get; set; }
		public uint OpacityMapN { get; set; }
		public bool SupportsOpacityMap => true;
		/// <summary>
		/// When true, primary rays are traced with Bvh.Intersect256Rays over 16x16 pixel blocks
		/// instead of one at a time. Shadow rays stay single. The packet traversal has no step
		/// counter, so the TraversalSteps display mode shows a constant zero while this is on.
		/// </summary>
		public bool Packets { get; set; }
		public float BuildMs { get; private set; }
		public float RefitMs { get; private set; }
		public long NodeCount => bvh.IsCreated ? bvh.UsedNodes : 0;

		private Bvh bvh;
		private NativeArray<float4> vertices;

		private Texture2D texture;
		private NativeArray<Color32> pixelBuffer;
		private int texWidth;
		private int texHeight;

		public void Build( NativeArray<float4> vertices, uint triCount )
		{
			this.vertices = vertices;
			if ( !bvh.IsCreated )
			{
				bvh = Bvh.Create( Allocator.Persistent );
			}
			Stopwatch stopwatch = Stopwatch.StartNew();
			bvh.UseThreadedBuild = ThreadedBuild;
			BvhBuilderOptions.Build( ref bvh, Builder, Presplit, vertices, triCount );
			if ( Optimize )
			{
				bvh.Optimize();
			}
			if ( OpacityMap.IsCreated && OpacityMapN > 0 && SupportsOpacityMap )
			{
				bvh.SetOpacityMicroMaps( OpacityMap, OpacityMapN );
			}
			else
			{
				bvh.ClearOpacityMicroMaps();
			}
			stopwatch.Stop();
			BuildMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
		}

		public void Refit()
		{
			Stopwatch stopwatch = Stopwatch.StartNew();
			bvh.Refit();
			stopwatch.Stop();
			RefitMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
		}

		public Texture Render( in RenderSettings settings, bool waitForGpu )
		{
			EnsureTexture( settings.Width, settings.Height );
			int blocksX = ( settings.Width + PacketBlockSide - 1 ) / PacketBlockSide;
			int blocksY = ( settings.Height + PacketBlockSide - 1 ) / PacketBlockSide;
			RenderJob job = new RenderJob
			{
				Vertices = vertices,
				Bvh = bvh,
				Pixels = pixelBuffer,
				Width = settings.Width,
				Height = settings.Height,
				Packets = Packets,
				BlocksX = blocksX,
				Origin = settings.Origin,
				TopLeftDir = settings.TopLeftDir,
				Horizontal = settings.Horizontal,
				Vertical = settings.Vertical,
				Mode = settings.Mode,
				Shadows = settings.Shadows,
				SunDir = settings.SunDir,
				SceneDiagonal = settings.SceneDiagonal
			};
			JobHandle handle = Packets ? job.Schedule( blocksX * blocksY, 1 ) : job.Schedule( settings.Height, 4 );
			handle.Complete();
			texture.Apply( false );
			return texture;
		}

		public void Dispose()
		{
			if ( bvh.IsCreated )
			{
				bvh.Dispose();
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

		/// <summary>
		/// Traces the output texture. One job index covers a full scanline, or, with Packets, one
		/// 16x16 pixel block traced as a single 256-ray packet.
		/// </summary>
		[BurstCompile]
		private struct RenderJob : IJobParallelFor
		{
			[ReadOnly] public NativeArray<float4> Vertices;
			public Bvh Bvh;
			[NativeDisableParallelForRestriction] public NativeArray<Color32> Pixels;
			public int Width;
			public int Height;
			public bool Packets;
			/// <summary>Number of 16x16 blocks per row of the output texture; only used with Packets.</summary>
			public int BlocksX;
			public float3 Origin;
			public float3 TopLeftDir;
			public float3 Horizontal;
			public float3 Vertical;
			public DisplayMode Mode;
			public bool Shadows;
			public float3 SunDir;
			public float SceneDiagonal;

			public void Execute( int index )
			{
				if ( Packets )
				{
					ExecuteBlock( index );
				}
				else
				{
					ExecuteRow( index );
				}
			}

			private void ExecuteRow( int row )
			{
				// Row 0 of the raw texture buffer is the bottom scanline; flip v to match.
				float v = 1f - ( ( row + 0.5f ) / Height );
				int rowOffset = row * Width;
				for ( int x = 0; x < Width; x++ )
				{
					float u = ( x + 0.5f ) / Width;
					float3 dir = TopLeftDir + ( Horizontal * u ) + ( Vertical * v );
					Ray ray = new Ray( Origin, dir );
					int steps = Bvh.Intersect( ref ray );
					Pixels[ rowOffset + x ] = Shade( ray, steps );
				}
			}

			/// <summary>
			/// Traces one 16x16 pixel block as a single packet. Intersect256Rays wants the 256 rays
			/// as a 4x4 grid of 4x4 tiles, so ray i sits in tile i >> 4 - at ( tile &amp; 3, tile >> 2 )
			/// in the grid - at ( i &amp; 3, ( i &amp; 15 ) >> 2 ) inside that tile. That places ray 0 at
			/// pixel ( 0, 0 ), ray 51 at ( 15, 0 ), ray 204 at ( 0, 15 ) and ray 255 at ( 15, 15 ) of
			/// the block, which is the corner rule the traversal builds its four frustum planes from.
			/// The block coordinates run from the top left of the image, the way the u/v of the
			/// scanline path do; the texture row is flipped back when the pixel is written.
			/// Rays that fall outside a partial block are clamped to the last valid pixel - the
			/// packet has to be full and the corner rays have to stay the extremes - and not written.
			/// The packet traversal returns no step count, so TraversalSteps shades a constant zero.
			/// </summary>
			private void ExecuteBlock( int block )
			{
				int blockX = ( block % BlocksX ) * PacketBlockSide;
				int blockY = ( block / BlocksX ) * PacketBlockSide;
				Ray* packet = stackalloc Ray[ Bvh.PacketSize ];
				for ( int i = 0; i < Bvh.PacketSize; i++ )
				{
					int tile = i >> 4;
					int x = blockX + ( ( tile & 3 ) * 4 ) + ( i & 3 );
					int y = blockY + ( ( tile >> 2 ) * 4 ) + ( ( i & 15 ) >> 2 );
					float u = ( math.min( x, Width - 1 ) + 0.5f ) / Width;
					float v = ( math.min( y, Height - 1 ) + 0.5f ) / Height;
					packet[ i ] = new Ray( Origin, TopLeftDir + ( Horizontal * u ) + ( Vertical * v ) );
				}
				Bvh.Intersect256Rays( packet );
				for ( int i = 0; i < Bvh.PacketSize; i++ )
				{
					int tile = i >> 4;
					int x = blockX + ( ( tile & 3 ) * 4 ) + ( i & 3 );
					int y = blockY + ( ( tile >> 2 ) * 4 ) + ( ( i & 15 ) >> 2 );
					if ( x >= Width || y >= Height )
					{
						continue;
					}
					Pixels[ ( ( Height - 1 - y ) * Width ) + x ] = Shade( packet[ i ], 0 );
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
					if ( Bvh.IsOccluded( shadowRay ) )
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
