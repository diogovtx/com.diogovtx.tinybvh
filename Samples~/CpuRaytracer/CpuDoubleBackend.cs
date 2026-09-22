using Stopwatch = System.Diagnostics.Stopwatch;
using TinyBVH;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace TinyBVH.Samples
{
	/// <summary>
	/// CPU backend that traces the scene with a TinyBVH.BvhDouble from a Burst job. The scene is a
	/// float triangle soup, so this backend owns a widened double copy of it and rebuilds that copy
	/// on every build; the widening is part of the reported build time.
	/// </summary>
	public sealed class CpuDoubleBackend : IRaytraceBackend
	{
		public string Name => "CPU BVH2 double (Burst)";
		public bool IsGpu => false;
		/// <summary>BvhDouble has the binned builder only, so this choice never reaches it.</summary>
		public BvhBuilder Builder { get; set; }
		/// <summary>BvhDouble has no presplitting, so this choice never reaches it.</summary>
		public bool Presplit { get; set; }
		/// <summary>BvhDouble has no threaded builder, so this choice never reaches it.</summary>
		public bool ThreadedBuild { get; set; }
		/// <summary>BvhDouble has no tree-rotation optimizer, so this choice never reaches it.</summary>
		public bool Optimize { get; set; }
		public NativeArray<uint> OpacityMap { get; set; }
		public uint OpacityMapN { get; set; }
		/// <summary>The double-precision traversal carries no opacity micromap.</summary>
		public bool SupportsOpacityMap => false;
		public float BuildMs { get; private set; }
		public long NodeCount => bvhDouble.IsCreated ? ( long )bvhDouble.UsedNodes : 0;

		private BvhDouble bvhDouble;
		/// <summary>The scene's vertices widened to double, owned by this backend.</summary>
		private NativeArray<double3> vertices;

		private Texture2D texture;
		private NativeArray<Color32> pixelBuffer;
		private int texWidth;
		private int texHeight;

		public void Build( NativeArray<float4> source, uint triCount )
		{
			if ( !bvhDouble.IsCreated )
			{
				bvhDouble = BvhDouble.Create( Allocator.Persistent );
			}
			Stopwatch stopwatch = Stopwatch.StartNew();
			Widen( source, ( int )triCount * 3 );
			bvhDouble.Build( vertices, triCount );
			stopwatch.Stop();
			BuildMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
		}

		/// <summary>Keeps the owned double copy sized for the scene and fills it from the float soup.</summary>
		private void Widen( NativeArray<float4> source, int count )
		{
			if ( vertices.IsCreated && vertices.Length != count )
			{
				vertices.Dispose();
			}
			if ( !vertices.IsCreated )
			{
				vertices = new NativeArray<double3>( count, Allocator.Persistent );
			}
			WidenJob job = new WidenJob
			{
				Source = source,
				Dest = vertices
			};
			job.Schedule( count, 1024 ).Complete();
		}

		public Texture Render( in RenderSettings settings, bool waitForGpu )
		{
			EnsureTexture( settings.Width, settings.Height );
			RenderJob job = new RenderJob
			{
				Vertices = vertices,
				Bvh = bvhDouble,
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
			if ( bvhDouble.IsCreated )
			{
				bvhDouble.Dispose();
			}
			if ( vertices.IsCreated )
			{
				vertices.Dispose();
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

		/// <summary>Widens one vertex of the float soup into the double copy.</summary>
		[BurstCompile]
		private struct WidenJob : IJobParallelFor
		{
			[ReadOnly] public NativeArray<float4> Source;
			public NativeArray<double3> Dest;

			public void Execute( int i )
			{
				float4 v = Source[ i ];
				Dest[ i ] = new double3( v.x, v.y, v.z );
			}
		}

		/// <summary>
		/// Traces one ray per pixel in double precision; one job index covers a full scanline. The
		/// camera comes in as floats and is widened here; the hit is narrowed back for the shading,
		/// which is the same as the other backends'.
		/// </summary>
		[BurstCompile]
		private struct RenderJob : IJobParallelFor
		{
			[ReadOnly] public NativeArray<double3> Vertices;
			public BvhDouble Bvh;
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
				double3 origin = new double3( Origin.x, Origin.y, Origin.z );
				double3 topLeftDir = new double3( TopLeftDir.x, TopLeftDir.y, TopLeftDir.z );
				double3 horizontal = new double3( Horizontal.x, Horizontal.y, Horizontal.z );
				double3 vertical = new double3( Vertical.x, Vertical.y, Vertical.z );
				// Row 0 of the raw texture buffer is the bottom scanline; flip v to match.
				double v = 1.0 - ( ( row + 0.5 ) / Height );
				int rowOffset = row * Width;
				for ( int x = 0; x < Width; x++ )
				{
					double u = ( x + 0.5 ) / Width;
					double3 dir = topLeftDir + ( horizontal * u ) + ( vertical * v );
					RayDouble ray = new RayDouble( origin, dir );
					int steps = Bvh.Intersect( ref ray );
					Pixels[ rowOffset + x ] = Shade( ray, steps );
				}
			}

			private Color32 Shade( in RayDouble ray, int steps )
			{
				if ( ray.Hit.T >= BvhDoubleConstants.Far )
				{
					return new Color32( 20, 20, 28, 255 );
				}
				switch ( Mode )
				{
					case DisplayMode.Normals:
						return ToColor32( ( ComputeNormal( ray ) * 0.5f ) + 0.5f );
					case DisplayMode.Depth:
						float depth = math.saturate( ( float )ray.Hit.T / SceneDiagonal );
						return ToColor32( new float3( depth, depth, depth ) );
					case DisplayMode.TraversalSteps:
						return HeatMap( steps );
					case DisplayMode.Barycentrics:
						float hu = ( float )ray.Hit.U, hv = ( float )ray.Hit.V;
						return ToColor32( new float3( hu, hv, 1f - hu - hv ) );
					default:
						return ShadeLit( ray );
				}
			}

			private Color32 ShadeLit( in RayDouble ray )
			{
				float3 n = ComputeNormal( ray );
				float lit = ( math.saturate( math.dot( n, -SunDir ) ) * 0.8f ) + 0.2f;
				if ( Shadows )
				{
					double3 hitPos = ray.O + ( ray.D * ray.Hit.T );
					double3 offset = new double3( n.x, n.y, n.z ) * ( 1e-3 * SceneDiagonal );
					double3 toSun = new double3( -SunDir.x, -SunDir.y, -SunDir.z );
					RayDouble shadowRay = new RayDouble( hitPos + offset, toSun );
					if ( Bvh.IsOccluded( shadowRay ) )
					{
						lit = 0.2f;
					}
				}
				float3 albedo = new float3( 0.65f, 0.65f, 0.65f );
				return ToColor32( albedo * lit );
			}

			private float3 ComputeNormal( in RayDouble ray )
			{
				int prim = ( int )( ray.Hit.Prim * 3ul );
				double3 a = Vertices[ prim ];
				double3 b = Vertices[ prim + 1 ];
				double3 c = Vertices[ prim + 2 ];
				double3 n = math.normalize( math.cross( b - a, c - a ) );
				if ( math.dot( n, ray.D ) > 0.0 )
				{
					n = -n;
				}
				return new float3( ( float )n.x, ( float )n.y, ( float )n.z );
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
