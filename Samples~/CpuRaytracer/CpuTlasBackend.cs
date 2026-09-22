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
	/// <summary>
	/// CPU backend that traces one BLAS placed in the scene through a TLAS, both TinyBVH.Bvh,
	/// from a Burst job. Moving the instances only rebuilds the TLAS, which is a BVH over as many
	/// boxes as there are instances rather than over the whole flattened scene.
	/// </summary>
	public sealed unsafe class CpuTlasBackend : IInstancedBackend, IRefittableBackend
	{
		public string Name => "CPU TLAS (Burst)";
		public bool IsGpu => false;
		public BvhBuilder Builder { get; set; }
		public bool Presplit { get; set; }
		public bool ThreadedBuild { get; set; }
		public bool Optimize { get; set; }
		public NativeArray<uint> OpacityMap { get; set; }
		public uint OpacityMapN { get; set; }
		public bool SupportsOpacityMap => true;
		public float BuildMs { get; private set; }
		public float UpdateMs { get; private set; }
		public float RefitMs { get; private set; }
		public long NodeCount => ( blas.IsCreated ? blas.UsedNodes : 0 ) + ( tlas.IsCreated ? tlas.UsedNodes : 0 );

		private Bvh blas;
		private Bvh tlas;
		private NativeArray<float4> vertices;
		/// <summary>One-element array holding the BLAS; Bvh.BuildTlas takes a pointer into it.</summary>
		private NativeArray<Bvh> blasArray;
		/// <summary>Single identity instance, used when the sample builds this backend without instancing.</summary>
		private NativeArray<BlasInstance> singleInstance;
		/// <summary>The instance list of the last build or update, so Refit can rebuild the TLAS over it. Not owned.</summary>
		private NativeArray<BlasInstance> instances;

		private Texture2D texture;
		private NativeArray<Color32> pixelBuffer;
		private int texWidth;
		private int texHeight;

		public void Build( NativeArray<float4> vertices, uint triCount )
		{
			if ( !singleInstance.IsCreated )
			{
				singleInstance = new NativeArray<BlasInstance>( 1, Allocator.Persistent );
			}
			singleInstance[ 0 ] = BlasInstance.Create( 0 );
			BuildInstanced( vertices, triCount, singleInstance );
		}

		public void BuildInstanced( NativeArray<float4> vertices, uint triCount, NativeArray<BlasInstance> instances )
		{
			this.vertices = vertices;
			if ( !blas.IsCreated )
			{
				blas = Bvh.Create( Allocator.Persistent );
			}
			if ( !tlas.IsCreated )
			{
				tlas = Bvh.Create( Allocator.Persistent );
			}
			if ( !blasArray.IsCreated )
			{
				blasArray = new NativeArray<Bvh>( 1, Allocator.Persistent );
			}
			Stopwatch stopwatch = Stopwatch.StartNew();
			blas.UseThreadedBuild = ThreadedBuild;
			BvhBuilderOptions.Build( ref blas, Builder, Presplit, vertices, triCount );
			if ( Optimize )
			{
				blas.Optimize();
			}
			if ( OpacityMap.IsCreated && OpacityMapN > 0 && SupportsOpacityMap )
			{
				blas.SetOpacityMicroMaps( OpacityMap, OpacityMapN );
			}
			else
			{
				blas.ClearOpacityMicroMaps();
			}
			blasArray[ 0 ] = blas;
			this.instances = instances;
			BuildTlas( instances );
			stopwatch.Stop();
			BuildMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
			UpdateMs = 0f;
		}

		public void UpdateInstances( NativeArray<BlasInstance> instances )
		{
			Stopwatch stopwatch = Stopwatch.StartNew();
			this.instances = instances;
			BuildTlas( instances );
			stopwatch.Stop();
			UpdateMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
		}

		/// <summary>
		/// Refits the BLAS over the changed vertex positions and rebuilds the TLAS, whose instance
		/// bounds moved with it.
		/// </summary>
		public void Refit()
		{
			Stopwatch stopwatch = Stopwatch.StartNew();
			blas.Refit();
			// The TLAS reads the BLAS root bounds through this copy, so refresh it after the refit.
			blasArray[ 0 ] = blas;
			BuildTlas( instances );
			stopwatch.Stop();
			RefitMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
		}

		public Texture Render( in RenderSettings settings, bool waitForGpu )
		{
			EnsureTexture( settings.Width, settings.Height );
			RenderJob job = new RenderJob
			{
				Vertices = vertices,
				Tlas = tlas,
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
			if ( tlas.IsCreated )
			{
				tlas.Dispose();
			}
			if ( blas.IsCreated )
			{
				blas.Dispose();
			}
			if ( blasArray.IsCreated )
			{
				blasArray.Dispose();
			}
			if ( singleInstance.IsCreated )
			{
				singleInstance.Dispose();
			}
			if ( texture != null )
			{
				Object.Destroy( texture );
				texture = null;
			}
		}

		private void BuildTlas( NativeArray<BlasInstance> instances )
		{
			tlas.BuildTlas( ( BlasInstance* )instances.GetUnsafePtr(), ( uint )instances.Length,
				( Bvh* )blasArray.GetUnsafePtr(), 1 );
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

		/// <summary>Traces one ray per pixel through the TLAS; one job index covers a full scanline.</summary>
		[BurstCompile]
		private struct RenderJob : IJobParallelFor
		{
			[ReadOnly] public NativeArray<float4> Vertices;
			public Bvh Tlas;
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
					int steps = Tlas.Intersect( ref ray );
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
					if ( Tlas.IsOccluded( shadowRay ) )
					{
						lit = 0.2f;
					}
				}
				float3 albedo = new float3( 0.65f, 0.65f, 0.65f );
				return ToColor32( albedo * lit );
			}

			/// <summary>
			/// Object-space triangle normal of the hit BLAS primitive, pushed to world space with
			/// the 3x3 part of the instance transform. That is correct for the rigid and uniformly
			/// scaled transforms the sample builds; a non-uniform scale would need the inverse transpose.
			/// </summary>
			private float3 ComputeNormal( in Ray ray )
			{
				int prim = ( int )ray.Hit.Prim * 3;
				float3 a = Vertices[ prim ].xyz;
				float3 b = Vertices[ prim + 1 ].xyz;
				float3 c = Vertices[ prim + 2 ].xyz;
				float3 nObj = math.cross( b - a, c - a );
				float3 n = math.normalize( Tlas.Instances[ ray.Hit.Inst ].Transform.TransformVector( nObj ) );
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
