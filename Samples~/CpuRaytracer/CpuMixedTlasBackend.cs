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
	/// CPU backend that traces a TLAS whose BLASses are in four different layouts. The same mesh is
	/// built four times - as a Bvh, a Bvh4Cpu, a Bvh8Cpu and a BvhSoa - and instance i enters BLAS
	/// i % 4, so one traversal walks all four leaf layouts through the tagged BlasRef dispatch of
	/// Bvh.BuildTlas( BlasInstance*, uint, BlasRef*, uint ). With a single instance only the Bvh
	/// BLAS is reached.
	/// </summary>
	public sealed unsafe class CpuMixedTlasBackend : IInstancedBackend
	{
		/// <summary>Number of BLAS layouts in the mix; instance i uses BLAS i % BlasCount.</summary>
		private const int BlasCount = 4;

		public string Name => "CPU TLAS mixed BLAS (Burst)";
		public bool IsGpu => false;
		public BvhBuilder Builder { get; set; }
		public bool Presplit { get; set; }
		public bool ThreadedBuild { get; set; }
		public bool Optimize { get; set; }
		public NativeArray<uint> OpacityMap { get; set; }
		public uint OpacityMapN { get; set; }
		/// <summary>One of the four layouts in the mix, BvhSoa, has no opacity micromap API.</summary>
		public bool SupportsOpacityMap => false;
		public float BuildMs { get; private set; }
		public float UpdateMs { get; private set; }

		public long NodeCount
		{
			get
			{
				long nodes = tlas.IsCreated ? tlas.UsedNodes : 0;
				nodes += blas.IsCreated ? blas.UsedNodes : 0;
				nodes += blas4.IsCreated ? blas4.Mbvh4.UsedNodes : 0;
				nodes += blas8.IsCreated ? blas8.Mbvh8.UsedNodes : 0;
				nodes += blasSoa.IsCreated ? blasSoa.UsedNodes : 0;
				return nodes;
			}
		}

		private Bvh tlas;
		private Bvh blas;
		private Bvh4Cpu blas4;
		private Bvh8Cpu blas8;
		private BvhSoa blasSoa;
		private NativeArray<float4> vertices;

		// The BlasRef list points at the BLAS structures themselves, so those need addresses that
		// outlive the build. Each one therefore lives in a one-element NativeArray, refreshed from
		// its field after every build exactly the way CpuTlasBackend refreshes its blasArray: the
		// layouts are plain structs of pointers, so a copy taken after the build traverses the same
		// tree. Nothing writes to a BLAS between the build and the traversal.
		private NativeArray<Bvh> blasStorage;
		private NativeArray<Bvh4Cpu> blas4Storage;
		private NativeArray<Bvh8Cpu> blas8Storage;
		private NativeArray<BvhSoa> blasSoaStorage;
		private NativeArray<BlasRef> blasRefs;

		/// <summary>Single identity instance, used when the sample builds this backend without instancing.</summary>
		private NativeArray<BlasInstance> singleInstance;
		/// <summary>
		/// This backend's own copy of the instance list: the shared one the sample hands out has
		/// every instance on BLAS 0, because the other TLAS backends only have one BLAS.
		/// </summary>
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
			EnsureStructures();
			Stopwatch stopwatch = Stopwatch.StartNew();
			blas.UseThreadedBuild = ThreadedBuild;
			BvhBuilderOptions.Build( ref blas, Builder, Presplit, vertices, triCount );
			BvhBuilderOptions.Build( ref blas4, Builder, Presplit, ThreadedBuild, vertices, triCount );
			BvhBuilderOptions.Build( ref blas8, Builder, Presplit, ThreadedBuild, vertices, triCount );
			BvhBuilderOptions.Build( ref blasSoa, Builder, Presplit, ThreadedBuild, vertices, triCount );
			if ( Optimize )
			{
				blas.Optimize();
				blas4.Optimize();
				blas8.Optimize();
				blasSoa.Optimize();
			}
			RefreshBlasList();
			CopyInstances( instances );
			BuildTlas();
			stopwatch.Stop();
			BuildMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
			UpdateMs = 0f;
		}

		public void UpdateInstances( NativeArray<BlasInstance> instances )
		{
			Stopwatch stopwatch = Stopwatch.StartNew();
			CopyInstances( instances );
			BuildTlas();
			stopwatch.Stop();
			UpdateMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
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
			if ( blasSoa.IsCreated )
			{
				blasSoa.Dispose();
			}
			if ( blas8.IsCreated )
			{
				blas8.Dispose();
			}
			if ( blas4.IsCreated )
			{
				blas4.Dispose();
			}
			if ( blas.IsCreated )
			{
				blas.Dispose();
			}
			if ( blasRefs.IsCreated )
			{
				blasRefs.Dispose();
			}
			if ( blasSoaStorage.IsCreated )
			{
				blasSoaStorage.Dispose();
			}
			if ( blas8Storage.IsCreated )
			{
				blas8Storage.Dispose();
			}
			if ( blas4Storage.IsCreated )
			{
				blas4Storage.Dispose();
			}
			if ( blasStorage.IsCreated )
			{
				blasStorage.Dispose();
			}
			if ( instances.IsCreated )
			{
				instances.Dispose();
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

		/// <summary>Creates the TLAS, the four BLASses and the storage the BlasRef list points into.</summary>
		private void EnsureStructures()
		{
			if ( !tlas.IsCreated )
			{
				tlas = Bvh.Create( Allocator.Persistent );
			}
			if ( !blas.IsCreated )
			{
				blas = Bvh.Create( Allocator.Persistent );
			}
			if ( !blas4.IsCreated )
			{
				blas4 = Bvh4Cpu.Create( Allocator.Persistent );
			}
			if ( !blas8.IsCreated )
			{
				blas8 = Bvh8Cpu.Create( Allocator.Persistent );
			}
			if ( !blasSoa.IsCreated )
			{
				blasSoa = BvhSoa.Create( Allocator.Persistent );
			}
			if ( !blasStorage.IsCreated )
			{
				blasStorage = new NativeArray<Bvh>( 1, Allocator.Persistent );
				blas4Storage = new NativeArray<Bvh4Cpu>( 1, Allocator.Persistent );
				blas8Storage = new NativeArray<Bvh8Cpu>( 1, Allocator.Persistent );
				blasSoaStorage = new NativeArray<BvhSoa>( 1, Allocator.Persistent );
				blasRefs = new NativeArray<BlasRef>( BlasCount, Allocator.Persistent );
			}
		}

		/// <summary>Republishes the four freshly built BLASses and the tagged pointers into them.</summary>
		private void RefreshBlasList()
		{
			blasStorage[ 0 ] = blas;
			blas4Storage[ 0 ] = blas4;
			blas8Storage[ 0 ] = blas8;
			blasSoaStorage[ 0 ] = blasSoa;
			blasRefs[ 0 ] = BlasRef.From( ( Bvh* )blasStorage.GetUnsafePtr() );
			blasRefs[ 1 ] = BlasRef.From( ( Bvh4Cpu* )blas4Storage.GetUnsafePtr() );
			blasRefs[ 2 ] = BlasRef.From( ( Bvh8Cpu* )blas8Storage.GetUnsafePtr() );
			blasRefs[ 3 ] = BlasRef.From( ( BvhSoa* )blasSoaStorage.GetUnsafePtr() );
		}

		/// <summary>Copies the sample's instance list, spreading the instances over the four BLASses.</summary>
		private void CopyInstances( NativeArray<BlasInstance> source )
		{
			if ( instances.IsCreated && instances.Length != source.Length )
			{
				instances.Dispose();
			}
			if ( !instances.IsCreated )
			{
				instances = new NativeArray<BlasInstance>( source.Length, Allocator.Persistent );
			}
			for ( int i = 0; i < source.Length; i++ )
			{
				BlasInstance instance = source[ i ];
				instance.BlasIdx = ( uint )( i % BlasCount );
				instances[ i ] = instance;
			}
		}

		private void BuildTlas()
		{
			tlas.BuildTlas( ( BlasInstance* )instances.GetUnsafePtr(), ( uint )instances.Length,
				( BlasRef* )blasRefs.GetUnsafePtr(), BlasCount );
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
			/// the 3x3 part of the instance transform. Every BLAS is built over the same mesh, so
			/// the primitive index means the same thing whichever layout the ray ended up in.
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
