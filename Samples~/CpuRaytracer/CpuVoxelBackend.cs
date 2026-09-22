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
	/// CPU backend that shows the scene as voxels: the source mesh is voxelised into one 256^3
	/// TinyBVH.VoxelSet mapped onto the unit cube, and that voxel object is placed through a TLAS
	/// built over the tagged BlasRef list, so a ray leaves the TLAS into the three-level DDA.
	/// The triangle build options mean nothing here; the stats line says so.
	/// </summary>
	public sealed unsafe class CpuVoxelBackend : IInstancedBackend
	{
		/// <summary>Empty margin around the mesh inside the unit cube, as a fraction of its largest extent.</summary>
		private const float CubeMargin = 0.01f;
		/// <summary>Surface samples per voxel along a triangle edge; three is dense enough to leave no gaps.</summary>
		private const float SamplesPerVoxel = 3f;
		/// <summary>Cap on the samples one triangle edge is cut into; the grid diagonal is about 443 voxels.</summary>
		private const int MaxSamplesPerEdge = 1536;

		public string Name => "CPU Voxels (Burst)";
		public bool IsGpu => false;
		/// <summary>Ignored: there is no triangle BVH below the TLAS, only the voxel DDA.</summary>
		public BvhBuilder Builder { get; set; }
		/// <summary>Ignored; see <see cref="Builder"/>.</summary>
		public bool Presplit { get; set; }
		/// <summary>Ignored; see <see cref="Builder"/>.</summary>
		public bool ThreadedBuild { get; set; }
		/// <summary>Ignored; see <see cref="Builder"/>.</summary>
		public bool Optimize { get; set; }
		public NativeArray<uint> OpacityMap { get; set; }
		public uint OpacityMapN { get; set; }
		/// <summary>Voxels carry no opacity micromap.</summary>
		public bool SupportsOpacityMap => false;
		public float BuildMs { get; private set; }
		public float UpdateMs { get; private set; }

		/// <summary>TLAS nodes plus the bricks the voxelisation handed out (brick 0 is never used).</summary>
		public long NodeCount
		{
			get
			{
				long nodes = tlas.IsCreated ? tlas.UsedNodes : 0;
				if ( voxelStorage.IsCreated )
				{
					VoxelSet* voxels = ( VoxelSet* )voxelStorage.GetUnsafePtr();
					nodes += voxels->FreeBrickPtr > 0 ? voxels->FreeBrickPtr - 1 : 0;
				}
				return nodes;
			}
		}

		private Bvh tlas;
		/// <summary>The voxel object. It lives in the array, not in a field: the BlasRef points at it.</summary>
		private NativeArray<VoxelSet> voxelStorage;
		private NativeArray<BlasRef> blasRefs;
		/// <summary>The unit-cube mapping the voxelisation derived: [ 0 ] = ( min, scale ), [ 1 ] = ( offset, 0 ).</summary>
		private NativeArray<float4> mapping;
		/// <summary>Maps the unit cube back onto the source mesh; composed into every instance transform.</summary>
		private float4x4 cubeToMesh = float4x4.identity;
		/// <summary>Edge of the cube in world units; see <see cref="RenderJob.RayScale"/>.</summary>
		private float cubeScale = 1f;

		/// <summary>Single identity instance, used when the sample builds this backend without instancing.</summary>
		private NativeArray<BlasInstance> singleInstance;
		/// <summary>This backend's own instance list: the sample's transforms times <see cref="cubeToMesh"/>.</summary>
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
			EnsureStructures();
			Stopwatch stopwatch = Stopwatch.StartNew();
			Voxelise( vertices, triCount );
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
				Tlas = tlas,
				Voxels = voxelStorage.IsCreated ? *( VoxelSet* )voxelStorage.GetUnsafePtr() : default,
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
				SceneDiagonal = settings.SceneDiagonal,
				RayScale = cubeScale,
				VoxelSize = cubeScale / VoxelSet.ObjectDim
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
			if ( voxelStorage.IsCreated )
			{
				VoxelSet* voxels = ( VoxelSet* )voxelStorage.GetUnsafePtr();
				if ( voxels->IsCreated )
				{
					voxels->Dispose();
				}
				voxelStorage.Dispose();
			}
			if ( blasRefs.IsCreated )
			{
				blasRefs.Dispose();
			}
			if ( mapping.IsCreated )
			{
				mapping.Dispose();
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

		private void EnsureStructures()
		{
			if ( !tlas.IsCreated )
			{
				tlas = Bvh.Create( Allocator.Persistent );
			}
			if ( !voxelStorage.IsCreated )
			{
				voxelStorage = new NativeArray<VoxelSet>( 1, Allocator.Persistent );
				blasRefs = new NativeArray<BlasRef>( 1, Allocator.Persistent );
				mapping = new NativeArray<float4>( 2, Allocator.Persistent );
				blasRefs[ 0 ] = BlasRef.From( ( VoxelSet* )voxelStorage.GetUnsafePtr() );
			}
		}

		/// <summary>
		/// Fills a fresh voxel object from the triangle soup. VoxelSet has no clear, and a brick
		/// handed out again would still hold the voxels of the previous build, so the object is
		/// disposed and recreated rather than reused. VoxelSet.Set is not thread safe, so the whole
		/// fill runs in one Burst job; the job also derives the unit-cube mapping from the bounds it
		/// walks, which is read back here for the instance transforms.
		/// </summary>
		private void Voxelise( NativeArray<float4> vertices, uint triCount )
		{
			VoxelSet* voxels = ( VoxelSet* )voxelStorage.GetUnsafePtr();
			if ( voxels->IsCreated )
			{
				voxels->Dispose();
			}
			*voxels = VoxelSet.Create( Allocator.Persistent );
			VoxeliseJob job = new VoxeliseJob
			{
				Vertices = vertices,
				Voxels = voxels,
				Mapping = mapping,
				TriCount = triCount,
				Margin = CubeMargin
			};
			job.Schedule().Complete();
			float3 min = mapping[ 0 ].xyz;
			float scale = mapping[ 0 ].w;
			float3 offset = mapping[ 1 ].xyz;
			// Inverse of the mapping the job applied, u = ( p - min ) / scale + offset: a uniform
			// scale, so pushing a voxel normal back with the 3x3 part of the transform stays correct.
			cubeToMesh = float4x4.TRS( min - ( offset * scale ), quaternion.identity, new float3( scale ) );
			cubeScale = scale;
		}

		/// <summary>Copies the sample's instance list, composing the unit-cube mapping into every transform.</summary>
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
				instance.BlasIdx = 0;
				instance.Transform = BvhMat4.FromFloat4x4( math.mul( instance.Transform.ToFloat4x4(), cubeToMesh ) );
				instances[ i ] = instance;
			}
		}

		private void BuildTlas()
		{
			tlas.BuildTlas( ( BlasInstance* )instances.GetUnsafePtr(), ( uint )instances.Length,
				( BlasRef* )blasRefs.GetUnsafePtr(), 1 );
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
		/// Voxelises the triangle soup into the unit cube. Every triangle is sampled on a
		/// barycentric grid whose spacing is a third of a voxel along both of its edges, so no
		/// voxel the surface passes through is skipped.
		/// </summary>
		[BurstCompile]
		private struct VoxeliseJob : IJob
		{
			[ReadOnly] public NativeArray<float4> Vertices;
			[NativeDisableUnsafePtrRestriction] public VoxelSet* Voxels;
			[WriteOnly] public NativeArray<float4> Mapping;
			public uint TriCount;
			public float Margin;

			public void Execute()
			{
				float3 min = new float3( BvhConstants.Far );
				float3 max = new float3( -BvhConstants.Far );
				int vertexCount = ( int )TriCount * 3;
				for ( int i = 0; i < vertexCount; i++ )
				{
					float3 p = Vertices[ i ].xyz;
					min = math.min( min, p );
					max = math.max( max, p );
				}
				float3 extent = max - min;
				// One scale for all three axes keeps the placement transform a uniform scale.
				float scale = math.max( math.cmax( extent ) * ( 1f + ( 2f * Margin ) ), 1e-6f );
				float3 offset = ( 1f - ( extent / scale ) ) * 0.5f;
				Mapping[ 0 ] = new float4( min, scale );
				Mapping[ 1 ] = new float4( offset, 0f );
				for ( uint t = 0; t < TriCount; t++ )
				{
					int v = ( int )t * 3;
					float3 a = ( ( Vertices[ v ].xyz - min ) / scale ) + offset;
					float3 b = ( ( Vertices[ v + 1 ].xyz - min ) / scale ) + offset;
					float3 c = ( ( Vertices[ v + 2 ].xyz - min ) / scale ) + offset;
					// Knuth's multiplicative hash of the triangle index, mapped to 1..255, so
					// neighbouring triangles land on different palette entries and zero - which
					// means "empty voxel" - is never written.
					uint value = 1u + ( ( ( t * 2654435761u ) >> 24 ) & 254u );
					FillTriangle( a, b, c, value );
				}
				Voxels->UpdateTopGrid();
			}

			private void FillTriangle( float3 a, float3 b, float3 c, uint value )
			{
				const float dim = VoxelSet.ObjectDim;
				float3 ab = b - a;
				float3 ac = c - a;
				int steps1 = math.clamp( ( int )math.ceil( math.length( ab ) * dim * SamplesPerVoxel ), 1, MaxSamplesPerEdge );
				int steps2 = math.clamp( ( int )math.ceil( math.length( ac ) * dim * SamplesPerVoxel ), 1, MaxSamplesPerEdge );
				for ( int i = 0; i <= steps1; i++ )
				{
					float s = ( float )i / steps1;
					for ( int j = 0; j <= steps2; j++ )
					{
						float u = ( float )j / steps2;
						if ( ( s + u ) > 1f )
						{
							break;
						}
						float3 p = a + ( ab * s ) + ( ac * u );
						uint x = ( uint )math.clamp( ( int )( p.x * dim ), 0, VoxelSet.ObjectDim - 1 );
						uint y = ( uint )math.clamp( ( int )( p.y * dim ), 0, VoxelSet.ObjectDim - 1 );
						uint z = ( uint )math.clamp( ( int )( p.z * dim ), 0, VoxelSet.ObjectDim - 1 );
						Voxels->Set( x, y, z, value );
					}
				}
			}
		}

		/// <summary>
		/// Traces one ray per pixel through the TLAS; one job index covers a full scanline.
		///
		/// Every ray leaves here with a direction of length <see cref="RayScale"/> rather than a unit
		/// one. The TLAS hands a BLAS the direction pushed through the instance's inverse transform
		/// without renormalising it, so a unit world direction would reach the voxel DDA 1/RayScale
		/// long, and the fixed entry-point epsilon of that DDA would stop working - see the comment on
		/// VoxelSet.Setup3DDDA. Scaling the direction up by the cube's edge length makes the
		/// object-space ray unit length, which is what the DDA assumes. Hit.T is then measured in
		/// units of RayScale, so 'origin + direction * Hit.T' is still the hit point, but anything
		/// that wants a world distance out of it has to multiply by RayScale. One length has to do
		/// for every instance; the sample's own instance transforms only scale by 1, 0.875 or 0.75 on
		/// top of the cube, so the object-space rays stay within a third of unit length.
		/// </summary>
		[BurstCompile]
		private struct RenderJob : IJobParallelFor
		{
			public Bvh Tlas;
			/// <summary>A value copy of the voxel object, for GetNormal; the traversal goes through the TLAS.</summary>
			public VoxelSet Voxels;
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
			/// <summary>Length every ray direction is given: the world edge of the unit cube.</summary>
			public float RayScale;
			/// <summary>World size of one voxel, RayScale / 256.</summary>
			public float VoxelSize;

			public void Execute( int row )
			{
				// Row 0 of the raw texture buffer is the bottom scanline; flip v to match.
				float v = 1f - ( ( row + 0.5f ) / Height );
				int rowOffset = row * Width;
				for ( int x = 0; x < Width; x++ )
				{
					float u = ( x + 0.5f ) / Width;
					float3 dir = TopLeftDir + ( Horizontal * u ) + ( Vertical * v );
					Ray ray = MakeRay( Origin, math.normalize( dir ) * RayScale );
					int steps = Tlas.Intersect( ref ray );
					Pixels[ rowOffset + x ] = Shade( ray, steps );
				}
			}

			/// <summary>
			/// A ray with the direction left exactly as given. The Ray constructor normalises, which
			/// is the one thing this backend must not do; see the note on the job.
			/// </summary>
			private static Ray MakeRay( float3 origin, float3 direction )
			{
				Ray ray = default;
				ray.O = origin;
				ray.D = direction;
				ray.RD = BvhMath.Rcp( direction );
				ray.Mask = BvhConstants.RayMaskIntersectAll;
				ray.Hit.T = BvhConstants.Far;
				return ray;
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
						// Hit.T counts in units of RayScale, not world units; see the note on the job.
						float depth = math.saturate( ( ray.Hit.T * RayScale ) / SceneDiagonal );
						return ToColor32( new float3( depth, depth, depth ) );
					case DisplayMode.TraversalSteps:
						return HeatMap( steps );
					case DisplayMode.Barycentrics:
						// A voxel hit has no u/v, so this mode stands in for it with the palette: one
						// colour per voxel value, i.e. per source triangle.
						return ToColor32( Palette( ray.Hit.Prim ) );
					default:
						return ShadeLit( ray );
				}
			}

			/// <summary>
			/// The same grey Lambert the triangle backends use, so the two pictures are comparable.
			/// The shadow ray starts half a voxel off the surface along the voxel normal, rather than
			/// at the shared 1e-3 * diagonal epsilon: the DDA reports the distance at which the ray
			/// entered the voxel, so the hit point lies exactly on a voxel face, and half a voxel is
			/// the largest offset that is certain to leave the voxel that was hit without reaching
			/// past the empty neighbour the primary ray came through.
			/// </summary>
			private Color32 ShadeLit( in Ray ray )
			{
				float3 n = ComputeNormal( ray );
				float lit = ( math.saturate( math.dot( n, -SunDir ) ) * 0.8f ) + 0.2f;
				if ( Shadows )
				{
					float3 hitPos = ray.O + ( ray.D * ray.Hit.T );
					Ray shadowRay = MakeRay( hitPos + ( n * ( 0.5f * VoxelSize ) ), -SunDir * RayScale );
					if ( Tlas.IsOccluded( shadowRay ) )
					{
						lit = 0.2f;
					}
				}
				float3 albedo = new float3( 0.65f, 0.65f, 0.65f );
				return ToColor32( albedo * lit );
			}

			/// <summary>
			/// Face normal of the voxel that was hit. VoxelSet.GetNormal works off the object-space
			/// ray, so the world ray is pushed through the instance's inverse transform exactly the
			/// way the TLAS does before it enters a BLAS - including leaving Hit.T alone, which is
			/// what the DDA measured along the transformed direction - and the resulting normal is
			/// rotated back. The transform is a rotation and a uniform scale, so the 3x3 part is
			/// enough; GetNormal already returns the face turned towards the ray.
			/// </summary>
			private float3 ComputeNormal( in Ray ray )
			{
				BlasInstance instance = Tlas.Instances[ ray.Hit.Inst ];
				Ray objectRay = default;
				objectRay.O = instance.InvTransform.TransformPoint( ray.O );
				objectRay.D = instance.InvTransform.TransformVector( ray.D );
				objectRay.Hit = ray.Hit;
				return math.normalize( instance.Transform.TransformVector( Voxels.GetNormal( objectRay ) ) );
			}

			/// <summary>
			/// Spreads the 1..255 voxel value over a light, well separated set of colours. Only the
			/// Barycentrics mode uses it: neighbouring triangles hash to unrelated entries, which
			/// reads as noise over a shaded surface.
			/// </summary>
			private static float3 Palette( uint value )
			{
				float3 c = new float3(
					( ( value * 37u ) & 255u ) / 255f,
					( ( value * 59u ) & 255u ) / 255f,
					( ( value * 83u ) & 255u ) / 255f );
				return ( c * 0.6f ) + 0.3f;
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
