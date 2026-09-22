using System.Diagnostics;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace TinyBVH.Samples
{
	/// <summary>
	/// GPU backend: builds the base BVH, converts it to one of the four GPU layouts and traces
	/// it with the compute kernels in Resources/TinyBvhTrace.compute.
	/// </summary>
	public sealed class GpuBackend : IRefittableBackend
	{
		private readonly GpuLayout layout;

		private GpuTracer tracer;
		private Bvh bvh;
		private Mbvh mbvh;
		private BvhGpu bvhGpu;
		private Bvh4Gpu bvh4Gpu;
		private BvhCwbvh cwbvh;

		private RenderTexture target;
		private int texWidth;
		private int texHeight;

		public GpuBackend( GpuLayout layout )
		{
			this.layout = layout;
		}

		public string Name
		{
			get
			{
				switch ( layout )
				{
					case GpuLayout.BvhGpu:
						return "GPU BVH_GPU (Aila-Laine)";
					case GpuLayout.Bvh4Gpu:
						return "GPU BVH4_GPU";
					case GpuLayout.Cwbvh:
						return "GPU CWBVH";
					default:
						return "GPU BVH2";
				}
			}
		}

		public bool IsGpu => true;
		public BvhBuilder Builder { get; set; }
		public bool Presplit { get; set; }
		public bool ThreadedBuild { get; set; }
		public bool Optimize { get; set; }
		public NativeArray<uint> OpacityMap { get; set; }
		public uint OpacityMapN { get; set; }
		public bool SupportsOpacityMap => layout == GpuLayout.Bvh2 || layout == GpuLayout.BvhGpu;
		public float BuildMs { get; private set; }
		public float RefitMs { get; private set; }
		public long NodeCount { get; private set; }

		public void Build( NativeArray<float4> vertices, uint triCount )
		{
			if ( tracer == null )
			{
				tracer = new GpuTracer();
			}
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
			// The map has to be on the base BVH before ConvertAndUpload: for the BvhGpu layout,
			// BvhGpu.Source is a value copy of bvh taken at conversion time, and that copy is what
			// GpuTracer uploads the map from.
			if ( OpacityMap.IsCreated && OpacityMapN > 0 && SupportsOpacityMap )
			{
				bvh.SetOpacityMicroMaps( OpacityMap, OpacityMapN );
			}
			else
			{
				bvh.ClearOpacityMicroMaps();
			}
			ConvertAndUpload( true );
			stopwatch.Stop();
			BuildMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
			NodeCount = tracer.NodeCount;
		}

		/// <summary>
		/// Refits the base BVH over the changed vertex positions and re-runs the layout conversion
		/// and the upload. For CWBVH the base was already reshaped by the build, so it is that
		/// prepared base - leaves of at most three triangles - that gets refit and re-converted.
		/// </summary>
		public void Refit()
		{
			Stopwatch stopwatch = Stopwatch.StartNew();
			bvh.Refit();
			ConvertAndUpload( false );
			stopwatch.Stop();
			RefitMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
			NodeCount = tracer.NodeCount;
		}

		public Texture Render( in RenderSettings settings, bool waitForGpu )
		{
			EnsureTarget( settings.Width, settings.Height );
			tracer.Render( target, settings.Origin, settings.TopLeftDir, settings.Horizontal, settings.Vertical,
				( int )settings.Mode, settings.Shadows, settings.SunDir, settings.SceneDiagonal );
			if ( waitForGpu )
			{
				AsyncGPUReadback.Request( target ).WaitForCompletion();
			}
			return target;
		}

		public void Dispose()
		{
			if ( tracer != null )
			{
				tracer.Dispose();
				tracer = null;
			}
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
			if ( bvh.IsCreated )
			{
				bvh.Dispose();
			}
			if ( target != null )
			{
				target.Release();
				Object.Destroy( target );
				target = null;
			}
		}

		/// <summary>
		/// Converts the base BVH to the backend's layout and uploads it. With prepareBase the leaf
		/// reshaping the CWBVH layout needs is applied first; a refit leaves it out, because the
		/// base it refitted is the reshaped one from the build.
		/// </summary>
		private void ConvertAndUpload( bool prepareBase )
		{
			switch ( layout )
			{
				case GpuLayout.BvhGpu:
					if ( !bvhGpu.IsCreated )
					{
						bvhGpu = BvhGpu.Create( Allocator.Persistent );
					}
					bvhGpu.ConvertFrom( ref bvh );
					tracer.Upload( ref bvhGpu );
					break;
				case GpuLayout.Bvh4Gpu:
					ConvertToMbvh( 4 );
					if ( !bvh4Gpu.IsCreated )
					{
						bvh4Gpu = Bvh4Gpu.Create( Allocator.Persistent );
					}
					bvh4Gpu.ConvertFrom( ref mbvh );
					tracer.Upload( ref bvh4Gpu );
					break;
				case GpuLayout.Cwbvh:
					if ( prepareBase )
					{
						// BVH8_CWBVH::Build prepares the base BVH this way: leaves of at most 3 triangles.
						bvh.Compact();
						bvh.SplitLeafs( 3 );
					}
					ConvertToMbvh( 8 );
					if ( !cwbvh.IsCreated )
					{
						cwbvh = BvhCwbvh.Create( Allocator.Persistent );
					}
					cwbvh.ConvertFrom( ref mbvh );
					tracer.Upload( ref cwbvh );
					break;
				default:
					tracer.Upload( ref bvh );
					break;
			}
		}

		/// <summary>Creates the m-wide intermediate on first use and converts the base BVH into it.</summary>
		private void ConvertToMbvh( int m )
		{
			if ( !mbvh.IsCreated )
			{
				mbvh = Mbvh.Create( m, Allocator.Persistent );
			}
			mbvh.ConvertFrom( ref bvh );
		}

		private void EnsureTarget( int width, int height )
		{
			if ( target != null && texWidth == width && texHeight == height )
			{
				return;
			}
			if ( target != null )
			{
				target.Release();
				Object.Destroy( target );
			}
			target = new RenderTexture( width, height, 0, RenderTextureFormat.ARGB32 );
			target.enableRandomWrite = true;
			target.filterMode = FilterMode.Point;
			target.Create();
			texWidth = width;
			texHeight = height;
		}
	}
}
