using System.Diagnostics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace TinyBVH.Samples
{
	/// <summary>
	/// GPU backend that traces a TLAS over one BLAS with the Render_Tlas kernel. The BLAS is
	/// converted to one of the two layouts the TLAS kernel can enter; moving the instances only
	/// re-uploads the TLAS nodes and the instance records, not the geometry.
	/// </summary>
	public sealed unsafe class GpuTlasBackend : IInstancedBackend, IRefittableBackend
	{
		private readonly GpuBlasType blasType;

		private GpuTracer tracer;
		private Bvh blas;
		private Bvh tlas;
		private Mbvh mbvh;
		private BvhGpu bvhGpu;
		private BvhCwbvh cwbvh;
		/// <summary>One-element array holding the BLAS; Bvh.BuildTlas takes a pointer into it.</summary>
		private NativeArray<Bvh> blasArray;
		/// <summary>Single identity instance, used when the sample builds this backend without instancing.</summary>
		private NativeArray<BlasInstance> singleInstance;
		/// <summary>The instance list of the last build or update, so Refit can rebuild the TLAS over it. Not owned.</summary>
		private NativeArray<BlasInstance> instances;

		private RenderTexture target;
		private int texWidth;
		private int texHeight;

		public GpuTlasBackend( GpuBlasType blasType )
		{
			this.blasType = blasType;
		}

		public string Name => blasType == GpuBlasType.Cwbvh ? "GPU TLAS over CWBVH" : "GPU TLAS over BVH_GPU";

		public bool IsGpu => true;
		public BvhBuilder Builder { get; set; }
		public bool Presplit { get; set; }
		public bool ThreadedBuild { get; set; }
		public bool Optimize { get; set; }
		public NativeArray<uint> OpacityMap { get; set; }
		public uint OpacityMapN { get; set; }
		public bool SupportsOpacityMap => blasType == GpuBlasType.BvhGpu;
		public float BuildMs { get; private set; }
		public float UpdateMs { get; private set; }
		public float RefitMs { get; private set; }
		public long NodeCount { get; private set; }

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
			if ( tracer == null )
			{
				tracer = new GpuTracer();
			}
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
			// The map has to be on the BLAS before ConvertBlas: BvhGpu.Source is a value copy of
			// blas taken at conversion time, and UploadTlas reads the map from that copy.
			if ( OpacityMap.IsCreated && OpacityMapN > 0 && SupportsOpacityMap )
			{
				blas.SetOpacityMicroMaps( OpacityMap, OpacityMapN );
			}
			else
			{
				blas.ClearOpacityMicroMaps();
			}
			if ( blasType == GpuBlasType.Cwbvh )
			{
				// BVH8_CWBVH::Build prepares the base BVH this way: leaves of at most 3 triangles.
				blas.Compact();
				blas.SplitLeafs( 3 );
			}
			ConvertBlas();
			// The leaf reshaping above reallocates the base BVH, so copy it afterwards.
			blasArray[ 0 ] = blas;
			this.instances = instances;
			BuildTlas( instances );
			UploadAll();
			stopwatch.Stop();
			BuildMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
			UpdateMs = 0f;
			NodeCount = tracer.NodeCount;
		}

		public void UpdateInstances( NativeArray<BlasInstance> instances )
		{
			Stopwatch stopwatch = Stopwatch.StartNew();
			this.instances = instances;
			BuildTlas( instances );
			tracer.UpdateTlas( ref tlas );
			stopwatch.Stop();
			UpdateMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
			NodeCount = tracer.NodeCount;
		}

		/// <summary>
		/// Refits the BLAS over the changed vertex positions, re-converts it and rebuilds the TLAS,
		/// whose instance bounds moved with it. The BLAS geometry has to go back to the GPU, so
		/// this is a full upload rather than the cheap top-level refresh UpdateInstances does.
		/// </summary>
		public void Refit()
		{
			Stopwatch stopwatch = Stopwatch.StartNew();
			blas.Refit();
			ConvertBlas();
			blasArray[ 0 ] = blas;
			BuildTlas( instances );
			UploadAll();
			stopwatch.Stop();
			RefitMs = ( float )stopwatch.Elapsed.TotalMilliseconds;
			NodeCount = tracer.NodeCount;
		}

		/// <summary>Converts the BLAS into the layout the TLAS kernel enters it through.</summary>
		private void ConvertBlas()
		{
			if ( blasType == GpuBlasType.Cwbvh )
			{
				if ( !mbvh.IsCreated )
				{
					mbvh = Mbvh.Create( 8, Allocator.Persistent );
				}
				mbvh.ConvertFrom( ref blas );
				if ( !cwbvh.IsCreated )
				{
					cwbvh = BvhCwbvh.Create( Allocator.Persistent );
				}
				cwbvh.ConvertFrom( ref mbvh );
			}
			else
			{
				if ( !bvhGpu.IsCreated )
				{
					bvhGpu = BvhGpu.Create( Allocator.Persistent );
				}
				bvhGpu.ConvertFrom( ref blas );
			}
		}

		/// <summary>Uploads the TLAS together with its one BLAS.</summary>
		private void UploadAll()
		{
			if ( blasType == GpuBlasType.Cwbvh )
			{
				tracer.UploadTlas( ref tlas, new BvhCwbvh[] { cwbvh } );
			}
			else
			{
				tracer.UploadTlas( ref tlas, new BvhGpu[] { bvhGpu } );
			}
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
			if ( bvhGpu.IsCreated )
			{
				bvhGpu.Dispose();
			}
			if ( mbvh.IsCreated )
			{
				mbvh.Dispose();
			}
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
			if ( target != null )
			{
				target.Release();
				Object.Destroy( target );
				target = null;
			}
		}

		private void BuildTlas( NativeArray<BlasInstance> instances )
		{
			tlas.BuildTlas( ( BlasInstance* )instances.GetUnsafePtr(), ( uint )instances.Length,
				( Bvh* )blasArray.GetUnsafePtr(), 1 );
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
