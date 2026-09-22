using System.Collections.Generic;

namespace TinyBVH.Samples
{
	/// <summary>Registry of the raytrace backends the sample offers. Register additional backends here.</summary>
	public static class BackendList
	{
		public static List<IRaytraceBackend> Create()
		{
			List<IRaytraceBackend> list = new List<IRaytraceBackend>
			{
				new CpuBvhBackend(),
				new CpuBvh4Backend(),
				new CpuBvh8Backend(),
				new CpuTlasBackend(),
				new CpuCwbvhBackend(),
				new CpuDoubleBackend(),
				new CpuSoaBackend(),
				new CpuMixedTlasBackend(),
				new CpuVoxelBackend()
			};
			if ( GpuTracer.IsSupported )
			{
				list.Add( new GpuBackend( GpuLayout.Bvh2 ) );
				list.Add( new GpuBackend( GpuLayout.BvhGpu ) );
				list.Add( new GpuBackend( GpuLayout.Bvh4Gpu ) );
				list.Add( new GpuBackend( GpuLayout.Cwbvh ) );
				list.Add( new GpuTlasBackend( GpuBlasType.BvhGpu ) );
				list.Add( new GpuTlasBackend( GpuBlasType.Cwbvh ) );
			}
			return list;
		}
	}
}
