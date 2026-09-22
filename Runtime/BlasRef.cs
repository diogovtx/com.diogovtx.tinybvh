using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Port of the subset of BVHBase::BVHType a TLAS can dispatch on. The C++ stores the layout in
	/// every BVH class and casts BVHBase* accordingly when it steps from a TLAS leaf into a BLAS;
	/// the port has no common base class, so the tag travels next to the pointer in a
	/// <see cref="BlasRef"/>. The numbering follows the C++ enum for the layouts that are valid
	/// BLASses, not the full list.
	/// </summary>
	public enum BlasLayout : uint
	{
		/// <summary>LAYOUT_BVH: the binary BVH, <see cref="Bvh"/>.</summary>
		Bvh = 1,
		/// <summary>LAYOUT_BVH4_CPU: the 4-wide SSE layout, <see cref="TinyBVH.Bvh4Cpu"/>.</summary>
		Bvh4Cpu,
		/// <summary>LAYOUT_BVH8_AVX2: the 8-wide AVX2 layout, <see cref="TinyBVH.Bvh8Cpu"/>.</summary>
		Bvh8Cpu,
		/// <summary>LAYOUT_BVH_SOA: the SoA binary layout, <see cref="TinyBVH.BvhSoa"/>.</summary>
		BvhSoa,
		/// <summary>LAYOUT_VOXELSET: the 256^3 voxel object, <see cref="TinyBVH.VoxelSet"/>.</summary>
		VoxelSet
	}

	/// <summary>
	/// One entry of the C++ 'BVHBase** blasList': a BLAS of any supported layout, as a tagged
	/// pointer. The pointed-at structure is not owned and must outlive the TLAS.
	/// </summary>
	[StructLayout( LayoutKind.Sequential )]
	public unsafe struct BlasRef
	{
		public BlasLayout Layout;
		[NativeDisableUnsafePtrRestriction] public void* Ptr;

		public static BlasRef From( Bvh* blas )
		{
			return new BlasRef { Layout = BlasLayout.Bvh, Ptr = blas };
		}

		public static BlasRef From( Bvh4Cpu* blas )
		{
			return new BlasRef { Layout = BlasLayout.Bvh4Cpu, Ptr = blas };
		}

		public static BlasRef From( Bvh8Cpu* blas )
		{
			return new BlasRef { Layout = BlasLayout.Bvh8Cpu, Ptr = blas };
		}

		public static BlasRef From( BvhSoa* blas )
		{
			return new BlasRef { Layout = BlasLayout.BvhSoa, Ptr = blas };
		}

		/// <summary>
		/// A voxel set as a BLAS. Note that the C++ VoxelSet constructor never sets
		/// BVHBase::layout, so the C++ caller has to tag it by hand before a TLAS build will reach
		/// it (voxeldump.cpp does exactly that); here the tag comes with the reference.
		/// </summary>
		public static BlasRef From( VoxelSet* blas )
		{
			return new BlasRef { Layout = BlasLayout.VoxelSet, Ptr = blas };
		}

		/// <summary>
		/// The BVHBase::aabbMin and aabbMax of the referenced BLAS, which is all
		/// BLASInstance::Update reads after the C++ casts the BVHBase pointer. An unsupported
		/// layout reports an empty box.
		/// Deviation: written as an if/else-if chain over typed local pointers, with the bounds
		/// handed back through out parameters. Returning a float3 straight out of a switch case on
		/// a cast void* makes the Burst compiler itself fail with BC0102, which drops the whole
		/// assembly to the Mono fallback.
		/// </summary>
		public static void GetBounds( in BlasRef blas, out float3 aabbMin, out float3 aabbMax )
		{
			if ( blas.Layout == BlasLayout.Bvh )
			{
				Bvh* b = ( Bvh* )blas.Ptr;
				aabbMin = b->AabbMin;
				aabbMax = b->AabbMax;
			}
			else if ( blas.Layout == BlasLayout.Bvh4Cpu )
			{
				Bvh4Cpu* b = ( Bvh4Cpu* )blas.Ptr;
				aabbMin = b->AabbMin;
				aabbMax = b->AabbMax;
			}
			else if ( blas.Layout == BlasLayout.Bvh8Cpu )
			{
				Bvh8Cpu* b = ( Bvh8Cpu* )blas.Ptr;
				aabbMin = b->AabbMin;
				aabbMax = b->AabbMax;
			}
			else if ( blas.Layout == BlasLayout.BvhSoa )
			{
				BvhSoa* b = ( BvhSoa* )blas.Ptr;
				aabbMin = b->AabbMin;
				aabbMax = b->AabbMax;
			}
			else if ( blas.Layout == BlasLayout.VoxelSet )
			{
				// a voxel object is always the unit cube in object space.
				VoxelSet* b = ( VoxelSet* )blas.Ptr;
				aabbMin = b->AabbMin;
				aabbMax = b->AabbMax;
			}
			else
			{
				aabbMin = new float3( BvhConstants.Far );
				aabbMax = new float3( -BvhConstants.Far );
			}
		}
	}
}
