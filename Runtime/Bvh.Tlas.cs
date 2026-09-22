using System;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;

namespace TinyBVH
{
	/// <summary>
	/// Mixed-layout TLAS support: the C++ TLAS keeps a 'BVHBase** blasList' and switches on the
	/// layout tag when it steps from a TLAS leaf into a BLAS. This file adds that list, the
	/// BuildTlas overload that takes it, and the two dispatch helpers the traversal calls.
	/// The single-layout Bvh* list stays: a TLAS built over it sets Blasses and takes the direct
	/// path, one built over a BlasRef list leaves Blasses null and takes the switch.
	/// </summary>
	public unsafe partial struct Bvh
	{
		/// <summary>
		/// Port of BVH::blasList: BLASses of mixed layouts (not owned). Null when the TLAS was
		/// built over the single-layout <see cref="Blasses"/> list instead. BlasCount counts
		/// whichever of the two is set.
		/// </summary>
		[NativeDisableUnsafePtrRestriction] public BlasRef* BlasRefs;

		/// <summary>
		/// TLAS builder over BLASses of mixed layouts, the C++
		/// BVH::Build( BLASInstance*, uint32_t, BVHBase**, uint32_t ).
		/// Deviation: the C++ updates each instance inside the fragment loop of its TLAS builder.
		/// Here the instances are updated up front and the shared builder is called with a null
		/// BLAS list - the C++ escape hatch for "the BLASInstances have been updated elsewhere" -
		/// so the single-layout build path is untouched. Update only writes to the instance it is
		/// called on, so the fragments, the root bounds and the resulting tree are unchanged.
		/// </summary>
		public void BuildTlas( BlasInstance* instances, uint instanceCount, BlasRef* blasses, uint blasCount )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "Bvh.BuildTlas( .. ), bvh was not created." );
			}
			if ( instances == null )
			{
				throw new ArgumentException( "Bvh.BuildTlas( .. ), instances == null.", nameof( instances ) );
			}
			if ( instanceCount == 0 )
			{
				throw new ArgumentException( "Bvh.BuildTlas( .. ), instanceCount == 0.", nameof( instanceCount ) );
			}
			if ( blasses != null ) // if a null pointer is passed, we'll assume the instances have been updated elsewhere.
			{
				BvhTlasBuilder.UpdateInstances( instances, instanceCount, blasses );
			}
			BvhBuilder.PrepareTlasBuild( ref this, instances, instanceCount, null, 0 );
			BlasRefs = blasses;
			BlasCount = blasCount;
			BvhBuilder.Build( ref this );
		}

		/// <summary>
		/// Port of the layout switch in BVH::IntersectTLAS. The transformed ray already carries its
		/// reciprocal direction, which every layout uses as-is; the octant predicates of the binary
		/// BVH are recomputed here, exactly as the BVH::Intersect dispatcher would.
		/// </summary>
		private static int IntersectBlasRef( in BlasRef blas, ref Ray tmpRay )
		{
			if ( blas.Layout == BlasLayout.Bvh )
			{
				// regular (triangle) BVH traversal
				Bvh* b = ( Bvh* )blas.Ptr;
				return b->IntersectBlas( ref tmpRay, tmpRay.D.x >= 0f, tmpRay.D.y >= 0f, tmpRay.D.z >= 0f );
			}
			if ( blas.Layout == BlasLayout.Bvh4Cpu )
			{
				Bvh4Cpu* b = ( Bvh4Cpu* )blas.Ptr;
				return b->Intersect( ref tmpRay );
			}
			if ( blas.Layout == BlasLayout.Bvh8Cpu )
			{
				Bvh8Cpu* b = ( Bvh8Cpu* )blas.Ptr;
				return b->Intersect( ref tmpRay );
			}
			if ( blas.Layout == BlasLayout.BvhSoa )
			{
				BvhSoa* b = ( BvhSoa* )blas.Ptr;
				return b->Intersect( ref tmpRay );
			}
			if ( blas.Layout == BlasLayout.VoxelSet )
			{
				// the DDA returns its step count, which the C++ adds to the traversal cost.
				VoxelSet* b = ( VoxelSet* )blas.Ptr;
				return b->Intersect( ref tmpRay );
			}
			// the C++ asserts on any other layout.
			return 0;
		}

		/// <summary>Port of the layout switch in BVH::IsOccludedTLAS; see <see cref="IntersectBlasRef"/>.</summary>
		private static bool IsOccludedBlasRef( in BlasRef blas, in Ray tmpRay )
		{
			if ( blas.Layout == BlasLayout.Bvh )
			{
				// regular (triangle) BVH traversal
				Bvh* b = ( Bvh* )blas.Ptr;
				return b->IsOccludedBlas( tmpRay, tmpRay.D.x >= 0f, tmpRay.D.y >= 0f, tmpRay.D.z >= 0f );
			}
			if ( blas.Layout == BlasLayout.Bvh4Cpu )
			{
				Bvh4Cpu* b = ( Bvh4Cpu* )blas.Ptr;
				return b->IsOccluded( tmpRay );
			}
			if ( blas.Layout == BlasLayout.Bvh8Cpu )
			{
				Bvh8Cpu* b = ( Bvh8Cpu* )blas.Ptr;
				return b->IsOccluded( tmpRay );
			}
			if ( blas.Layout == BlasLayout.BvhSoa )
			{
				BvhSoa* b = ( BvhSoa* )blas.Ptr;
				return b->IsOccluded( tmpRay );
			}
			if ( blas.Layout == BlasLayout.VoxelSet )
			{
				VoxelSet* b = ( VoxelSet* )blas.Ptr;
				return b->IsOccluded( tmpRay );
			}
			// the C++ asserts on any other layout.
			return false;
		}
	}

	/// <summary>
	/// Burst-compiled helper for the mixed-layout TLAS build. Direct calls must be synchronous,
	/// otherwise the instance transforms are inverted under Mono, which evaluates float math in
	/// double, and the resulting world-space bounds - and the tree built over them - drift.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhTlasBuilder
	{
		/// <summary>Port of the 'instList[i].Update( blasList[instList[i].blasIdx] )' line of the C++ TLAS builder.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void UpdateInstances( BlasInstance* instances, uint instCount, BlasRef* blasses )
		{
			for ( uint i = 0; i < instCount; i++ )
			{
				instances[ i ].Update( blasses[ instances[ i ].BlasIdx ] );
			}
		}
	}
}
