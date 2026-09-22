using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Fills the bounds of one custom primitive. Port of the customGetAABB callback of
	/// BVH::Build( void (*customGetAABB)( .. ), primCount ). The bounds are written through
	/// pointers because Burst function pointers only take blittable arguments.
	/// </summary>
	[UnmanagedFunctionPointer( CallingConvention.Cdecl )]
	public unsafe delegate void GetAabbDelegate( uint prim, float3* aabbMin, float3* aabbMax );

	/// <summary>
	/// Port of tinybvh's customIntersect callback. Intersects one custom primitive and updates
	/// ray->Hit when the hit is closer than ray->Hit.T; returns 1 when it registered a hit.
	/// Returns a byte rather than a bool: bool is not in the list of types Burst accepts as a
	/// function pointer return type.
	/// </summary>
	[UnmanagedFunctionPointer( CallingConvention.Cdecl )]
	public unsafe delegate byte CustomIntersectDelegate( Ray* ray, uint prim );

	/// <summary>
	/// Port of tinybvh's customIsOccluded callback. Returns 1 when the primitive blocks the ray
	/// within ray->Hit.T. The ray is passed by pointer for symmetry with CustomIntersectDelegate,
	/// but, as in the C++ (which takes a const Ray&amp;), it must not be modified.
	/// </summary>
	[UnmanagedFunctionPointer( CallingConvention.Cdecl )]
	public unsafe delegate byte CustomOccludedDelegate( Ray* ray, uint prim );

	/// <summary>
	/// BVHs over custom geometry: the tree is built over user-supplied AABBs and traversal hands
	/// the primitive indices in a leaf to the CustomIntersect / CustomIsOccluded callbacks.
	/// Port of BVH::BuildAABB and BVH::Build( customGetAABB, primCount ); both fill the fragment
	/// array and then run the binned SAH reference builder, so UseSpatialSplits is ignored here,
	/// exactly as in the C++. Such a tree is marked refittable, as in the C++, but Refit cannot
	/// actually be used on it: it reads the vertex buffer for every leaf, and there is none.
	/// </summary>
	public unsafe partial struct Bvh
	{
		/// <summary>
		/// Builds over a list of AABBs: two float4 per primitive, min then max; only xyz is used.
		/// The AABBs are not referenced after the build.
		/// </summary>
		public void BuildAabbs( float4* aabbs, uint primCount )
		{
			ValidateCustomBuildInput( primCount );
			if ( aabbs == null )
			{
				throw new ArgumentException( "Bvh.BuildAabbs( .. ), aabbs == null.", nameof( aabbs ) );
			}
			BvhCustomBuilder.PrepareAabbBuild( ref this, aabbs, primCount );
			RunBinnedBuild();
		}

		public void BuildAabbs( NativeArray<float4> aabbs, uint primCount )
		{
			if ( aabbs.Length < primCount * 2 )
			{
				throw new ArgumentException( "Bvh.BuildAabbs( .. ), aabbs holds fewer than two float4 per primitive.", nameof( aabbs ) );
			}
			BuildAabbs( ( float4* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( aabbs ), primCount );
		}

		/// <summary>
		/// Builds over custom geometry: the bounds of every primitive are obtained from a
		/// Burst-compiled callback (BurstCompiler.CompileFunctionPointer).
		/// </summary>
		public void Build( FunctionPointer<GetAabbDelegate> getAabb, uint primCount )
		{
			ValidateCustomBuildInput( primCount );
			if ( !getAabb.IsCreated )
			{
				throw new ArgumentException( "Bvh.Build( .. ), getAabb was not compiled.", nameof( getAabb ) );
			}
			BvhCustomBuilder.PrepareCustomBuild( ref this, getAabb, primCount );
			RunBinnedBuild();
		}

		/// <summary>Managed-side replacement for the BVH_FATAL_ERROR checks of the custom builders.</summary>
		private void ValidateCustomBuildInput( uint primCount )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "Bvh.BuildAabbs( .. ), bvh was not created." );
			}
			if ( primCount == 0 )
			{
				throw new ArgumentException( "Bvh.BuildAabbs( .. ), primCount == 0.", nameof( primCount ) );
			}
		}
	}

	/// <summary>
	/// Burst-compiled fragment setup for the custom-geometry builders. Kept apart from BvhBuilder
	/// so the shared binned builder stays untouched; both prepare steps hand over to
	/// BvhBuilder.Build, which derives BvhOverAabbs from Verts == null.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhCustomBuilder
	{
		/// <summary>Port of the allocation and fragment loop of BVH::BuildAABB.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void PrepareAabbBuild( ref Bvh bvh, float4* aabbs, uint primCount )
		{
			BvhNode* root = PrepareCustomRoot( ref bvh, primCount );
			Fragment* fragment = bvh.Fragments;
			uint* primIdx = bvh.PrimIdx;
			for ( uint i = 0; i < primCount; i++ )
			{
				fragment[ i ].BMin = aabbs[ i * 2 ].xyz;
				fragment[ i ].BMax = aabbs[ ( i * 2 ) + 1 ].xyz;
				fragment[ i ].PrimIdx = i;
				fragment[ i ].Clipped = 0;
				primIdx[ i ] = i;
				root->AabbMin = math.min( root->AabbMin, fragment[ i ].BMin );
				root->AabbMax = math.max( root->AabbMax, fragment[ i ].BMax );
			}
		}

		/// <summary>Port of the allocation and fragment loop of BVH::Build( customGetAABB, primCount ).</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void PrepareCustomBuild( ref Bvh bvh, FunctionPointer<GetAabbDelegate> getAabb, uint primCount )
		{
			BvhNode* root = PrepareCustomRoot( ref bvh, primCount );
			Fragment* fragment = bvh.Fragments;
			uint* primIdx = bvh.PrimIdx;
			for ( uint i = 0; i < primCount; i++ )
			{
				float3 bmin, bmax;
				getAabb.Invoke( i, &bmin, &bmax );
				fragment[ i ].BMin = bmin;
				fragment[ i ].BMax = bmax;
				fragment[ i ].PrimIdx = i;
				fragment[ i ].Clipped = 0;
				primIdx[ i ] = i;
				root->AabbMin = math.min( root->AabbMin, fragment[ i ].BMin );
				root->AabbMax = math.max( root->AabbMax, fragment[ i ].BMax );
			}
		}

		/// <summary>Shared allocation and root setup of both custom builders.</summary>
		private static BvhNode* PrepareCustomRoot( ref Bvh bvh, uint primCount )
		{
			bvh.TriCount = primCount;
			bvh.IdxCount = primCount;
			uint spaceNeeded = primCount * 2; // upper limit
			bvh.AllocateNodes( spaceNeeded );
			bvh.AllocatePrimIdx( primCount );
			bvh.AllocateFragments( primCount );
			bvh.Nodes[ 1 ] = default; // node 1 remains unused, for cache line alignment.
			// there is no vertex data: leaf primitives are handed to the custom callbacks. The C++
			// leaves verts untouched here and relies on it being null to derive bvh_over_aabbs;
			// clear it so rebuilding over a used Bvh stays correct, as PrepareTlasBuild does.
			bvh.Verts = null;
			bvh.VertCount = 0;
			bvh.VertIdx = null;
			bvh.BvhOverIndices = false;
			BvhNode* root = bvh.Nodes;
			root->LeftFirst = 0;
			root->TriCount = primCount;
			root->AabbMin = new float3( BvhConstants.Far );
			root->AabbMax = new float3( -BvhConstants.Far );
			// start build
			bvh.UsedNodes = 2;
			return root;
		}
	}
}
