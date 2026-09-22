using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Port of tinybvh's BVH_Double class (DOUBLE_PRECISION_SUPPORT): a binary BVH in the
	/// traditional node layout, with 64-bit child and primitive indices and every coordinate in
	/// double precision, for scenes that a single-precision tree cannot represent.
	/// This file holds the data and memory management; construction lives in BvhDouble.Build.cs
	/// and traversal in BvhDouble.Intersect.cs. The struct is unmanaged so it can be used from Burst.
	/// Input geometry is referenced, not owned: keep the vertex buffer alive while the BVH is in use.
	/// </summary>
	public unsafe partial struct BvhDouble : IDisposable
	{
		// Input primitives (not owned): 24-byte double3 vertices, 3 per triangle, optionally
		// addressed through VertIdx (3 indices per primitive).
		[NativeDisableUnsafePtrRestriction] public double3* Verts;
		[NativeDisableUnsafePtrRestriction] public uint* VertIdx;

		// Input primitive bounds (owned).
		[NativeDisableUnsafePtrRestriction] public FragmentDouble* Fragments;

		// Node pool (owned). Root is always node 0. Unlike the single-precision builder, the
		// double builder starts handing out children at node 1: there is no unused node 1 here.
		[NativeDisableUnsafePtrRestriction] public BvhDoubleNode* Nodes;

		// Primitive index array (owned).
		[NativeDisableUnsafePtrRestriction] public ulong* PrimIdx;

		// TLAS data (not owned): set when this BVH was built over BLAS instances.
		[NativeDisableUnsafePtrRestriction] public BlasInstanceDouble* Instances;
		[NativeDisableUnsafePtrRestriction] public BvhDouble* Blasses;
		/// <summary>Number of blasses in Blasses.</summary>
		public ulong BlasCount;

		// 64-bit pool bookkeeping, as in the C++.
		public ulong UsedNodes;
		public ulong AllocatedNodes;
		/// <summary>Number of primitives in the BVH.</summary>
		public ulong TriCount;
		/// <summary>Number of primitive indices; equals TriCount for this builder.</summary>
		public ulong IdxCount;

		/// <summary>Bounds of the root node.</summary>
		public double3 AabbMin;
		public double3 AabbMax;

		// Flags maintained by the builder.
		[MarshalAs( UnmanagedType.U1 )] public bool Refittable;
		[MarshalAs( UnmanagedType.U1 )] public bool MayHaveHoles;
		[MarshalAs( UnmanagedType.U1 )] public bool BvhOverAabbs;
		[MarshalAs( UnmanagedType.U1 )] public bool BvhOverIndices;

		// SAH cost parameters.
		public float TraversalCost;
		public float IntersectionCost;

		// Custom geometry callbacks (not owned), tinybvh's customIntersect / customIsOccluded.
		// When set, the leaf loops of Intersect / IsOccluded hand every primitive index in the
		// leaf to these instead of intersecting a triangle. See BvhDouble.Intersect.cs.
		public FunctionPointer<CustomIntersectDoubleDelegate> CustomIntersect;
		public FunctionPointer<CustomOccludedDoubleDelegate> CustomIsOccluded;

		public Allocator Allocator;

		public static BvhDouble Create( Allocator allocator )
		{
			return new BvhDouble
			{
				TraversalCost = BvhConstants.DefaultTraversalCost,
				IntersectionCost = BvhConstants.DefaultIntersectionCost,
				Allocator = allocator
			};
		}

		public bool IsCreated => Allocator > Allocator.None;

		public bool IsTlas => Instances != null;

		/// <summary>Port of BVH_Double::isIndexed.</summary>
		public bool IsIndexed => VertIdx != null;

		public void Dispose()
		{
			Free( Fragments );
			Free( Nodes );
			Free( PrimIdx );
			Fragments = null;
			Nodes = null;
			PrimIdx = null;
			AllocatedNodes = 0;
			UsedNodes = 0;
			TriCount = 0;
			IdxCount = 0;
		}

		/// <summary>Port of GET_PRIM_INDICES_I0_I1_I2 for the double-precision BVH.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public void GetPrimIndices( ulong prim, out ulong i0, out ulong i1, out ulong i2 )
		{
			if ( VertIdx != null )
			{
				i0 = VertIdx[ prim * 3 ];
				i1 = VertIdx[ ( prim * 3 ) + 1 ];
				i2 = VertIdx[ ( prim * 3 ) + 2 ];
			}
			else
			{
				i0 = prim * 3;
				i1 = ( prim * 3 ) + 1;
				i2 = ( prim * 3 ) + 2;
			}
		}

		/// <summary>
		/// Port of the allocation block the three C++ builders share: the node pool sizes the
		/// decision, and all three arrays are reallocated together when it has to grow.
		/// Contents are not preserved.
		/// </summary>
		internal void AllocateArrays( ulong primCount )
		{
			ulong spaceNeeded = primCount * 2; // upper limit
			if ( AllocatedNodes < spaceNeeded )
			{
				Free( Nodes );
				Free( PrimIdx );
				Free( Fragments );
				Nodes = ( BvhDoubleNode* )Alloc( ( long )spaceNeeded * sizeof( BvhDoubleNode ) );
				AllocatedNodes = spaceNeeded;
				PrimIdx = ( ulong* )Alloc( ( long )primCount * sizeof( ulong ) );
				Fragments = ( FragmentDouble* )Alloc( ( long )primCount * sizeof( FragmentDouble ) );
			}
		}

		internal void* Alloc( long bytes )
		{
			return UnsafeUtility.Malloc( bytes, 64, Allocator );
		}

		internal void Free( void* ptr )
		{
			if ( ptr != null )
			{
				UnsafeUtility.Free( ptr, Allocator );
			}
		}
	}
}
