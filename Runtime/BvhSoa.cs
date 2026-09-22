using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Port of BVH_SoA::BVHNode: the second alternative 64-byte BVH node layout, same as
	/// BVHAilaLaine but with the child AABBs stored in SoA order. Xxxx holds
	/// (left.min.x, left.max.x, right.min.x, right.max.x), and Yyyy and Zzzz the same for the
	/// other two axes.
	/// </summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct BvhSoaNode
	{
		public float4 Xxxx;
		public float4 Yyyy;
		public float4 Zzzz;
		public uint Left;
		public uint Right;
		public uint TriCount;
		public uint FirstTri; // total: 64 bytes

		public bool IsLeaf => TriCount > 0;
	}

	/// <summary>
	/// Port of tinybvh's BVH_SoA class: a binary BVH whose two child slabs are interleaved over
	/// four SIMD lanes, so one 128-bit slab test covers both children. The C++ only compiles the
	/// traversal under BVH_USEAVX; it lives in BvhSoa.Intersect.cs here.
	/// Like BVH_SoA, this layout keeps the base BVH it was converted from and reads its vertices
	/// and primitive indices during traversal: keep that BVH - and the geometry - alive.
	/// </summary>
	public unsafe partial struct BvhSoa : IDisposable
	{
		/// <summary>
		/// Port of BVH_SoA::bvh: "BVH_SoA is created from BVH and uses its data". Owned when this
		/// layout was built through <see cref="Build"/>, otherwise a shallow copy of the BVH passed
		/// to <see cref="ConvertFrom"/> that shares - and does not own - its memory.
		/// </summary>
		public Bvh Source;

		/// <summary>Port of BVH_SoA::bvhNode: BVH node in 'structure of arrays' format (owned).</summary>
		[NativeDisableUnsafePtrRestriction] public BvhSoaNode* Nodes;
		public uint AllocatedNodes;
		public uint UsedNodes;

		// Properties copied from the source by CopyBasePropertiesFrom.
		public uint TriCount;
		public uint IdxCount;
		public float3 AabbMin;
		public float3 AabbMax;
		[MarshalAs( UnmanagedType.U1 )] public bool Refittable;
		[MarshalAs( UnmanagedType.U1 )] public bool MayHaveHoles;
		[MarshalAs( UnmanagedType.U1 )] public bool BvhOverAabbs;
		[MarshalAs( UnmanagedType.U1 )] public bool BvhOverIndices;

		/// <summary>Port of BVH_SoA::ownBVH: set by <see cref="Build"/>, cleared by <see cref="ConvertFrom"/>.</summary>
		[MarshalAs( UnmanagedType.U1 )] public bool OwnsSource;

		/// <summary>Forwarded to Source before the base build; the C++ BVH_SoA::BuildHQ equivalent.</summary>
		[MarshalAs( UnmanagedType.U1 )] public bool UseSpatialSplits;

		/// <summary>Forwarded to Source before the base build; builds with the job system for large inputs.</summary>
		[MarshalAs( UnmanagedType.U1 )] public bool UseThreadedBuild;

		// SAH cost parameters. Unlike BVH4_CPU and BVH8_CPU, the C++ BVH_SoA constructor leaves
		// both at their defaults.
		public float TraversalCost;
		public float IntersectionCost;

		public Allocator Allocator;

		public static BvhSoa Create( Allocator allocator )
		{
			return new BvhSoa
			{
				TraversalCost = BvhConstants.DefaultTraversalCost,
				IntersectionCost = BvhConstants.DefaultIntersectionCost,
				Allocator = allocator
			};
		}

		public bool IsCreated => Allocator > Allocator.None;

		public void Dispose()
		{
			Free( Nodes );
			Nodes = null;
			AllocatedNodes = 0;
			UsedNodes = 0;
			TriCount = 0;
			IdxCount = 0;
			if ( OwnsSource )
			{
				Source.Dispose();
				OwnsSource = false;
			}
		}

		/// <summary>
		/// Port of BVH_SoA::Build: propagates the SAH costs of this layout into a base BVH of its
		/// own, builds it and converts. The base BVH is owned and is freed by <see cref="Dispose"/>.
		/// </summary>
		public void Build( NativeArray<float4> vertices, uint triCount )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "BvhSoa.Build( .. ), bvhSoa was not created." );
			}
			if ( !Source.IsCreated )
			{
				Source = Bvh.Create( Allocator );
			}
			// propagate settings for this layout to the underlying layout
			Source.TraversalCost = TraversalCost;
			Source.IntersectionCost = IntersectionCost;
			// build underlying layout
			Source.UseSpatialSplits = UseSpatialSplits;
			Source.UseThreadedBuild = UseThreadedBuild;
			Source.Build( vertices, triCount );
			// convert to BVH_SoA layout
			Bvh source = Source;
			ConvertFrom( ref source, false );
			// the C++ compares &original against &bvh to decide this; here Build simply claims
			// ownership of the BVH it just created, after ConvertFrom cleared the flag.
			OwnsSource = true;
		}

		/// <summary>
		/// Port of BVH_SoA::ConvertFrom( const BVH&amp;, bool ). Takes a shallow copy of the source,
		/// which shares its node, index and vertex memory, and emits the SoA nodes.
		/// </summary>
		public void ConvertFrom( ref Bvh original, bool compact = true )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "BvhSoa.ConvertFrom( .. ), bvhSoa was not created." );
			}
			if ( original.Nodes == null || original.UsedNodes == 0 )
			{
				throw new ArgumentException( "BvhSoa.ConvertFrom( .. ), source has no nodes.", nameof( original ) );
			}
			// bvh isn't ours; don't delete in Dispose. Build re-claims ownership afterwards.
			OwnsSource = false;
			// get a copy of the original bvh
			Source = original;
			// The C++ passes bvh to its own ConvertFrom, which assigns it back to itself; a local
			// copy of the same value keeps that behaviour without aliasing 'ref this' in Burst.
			Bvh source = Source;
			BvhSoaConverter.ConvertFrom( ref this, ref source, compact );
		}

		/// <summary>
		/// Port of BVH_SoA::Optimize: optimizes the underlying BVH and re-converts this layout
		/// from it. Unavailable on a layout that has no base BVH.
		/// </summary>
		public void Optimize( uint iterations = 25, bool extreme = false )
		{
			if ( Source.Nodes == null )
			{
				throw new InvalidOperationException( "BvhSoa.Optimize( .. ), this layout has no source BVH." );
			}
			// ConvertFrom clears the flag; Optimize does not change who owns the source BVH.
			bool owns = OwnsSource;
			Source.Optimize( iterations, extreme );
			Bvh source = Source;
			ConvertFrom( ref source, false );
			OwnsSource = owns;
		}

		/// <summary>
		/// Port of BVH_SoA::SAHCost, which forwards to the underlying BVH.
		/// Unavailable on a layout that has no base BVH.
		/// </summary>
		public float SahCost( uint nodeIdx = 0 )
		{
			if ( Source.Nodes == null )
			{
				throw new InvalidOperationException( "BvhSoa.SahCost( .. ), this layout has no source BVH." );
			}
			return Source.SahCost( nodeIdx );
		}

		/// <summary>Ensures the node pool can hold count nodes. Contents are not preserved when it grows.</summary>
		internal void AllocateNodes( uint count )
		{
			if ( AllocatedNodes < count )
			{
				Free( Nodes );
				Nodes = ( BvhSoaNode* )Alloc( ( long )count * sizeof( BvhSoaNode ) );
				AllocatedNodes = count;
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

	/// <summary>
	/// Burst-compiled implementation of the BVH_SoA conversion. Direct calls must be synchronous,
	/// otherwise editor tests silently run the Mono fallback.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhSoaConverter
	{
		/// <summary>Stack depth of the conversion walk, the C++ 'uint32_t stack[128]'.</summary>
		private const int StackSize = 128;

		/// <summary>Port of the BVH_SoA::ConvertFrom node-emitting loop.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void ConvertFrom( ref BvhSoa soa, ref Bvh bvh, [MarshalAs( UnmanagedType.U1 )] bool compact )
		{
			// allocate space
			uint spaceNeeded = compact ? bvh.UsedNodes : bvh.AllocatedNodes;
			soa.AllocateNodes( spaceNeeded );
			UnsafeUtility.MemClear( soa.Nodes, ( long )spaceNeeded * sizeof( BvhSoaNode ) );
			CopyBasePropertiesFrom( ref soa, ref bvh );
			// recursively convert nodes
			uint* stack = stackalloc uint[ StackSize ];
			BvhSoaNode* soaNode = soa.Nodes;
			uint newAlt2Node = 0, nodeIdx = 0, stackPtr = 0;
			while ( true )
			{
				BvhNode* node = bvh.Nodes + nodeIdx;
				uint idx = newAlt2Node++;
				if ( node->IsLeaf )
				{
					soaNode[ idx ].TriCount = node->TriCount;
					soaNode[ idx ].FirstTri = node->LeftFirst;
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
					uint newNodeParent = stack[ --stackPtr ];
					soaNode[ newNodeParent ].Right = newAlt2Node;
				}
				else
				{
					BvhNode* left = bvh.Nodes + node->LeftFirst;
					BvhNode* right = bvh.Nodes + node->LeftFirst + 1;
					// This BVH layout requires BVH_USEAVX/BVH_USENEON for traversal, but at least we
					// can convert to it without SSE/AVX/NEON support.
					soaNode[ idx ].Xxxx = new float4( left->AabbMin.x, left->AabbMax.x, right->AabbMin.x, right->AabbMax.x );
					soaNode[ idx ].Yyyy = new float4( left->AabbMin.y, left->AabbMax.y, right->AabbMin.y, right->AabbMax.y );
					soaNode[ idx ].Zzzz = new float4( left->AabbMin.z, left->AabbMax.z, right->AabbMin.z, right->AabbMax.z );
					soaNode[ idx ].Left = newAlt2Node; // right will be filled when popped
					stack[ stackPtr++ ] = idx;
					stack[ stackPtr++ ] = node->LeftFirst + 1;
					nodeIdx = node->LeftFirst;
				}
			}
			soa.UsedNodes = newAlt2Node;
		}

		/// <summary>Port of BVHBase::CopyBasePropertiesFrom for the Bvh -&gt; BvhSoa direction.</summary>
		private static void CopyBasePropertiesFrom( ref BvhSoa soa, ref Bvh original )
		{
			soa.Refittable = original.Refittable;
			soa.MayHaveHoles = original.MayHaveHoles;
			soa.BvhOverAabbs = original.BvhOverAabbs;
			soa.BvhOverIndices = original.BvhOverIndices;
			soa.TriCount = original.TriCount;
			soa.IdxCount = original.IdxCount;
			soa.AabbMin = original.AabbMin;
			soa.AabbMax = original.AabbMax;
		}
	}
}
