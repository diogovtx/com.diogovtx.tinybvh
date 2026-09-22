using System;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Leaf and node-pool maintenance of tinybvh's BVH class: Compact, SplitLeafs and the two
	/// CombineLeafs overloads. These are the preparation steps the wide layouts require, e.g.
	/// BVH8_CWBVH::Build does Compact() + SplitLeafs( 3 ) before converting, because the CWBVH
	/// leaf encoding assumes at most three triangles per leaf.
	/// </summary>
	public unsafe partial struct Bvh
	{
		/// <summary>
		/// Port of BVH::Compact. Reduce the size of a BVH by removing any unused nodes.
		/// This is useful after an SBVH build or multi-threaded build, but also after
		/// calling MergeLeafs. Some operations, such as Optimize, *require* a compacted tree
		/// to work correctly. Also reorders PrimIdx so leaf ranges are laid out in traversal order.
		/// </summary>
		public void Compact()
		{
			if ( Nodes == null )
			{
				throw new InvalidOperationException( "Bvh.Compact(), nodes == null." );
			}
			BvhLeafTools.Compact( ref this );
		}

		/// <summary>
		/// Port of BVH::SplitLeafs. Splits every leaf with more than maxPrims primitives into a
		/// chain of leaves of at most maxPrims each. Requires a compacted BVH: the leaf ranges must
		/// be contiguous in PrimIdx, which Compact guarantees.
		/// </summary>
		public void SplitLeafs( uint maxPrims )
		{
			if ( Nodes == null )
			{
				throw new InvalidOperationException( "Bvh.SplitLeafs( .. ), nodes == null." );
			}
			if ( maxPrims == 0 )
			{
				throw new ArgumentException( "Bvh.SplitLeafs( .. ), maxPrims == 0.", nameof( maxPrims ) );
			}
			BvhLeafTools.SplitLeafs( ref this, maxPrims );
		}

		/// <summary>
		/// Port of BVH::CombineLeafs( primCount, firstIdx, nodeIdx ): collapse subtrees if the
		/// summed leaf prim count does not exceed the specified number. For BVH8_CPU construction.
		/// Recursive and managed, matching the other recursive tree walks of this port.
		/// </summary>
		public uint CombineLeafs( uint primCount, ref uint firstIdx, uint nodeIdx )
		{
			BvhNode* node = Nodes + nodeIdx;
			if ( node->IsLeaf )
			{
				firstIdx = node->LeftFirst;
				return node->TriCount;
			}
			uint firstLeft = 0;
			uint leftCount = CombineLeafs( primCount, ref firstLeft, node->LeftFirst );
			uint firstRight = 0;
			uint rightCount = CombineLeafs( primCount, ref firstRight, node->LeftFirst + 1 );
			firstIdx = math.min( firstLeft, firstRight );
			if ( ( leftCount + rightCount ) <= primCount )
			{
				node->TriCount = leftCount + rightCount;
				node->LeftFirst = firstIdx;
			}
			return leftCount + rightCount;
		}

		/// <summary>
		/// Port of BVH::CombineLeafs( nodeIdx ): combine leaf nodes if this improves tree SAH cost.
		/// For HPLOC postprocessing. Recursive and managed.
		/// </summary>
		public void CombineLeafs( uint nodeIdx = 0 )
		{
			BvhNode* node = Nodes + nodeIdx;
			if ( node->IsLeaf )
			{
				return;
			}
			BvhNode* left = Nodes + node->LeftFirst;
			BvhNode* right = Nodes + node->LeftFirst + 1;
			if ( left->IsLeaf && right->IsLeaf )
			{
				int combinedCount = ( int )( left->TriCount + right->TriCount );
				float rAnode = 1.0f / BvhLeafTools.HalfArea( node->AabbMax - node->AabbMin );
				float Cnode = IntersectionCost * combinedCount;
				float Cleft = IntersectionCost * left->TriCount * BvhLeafTools.HalfArea( left->AabbMax - left->AabbMin ) * rAnode;
				float Cright = IntersectionCost * right->TriCount * BvhLeafTools.HalfArea( right->AabbMax - right->AabbMin ) * rAnode;
				float Csplit = Cleft + Cright + TraversalCost;
				if ( Cnode < Csplit )
				{
					if ( right->LeftFirst == ( left->LeftFirst + left->TriCount ) )
					{
						node->LeftFirst = left->LeftFirst;
						node->TriCount = ( uint )combinedCount;
					}
				}
				return;
			}
			CombineLeafs( node->LeftFirst );
			CombineLeafs( node->LeftFirst + 1 );
		}
	}

	/// <summary>
	/// Burst-compiled implementation of the iterative leaf tools. Direct calls must be synchronous,
	/// otherwise editor tests silently run the Mono fallback.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhLeafTools
	{
		/// <summary>Port of BVH::Compact.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void Compact( ref Bvh bvh )
		{
			if ( bvh.Nodes[ 0 ].IsLeaf )
			{
				return; // nothing to compact.
			}
			BvhNode* tmpNodes = ( BvhNode* )bvh.Alloc( ( long )sizeof( BvhNode ) * bvh.AllocatedNodes ); // do *not* trim
			uint* idx = ( uint* )bvh.Alloc( ( long )sizeof( uint ) * bvh.IdxCount );
			UnsafeUtility.MemCpy( tmpNodes, bvh.Nodes, 2 * sizeof( BvhNode ) );
			uint newNodePtr = 2;
			uint newIdxPtr = 0, nodeIdx = 0, stackPtr = 0;
			uint* stack = stackalloc uint[ 128 ];
			while ( true )
			{
				BvhNode* node = tmpNodes + nodeIdx;
				if ( node->IsLeaf )
				{
					uint leafStart = newIdxPtr;
					for ( uint i = 0; i < node->TriCount; i++ )
					{
						idx[ newIdxPtr++ ] = bvh.PrimIdx[ node->LeftFirst + i ];
					}
					node->LeftFirst = leafStart;
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
				}
				else
				{
					BvhNode* left = bvh.Nodes + node->LeftFirst;
					BvhNode* right = bvh.Nodes + node->LeftFirst + 1;
					tmpNodes[ newNodePtr ] = *left;
					tmpNodes[ newNodePtr + 1 ] = *right;
					uint todo1 = newNodePtr, todo2 = newNodePtr + 1;
					node->LeftFirst = newNodePtr;
					newNodePtr += 2;
					nodeIdx = todo1;
					stack[ stackPtr++ ] = todo2;
				}
			}
			bvh.Free( bvh.Nodes );
			bvh.Free( bvh.PrimIdx );
			bvh.UsedNodes = newNodePtr;
			bvh.Nodes = tmpNodes;
			bvh.PrimIdx = idx;
			// The C++ does not track the capacity of primIdx; this port does, so keep it in sync
			// with the freshly allocated array. The node pool is deliberately not trimmed.
			bvh.AllocatedPrimIdx = bvh.IdxCount;
		}

		/// <summary>
		/// Port of BVH::SplitLeafs. The C++ member 'newNodePtr' is UsedNodes in this port: the
		/// reference builder and Compact both leave it equal to the first free node.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void SplitLeafs( ref Bvh bvh, uint maxPrims )
		{
			BvhNode* bvhNode = bvh.Nodes;
			uint* stack = stackalloc uint[ 64 ];
			uint stackPtr = 0, nodeIdx = 0;
			uint newNodePtr = bvh.UsedNodes;
			while ( true )
			{
				BvhNode* node = bvhNode + nodeIdx;
				if ( node->IsLeaf )
				{
					if ( node->TriCount > maxPrims )
					{
						BvhNode* left = bvhNode + newNodePtr;
						BvhNode* right = bvhNode + newNodePtr + 1;
						*left = *node;
						*right = *node;
						right->LeftFirst = node->LeftFirst + maxPrims;
						right->TriCount = node->TriCount - maxPrims;
						left->TriCount = maxPrims;
						node->LeftFirst = newNodePtr;
						node->TriCount = 0;
						newNodePtr += 2;
					}
					else
					{
						if ( stackPtr == 0 )
						{
							break;
						}
						nodeIdx = stack[ --stackPtr ];
					}
				}
				else
				{
					nodeIdx = node->LeftFirst;
					stack[ stackPtr++ ] = node->LeftFirst + 1;
				}
			}
			bvh.UsedNodes = newNodePtr;
		}

		/// <summary>Port of tinybvh_halfarea, including the guard for empty (inverted) boxes.</summary>
		internal static float HalfArea( float3 v )
		{
			return v.x < -BvhConstants.Far ? 0f : ( ( v.x * v.y ) + ( v.y * v.z ) + ( v.z * v.x ) );
		}
	}
}
