using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// M-wide ('shallow') BVH node. Port of MBVH&lt;M&gt;::MBVHNode.
	/// Deviation: the C++ sizes child[] by the template parameter and pads with dummies; here the
	/// array is always eight wide so a single struct can serve both M = 4 and M = 8. Slots beyond
	/// M are never written and stay zero, so the first M entries are the only meaningful ones.
	/// </summary>
	[StructLayout( LayoutKind.Sequential )]
	public unsafe struct MbvhNode
	{
		public float3 AabbMin;
		public uint FirstTri;
		public float3 AabbMax;
		public uint TriCount;
		public fixed uint Child[ 8 ];
		public uint ChildCount;

		public bool IsLeaf => TriCount > 0;
	}

	/// <summary>
	/// Port of tinybvh's MBVH&lt;M&gt; class, with M as a runtime field (only 4 and 8 are used).
	/// The C++ embeds a BVH member and reads bvh.primIdx / bvh.verts through it; here that member
	/// is <see cref="Source"/>, a value copy of the source Bvh that shares - and does not own - its
	/// memory. Keep the source Bvh alive for as long as this Mbvh is used.
	/// </summary>
	public unsafe partial struct Mbvh : IDisposable
	{
		/// <summary>Branching factor; the C++ template parameter.</summary>
		public int M;
		/// <summary>Value copy of the source Bvh. Shares its memory; Dispose does not free it.</summary>
		public Bvh Source;

		// Node pool (owned).
		[NativeDisableUnsafePtrRestriction] public MbvhNode* Nodes;
		public uint UsedNodes;
		public uint AllocatedNodes;

		// Properties copied from the source by CopyBasePropertiesFrom.
		public uint TriCount;
		public uint IdxCount;
		public float3 AabbMin;
		public float3 AabbMax;
		[MarshalAs( UnmanagedType.U1 )] public bool Refittable;
		[MarshalAs( UnmanagedType.U1 )] public bool MayHaveHoles;
		[MarshalAs( UnmanagedType.U1 )] public bool BvhOverAabbs;
		[MarshalAs( UnmanagedType.U1 )] public bool BvhOverIndices;

		// SAH cost parameters. Not copied by CopyBasePropertiesFrom in the C++ either, so these
		// keep the BVHBase defaults unless the caller changes them.
		public float TraversalCost;
		public float IntersectionCost;

		public Allocator Allocator;

		public static Mbvh Create( int m, Allocator allocator )
		{
			if ( m != 4 && m != 8 )
			{
				throw new ArgumentException( "Mbvh.Create( .. ), only M == 4 and M == 8 are supported.", nameof( m ) );
			}
			return new Mbvh
			{
				M = m,
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
		}

		/// <summary>Port of MBVH&lt;M&gt;::ConvertFrom: collapses a binary BVH into an M-wide one.</summary>
		public void ConvertFrom( ref Bvh original, bool compact = true )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "Mbvh.ConvertFrom( .. ), mbvh was not created." );
			}
			if ( original.Nodes == null )
			{
				throw new ArgumentException( "Mbvh.ConvertFrom( .. ), original.Nodes == null.", nameof( original ) );
			}
			if ( original.UsedNodes == 0 )
			{
				throw new ArgumentException( "Mbvh.ConvertFrom( .. ), original.UsedNodes == 0.", nameof( original ) );
			}
			MbvhConverter.ConvertFrom( ref this, ref original, compact );
		}

		/// <summary>
		/// Port of MBVH&lt;M&gt;::Optimize: optimizes the underlying binary BVH and re-converts.
		/// </summary>
		public void Optimize( uint iterations = 25, bool extreme = false )
		{
			Source.Optimize( iterations, extreme );
			// A local copy keeps ConvertFrom from aliasing a field of 'this' in Burst; the
			// converter assigns it back to Source, exactly like the C++ 'bvh = original'.
			Bvh source = Source;
			ConvertFrom( ref source, true );
		}

		/// <summary>
		/// Port of MBVH&lt;M&gt;::Refit. Adjusts node bounds to the current vertex positions, bottom up.
		/// Recursive and managed, like Bvh.SahCost; the conversions do not use it.
		/// </summary>
		public void Refit( uint nodeIdx = 0 )
		{
			MbvhNode* node = Nodes + nodeIdx;
			if ( node->IsLeaf )
			{
				float3 bmin = new float3( BvhConstants.Far ), bmax = new float3( -BvhConstants.Far );
				if ( Source.VertIdx != null )
				{
					for ( uint first = node->FirstTri, j = 0; j < node->TriCount; j++ )
					{
						uint vidx = Source.PrimIdx[ first + j ] * 3;
						uint i0 = Source.VertIdx[ vidx ], i1 = Source.VertIdx[ vidx + 1 ], i2 = Source.VertIdx[ vidx + 2 ];
						float3 v0 = Source.Vertex( i0 ).xyz, v1 = Source.Vertex( i1 ).xyz, v2 = Source.Vertex( i2 ).xyz;
						bmin = math.min( bmin, math.min( math.min( v0, v1 ), v2 ) );
						bmax = math.max( bmax, math.max( math.max( v0, v1 ), v2 ) );
					}
				}
				else
				{
					for ( uint first = node->FirstTri, j = 0; j < node->TriCount; j++ )
					{
						uint vidx = Source.PrimIdx[ first + j ] * 3;
						float3 v0 = Source.Vertex( vidx ).xyz, v1 = Source.Vertex( vidx + 1 ).xyz, v2 = Source.Vertex( vidx + 2 ).xyz;
						bmin = math.min( bmin, math.min( math.min( v0, v1 ), v2 ) );
						bmax = math.max( bmax, math.max( math.max( v0, v1 ), v2 ) );
					}
				}
				node->AabbMin = bmin;
				node->AabbMax = bmax;
			}
			else
			{
				for ( uint i = 0; i < node->ChildCount; i++ )
				{
					Refit( node->Child[ i ] );
				}
				MbvhNode* firstChild = Nodes + node->Child[ 0 ];
				float3 bmin = firstChild->AabbMin, bmax = firstChild->AabbMax;
				for ( uint i = 1; i < node->ChildCount; i++ )
				{
					MbvhNode* child = Nodes + node->Child[ i ];
					bmin = math.min( bmin, child->AabbMin );
					bmax = math.max( bmax, child->AabbMax );
				}
				node->AabbMin = bmin;
				node->AabbMax = bmax;
			}
			if ( nodeIdx == 0 )
			{
				AabbMin = node->AabbMin;
				AabbMax = node->AabbMax;
			}
		}

		/// <summary>
		/// Port of MBVH&lt;M&gt;::SAHCost. Determine the SAH cost of the tree. This provides an
		/// indication of the quality of the BVH: Lower is better.
		/// </summary>
		public float SahCost( uint nodeIdx = 0 )
		{
			MbvhNode* n = Nodes + nodeIdx;
			float sa = BvhMath.SurfaceArea( n->AabbMin, n->AabbMax );
			if ( n->IsLeaf )
			{
				return IntersectionCost * sa * n->TriCount;
			}
			float cost = TraversalCost * sa;
			for ( int i = 0; i < M; i++ )
			{
				if ( n->Child[ i ] != 0 )
				{
					cost += SahCost( n->Child[ i ] );
				}
			}
			return nodeIdx == 0 ? ( cost / sa ) : cost;
		}

		/// <summary>Port of MBVH&lt;M&gt;::LeafCount.</summary>
		public uint LeafCount( uint nodeIdx = 0 )
		{
			MbvhNode* node = Nodes + nodeIdx;
			if ( node->IsLeaf )
			{
				return 1;
			}
			uint count = 0;
			for ( uint i = 0; i < node->ChildCount; i++ )
			{
				count += LeafCount( node->Child[ i ] );
			}
			return count;
		}

		/// <summary>Ensures the node pool can hold count nodes. Contents are not preserved when it grows.</summary>
		internal void AllocateNodes( uint count )
		{
			if ( AllocatedNodes < count )
			{
				Free( Nodes );
				Nodes = ( MbvhNode* )Alloc( ( long )count * sizeof( MbvhNode ) );
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
	/// Burst-compiled implementation of the MBVH conversion routines. Direct calls must be
	/// synchronous, otherwise editor tests silently run the Mono fallback.
	/// </summary>
	[BurstCompile]
	internal static unsafe class MbvhConverter
	{
		/// <summary>Port of MBVH&lt;M&gt;::ConvertFrom.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void ConvertFrom( ref Mbvh mbvh, ref Bvh original, [MarshalAs( UnmanagedType.U1 )] bool compact )
		{
			int m = mbvh.M;
			// get a copy of the original bvh
			mbvh.Source = original;
			// allocate space
			uint spaceNeeded = compact ? original.UsedNodes : original.AllocatedNodes;
			bool m8 = m == 8;
			if ( m8 )
			{
				spaceNeeded += original.UsedNodes >> 1; // cwbvh / SplitLeafs
			}
			mbvh.AllocateNodes( spaceNeeded );
			MbvhNode* mbvhNode = mbvh.Nodes;
			UnsafeUtility.MemClear( mbvhNode, ( long )spaceNeeded * sizeof( MbvhNode ) );
			CopyBasePropertiesFrom( ref mbvh, ref original );
			// create an mbvh node for each bvh2 node
			for ( uint i = 0; i < original.UsedNodes; i++ )
			{
				if ( i != 1 )
				{
					BvhNode* orig = original.Nodes + i;
					MbvhNode* node = mbvhNode + i;
					node->AabbMin = orig->AabbMin;
					node->AabbMax = orig->AabbMax;
					if ( orig->IsLeaf )
					{
						node->TriCount = orig->TriCount;
						node->FirstTri = orig->LeftFirst;
					}
					else
					{
						node->Child[ 0 ] = orig->LeftFirst;
						node->Child[ 1 ] = orig->LeftFirst + 1;
						node->ChildCount = 2;
					}
				}
			}
			// collapse
			uint* stack = stackalloc uint[ 128 ];
			uint stackPtr = 0, nodeIdx = 0; // i.e., root node
			while ( true )
			{
				MbvhNode* node = mbvhNode + nodeIdx;
				while ( node->ChildCount < m )
				{
					int bestChild = -1;
					float bestChildSA = 0;
					for ( uint i = 0; i < node->ChildCount; i++ )
					{
						// see if we can adopt child i
						MbvhNode* candidate = mbvhNode + node->Child[ i ];
						if ( !candidate->IsLeaf && ( int )( node->ChildCount - 1 + candidate->ChildCount ) <= m )
						{
							float childSA = BvhMath.SurfaceArea( candidate->AabbMin, candidate->AabbMax );
							if ( childSA > bestChildSA )
							{
								bestChild = ( int )i;
								bestChildSA = childSA;
							}
						}
					}
					if ( bestChild == -1 )
					{
						break; // could not adopt
					}
					MbvhNode* child = mbvhNode + node->Child[ bestChild ];
					node->Child[ bestChild ] = child->Child[ 0 ];
					for ( uint i = 1; i < child->ChildCount; i++ )
					{
						node->Child[ node->ChildCount++ ] = child->Child[ i ];
					}
				}
				// we're done with the node; proceed with the children.
				for ( uint i = 0; i < node->ChildCount; i++ )
				{
					uint childIdx = node->Child[ i ];
					MbvhNode* child = mbvhNode + childIdx;
					if ( !child->IsLeaf )
					{
						stack[ stackPtr++ ] = childIdx;
					}
				}
				if ( stackPtr == 0 )
				{
					break;
				}
				nodeIdx = stack[ --stackPtr ];
			}
			// special case where root is leaf: add extra level - cwbvh needs this.
			MbvhNode* root = mbvhNode;
			if ( root->IsLeaf )
			{
				mbvhNode[ 1 ] = *root;
				root->ChildCount = 1;
				root->Child[ 0 ] = 1;
				root->TriCount = 0;
			}
			// finalize
			mbvh.UsedNodes = original.UsedNodes;
			mbvh.MayHaveHoles = true;
		}

		/// <summary>Port of BVHBase::CopyBasePropertiesFrom for the Bvh -&gt; Mbvh direction.</summary>
		private static void CopyBasePropertiesFrom( ref Mbvh mbvh, ref Bvh original )
		{
			mbvh.Refittable = original.Refittable;
			mbvh.MayHaveHoles = original.MayHaveHoles;
			mbvh.BvhOverAabbs = original.BvhOverAabbs;
			mbvh.BvhOverIndices = original.BvhOverIndices;
			mbvh.TriCount = original.TriCount;
			mbvh.IdxCount = original.IdxCount;
			mbvh.AabbMin = original.AabbMin;
			mbvh.AabbMax = original.AabbMax;
		}
	}
}
