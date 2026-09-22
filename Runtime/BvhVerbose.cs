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
	/// Port of BVH_Verbose::BVHNode. This node layout has some extra data per node: It stores left
	/// and right child node indices explicitly, and stores the index of the parent node.
	/// This format exists primarily for the BVH optimizer. Total: 64 bytes.
	/// </summary>
	[StructLayout( LayoutKind.Sequential )]
	public unsafe struct BvhVerboseNode
	{
		public float3 AabbMin;
		public uint Left;
		public float3 AabbMax;
		public uint Right;
		public uint TriCount;
		public uint FirstTri;
		public uint Parent;
		/// <summary>Padding to 64 bytes, as in the C++ node. Never read or written after ConvertFrom clears it.</summary>
		public fixed float Dummy[ 5 ];

		public bool IsLeaf => TriCount > 0;
		public float SurfaceArea => BvhMath.SurfaceArea( AabbMin, AabbMax );
	}

	/// <summary>
	/// Port of tinybvh's BVH_Verbose class: a binary BVH in a 64-byte node layout with explicit
	/// child and parent indices, used by the tree-rotation optimizer. Build a regular
	/// <see cref="Bvh"/>, convert, optimize and convert back; <see cref="Bvh.Optimize"/> does that
	/// in one call. The input geometry, the fragment array and (until MergeLeafs runs) the
	/// primitive index array are referenced, not owned: keep the source Bvh alive.
	/// </summary>
	public unsafe partial struct BvhVerbose : IDisposable
	{
		// Input primitives (not owned): pointers copied from the source Bvh by ConvertFrom.
		[NativeDisableUnsafePtrRestriction] public byte* Verts;
		public int VertStride;
		[NativeDisableUnsafePtrRestriction] public Fragment* Fragments;
		/// <summary>
		/// Primitive index array. Shared with the source Bvh, except after MergeLeafs, which frees
		/// that array and installs a fresh one here - see <see cref="MergeLeafs"/>.
		/// </summary>
		[NativeDisableUnsafePtrRestriction] public uint* PrimIdx;

		// Node pool (owned). Root is always node 0; node 1 is unused for alignment.
		[NativeDisableUnsafePtrRestriction] public BvhVerboseNode* Nodes;
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

		/// <summary>
		/// True once MergeLeafs has replaced <see cref="PrimIdx"/> with an array of its own.
		/// Ownership moves back to a Bvh on the next Bvh.ConvertFrom( ref BvhVerbose, bool ).
		/// Deviation: the C++ leaks the array instead of tracking this.
		/// </summary>
		[MarshalAs( UnmanagedType.U1 )] public bool OwnsPrimIdx;

		/// <summary>
		/// SAH cost parameters. Deviation-free: BVHBase::CopyBasePropertiesFrom does not copy
		/// c_trav / c_int, so a BVH_Verbose always uses the defaults, whatever the source BVH had.
		/// MergeLeafs is the only member that reads them.
		/// </summary>
		public float TraversalCost;
		public float IntersectionCost;

		/// <summary>
		/// State of the pseudo random generator the stochastic variant of <see cref="Optimize"/>
		/// draws from. tinybvh calls the C runtime's rand() there; the reference tool is MSVC and
		/// never seeds it, so this reproduces the MSVC generator from its initial state: state
		/// starts at 1, state = state * 214013 + 2531011, rand() = ( state >&gt; 16 ) &amp; 0x7fff.
		/// <para>
		/// Deviation: the C++ has one process-wide rand() stream, so a second Optimize call there
		/// continues where the first left off. Here the state lives in the BvhVerbose, so it is
		/// per-object and <see cref="Bvh.Optimize"/>, which creates a temporary one, restarts from
		/// its randomSeed on every call. Set this field to continue a stream by hand.
		/// </para>
		/// </summary>
		public uint RandomState;

		public Allocator Allocator;

		public static BvhVerbose Create( Allocator allocator )
		{
			return new BvhVerbose
			{
				VertStride = 16,
				Refittable = true,
				TraversalCost = BvhConstants.DefaultTraversalCost,
				IntersectionCost = BvhConstants.DefaultIntersectionCost,
				RandomState = 1,
				Allocator = allocator
			};
		}

		public bool IsCreated => Allocator > Allocator.None;

		public void Dispose()
		{
			Free( Nodes );
			if ( OwnsPrimIdx )
			{
				Free( PrimIdx );
				OwnsPrimIdx = false;
			}
			Nodes = null;
			PrimIdx = null;
			AllocatedNodes = 0;
			UsedNodes = 0;
			TriCount = 0;
			IdxCount = 0;
		}

		/// <summary>
		/// Port of BVH_Verbose::ConvertFrom( const BVH&amp; ). The node pool is sized
		/// triCount * (refittable ? 2 : 3) as in the C++, using this object's own refittable flag,
		/// which is true for a freshly created BvhVerbose. That is enough room for the source tree
		/// plus a later SplitLeafs( 1 ), which needs at most 2 * triCount nodes.
		/// </summary>
		public void ConvertFrom( ref Bvh original )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "BvhVerbose.ConvertFrom( .. ), bvhVerbose was not created." );
			}
			if ( original.Nodes == null )
			{
				throw new ArgumentException( "BvhVerbose.ConvertFrom( .. ), original.Nodes == null.", nameof( original ) );
			}
			if ( original.UsedNodes == 0 )
			{
				throw new ArgumentException( "BvhVerbose.ConvertFrom( .. ), original.UsedNodes == 0.", nameof( original ) );
			}
			// The C++ allocates blindly here; a tree with more nodes than the formula allows - an
			// SBVH, say - would run off the end of the pool, so reject it instead.
			uint spaceNeeded = original.TriCount * ( Refittable ? 2u : 3u );
			if ( original.UsedNodes > spaceNeeded )
			{
				throw new ArgumentException( "BvhVerbose.ConvertFrom( .. ), original.UsedNodes exceeds the verbose node pool.", nameof( original ) );
			}
			BvhVerboseTools.ConvertFrom( ref this, ref original );
		}

		/// <summary>Port of BVH_Verbose::SAHCost. Lower is better.</summary>
		public float SahCost( uint nodeIdx = 0 )
		{
			BvhVerboseTools.SahCost( ref this, nodeIdx, out float cost );
			return cost;
		}

		/// <summary>Port of BVH_Verbose::NodeCount.</summary>
		public int NodeCount()
		{
			uint retVal = 0, nodeIdx = 0, stackPtr = 0;
			uint* stack = stackalloc uint[ 64 ];
			while ( true )
			{
				BvhVerboseNode* n = Nodes + nodeIdx;
				retVal++;
				if ( n->IsLeaf )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
				}
				else
				{
					nodeIdx = n->Left;
					stack[ stackPtr++ ] = n->Right;
				}
			}
			return ( int )retVal;
		}

		/// <summary>Port of BVH_Verbose::PrimCount.</summary>
		public int PrimCount( uint nodeIdx = 0 )
		{
			BvhVerboseNode* n = Nodes + nodeIdx;
			return n->IsLeaf ? ( int )n->TriCount : ( PrimCount( n->Left ) + PrimCount( n->Right ) );
		}

		/// <summary>
		/// Port of BVH_Verbose::Refit. With skipLeafs the leaf bounds are left alone and only the
		/// interior nodes are recomputed bottom up, which is what the optimizer needs.
		/// </summary>
		public void Refit( uint nodeIdx = 0, bool skipLeafs = false )
		{
			if ( !Refittable && !skipLeafs )
			{
				throw new InvalidOperationException( "BvhVerbose.Refit( .. ), refitting an SBVH." );
			}
			if ( Nodes == null )
			{
				throw new InvalidOperationException( "BvhVerbose.Refit( .. ), nodes == null." );
			}
			if ( BvhOverIndices && !skipLeafs )
			{
				throw new InvalidOperationException( "BvhVerbose.Refit( .. ), bvh used indexed tris." );
			}
			BvhVerboseTools.Refit( ref this, nodeIdx, skipLeafs );
		}

		/// <summary>
		/// Port of BVH_Verbose::Compact. Deviation: the C++ leaves allocatedNodes at its old value
		/// although the new pool is smaller; this port corrects it. Note that, as in the C++, the
		/// parent indices are not remapped, so the result is only usable for a read-only walk.
		/// </summary>
		public void Compact()
		{
			if ( Nodes == null )
			{
				throw new InvalidOperationException( "BvhVerbose.Compact(), nodes == null." );
			}
			BvhVerboseTools.Compact( ref this );
		}

		/// <summary>
		/// Port of BVH_Verbose::SortIndices: rewrites PrimIdx so the primitive indices are in
		/// depth-first traversal order, and updates the leaf offsets to match.
		/// </summary>
		public void SortIndices()
		{
			if ( Nodes == null )
			{
				throw new InvalidOperationException( "BvhVerbose.SortIndices(), nodes == null." );
			}
			BvhVerboseTools.SortIndices( ref this );
		}

		/// <summary>
		/// Port of BVH_Verbose::SplitLeafs. Single-primitive leafs: Prepare the BVH for
		/// optimization. While it is not strictly necessary to have a single primitive per leaf,
		/// it will yield a slightly better optimized BVH. The leafs of the optimized BVH should be
		/// collapsed ('MergeLeafs') to obtain the final tree. Uses the fragment bounds of the
		/// source Bvh, so that array must still be alive.
		/// </summary>
		public void SplitLeafs( uint maxPrims = 1 )
		{
			if ( Nodes == null )
			{
				throw new InvalidOperationException( "BvhVerbose.SplitLeafs( .. ), nodes == null." );
			}
			if ( Fragments == null )
			{
				throw new InvalidOperationException( "BvhVerbose.SplitLeafs( .. ), fragments == null." );
			}
			if ( maxPrims == 0 )
			{
				throw new ArgumentException( "BvhVerbose.SplitLeafs( .. ), maxPrims == 0.", nameof( maxPrims ) );
			}
			BvhVerboseTools.SplitLeafs( ref this, maxPrims );
		}

		/// <summary>
		/// Port of BVH_Verbose::MergeLeafs. After optimizing a BVH, single-primitive leafs should
		/// be merged whenever SAH indicates this is an improvement. Frees the primitive index array
		/// it shares with the source Bvh and installs a fresh one, so that Bvh now holds a dangling
		/// pointer until Bvh.ConvertFrom( ref BvhVerbose, bool ) copies the new array back into it.
		/// That is exactly what the C++ does; this port additionally frees the new array in Dispose
		/// when no conversion back has claimed it.
		/// </summary>
		public void MergeLeafs()
		{
			if ( Nodes == null )
			{
				throw new InvalidOperationException( "BvhVerbose.MergeLeafs(), nodes == null." );
			}
			BvhVerboseTools.MergeLeafs( ref this );
		}

		/// <summary>
		/// Port of BVH_Verbose::Optimize: optimize by reinserting subtrees with a high cost -
		/// Section 3.4 of "Fast Insertion-Based Optimization of Bounding Volume Hierarchies".
		/// With extreme, later iterations process more nodes. With stochastic, half of the
		/// candidates are considered each iteration but the walk over them starts at a random
		/// offset and skips a random number of entries; see <see cref="RandomState"/>. The C++
		/// ignores extreme when stochastic is set and so does this.
		/// </summary>
		public void Optimize( uint iterations = 25, bool extreme = false, bool stochastic = false )
		{
			if ( Nodes == null )
			{
				throw new InvalidOperationException( "BvhVerbose.Optimize( .. ), nodes == null." );
			}
			if ( UsedNodes == 0 )
			{
				throw new InvalidOperationException( "BvhVerbose.Optimize( .. ), usedNodes == 0." );
			}
			BvhVerboseTools.Optimize( ref this, iterations, extreme, stochastic );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public float4 Vertex( uint i )
		{
			return *( float4* )( Verts + ( i * VertStride ) );
		}

		/// <summary>Ensures the node pool can hold count nodes. Contents are not preserved when it grows.</summary>
		internal void AllocateNodes( uint count )
		{
			if ( AllocatedNodes < count )
			{
				Free( Nodes );
				Nodes = ( BvhVerboseNode* )Alloc( ( long )count * sizeof( BvhVerboseNode ) );
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
	/// Burst-compiled implementation of BVH_Verbose. Direct calls must be synchronous, otherwise
	/// editor tests silently run the Mono fallback, which evaluates float math in double and so
	/// produces a different tree than the C++ reference.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhVerboseTools
	{
		/// <summary>Port of the private BVH_Verbose::SortItem.</summary>
		private struct SortItem
		{
			public uint Idx;
			public float Cost;
		}

		/// <summary>Port of the 'struct Task { uint32_t first, last; }' of the partial quick sort.</summary>
		private struct SortTask
		{
			public int First;
			public int Last;
		}

		/// <summary>Port of the 'struct Task { float ci; uint32_t node; }' of FindBestNewPosition.</summary>
		private struct FindTask
		{
			public float Ci;
			public uint Node;
		}

		/// <summary>C++: 'struct Task { .. } stack[4096]' in BVH_Verbose::Optimize.</summary>
		private const int SortStackSize = 4096;

		/// <summary>C++: 'ALIGNED( 64 ) Task task[512]' in BVH_Verbose::FindBestNewPosition.</summary>
		private const int FindTaskCount = 512;

		/// <summary>Port of BVHBase::CopyBasePropertiesFrom for the Bvh -&gt; BvhVerbose direction.</summary>
		private static void CopyBasePropertiesFrom( ref BvhVerbose bvh, ref Bvh original )
		{
			bvh.Refittable = original.Refittable;
			bvh.MayHaveHoles = original.MayHaveHoles;
			bvh.BvhOverAabbs = original.BvhOverAabbs;
			bvh.BvhOverIndices = original.BvhOverIndices;
			bvh.TriCount = original.TriCount;
			bvh.IdxCount = original.IdxCount;
			bvh.AabbMin = original.AabbMin;
			bvh.AabbMax = original.AabbMax;
		}

		/// <summary>Port of BVH_Verbose::ConvertFrom( const BVH&amp; ).</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void ConvertFrom( ref BvhVerbose bvh, ref Bvh original )
		{
			// allocate space
			uint spaceNeeded = original.TriCount * ( bvh.Refittable ? 2u : 3u );
			bvh.AllocateNodes( spaceNeeded );
			BvhVerboseNode* bvhNode = bvh.Nodes;
			UnsafeUtility.MemClear( bvhNode, ( long )spaceNeeded * sizeof( BvhVerboseNode ) );
			CopyBasePropertiesFrom( ref bvh, ref original );
			bvh.Verts = original.Verts;
			bvh.VertStride = original.VertStride;
			bvh.Fragments = original.Fragments;
			bvh.PrimIdx = original.PrimIdx;
			bvh.OwnsPrimIdx = false;
			bvhNode[ 0 ].Parent = 0xffffffff; // root sentinel
			// convert
			uint nodeIdx = 0, parent = 0xffffffff, stackPtr = 0;
			uint* stack = stackalloc uint[ 128 ];
			while ( true )
			{
				BvhNode* orig = original.Nodes + nodeIdx;
				bvhNode[ nodeIdx ].AabbMin = orig->AabbMin;
				bvhNode[ nodeIdx ].AabbMax = orig->AabbMax;
				bvhNode[ nodeIdx ].TriCount = orig->TriCount;
				bvhNode[ nodeIdx ].Parent = parent;
				if ( orig->IsLeaf )
				{
					bvhNode[ nodeIdx ].FirstTri = orig->LeftFirst;
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
					parent = stack[ --stackPtr ];
				}
				else
				{
					bvhNode[ nodeIdx ].Left = orig->LeftFirst;
					bvhNode[ nodeIdx ].Right = orig->LeftFirst + 1;
					stack[ stackPtr++ ] = nodeIdx;
					stack[ stackPtr++ ] = orig->LeftFirst + 1;
					parent = nodeIdx;
					nodeIdx = orig->LeftFirst;
				}
			}
			bvh.UsedNodes = original.UsedNodes;
		}

		/// <summary>Port of BVH_Verbose::SAHCost.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void SahCost( ref BvhVerbose bvh, uint nodeIdx, out float result )
		{
			result = SahCostRec( bvh.Nodes, nodeIdx, bvh.TraversalCost, bvh.IntersectionCost );
		}

		private static float SahCostRec( BvhVerboseNode* bvhNode, uint nodeIdx, float cTrav, float cInt )
		{
			BvhVerboseNode* n = bvhNode + nodeIdx;
			float SAn = BvhMath.SurfaceArea( n->AabbMin, n->AabbMax );
			if ( n->IsLeaf )
			{
				return cInt * SAn * n->TriCount;
			}
			float cost = ( cTrav * SAn ) + SahCostRec( bvhNode, n->Left, cTrav, cInt ) + SahCostRec( bvhNode, n->Right, cTrav, cInt );
			return nodeIdx == 0 ? ( cost / SAn ) : cost;
		}

		/// <summary>Port of BVH_Verbose::Refit.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void Refit( ref BvhVerbose bvh, uint nodeIdx, [MarshalAs( UnmanagedType.U1 )] bool skipLeafs )
		{
			RefitRec( ref bvh, nodeIdx, skipLeafs );
		}

		private static void RefitRec( ref BvhVerbose bvh, uint nodeIdx, bool skipLeafs )
		{
			BvhVerboseNode* bvhNode = bvh.Nodes;
			BvhVerboseNode* node = bvhNode + nodeIdx;
			if ( node->IsLeaf ) // leaf: adjust to current triangle vertex positions
			{
				if ( skipLeafs )
				{
					return;
				}
				float3 bmin = new float3( BvhConstants.Far ), bmax = new float3( -BvhConstants.Far );
				for ( uint first = node->FirstTri, j = 0; j < node->TriCount; j++ )
				{
					uint vertIdx = bvh.PrimIdx[ first + j ] * 3;
					float3 v0 = bvh.Vertex( vertIdx ).xyz;
					float3 v1 = bvh.Vertex( vertIdx + 1 ).xyz;
					float3 v2 = bvh.Vertex( vertIdx + 2 ).xyz;
					bmin = math.min( bmin, v0 );
					bmax = math.max( bmax, v0 );
					bmin = math.min( bmin, v1 );
					bmax = math.max( bmax, v1 );
					bmin = math.min( bmin, v2 );
					bmax = math.max( bmax, v2 );
				}
				node->AabbMin = bmin;
				node->AabbMax = bmax;
			}
			else
			{
				RefitRec( ref bvh, node->Left, skipLeafs );
				RefitRec( ref bvh, node->Right, skipLeafs );
				node->AabbMin = math.min( bvhNode[ node->Left ].AabbMin, bvhNode[ node->Right ].AabbMin );
				node->AabbMax = math.max( bvhNode[ node->Left ].AabbMax, bvhNode[ node->Right ].AabbMax );
			}
			if ( nodeIdx == 0 )
			{
				bvh.AabbMin = node->AabbMin;
				bvh.AabbMax = node->AabbMax;
			}
		}

		/// <summary>Port of BVH_Verbose::Compact.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void Compact( ref BvhVerbose bvh )
		{
			if ( bvh.Nodes[ 0 ].IsLeaf )
			{
				return; // nothing to compact.
			}
			uint capacity = bvh.UsedNodes;
			BvhVerboseNode* tmp = ( BvhVerboseNode* )bvh.Alloc( ( long )capacity * sizeof( BvhVerboseNode ) );
			UnsafeUtility.MemCpy( tmp, bvh.Nodes, 2 * sizeof( BvhVerboseNode ) );
			uint newNodePtr = 2, nodeIdx = 0, stackPtr = 0;
			uint* stack = stackalloc uint[ 64 ];
			while ( true )
			{
				BvhVerboseNode* node = tmp + nodeIdx;
				BvhVerboseNode* left = bvh.Nodes + node->Left;
				BvhVerboseNode* right = bvh.Nodes + node->Right;
				tmp[ newNodePtr ] = *left;
				tmp[ newNodePtr + 1 ] = *right;
				uint todo1 = newNodePtr, todo2 = newNodePtr + 1;
				node->Left = newNodePtr++;
				node->Right = newNodePtr++;
				if ( !left->IsLeaf )
				{
					stack[ stackPtr++ ] = todo1;
				}
				if ( !right->IsLeaf )
				{
					stack[ stackPtr++ ] = todo2;
				}
				if ( stackPtr == 0 )
				{
					break;
				}
				nodeIdx = stack[ --stackPtr ];
			}
			bvh.UsedNodes = newNodePtr;
			bvh.Free( bvh.Nodes );
			bvh.Nodes = tmp;
			// The C++ does not track that the pool shrank; this port must, or a later
			// AllocateNodes would believe there is more room than the buffer has.
			bvh.AllocatedNodes = capacity;
		}

		/// <summary>Port of BVH_Verbose::SortIndices.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void SortIndices( ref BvhVerbose bvh )
		{
			// create a new primIdx array which has the primitive indices sorted by depth-first traversal order.
			BvhVerboseNode* bvhNode = bvh.Nodes;
			uint nodeIdx = 0, stackPtr = 0, nextIdx = 0;
			uint* stack = stackalloc uint[ 256 ];
			uint* tmp = ( uint* )bvh.Alloc( ( long )bvh.TriCount * sizeof( uint ) );
			while ( true )
			{
				BvhVerboseNode* node = bvhNode + nodeIdx;
				if ( node->IsLeaf )
				{
					uint tmpFirst = nextIdx;
					for ( uint i = 0; i < node->TriCount; i++ )
					{
						tmp[ nextIdx++ ] = bvh.PrimIdx[ node->FirstTri + i ];
					}
					node->FirstTri = tmpFirst;
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
					continue;
				}
				nodeIdx = node->Left;
				stack[ stackPtr++ ] = node->Right;
			}
			UnsafeUtility.MemCpy( bvh.PrimIdx, tmp, ( long )bvh.TriCount * sizeof( uint ) );
			bvh.Free( tmp );
		}

		/// <summary>Port of BVH_Verbose::SplitLeafs.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void SplitLeafs( ref BvhVerbose bvh, uint maxPrims )
		{
			BvhVerboseNode* bvhNode = bvh.Nodes;
			Fragment* fragment = bvh.Fragments;
			uint nodeIdx = 0, stackPtr = 0;
			uint* stack = stackalloc uint[ 64 ];
			while ( true )
			{
				BvhVerboseNode* node = bvhNode + nodeIdx;
				if ( !node->IsLeaf )
				{
					nodeIdx = node->Left;
					stack[ stackPtr++ ] = node->Right;
				}
				else
				{
					// split this leaf
					if ( node->TriCount > maxPrims )
					{
						uint newIdx1 = bvh.UsedNodes++, newIdx2 = bvh.UsedNodes++;
						BvhVerboseNode* new1 = bvhNode + newIdx1;
						BvhVerboseNode* new2 = bvhNode + newIdx2;
						new1->FirstTri = node->FirstTri;
						new1->TriCount = node->TriCount / 2;
						new1->Parent = new2->Parent = nodeIdx;
						new1->Left = new1->Right = 0;
						new2->FirstTri = node->FirstTri + new1->TriCount;
						new2->TriCount = node->TriCount - new1->TriCount;
						new2->Left = new2->Right = 0;
						node->Left = newIdx1;
						node->Right = newIdx2;
						node->TriCount = 0;
						new1->AabbMin = new2->AabbMin = new float3( BvhConstants.Far );
						new1->AabbMax = new2->AabbMax = new float3( -BvhConstants.Far );
						for ( uint i = 0; i < new1->TriCount; i++ )
						{
							uint fi = bvh.PrimIdx[ new1->FirstTri + i ];
							new1->AabbMin = math.min( new1->AabbMin, fragment[ fi ].BMin );
							new1->AabbMax = math.max( new1->AabbMax, fragment[ fi ].BMax );
						}
						for ( uint i = 0; i < new2->TriCount; i++ )
						{
							uint fi = bvh.PrimIdx[ new2->FirstTri + i ];
							new2->AabbMin = math.min( new2->AabbMin, fragment[ fi ].BMin );
							new2->AabbMax = math.max( new2->AabbMax, fragment[ fi ].BMax );
						}
						// recurse
						if ( new1->TriCount > 1 )
						{
							stack[ stackPtr++ ] = newIdx1;
						}
						if ( new2->TriCount > 1 )
						{
							stack[ stackPtr++ ] = newIdx2;
						}
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
				}
			}
		}

		/// <summary>Port of BVH_Verbose::MergeLeafs.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void MergeLeafs( ref BvhVerbose bvh )
		{
			// allocate some working space
			BvhVerboseNode* bvhNode = bvh.Nodes;
			uint* subtreeTriCount = ( uint* )bvh.Alloc( ( long )bvh.UsedNodes * 4 );
			uint* newIdx = ( uint* )bvh.Alloc( ( long )bvh.IdxCount * 4 );
			UnsafeUtility.MemClear( subtreeTriCount, ( long )bvh.UsedNodes * 4 );
			CountSubtreeTris( bvhNode, 0, subtreeTriCount );
			uint stackPtr = 0, nodeIdx = 0, newIdxPtr = 0;
			uint* stack = stackalloc uint[ 64 ];
			while ( true )
			{
				BvhVerboseNode* node = bvhNode + nodeIdx;
				if ( node->IsLeaf )
				{
					uint start = newIdxPtr;
					MergeSubtree( bvhNode, nodeIdx, bvh.PrimIdx, newIdx, ref newIdxPtr );
					node->FirstTri = start;
					// pop new task
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
				}
				else
				{
					uint leftCount = subtreeTriCount[ node->Left ];
					uint rightCount = subtreeTriCount[ node->Right ];
					uint mergedCount = leftCount + rightCount;
					// cost of unsplit
					float Cunsplit = BvhMath.SurfaceArea( node->AabbMin, node->AabbMax ) * mergedCount * bvh.IntersectionCost;
					// cost of leaving things as they are
					BvhVerboseNode* left = bvhNode + node->Left;
					BvhVerboseNode* right = bvhNode + node->Right;
					float Ckeepsplit = bvh.TraversalCost + ( bvh.IntersectionCost * ( ( left->SurfaceArea * leftCount ) + ( right->SurfaceArea * rightCount ) ) );
					if ( Cunsplit <= Ckeepsplit )
					{
						// collapse the subtree
						uint start = newIdxPtr;
						MergeSubtree( bvhNode, nodeIdx, bvh.PrimIdx, newIdx, ref newIdxPtr );
						node->FirstTri = start;
						node->TriCount = mergedCount;
						node->Left = node->Right = 0;
						// pop new task
						if ( stackPtr == 0 )
						{
							break;
						}
						nodeIdx = stack[ --stackPtr ];
					}
					else // recurse
					{
						nodeIdx = node->Left;
						stack[ stackPtr++ ] = node->Right;
					}
				}
			}
			// cleanup
			bvh.Free( subtreeTriCount );
			// Unless a previous MergeLeafs already handed us one, this array belongs to the source
			// Bvh, which the C++ frees here as well; that Bvh keeps a dangling pointer until
			// Bvh.ConvertFrom( ref BvhVerbose, bool ) copies the new array back into it.
			bvh.Free( bvh.PrimIdx );
			bvh.PrimIdx = newIdx;
			bvh.OwnsPrimIdx = true;
			bvh.MayHaveHoles = true; // all over the place, in fact
		}

		/// <summary>
		/// Port of BVH_Verbose::CountSubtreeTris. Determine for each node in the tree the number of
		/// primitives stored in that subtree. Helper function for MergeLeafs.
		/// </summary>
		private static uint CountSubtreeTris( BvhVerboseNode* bvhNode, uint nodeIdx, uint* counters )
		{
			BvhVerboseNode* node = bvhNode + nodeIdx;
			uint result = node->TriCount;
			if ( result == 0 )
			{
				result = CountSubtreeTris( bvhNode, node->Left, counters ) + CountSubtreeTris( bvhNode, node->Right, counters );
			}
			counters[ nodeIdx ] = result;
			return result;
		}

		/// <summary>
		/// Port of BVH_Verbose::MergeSubtree. Write the triangle indices stored in a subtree to a
		/// continuous slice in the 'newIdx' array. Helper function for MergeLeafs.
		/// </summary>
		private static void MergeSubtree( BvhVerboseNode* bvhNode, uint nodeIdx, uint* primIdx, uint* newIdx, ref uint newIdxPtr )
		{
			BvhVerboseNode* node = bvhNode + nodeIdx;
			if ( node->IsLeaf )
			{
				UnsafeUtility.MemCpy( newIdx + newIdxPtr, primIdx + node->FirstTri, ( long )node->TriCount * 4 );
				newIdxPtr += node->TriCount;
				return;
			}
			MergeSubtree( bvhNode, node->Left, primIdx, newIdx, ref newIdxPtr );
			MergeSubtree( bvhNode, node->Right, primIdx, newIdx, ref newIdxPtr );
		}

		/// <summary>
		/// Port of the MSVC C runtime's rand(): a 32-bit LCG seeded to 1, of which the middle 15
		/// bits are returned. <see cref="RandMax"/> is its RAND_MAX.
		/// </summary>
		private static int Rand( ref BvhVerbose bvh )
		{
			bvh.RandomState = ( bvh.RandomState * 214013u ) + 2531011u;
			return ( int )( ( bvh.RandomState >> 16 ) & 0x7fffu );
		}

		/// <summary>RAND_MAX of the MSVC C runtime.</summary>
		private const float RandMax = 32767f;

		/// <summary>Port of BVH_Verbose::Optimize, both the deterministic and the stochastic path.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void Optimize( ref BvhVerbose bvh, uint iterations, [MarshalAs( UnmanagedType.U1 )] bool extreme,
			[MarshalAs( UnmanagedType.U1 )] bool stochastic )
		{
			BvhVerboseNode* bvhNode = bvh.Nodes;
			// allocate array for sorting; size is upper-bound.
			SortItem* sortList = ( SortItem* )bvh.Alloc( ( long )bvh.UsedNodes * sizeof( SortItem ) );
			// The C++ puts these two on the stack; 32KB plus 4KB is too much for one here, and
			// FindBestNewPosition would re-reserve its buffer on every call.
			SortTask* stack = ( SortTask* )bvh.Alloc( SortStackSize * sizeof( SortTask ) );
			FindTask* task = ( FindTask* )bvh.Alloc( FindTaskCount * sizeof( FindTask ) );
			BvhVerboseNode* bckp = ( BvhVerboseNode* )bvh.Alloc( 5 * sizeof( BvhVerboseNode ) );
			// optimize by reinserting subtrees with a high cost - Section 3.4 of the paper.
			for ( uint i = 0; i < iterations; i++ )
			{
				// calculate combined cost for all nodes
				uint interiorNodes = 0;
				for ( uint j = 2; j < bvh.UsedNodes; j++ )
				{
					BvhVerboseNode* node = bvhNode + j;
					if ( node->IsLeaf )
					{
						continue;
					}
					if ( node->Parent == 0 )
					{
						continue;
					}
					if ( bvhNode[ node->Parent ].Parent == 0 )
					{
						continue;
					}
					float A = node->SurfaceArea, AL = bvhNode[ node->Left ].SurfaceArea, AR = bvhNode[ node->Right ].SurfaceArea;
					float Mmin = A / math.min( 1e-10f, math.min( AL, AR ) );
					float Msum = A / math.min( 1e-10f, 0.5f * ( AL + AR ) );
					float Mcomb = A * Msum * Mmin;
					sortList[ interiorNodes ].Idx = j;
					sortList[ interiorNodes++ ].Cost = Mcomb;
				}
				// last couple of iterations we will process more nodes.
				float portion = stochastic ? 0.5f : ( extreme ? ( 0.01f + ( ( 0.6f * ( float )i ) / ( float )iterations ) ) : 0.01f );
				int limit = ( int )( uint )( portion * ( float )interiorNodes );
				int step = math.max( 1, ( int )( portion / 0.02f ) );
				// sort list - partial quick sort.
				int pivot, first = 0, last = ( int )interiorNodes - 1, stackPtr = 0;
				while ( true )
				{
					if ( first >= last )
					{
						if ( stackPtr == 0 )
						{
							break;
						}
						first = stack[ --stackPtr ].First;
						last = stack[ stackPtr ].Last;
						continue;
					}
					pivot = first;
					SortItem t, e = sortList[ first ];
					for ( int j = first + 1; j <= last; j++ )
					{
						if ( sortList[ j ].Cost > e.Cost )
						{
							t = sortList[ j ];
							sortList[ j ] = sortList[ ++pivot ];
							sortList[ pivot ] = t;
						}
					}
					t = sortList[ pivot ];
					sortList[ pivot ] = sortList[ first ];
					sortList[ first ] = t;
					if ( pivot < limit )
					{
						stack[ stackPtr ].First = pivot + 1;
						stack[ stackPtr++ ].Last = last;
					}
					last = pivot - 1;
				}
				// reinsert selected nodes
				int start = 0;
				if ( stochastic )
				{
					float r = ( float )Rand( ref bvh ) / RandMax;
					r = math.max( 0f, ( r * 1.2f ) - 0.3f ); // 0 .. 0.9f
					start = ( int )( ( float )limit * r );
				}
				// Dead in v1.8.0: only the removed 'finishing touch' pass ever set this. Kept as
				// the C++ has it so the loop below stays a literal transcription.
				bool finishingTouch = false;
				for ( int j = start; j < limit; j += stochastic ? ( ( Rand( ref bvh ) & 63 ) + 1 ) : step )
				{
					// prepare change
					uint Nid = sortList[ j ].Idx;
					BvhVerboseNode* N = bvhNode + Nid;
					if ( N->Parent == 0 )
					{
						continue;
					}
					uint Pid = N->Parent;
					BvhVerboseNode* P = bvhNode + Pid;
					if ( P->Parent == 0 )
					{
						continue;
					}
					uint X1 = P->Parent, X2 = P->Left == Nid ? P->Right : P->Left;
					// compute SAH before change
					float sahBefore = SahCostUp( bvhNode, Nid );
					// execute change
					bckp[ 0 ] = bvhNode[ X1 ];
					if ( bvhNode[ X1 ].Left == Pid )
					{
						bvhNode[ X1 ].Left = X2;
					}
					else // verbose[X1].right == Pid
					{
						bvhNode[ X1 ].Right = X2;
					}
					uint p2 = bvhNode[ X2 ].Parent;
					bvhNode[ X2 ].Parent = X1;
					uint Lid = N->Left, Rid = N->Right;
					RefitUp( bvhNode, X2 );
					// ReinsertNode( L, Nid ); ReinsertNode( R, Pid );
					uint Xbest1 = FindBestNewPosition( bvhNode, Lid, task ), XA = bvhNode[ Xbest1 ].Parent;
					sahBefore += SahCostUp( bvhNode, Xbest1 );
					bckp[ 1 ] = bvhNode[ Nid ];
					N->Left = Xbest1;
					N->Right = Lid;
					N->Parent = XA;
					bckp[ 2 ] = bvhNode[ XA ];
					if ( bvhNode[ XA ].Left == Xbest1 )
					{
						bvhNode[ XA ].Left = Nid;
					}
					else
					{
						bvhNode[ XA ].Right = Nid;
					}
					uint p3 = bvhNode[ Xbest1 ].Parent, p4 = bvhNode[ Lid ].Parent;
					bvhNode[ Xbest1 ].Parent = Nid;
					bvhNode[ Lid ].Parent = Nid;
					RefitUp( bvhNode, Nid );
					uint Xbest2 = FindBestNewPosition( bvhNode, Rid, task ), XB = bvhNode[ Xbest2 ].Parent;
					sahBefore += SahCostUp( bvhNode, Xbest2 );
					bckp[ 3 ] = bvhNode[ Pid ];
					P->Left = Xbest2;
					P->Right = Rid;
					P->Parent = XB;
					bckp[ 4 ] = bvhNode[ XB ];
					if ( bvhNode[ XB ].Left == Xbest2 )
					{
						bvhNode[ XB ].Left = Pid;
					}
					else
					{
						bvhNode[ XB ].Right = Pid;
					}
					uint p1 = bvhNode[ Xbest2 ].Parent, p0 = bvhNode[ Rid ].Parent;
					bvhNode[ Xbest2 ].Parent = Pid;
					bvhNode[ Rid ].Parent = Pid;
					RefitUp( bvhNode, Pid );
					// compute SAH after change
					float sahAfter = SahCostUp( bvhNode, X1 ) + SahCostUp( bvhNode, Nid ) + SahCostUp( bvhNode, Pid );
					if ( finishingTouch && ( ( sahBefore / sahAfter ) > 1.01f ) )
					{
						break;
					}
					if ( sahAfter < sahBefore )
					{
						continue;
					}
					// undo change, mind the order.
					bvhNode[ Rid ].Parent = p0;
					bvhNode[ Xbest2 ].Parent = p1;
					bvhNode[ XB ] = bckp[ 4 ];
					bvhNode[ Pid ] = bckp[ 3 ];
					bvhNode[ Lid ].Parent = p4;
					bvhNode[ Xbest1 ].Parent = p3;
					bvhNode[ XA ] = bckp[ 2 ];
					bvhNode[ Nid ] = bckp[ 1 ];
					bvhNode[ X2 ].Parent = p2;
					bvhNode[ X1 ] = bckp[ 0 ];
					RefitUp( bvhNode, XB );
					RefitUp( bvhNode, XA );
					RefitUp( bvhNode, Nid );
				}
				RefitRec( ref bvh, 0, true );
			}
			bvh.Free( bckp );
			bvh.Free( task );
			bvh.Free( stack );
			bvh.Free( sortList );
		}

		/// <summary>
		/// Port of BVH_Verbose::RefitUp: Update bounding boxes of ancestors of the specified node.
		/// </summary>
		private static void RefitUp( BvhVerboseNode* bvhNode, uint nodeIdx )
		{
			while ( true )
			{
				BvhVerboseNode* node = bvhNode + nodeIdx;
				if ( !node->IsLeaf )
				{
					BvhVerboseNode* left = bvhNode + node->Left;
					BvhVerboseNode* right = bvhNode + node->Right;
					node->AabbMin = math.min( left->AabbMin, right->AabbMin );
					node->AabbMax = math.max( left->AabbMax, right->AabbMax );
				}
				if ( nodeIdx == 0 )
				{
					break;
				}
				nodeIdx = node->Parent;
			}
		}

		/// <summary>
		/// Port of BVH_Verbose::SAHCostUp: Calculate the SAH cost of a node and its ancestry.
		/// </summary>
		private static float SahCostUp( BvhVerboseNode* bvhNode, uint nodeIdx )
		{
			float sum = 0f;
			while ( nodeIdx != 0xffffffff )
			{
				BvhVerboseNode* node = bvhNode + nodeIdx;
				sum += BvhMath.SurfaceArea( node->AabbMin, node->AabbMax );
				nodeIdx = node->Parent;
			}
			return sum;
		}

		/// <summary>
		/// Port of BVH_Verbose::FindBestNewPosition. Part of "Fast Insertion-Based Optimization of
		/// Bounding Volume Hierarchies". K.I.S.S. version with brute-force array search.
		/// </summary>
		private static uint FindBestNewPosition( BvhVerboseNode* bvhNode, uint Lid, FindTask* task )
		{
			float Cbest = BvhConstants.Far;
			int tasks = 1; // doesn't exceed 70 for Crytek Sponza
			uint Xbest = 0;
			BvhVerboseNode* L = bvhNode + Lid;
			// reinsert L into BVH
			task[ 0 ].Node = 0;
			task[ 0 ].Ci = 0f;
			while ( tasks > 0 )
			{
				// 'pop' task with smallest Ci
				int bestTask = 0;
				float minCi = task[ 0 ].Ci; // tnx Brian
				for ( int j = 1; j < tasks; j++ )
				{
					if ( task[ j ].Ci < minCi )
					{
						minCi = task[ j ].Ci;
						bestTask = j;
					}
				}
				uint Xid = task[ bestTask ].Node;
				float CiLX = task[ bestTask ].Ci;
				if ( --tasks > 0 )
				{
					task[ bestTask ] = task[ tasks ];
				}
				// execute task
				BvhVerboseNode* X = bvhNode + Xid;
				if ( ( CiLX + L->SurfaceArea ) >= Cbest )
				{
					break;
				}
				float CdLX = BvhMath.SurfaceArea( math.min( L->AabbMin, X->AabbMin ), math.max( L->AabbMax, X->AabbMax ) );
				float CLX = CiLX + CdLX;
				if ( CLX < Cbest && Xid != 0 )
				{
					Cbest = CLX;
					Xbest = Xid;
				}
				float Ci = CLX - X->SurfaceArea;
				if ( ( Ci + L->SurfaceArea ) >= Cbest || X->IsLeaf )
				{
					continue;
				}
				task[ tasks ].Node = X->Left;
				task[ tasks++ ].Ci = Ci;
				task[ tasks ].Node = X->Right;
				task[ tasks++ ].Ci = Ci;
			}
			return Xbest;
		}
	}
}
