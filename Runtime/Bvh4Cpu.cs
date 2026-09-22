using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Port of BVH4_CPU::BVHNode: a 4-wide interior node in SoA form, 128 bytes, i.e. two
	/// 64-byte cache lines. Child4 holds, per lane, either a block index of the child node,
	/// a block index of a quad-triangle leaf with Bvh4Cpu.LeafBit set, or Bvh4Cpu.EmptyBit for
	/// a padding lane. Perm4 packs eight 2-bit child orderings, one per ray direction octant.
	/// </summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct Bvh4CpuNode
	{
		public float4 XMin4;
		public float4 XMax4;
		public float4 YMin4;
		public float4 YMax4;
		public float4 ZMin4;
		public float4 ZMax4;
		public uint4 Child4;
		public uint4 Perm4;
	}

	/// <summary>
	/// Port of BVHTri4Leaf: storage for up to four triangles in SoA layout. 192 bytes, i.e.
	/// three 64-byte cache lines; the last 32 bytes are the C++ dummy0/dummy1 padding.
	/// </summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct Bvh4CpuTri4Leaf
	{
		public float4 V0x4;
		public float4 V0y4;
		public float4 V0z4;
		public float4 E1x4;
		public float4 E1y4;
		public float4 E1z4;
		public float4 E2x4;
		public float4 E2y4;
		public float4 E2z4;
		public uint4 PrimIdx;
		public float4 Dummy0;
		public float4 Dummy1;
	}

	/// <summary>
	/// Port of tinybvh's BVH4_CPU class: a 4-wide BVH in the 'WiVe' layout, laid out as a flat
	/// blob of 64-byte cache-line blocks holding interleaved 128-byte interior nodes and
	/// 192-byte quad-triangle leaves. Traversal lives in Bvh4Cpu.Intersect.cs.
	/// </summary>
	public unsafe partial struct Bvh4Cpu : IDisposable
	{
		/// <summary>Port of BVH4_CPU::EMPTY_BIT: marks an unused child lane.</summary>
		public const uint EmptyBit = 1u << 31;
		/// <summary>Port of BVH4_CPU::LEAF_BIT: marks a child lane that points at a quad-triangle leaf.</summary>
		public const uint LeafBit = 1u << 30;
		/// <summary>Mask the traversal applies to a child entry to recover the leaf's block index.</summary>
		internal const uint LeafOffsetMask = 0x1fffffff;
		/// <summary>Allocation unit: one cache line, the C++ BVH4_CPU::CacheLine.</summary>
		internal const int BlockSize = 64;

		/// <summary>
		/// The base binary BVH. Owned when this layout was built through <see cref="Build"/>,
		/// otherwise a shallow copy of Mbvh4.Source that shares - and does not own - its memory.
		/// </summary>
		public Bvh Base;
		/// <summary>
		/// The intermediate 4-wide BVH. Owned when this layout was built through <see cref="Build"/>,
		/// otherwise a shallow copy of the Mbvh passed to <see cref="ConvertFrom"/>.
		/// </summary>
		public Mbvh Mbvh4;

		/// <summary>Node and leaf data, in 64-byte blocks (owned).</summary>
		[NativeDisableUnsafePtrRestriction] public byte* Data;
		public uint UsedBlocks;
		public uint AllocatedBlocks;

		// Properties copied from the source by CopyBasePropertiesFrom.
		public uint TriCount;
		public uint IdxCount;
		public float3 AabbMin;
		public float3 AabbMax;
		[MarshalAs( UnmanagedType.U1 )] public bool Refittable;
		[MarshalAs( UnmanagedType.U1 )] public bool MayHaveHoles;
		[MarshalAs( UnmanagedType.U1 )] public bool BvhOverAabbs;
		[MarshalAs( UnmanagedType.U1 )] public bool BvhOverIndices;

		/// <summary>Port of BVH4_CPU::ownBVH4, extended to the base BVH: set by <see cref="Build"/>.</summary>
		[MarshalAs( UnmanagedType.U1 )] public bool OwnsSource;

		/// <summary>Forwarded to Base before the base build; the C++ BVH4_CPU::BuildHQ equivalent.</summary>
		[MarshalAs( UnmanagedType.U1 )] public bool UseSpatialSplits;

		/// <summary>Forwarded to Base before the base build; builds with the job system for large inputs.</summary>
		[MarshalAs( UnmanagedType.U1 )] public bool UseThreadedBuild;

		// SAH cost parameters. The C++ constructor sets c_int = 2 for this layout.
		public float TraversalCost;
		public float IntersectionCost;

		public Allocator Allocator;

		public static Bvh4Cpu Create( Allocator allocator )
		{
			return new Bvh4Cpu
			{
				TraversalCost = BvhConstants.DefaultTraversalCost,
				IntersectionCost = 2f,
				Allocator = allocator
			};
		}

		public bool IsCreated => Allocator > Allocator.None;

		public void Dispose()
		{
			Free( Data );
			Data = null;
			AllocatedBlocks = 0;
			UsedBlocks = 0;
			TriCount = 0;
			IdxCount = 0;
			if ( OwnsSource )
			{
				Mbvh4.Dispose();
				Base.Dispose();
				OwnsSource = false;
			}
		}

		/// <summary>
		/// Port of BVH4_CPU::Build: propagates the SAH costs of this layout into a base BVH and a
		/// 4-wide MBVH of its own, builds the base and converts. Both are owned and are freed by
		/// <see cref="Dispose"/>.
		/// </summary>
		public void Build( NativeArray<float4> vertices, uint triCount )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "Bvh4Cpu.Build( .. ), bvh4Cpu was not created." );
			}
			if ( !Mbvh4.IsCreated )
			{
				Mbvh4 = Mbvh.Create( 4, Allocator );
			}
			if ( !Mbvh4.Source.IsCreated )
			{
				Mbvh4.Source = Bvh.Create( Allocator );
			}
			// propagate settings for this layout to the underlying layout
			Mbvh4.TraversalCost = TraversalCost;
			Mbvh4.IntersectionCost = IntersectionCost;
			Mbvh4.Source.TraversalCost = TraversalCost;
			Mbvh4.Source.IntersectionCost = IntersectionCost;
			// build underlying layout
			Mbvh4.Source.UseSpatialSplits = UseSpatialSplits;
			Mbvh4.Source.UseThreadedBuild = UseThreadedBuild;
			Mbvh4.Source.Build( vertices, triCount );
			// convert to BVH4_CPU layout
			Mbvh mbvh4 = Mbvh4;
			ConvertFrom( ref mbvh4 );
			// the C++ compares &original against &bvh4 to decide this; here Build simply claims
			// ownership of the two structures it just created, after ConvertFrom cleared the flag.
			OwnsSource = true;
		}

		/// <summary>
		/// Port of BVH4_CPU::ConvertFrom( MBVH&lt;4&gt;&amp; ). Takes a shallow copy of the source,
		/// reshapes the base BVH it shares with CombineLeafs( 4 ) and SplitLeafs( 4 ), re-converts
		/// the 4-wide BVH from that reshaped base and emits the node/leaf blocks.
		/// </summary>
		public void ConvertFrom( ref Mbvh original )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "Bvh4Cpu.ConvertFrom( .. ), bvh4Cpu was not created." );
			}
			if ( original.M != 4 )
			{
				throw new ArgumentException( "Bvh4Cpu.ConvertFrom( .. ), source is not a 4-wide MBVH.", nameof( original ) );
			}
			if ( original.Source.Nodes == null || original.Source.UsedNodes == 0 )
			{
				throw new ArgumentException( "Bvh4Cpu.ConvertFrom( .. ), source has no base bvh.", nameof( original ) );
			}
			// bvh isn't ours; don't delete in Dispose. Build re-claims ownership afterwards.
			OwnsSource = false;
			// get a copy of the input bvh4; this shares the node and index memory, exactly like
			// the C++ 'bvh4 = original', so the reshaping below is visible through both.
			Mbvh4 = original;
			// prepare input bvh4
			uint firstIdx = 0;
			Mbvh4.Source.CombineLeafs( 4, ref firstIdx, 0 );
			Mbvh4.Source.SplitLeafs( 4 );
			// The C++ passes bvh4.bvh to bvh4.ConvertFrom, which assigns it back to itself; a local
			// copy of the same value keeps that behaviour without aliasing 'ref this' in Burst.
			Bvh reshaped = Mbvh4.Source;
			Mbvh4.ConvertFrom( ref reshaped, true );
			Base = Mbvh4.Source;
			Mbvh source = Mbvh4;
			Bvh4CpuConverter.ConvertFrom( ref this, ref source );
		}

		/// <summary>
		/// Port of BVH4_CPU::Optimize: optimizes the underlying MBVH&lt;4&gt; - which optimizes the
		/// base BVH and re-converts - and then re-converts this layout from it.
		/// Unavailable on a tree loaded from a file.
		/// </summary>
		public void Optimize( uint iterations = 25, bool extreme = false )
		{
			if ( Mbvh4.Source.Nodes == null )
			{
				throw new InvalidOperationException( "Bvh4Cpu.Optimize( .. ), this tree was loaded from a file and has no source BVH." );
			}
			// ConvertFrom clears the flag; Optimize does not change who owns the source structures.
			bool owns = OwnsSource;
			Mbvh4.Optimize( iterations, extreme );
			Mbvh mbvh4 = Mbvh4;
			ConvertFrom( ref mbvh4 );
			OwnsSource = owns;
		}

		/// <summary>
		/// Port of BVH4_CPU::Refit: refits the underlying tree over the current vertex positions and
		/// re-converts this layout from it. Throws when the base is not refittable, e.g. after a
		/// spatial-split build, with the exception <see cref="Bvh.Refit"/> raises.
		/// Deviation: the C++ refits the MBVH&lt;4&gt;, but ConvertFrom re-derives the 4-wide nodes
		/// from the base BVH right after, which discards those bounds again; the base BVH is what
		/// has to be refit. CombineLeafs / SplitLeafs leave Refittable and MayHaveHoles untouched,
		/// exactly as in the C++, so Bvh.Refit accepts the reshaped base the converter produced.
		/// Unavailable on a tree loaded from a file.
		/// </summary>
		public void Refit()
		{
			if ( Mbvh4.Source.Nodes == null )
			{
				throw new InvalidOperationException( "Bvh4Cpu.Refit( .. ), this tree was loaded from a file and has no source BVH." );
			}
			// ConvertFrom clears the flag; Refit does not change who owns the source structures.
			bool owns = OwnsSource;
			Mbvh4.Source.Refit();
			Mbvh mbvh4 = Mbvh4;
			ConvertFrom( ref mbvh4 );
			OwnsSource = owns;
		}

		/// <summary>
		/// Port of BVH4_CPU::SAHCost, which forwards to the underlying MBVH&lt;4&gt;.
		/// Unavailable on a tree loaded from a file.
		/// </summary>
		public float SahCost( uint nodeIdx = 0 )
		{
			if ( Mbvh4.Source.Nodes == null )
			{
				throw new InvalidOperationException( "Bvh4Cpu.SahCost( .. ), this tree was loaded from a file and has no source BVH." );
			}
			return Mbvh4.SahCost( nodeIdx );
		}

		/// <summary>Ensures the blob can hold count 64-byte blocks. Contents are not preserved when it grows.</summary>
		internal void AllocateBlocks( uint count )
		{
			if ( AllocatedBlocks < count )
			{
				Free( Data );
				Data = ( byte* )Alloc( ( long )count * BlockSize );
				AllocatedBlocks = count;
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
	/// Burst-compiled implementation of the BVH4_CPU conversion. Direct calls must be synchronous,
	/// otherwise editor tests silently run the Mono fallback.
	/// </summary>
	[BurstCompile]
	internal static unsafe class Bvh4CpuConverter
	{
		/// <summary>Stack depth of the conversion walk. The C++ uses 256; see the note in ConvertFrom.</summary>
		private const int StackSize = 1024;

		/// <summary>Port of the BVH4_CPU::ConvertFrom block-emitting loop.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void ConvertFrom( ref Bvh4Cpu bvh4Cpu, ref Mbvh bvh4 )
		{
			// allocate if needed
			uint nodesNeeded = bvh4.UsedNodes, leafsNeeded = LeafCount( bvh4.Nodes );
			uint blocksNeeded = nodesNeeded * ( 128 / Bvh4Cpu.BlockSize ); // here, block = cacheline.
			blocksNeeded += leafsNeeded * ( 192 / Bvh4Cpu.BlockSize );
			bvh4Cpu.AllocateBlocks( blocksNeeded );
			byte* data = bvh4Cpu.Data;
			// Deviation: the C++ clears each interior node but leaves BVHTri4Leaf::dummy0/dummy1
			// (the last 32 bytes of every leaf) at whatever malloc64 returned. Clear the whole
			// blob so the emitted bytes are deterministic.
			UnsafeUtility.MemClear( data, ( long )blocksNeeded * Bvh4Cpu.BlockSize );
			CopyBasePropertiesFrom( ref bvh4Cpu, ref bvh4 );
			// start conversion
			uint* stack = stackalloc uint[ StackSize ];
			// The C++ 'union { float dist[4]; uint32_t idist[4]; }': the sorting network swaps the
			// float view, so the lane indices packed into the mantissa move with the distances.
			// Stored as uint here to avoid type punning through a pointer under Burst.
			uint* idist = stackalloc uint[ 4 ];
			uint newBlockPtr = 0, nodeIdx = 0, stackPtr = 0;
			while ( true )
			{
				MbvhNode* orig = bvh4.Nodes + nodeIdx;
				Bvh4CpuNode* newNode = ( Bvh4CpuNode* )( data + ( newBlockPtr * Bvh4Cpu.BlockSize ) );
				newBlockPtr += 128 / Bvh4Cpu.BlockSize;
				UnsafeUtility.MemClear( newNode, sizeof( Bvh4CpuNode ) );
				float* xmin = ( float* )&newNode->XMin4, xmax = ( float* )&newNode->XMax4;
				float* ymin = ( float* )&newNode->YMin4, ymax = ( float* )&newNode->YMax4;
				float* zmin = ( float* )&newNode->ZMin4, zmax = ( float* )&newNode->ZMax4;
				uint* child4 = ( uint* )&newNode->Child4;
				uint* perm4 = ( uint* )&newNode->Perm4;
				// calculate the permutation offsets for the node
				for ( int q = 0; q < 8; q++ )
				{
					float dx = ( q & 1 ) != 0 ? 1.0f : -1.0f;
					float dy = ( q & 2 ) != 0 ? 1.0f : -1.0f;
					float dz = ( q & 4 ) != 0 ? 1.0f : -1.0f;
					for ( int i = 0; i < 4; i++ )
					{
						if ( orig->Child[ i ] == 0 )
						{
							idist[ i ] = ( math.asuint( BvhConstants.Far ) & 0xfffffffc ) + ( uint )i;
						}
						else
						{
							MbvhNode* c = bvh4.Nodes + orig->Child[ i ];
							float px = ( q & 1 ) != 0 ? c->AabbMin.x : c->AabbMax.x;
							float py = ( q & 2 ) != 0 ? c->AabbMin.y : c->AabbMax.y;
							float pz = ( q & 4 ) != 0 ? c->AabbMin.z : c->AabbMax.z;
							float d = ( ( dx * px ) + ( dy * py ) ) + ( dz * pz );
							idist[ i ] = ( math.asuint( d ) & 0xfffffff8 ) + ( uint )i;
						}
					}
					// apply sorting network - https://bertdobbelaere.github.io/sorting_networks.html#N4L5D3
					Sort( idist, 0, 2 );
					Sort( idist, 1, 3 );
					Sort( idist, 0, 1 );
					Sort( idist, 2, 3 );
					Sort( idist, 1, 2 );
					for ( int i = 0; i < 4; i++ )
					{
						perm4[ i ] += ( idist[ i ] & 3 ) << ( q * 2 );
					}
				}
				// fill remaining fields
				int cidx = 0;
				for ( int i = 0; i < 4; i++ )
				{
					if ( orig->Child[ i ] != 0 )
					{
						MbvhNode* child = bvh4.Nodes + orig->Child[ i ];
						xmin[ cidx ] = child->AabbMin.x;
						xmax[ cidx ] = child->AabbMax.x;
						ymin[ cidx ] = child->AabbMin.y;
						ymax[ cidx ] = child->AabbMax.y;
						zmin[ cidx ] = child->AabbMin.z;
						zmax[ cidx ] = child->AabbMax.z;
						if ( child->IsLeaf )
						{
							// emit leaf node: group of up to 4 triangles in SoA format.
							child4[ cidx ] = newBlockPtr + Bvh4Cpu.LeafBit;
							Bvh4CpuTri4Leaf* leaf = ( Bvh4CpuTri4Leaf* )( data + ( newBlockPtr * Bvh4Cpu.BlockSize ) );
							newBlockPtr += 192 / Bvh4Cpu.BlockSize;
							for ( uint l = 0; l < 4; l++ )
							{
								uint primIdx = bvh4.Source.PrimIdx[ child->FirstTri + math.min( l, child->TriCount - 1u ) ];
								bvh4.Source.GetPrimIndices( primIdx, out uint i0, out uint i1, out uint i2 );
								float4 v0 = bvh4.Source.Vertex( i0 );
								float4 e1 = bvh4.Source.Vertex( i1 ) - v0;
								float4 e2 = bvh4.Source.Vertex( i2 ) - v0;
								SetLeafData( leaf, v0.xyz, e1.xyz, e2.xyz, primIdx, l );
							}
						}
						else
						{
							uint* slot = child4 + cidx;
							stack[ stackPtr++ ] = ( uint )( slot - ( uint* )data );
							stack[ stackPtr++ ] = orig->Child[ i ];
						}
						cidx++;
					}
				}
				for ( ; cidx < 4; cidx++ )
				{
					xmin[ cidx ] = 1e30f;
					xmax[ cidx ] = 1.00001e30f;
					ymin[ cidx ] = 1e30f;
					ymax[ cidx ] = 1.00001e30f;
					zmin[ cidx ] = 1e30f;
					zmax[ cidx ] = 1.00001e30f;
					child4[ cidx ] |= Bvh4Cpu.EmptyBit;
				}
				// pop next task
				if ( stackPtr == 0 )
				{
					break;
				}
				nodeIdx = stack[ --stackPtr ];
				uint offset = stack[ --stackPtr ];
				( ( uint* )data )[ offset ] = newBlockPtr;
			}
			bvh4Cpu.UsedBlocks = newBlockPtr;
		}

		/// <summary>
		/// Port of the SORT macro: a descending compare-exchange on the float view of the
		/// dist/idist union, so the packed lane indices travel with the distances.
		/// </summary>
		internal static void Sort( uint* idist, int a, int b )
		{
			if ( math.asfloat( idist[ a ] ) < math.asfloat( idist[ b ] ) )
			{
				uint h = idist[ a ];
				idist[ a ] = idist[ b ];
				idist[ b ] = h;
			}
		}

		/// <summary>Port of BVHTri4Leaf::SetData.</summary>
		internal static void SetLeafData( Bvh4CpuTri4Leaf* leaf, float3 v0, float3 e1, float3 e2, uint pidx, uint slot )
		{
			( ( float* )&leaf->V0x4 )[ slot ] = v0.x;
			( ( float* )&leaf->V0y4 )[ slot ] = v0.y;
			( ( float* )&leaf->V0z4 )[ slot ] = v0.z;
			( ( float* )&leaf->E1x4 )[ slot ] = e1.x;
			( ( float* )&leaf->E1y4 )[ slot ] = e1.y;
			( ( float* )&leaf->E1z4 )[ slot ] = e1.z;
			( ( float* )&leaf->E2x4 )[ slot ] = e2.x;
			( ( float* )&leaf->E2y4 )[ slot ] = e2.y;
			( ( float* )&leaf->E2z4 )[ slot ] = e2.z;
			( ( uint* )&leaf->PrimIdx )[ slot ] = pidx;
		}

		/// <summary>
		/// Iterative form of MBVH&lt;M&gt;::LeafCount over the whole tree. The managed Mbvh.LeafCount
		/// is recursive, which this Burst-compiled path avoids.
		/// </summary>
		internal static uint LeafCount( MbvhNode* nodes )
		{
			uint* stack = stackalloc uint[ StackSize ];
			uint stackPtr = 0, nodeIdx = 0, count = 0;
			while ( true )
			{
				MbvhNode* node = nodes + nodeIdx;
				if ( node->IsLeaf )
				{
					count++;
				}
				else
				{
					for ( uint i = 0; i < node->ChildCount; i++ )
					{
						stack[ stackPtr++ ] = node->Child[ i ];
					}
				}
				if ( stackPtr == 0 )
				{
					break;
				}
				nodeIdx = stack[ --stackPtr ];
			}
			return count;
		}

		/// <summary>Port of BVHBase::CopyBasePropertiesFrom for the Mbvh -&gt; Bvh4Cpu direction.</summary>
		private static void CopyBasePropertiesFrom( ref Bvh4Cpu bvh4Cpu, ref Mbvh original )
		{
			bvh4Cpu.Refittable = original.Refittable;
			bvh4Cpu.MayHaveHoles = original.MayHaveHoles;
			bvh4Cpu.BvhOverAabbs = original.BvhOverAabbs;
			bvh4Cpu.BvhOverIndices = original.BvhOverIndices;
			bvh4Cpu.TriCount = original.TriCount;
			bvh4Cpu.IdxCount = original.IdxCount;
			bvh4Cpu.AabbMin = original.AabbMin;
			bvh4Cpu.AabbMax = original.AabbMax;
		}
	}
}
