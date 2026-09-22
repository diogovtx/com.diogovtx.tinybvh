using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Port of BVH8_CPU::BVHNode: an 8-wide interior node in SoA form, 256 bytes, i.e. four
	/// 64-byte cache lines. Child8 holds, per lane, either a block index of the child node,
	/// a block index of a quad-triangle leaf with Bvh8Cpu.LeafBit set, or Bvh8Cpu.EmptyBit for
	/// a padding lane. Perm8 packs eight 3-bit child orderings, one per ray direction octant.
	/// The C++ types the six bound arrays as SIMDVEC8 and the two index arrays as SIMDIVEC8;
	/// under TINYBVH_NO_SIMD those are plain 8-element structs, so the byte layout is the same
	/// as the fixed buffers used here.
	/// </summary>
	[StructLayout( LayoutKind.Sequential )]
	public unsafe struct Bvh8CpuNode
	{
		public fixed float XMin8[ 8 ];
		public fixed float XMax8[ 8 ];
		public fixed float YMin8[ 8 ];
		public fixed float YMax8[ 8 ];
		public fixed float ZMin8[ 8 ];
		public fixed float ZMax8[ 8 ];
		public fixed uint Child8[ 8 ];
		public fixed uint Perm8[ 8 ];
	}

	/// <summary>
	/// Port of tinybvh's BVH8_CPU class: an 8-wide BVH for AVX2 traversal, laid out as a flat
	/// blob of 64-byte cache-line blocks holding interleaved 256-byte interior nodes and
	/// 192-byte quad-triangle leaves. The leaves are the same BVHTri4Leaf the 4-wide layout
	/// uses, so <see cref="Bvh4CpuTri4Leaf"/> is reused. Traversal lives in Bvh8Cpu.Intersect.cs.
	/// </summary>
	public unsafe partial struct Bvh8Cpu : IDisposable
	{
		/// <summary>Port of BVH8_CPU::EMPTY_BIT: marks an unused child lane.</summary>
		public const uint EmptyBit = 1u << 31;
		/// <summary>Port of BVH8_CPU::LEAF_BIT: marks a child lane that points at a quad-triangle leaf.</summary>
		public const uint LeafBit = 1u << 30;
		/// <summary>Mask the traversal applies to a child entry to recover the leaf's block index.</summary>
		internal const uint LeafOffsetMask = 0x1fffffff;
		/// <summary>Allocation unit: one cache line. The C++ BVH8_CPU::CacheLine is two SIMDVEC8.</summary>
		internal const int BlockSize = 64;

		/// <summary>
		/// The base binary BVH. Owned when this layout was built through <see cref="Build"/>,
		/// otherwise a shallow copy of Mbvh8.Source that shares - and does not own - its memory.
		/// </summary>
		public Bvh Base;
		/// <summary>
		/// The intermediate 8-wide BVH. Owned when this layout was built through <see cref="Build"/>,
		/// otherwise a shallow copy of the Mbvh passed to <see cref="ConvertFrom"/>.
		/// </summary>
		public Mbvh Mbvh8;

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

		/// <summary>Port of BVH8_CPU::ownBVH8, extended to the base BVH: set by <see cref="Build"/>.</summary>
		[MarshalAs( UnmanagedType.U1 )] public bool OwnsSource;

		/// <summary>Forwarded to Base before the base build; the C++ BVH8_CPU::BuildHQ equivalent.</summary>
		[MarshalAs( UnmanagedType.U1 )] public bool UseSpatialSplits;

		/// <summary>Forwarded to Base before the base build; builds with the job system for large inputs.</summary>
		[MarshalAs( UnmanagedType.U1 )] public bool UseThreadedBuild;

		// SAH cost parameters. The C++ constructor sets c_int = 2 for this layout.
		public float TraversalCost;
		public float IntersectionCost;

		public Allocator Allocator;

		public static Bvh8Cpu Create( Allocator allocator )
		{
			return new Bvh8Cpu
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
				Mbvh8.Dispose();
				Base.Dispose();
				OwnsSource = false;
			}
		}

		/// <summary>
		/// Port of BVH8_CPU::Build: propagates the SAH costs of this layout into a base BVH and an
		/// 8-wide MBVH of its own, builds and compacts the base and converts. Both are owned and
		/// are freed by <see cref="Dispose"/>.
		/// </summary>
		public void Build( NativeArray<float4> vertices, uint triCount )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "Bvh8Cpu.Build( .. ), bvh8Cpu was not created." );
			}
			if ( !Mbvh8.IsCreated )
			{
				Mbvh8 = Mbvh.Create( 8, Allocator );
			}
			if ( !Mbvh8.Source.IsCreated )
			{
				Mbvh8.Source = Bvh.Create( Allocator );
			}
			// propagate settings for this layout to the underlying layout
			Mbvh8.TraversalCost = TraversalCost;
			Mbvh8.IntersectionCost = IntersectionCost;
			Mbvh8.Source.TraversalCost = TraversalCost;
			Mbvh8.Source.IntersectionCost = IntersectionCost;
			// build underlying layout; unlike BVH4_CPU::Build, the 8-wide one compacts the base.
			Mbvh8.Source.UseSpatialSplits = UseSpatialSplits;
			Mbvh8.Source.UseThreadedBuild = UseThreadedBuild;
			Mbvh8.Source.Build( vertices, triCount );
			Mbvh8.Source.Compact();
			// convert to BVH8_CPU layout
			Mbvh mbvh8 = Mbvh8;
			ConvertFrom( ref mbvh8 );
			// the C++ compares &original against &bvh8 to decide this; here Build simply claims
			// ownership of the two structures it just created, after ConvertFrom cleared the flag.
			OwnsSource = true;
		}

		/// <summary>
		/// Port of BVH8_CPU::ConvertFrom( MBVH&lt;8&gt;&amp; ). Takes a shallow copy of the source,
		/// reshapes the base BVH it shares with CombineLeafs( 4 ) and SplitLeafs( 4 ), re-converts
		/// the 8-wide BVH from that reshaped base and emits the node/leaf blocks.
		/// </summary>
		public void ConvertFrom( ref Mbvh original )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "Bvh8Cpu.ConvertFrom( .. ), bvh8Cpu was not created." );
			}
			if ( original.M != 8 )
			{
				throw new ArgumentException( "Bvh8Cpu.ConvertFrom( .. ), source is not an 8-wide MBVH.", nameof( original ) );
			}
			if ( original.Source.Nodes == null || original.Source.UsedNodes == 0 )
			{
				throw new ArgumentException( "Bvh8Cpu.ConvertFrom( .. ), source has no base bvh.", nameof( original ) );
			}
			// bvh isn't ours; don't delete in Dispose. Build re-claims ownership afterwards.
			OwnsSource = false;
			// get a copy of the input bvh8; this shares the node and index memory, exactly like
			// the C++ 'bvh8 = original', so the reshaping below is visible through both.
			Mbvh8 = original;
			// prepare input bvh8
			uint firstIdx = 0;
			Mbvh8.Source.CombineLeafs( 4, ref firstIdx, 0 );
			Mbvh8.Source.SplitLeafs( 4 );
			// The C++ passes bvh8.bvh to bvh8.ConvertFrom, which assigns it back to itself; a local
			// copy of the same value keeps that behaviour without aliasing 'ref this' in Burst.
			Bvh reshaped = Mbvh8.Source;
			Mbvh8.ConvertFrom( ref reshaped, true );
			Base = Mbvh8.Source;
			Mbvh source = Mbvh8;
			Bvh8CpuConverter.ConvertFrom( ref this, ref source );
		}

		/// <summary>
		/// Port of BVH8_CPU::Optimize: optimizes the underlying MBVH&lt;8&gt; - which optimizes the
		/// base BVH and re-converts - and then re-converts this layout from it.
		/// Unavailable on a tree loaded from a file.
		/// </summary>
		public void Optimize( uint iterations = 25, bool extreme = false )
		{
			if ( Mbvh8.Source.Nodes == null )
			{
				throw new InvalidOperationException( "Bvh8Cpu.Optimize( .. ), this tree was loaded from a file and has no source BVH." );
			}
			// ConvertFrom clears the flag; Optimize does not change who owns the source structures.
			bool owns = OwnsSource;
			Mbvh8.Optimize( iterations, extreme );
			Mbvh mbvh8 = Mbvh8;
			ConvertFrom( ref mbvh8 );
			OwnsSource = owns;
		}

		/// <summary>
		/// Port of BVH8_CPU::Refit: refits the underlying tree over the current vertex positions and
		/// re-converts this layout from it. Throws when the base is not refittable, e.g. after a
		/// spatial-split build, with the exception <see cref="Bvh.Refit"/> raises.
		/// Deviation: the C++ refits the MBVH&lt;8&gt;, but ConvertFrom re-derives the 8-wide nodes
		/// from the base BVH right after, which discards those bounds again; the base BVH is what
		/// has to be refit. CombineLeafs / SplitLeafs leave Refittable and MayHaveHoles untouched,
		/// exactly as in the C++, so Bvh.Refit accepts the reshaped base the converter produced.
		/// Unavailable on a tree loaded from a file.
		/// </summary>
		public void Refit()
		{
			if ( Mbvh8.Source.Nodes == null )
			{
				throw new InvalidOperationException( "Bvh8Cpu.Refit( .. ), this tree was loaded from a file and has no source BVH." );
			}
			// ConvertFrom clears the flag; Refit does not change who owns the source structures.
			bool owns = OwnsSource;
			Mbvh8.Source.Refit();
			Mbvh mbvh8 = Mbvh8;
			ConvertFrom( ref mbvh8 );
			OwnsSource = owns;
		}

		/// <summary>
		/// Port of BVH8_CPU::SAHCost, which forwards to the underlying MBVH&lt;8&gt;.
		/// Unavailable on a tree loaded from a file.
		/// </summary>
		public float SahCost( uint nodeIdx = 0 )
		{
			if ( Mbvh8.Source.Nodes == null )
			{
				throw new InvalidOperationException( "Bvh8Cpu.SahCost( .. ), this tree was loaded from a file and has no source BVH." );
			}
			return Mbvh8.SahCost( nodeIdx );
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
	/// Burst-compiled implementation of the BVH8_CPU conversion. Direct calls must be synchronous,
	/// otherwise editor tests silently run the Mono fallback.
	/// </summary>
	[BurstCompile]
	internal static unsafe class Bvh8CpuConverter
	{
		/// <summary>Stack depth of the conversion walk. The C++ uses 256; see the note in Bvh4Cpu.</summary>
		private const int StackSize = 1024;

		/// <summary>Port of the BVH8_CPU::ConvertFrom block-emitting loop.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void ConvertFrom( ref Bvh8Cpu bvh8Cpu, ref Mbvh bvh8 )
		{
			// allocate if needed
			uint nodesNeeded = bvh8.UsedNodes, leafsNeeded = Bvh4CpuConverter.LeafCount( bvh8.Nodes );
			uint blocksNeeded = nodesNeeded * ( 256 / Bvh8Cpu.BlockSize ); // here, block = cacheline.
			blocksNeeded += leafsNeeded * ( 192 / Bvh8Cpu.BlockSize );
			bvh8Cpu.AllocateBlocks( blocksNeeded );
			byte* data = bvh8Cpu.Data;
			// Deviation: the C++ clears each interior node but leaves BVHTri4Leaf::dummy0/dummy1
			// (the last 32 bytes of every leaf) at whatever malloc64 returned. Clear the whole
			// blob so the emitted bytes are deterministic.
			UnsafeUtility.MemClear( data, ( long )blocksNeeded * Bvh8Cpu.BlockSize );
			CopyBasePropertiesFrom( ref bvh8Cpu, ref bvh8 );
			// start conversion
			uint* stack = stackalloc uint[ StackSize ];
			// The C++ 'union { float dist[8]; uint32_t idist[8]; }': the sorting network swaps the
			// float view, so the lane indices packed into the mantissa move with the distances.
			// Stored as uint here to avoid type punning through a pointer under Burst.
			uint* idist = stackalloc uint[ 8 ];
			uint newBlockPtr = 0, nodeIdx = 0, stackPtr = 0;
			while ( true )
			{
				MbvhNode* orig = bvh8.Nodes + nodeIdx;
				Bvh8CpuNode* newNode = ( Bvh8CpuNode* )( data + ( newBlockPtr * Bvh8Cpu.BlockSize ) );
				newBlockPtr += 256 / Bvh8Cpu.BlockSize;
				UnsafeUtility.MemClear( newNode, sizeof( Bvh8CpuNode ) );
				float* xmin = newNode->XMin8, xmax = newNode->XMax8;
				float* ymin = newNode->YMin8, ymax = newNode->YMax8;
				float* zmin = newNode->ZMin8, zmax = newNode->ZMax8;
				uint* child8 = newNode->Child8;
				uint* perm8 = newNode->Perm8;
				// calculate the permutation offsets for the node
				for ( int q = 0; q < 8; q++ )
				{
					float dx = ( q & 1 ) != 0 ? 1.0f : -1.0f;
					float dy = ( q & 2 ) != 0 ? 1.0f : -1.0f;
					float dz = ( q & 4 ) != 0 ? 1.0f : -1.0f;
					for ( int i = 0; i < 8; i++ )
					{
						if ( orig->Child[ i ] == 0 )
						{
							idist[ i ] = ( math.asuint( BvhConstants.Far ) & 0xfffffff8 ) + ( uint )i;
						}
						else
						{
							MbvhNode* c = bvh8.Nodes + orig->Child[ i ];
							float px = ( q & 1 ) != 0 ? c->AabbMin.x : c->AabbMax.x;
							float py = ( q & 2 ) != 0 ? c->AabbMin.y : c->AabbMax.y;
							float pz = ( q & 4 ) != 0 ? c->AabbMin.z : c->AabbMax.z;
							float d = ( ( dx * px ) + ( dy * py ) ) + ( dz * pz );
							idist[ i ] = ( math.asuint( d ) & 0xfffffff8 ) + ( uint )i;
						}
					}
					// apply sorting network - https://bertdobbelaere.github.io/sorting_networks.html#N8L19D6
					Bvh4CpuConverter.Sort( idist, 0, 2 );
					Bvh4CpuConverter.Sort( idist, 1, 3 );
					Bvh4CpuConverter.Sort( idist, 4, 6 );
					Bvh4CpuConverter.Sort( idist, 5, 7 );
					Bvh4CpuConverter.Sort( idist, 0, 4 );
					Bvh4CpuConverter.Sort( idist, 1, 5 );
					Bvh4CpuConverter.Sort( idist, 2, 6 );
					Bvh4CpuConverter.Sort( idist, 3, 7 );
					Bvh4CpuConverter.Sort( idist, 0, 1 );
					Bvh4CpuConverter.Sort( idist, 2, 3 );
					Bvh4CpuConverter.Sort( idist, 4, 5 );
					Bvh4CpuConverter.Sort( idist, 6, 7 );
					Bvh4CpuConverter.Sort( idist, 2, 4 );
					Bvh4CpuConverter.Sort( idist, 3, 5 );
					Bvh4CpuConverter.Sort( idist, 1, 4 );
					Bvh4CpuConverter.Sort( idist, 3, 6 );
					Bvh4CpuConverter.Sort( idist, 1, 2 );
					Bvh4CpuConverter.Sort( idist, 3, 4 );
					Bvh4CpuConverter.Sort( idist, 5, 6 );
					for ( int i = 0; i < 8; i++ )
					{
						perm8[ i ] += ( idist[ i ] & 7 ) << ( q * 3 );
					}
				}
				// fill remaining fields
				int cidx = 0;
				for ( int i = 0; i < 8; i++ )
				{
					if ( orig->Child[ i ] != 0 )
					{
						MbvhNode* child = bvh8.Nodes + orig->Child[ i ];
						xmin[ cidx ] = child->AabbMin.x;
						xmax[ cidx ] = child->AabbMax.x;
						ymin[ cidx ] = child->AabbMin.y;
						ymax[ cidx ] = child->AabbMax.y;
						zmin[ cidx ] = child->AabbMin.z;
						zmax[ cidx ] = child->AabbMax.z;
						if ( child->IsLeaf )
						{
							// emit leaf node: group of up to 4 triangles in SoA format.
							child8[ cidx ] = newBlockPtr + Bvh8Cpu.LeafBit;
							Bvh4CpuTri4Leaf* leaf = ( Bvh4CpuTri4Leaf* )( data + ( newBlockPtr * Bvh8Cpu.BlockSize ) );
							newBlockPtr += 192 / Bvh8Cpu.BlockSize;
							for ( uint l = 0; l < 4; l++ )
							{
								uint primIdx = bvh8.Source.PrimIdx[ child->FirstTri + math.min( l, child->TriCount - 1u ) ];
								bvh8.Source.GetPrimIndices( primIdx, out uint i0, out uint i1, out uint i2 );
								float4 v0 = bvh8.Source.Vertex( i0 );
								float4 e1 = bvh8.Source.Vertex( i1 ) - v0;
								float4 e2 = bvh8.Source.Vertex( i2 ) - v0;
								Bvh4CpuConverter.SetLeafData( leaf, v0.xyz, e1.xyz, e2.xyz, primIdx, l );
							}
						}
						else
						{
							uint* slot = child8 + cidx;
							stack[ stackPtr++ ] = ( uint )( slot - ( uint* )data );
							stack[ stackPtr++ ] = orig->Child[ i ];
						}
						cidx++;
					}
				}
				for ( ; cidx < 8; cidx++ )
				{
					xmin[ cidx ] = 1e30f;
					xmax[ cidx ] = 1.00001e30f;
					ymin[ cidx ] = 1e30f;
					ymax[ cidx ] = 1.00001e30f;
					zmin[ cidx ] = 1e30f;
					zmax[ cidx ] = 1.00001e30f;
					child8[ cidx ] |= Bvh8Cpu.EmptyBit;
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
			bvh8Cpu.UsedBlocks = newBlockPtr;
		}

		/// <summary>Port of BVHBase::CopyBasePropertiesFrom for the Mbvh -&gt; Bvh8Cpu direction.</summary>
		private static void CopyBasePropertiesFrom( ref Bvh8Cpu bvh8Cpu, ref Mbvh original )
		{
			bvh8Cpu.Refittable = original.Refittable;
			bvh8Cpu.MayHaveHoles = original.MayHaveHoles;
			bvh8Cpu.BvhOverAabbs = original.BvhOverAabbs;
			bvh8Cpu.BvhOverIndices = original.BvhOverIndices;
			bvh8Cpu.TriCount = original.TriCount;
			bvh8Cpu.IdxCount = original.IdxCount;
			bvh8Cpu.AabbMin = original.AabbMin;
			bvh8Cpu.AabbMax = original.AabbMax;
		}
	}
}
