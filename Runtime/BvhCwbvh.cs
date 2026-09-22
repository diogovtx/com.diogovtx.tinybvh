using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Port of tinybvh's BVH8_CWBVH class: an 8-wide BVH in the format specified in "Efficient
	/// Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs", Ylitie et al. 2017.
	/// Nodes are 80 bytes (five 16-byte blocks) in <see cref="Data"/>; triangles are three 16-byte
	/// blocks each in <see cref="Tris"/>. Traversal is AVX-only in the C++ and is not ported.
	///
	/// The leaf encoding assumes at most three triangles per leaf, so the canonical build sequence
	/// (BVH8_CWBVH::Build) prepares the base BVH with Compact() + SplitLeafs( 3 ) before building
	/// the MBVH&lt;8&gt;. Note that those two calls mutate the base Bvh that Mbvh.Source shares by
	/// value, so they have to happen before Mbvh.ConvertFrom, not after.
	///
	/// This conversion also mutates the source MBVH&lt;8&gt;: the greedy child ordering permutes
	/// each node's child array in place. Hence the 'ref Mbvh' parameter.
	/// </summary>
	public unsafe partial struct BvhCwbvh : IDisposable
	{
		/// <summary>Value copy of the source Mbvh. Shares its memory; Dispose does not free it.</summary>
		public Mbvh Source;

		/// <summary>Nodes in CWBVH format, in 16-byte blocks (owned).</summary>
		[NativeDisableUnsafePtrRestriction] public float4* Data;
		/// <summary>Triangle data for the CWBVH nodes, in 16-byte blocks (owned).</summary>
		[NativeDisableUnsafePtrRestriction] public float4* Tris;

		public uint UsedBlocks;
		public uint AllocatedBlocks;
		/// <summary>Capacity of Tris in 16-byte blocks; sized on IdxCount, which grows when the source was built with spatial splits.</summary>
		public uint AllocatedTriBlocks;
		/// <summary>Number of valid float4s in <see cref="Tris"/>: three per triangle in a leaf.</summary>
		public uint TriBlocks;

		// Properties copied from the source by CopyBasePropertiesFrom.
		public uint TriCount;
		public uint IdxCount;
		public float3 AabbMin;
		public float3 AabbMax;
		[MarshalAs( UnmanagedType.U1 )] public bool Refittable;
		[MarshalAs( UnmanagedType.U1 )] public bool MayHaveHoles;
		[MarshalAs( UnmanagedType.U1 )] public bool BvhOverAabbs;
		[MarshalAs( UnmanagedType.U1 )] public bool BvhOverIndices;

		public Allocator Allocator;

		public static BvhCwbvh Create( Allocator allocator )
		{
			return new BvhCwbvh
			{
				Allocator = allocator
			};
		}

		public bool IsCreated => Allocator > Allocator.None;

		public void Dispose()
		{
			Free( Data );
			Free( Tris );
			Data = null;
			Tris = null;
			AllocatedBlocks = 0;
			AllocatedTriBlocks = 0;
			UsedBlocks = 0;
			TriBlocks = 0;
			TriCount = 0;
			IdxCount = 0;
		}

		/// <summary>
		/// Port of BVH8_CWBVH::ConvertFrom. The C++ ignores its second parameter; it is kept for
		/// symmetry with the other layouts.
		/// </summary>
		public void ConvertFrom( ref Mbvh bvh8, bool compact = true )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "BvhCwbvh.ConvertFrom( .. ), bvhCwbvh was not created." );
			}
			if ( bvh8.M != 8 )
			{
				throw new ArgumentException( "BvhCwbvh.ConvertFrom( .. ), source is not an 8-wide MBVH.", nameof( bvh8 ) );
			}
			if ( bvh8.Nodes == null || bvh8.UsedNodes == 0 )
			{
				throw new ArgumentException( "BvhCwbvh.ConvertFrom( .. ), source has no nodes.", nameof( bvh8 ) );
			}
			if ( bvh8.Nodes[ 0 ].IsLeaf )
			{
				throw new ArgumentException( "BvhCwbvh.ConvertFrom( .. ), converting a single-node bvh.", nameof( bvh8 ) );
			}
			BvhCwbvhConverter.ConvertFrom( ref this, ref bvh8, compact );
		}

		/// <summary>
		/// Port of BVH8_CWBVH::SAHCost, which forwards to the underlying MBVH&lt;8&gt;.
		/// Unavailable on a tree loaded from a file.
		/// </summary>
		public float SahCost( uint nodeIdx = 0 )
		{
			if ( Source.Source.Nodes == null )
			{
				throw new InvalidOperationException( "BvhCwbvh.SahCost( .. ), this tree was loaded from a file and has no source BVH." );
			}
			return Source.SahCost( nodeIdx );
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
	/// Burst-compiled implementation of the CWBVH conversion. It must run under Burst: the
	/// quantisation uses log2 / exp2 / ceil and the results have to match the C++ bit for bit.
	/// Direct calls must be synchronous, otherwise editor tests silently run the Mono fallback.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhCwbvhConverter
	{
		/// <summary>Port of BVH8_CWBVH::ConvertFrom. Adapted from code by "AlanWBFT".</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void ConvertFrom( ref BvhCwbvh cwbvh, ref Mbvh bvh8, [MarshalAs( UnmanagedType.U1 )] bool compact )
		{
			// get a copy of the original bvh8
			cwbvh.Source = bvh8;
			// allocate memory
			uint spaceNeeded = bvh8.TriCount * 5; // CWBVH nodes use 80 bytes each.
			if ( spaceNeeded > cwbvh.AllocatedBlocks )
			{
				// Deviation: the C++ overwrites the pointer without freeing it first.
				cwbvh.Free( cwbvh.Data );
				cwbvh.Data = ( float4* )cwbvh.Alloc( ( long )spaceNeeded * 16 );
				cwbvh.AllocatedBlocks = spaceNeeded;
			}
			// Deviation: the C++ sizes the triangle buffer together with the node buffer, on the first
			// conversion only. A later source with more index entries (an SBVH of the same mesh) then
			// overruns it, so the triangle capacity is tracked on its own here.
			uint triBlocksNeeded = bvh8.IdxCount * 4;
			if ( triBlocksNeeded > cwbvh.AllocatedTriBlocks )
			{
				cwbvh.Free( cwbvh.Tris );
				cwbvh.Tris = ( float4* )cwbvh.Alloc( ( long )triBlocksNeeded * 16 );
				cwbvh.AllocatedTriBlocks = triBlocksNeeded;
			}
			float4* bvh8Data = cwbvh.Data;
			float4* bvh8Tris = cwbvh.Tris;
			UnsafeUtility.MemClear( bvh8Data, ( long )spaceNeeded * 16 );
			// Deviation: the C++ allocates idxCount * 4 blocks here but only clears idxCount * 3 of
			// them, so the last quarter is whatever the allocator handed out - zero in practice,
			// since an allocation this size comes from fresh OS pages. The whole buffer is cleared
			// here so those trailing blocks are defined: the reference tri hash covers
			// idxCount * 64 bytes, and Save writes the same range.
			UnsafeUtility.MemClear( bvh8Tris, ( long )bvh8.IdxCount * 4 * 16 );
			CopyBasePropertiesFrom( ref cwbvh, ref bvh8 );
			MbvhNode** stackNodePtr = stackalloc MbvhNode*[ 256 ];
			uint* stackNodeAddr = stackalloc uint[ 256 ];
			float* cost = stackalloc float[ 8 * 8 ];
			int* assignment = stackalloc int[ 8 ];
			bool* isSlotEmpty = stackalloc bool[ 8 ];
			uint stackPtr = 1, nodeDataPtr = 5, triDataPtr = 0;
			stackNodePtr[ 0 ] = bvh8.Nodes;
			stackNodeAddr[ 0 ] = 0;
			// start conversion
			while ( stackPtr > 0 )
			{
				MbvhNode* orig = stackNodePtr[ --stackPtr ];
				int currentNodeAddr = ( int )stackNodeAddr[ stackPtr ];
				float3 nodeLo = orig->AabbMin, nodeHi = orig->AabbMax;
				// greedy child node ordering
				float3 nodeCentroid = ( nodeLo + nodeHi ) * 0.5f;
				for ( int s = 0; s < 8; s++ )
				{
					isSlotEmpty[ s ] = true;
					assignment[ s ] = -1;
					float3 ds = new float3(
						( ( ( s >> 2 ) & 1 ) == 1 ) ? -1.0f : 1.0f,
						( ( ( s >> 1 ) & 1 ) == 1 ) ? -1.0f : 1.0f,
						( ( ( s >> 0 ) & 1 ) == 1 ) ? -1.0f : 1.0f
					);
					for ( int i = 0; i < 8; i++ )
					{
						if ( orig->Child[ i ] == 0 )
						{
							cost[ ( s * 8 ) + i ] = BvhConstants.Far;
						}
						else
						{
							MbvhNode* child = bvh8.Nodes + orig->Child[ i ];
							float3 childCentroid = ( child->AabbMin + child->AabbMax ) * 0.5f;
							cost[ ( s * 8 ) + i ] = math.dot( childCentroid - nodeCentroid, ds );
						}
					}
				}
				while ( true )
				{
					float minCost = BvhConstants.Far;
					int minEntryx = -1, minEntryy = -1;
					for ( int s = 0; s < 8; s++ )
					{
						for ( int i = 0; i < 8; i++ )
						{
							if ( assignment[ i ] == -1 && isSlotEmpty[ s ] && cost[ ( s * 8 ) + i ] < minCost )
							{
								minCost = cost[ ( s * 8 ) + i ];
								minEntryx = s;
								minEntryy = i;
							}
						}
					}
					if ( minEntryx == -1 && minEntryy == -1 )
					{
						break;
					}
					isSlotEmpty[ minEntryx ] = false;
					assignment[ minEntryy ] = minEntryx;
				}
				for ( int i = 0; i < 8; i++ )
				{
					if ( assignment[ i ] == -1 )
					{
						for ( int s = 0; s < 8; s++ )
						{
							if ( isSlotEmpty[ s ] )
							{
								isSlotEmpty[ s ] = false;
								assignment[ i ] = s;
								break;
							}
						}
					}
				}
				MbvhNode oldNode = *orig;
				for ( int i = 0; i < 8; i++ )
				{
					orig->Child[ assignment[ i ] ] = oldNode.Child[ i ];
				}
				// calculate quantization parameters for each axis
				int ex = ToInt8( math.ceil( math.log2( ( nodeHi.x - nodeLo.x ) / 255.0f ) ) );
				int ey = ToInt8( math.ceil( math.log2( ( nodeHi.y - nodeLo.y ) / 255.0f ) ) );
				int ez = ToInt8( math.ceil( math.log2( ( nodeHi.z - nodeLo.z ) / 255.0f ) ) );
				// encode output
				int internalChildCount = 0, leafChildTriCount = 0, childBaseIndex = 0, triangleBaseIndex = 0;
				byte imask = 0;
				for ( int i = 0; i < 8; i++ )
				{
					if ( orig->Child[ i ] == 0 )
					{
						continue;
					}
					MbvhNode* child = bvh8.Nodes + orig->Child[ i ];
					// The C++ uses powf( 2, e ); exp2 of an integral exponent is the same exact value.
					int qlox = ( int )math.floor( ( child->AabbMin.x - nodeLo.x ) / math.exp2( ( float )ex ) );
					int qloy = ( int )math.floor( ( child->AabbMin.y - nodeLo.y ) / math.exp2( ( float )ey ) );
					int qloz = ( int )math.floor( ( child->AabbMin.z - nodeLo.z ) / math.exp2( ( float )ez ) );
					int qhix = ( int )math.ceil( ( child->AabbMax.x - nodeLo.x ) / math.exp2( ( float )ex ) );
					int qhiy = ( int )math.ceil( ( child->AabbMax.y - nodeLo.y ) / math.exp2( ( float )ey ) );
					int qhiz = ( int )math.ceil( ( child->AabbMax.z - nodeLo.z ) / math.exp2( ( float )ez ) );
					byte* baseAddr = ( byte* )( bvh8Data + currentNodeAddr + 2 );
					baseAddr[ i + 0 ] = ( byte )qlox;
					baseAddr[ i + 24 ] = ( byte )qhix;
					baseAddr[ i + 8 ] = ( byte )qloy;
					baseAddr[ i + 32 ] = ( byte )qhiy;
					baseAddr[ i + 16 ] = ( byte )qloz;
					baseAddr[ i + 40 ] = ( byte )qhiz;
					// set the meta field - This calculation assumes children are stored contiguously.
					byte* childMetaField = ( ( byte* )( bvh8Data + currentNodeAddr + 1 ) ) + 8;
					if ( !child->IsLeaf )
					{
						// interior node, set params and push onto stack
						int childNodeAddr = ( int )nodeDataPtr;
						if ( internalChildCount++ == 0 )
						{
							childBaseIndex = childNodeAddr / 5;
						}
						nodeDataPtr += 5;
						imask |= ( byte )( 1 << i );
						childMetaField[ i ] = ( byte )( ( 1 << 5 ) | ( 24 + ( byte )i ) ); // I don't see how this accounts for empty children?
						stackNodePtr[ stackPtr ] = child;
						stackNodeAddr[ stackPtr++ ] = ( uint )childNodeAddr; // counted in float4s
						continue;
					}
					// leaf node
					uint tcount = child->TriCount; // will not exceed 3.
					if ( leafChildTriCount == 0 )
					{
						triangleBaseIndex = ( int )triDataPtr;
					}
					int unaryEncodedTriCount = tcount == 1 ? 0x1 : tcount == 2 ? 0x3 : 0x7;
					childMetaField[ i ] = ( byte )( ( unaryEncodedTriCount << 5 ) | leafChildTriCount );
					leafChildTriCount += ( int )tcount;
					for ( uint j = 0; j < tcount; j++ )
					{
						int triIdx = ( int )bvh8.Source.PrimIdx[ child->FirstTri + j ];
						bvh8.Source.GetPrimIndices( ( uint )triIdx, out uint ti0, out uint ti1, out uint ti2 );
						float4 t = bvh8.Source.Vertex( ti0 );
						bvh8Tris[ triDataPtr + 0 ] = bvh8.Source.Vertex( ti2 ) - t;
						bvh8Tris[ triDataPtr + 1 ] = bvh8.Source.Vertex( ti1 ) - t;
						t.w = math.asfloat( triIdx );
						bvh8Tris[ triDataPtr + 2 ] = t;
						triDataPtr += 3;
					}
				}
				uint exyzAndimask = ( uint )( ex & 0xFF )
					| ( ( uint )( ey & 0xFF ) << 8 )
					| ( ( uint )( ez & 0xFF ) << 16 )
					| ( ( uint )imask << 24 );
				bvh8Data[ currentNodeAddr + 0 ] = new float4( nodeLo, math.asfloat( exyzAndimask ) );
				bvh8Data[ currentNodeAddr + 1 ].x = math.asfloat( childBaseIndex );
				bvh8Data[ currentNodeAddr + 1 ].y = math.asfloat( triangleBaseIndex );
			}
			cwbvh.UsedBlocks = nodeDataPtr;
			cwbvh.TriBlocks = triDataPtr;
		}

		/// <summary>
		/// Port of the C++ (int32_t)((int8_t)ceilf( .. )) conversion: the float is truncated to an
		/// integer, the low byte is reinterpreted as a signed char and sign-extended back to int.
		/// </summary>
		private static int ToInt8( float v )
		{
			return ( sbyte )( int )v;
		}

		/// <summary>Port of BVHBase::CopyBasePropertiesFrom for the Mbvh -&gt; BvhCwbvh direction.</summary>
		private static void CopyBasePropertiesFrom( ref BvhCwbvh cwbvh, ref Mbvh original )
		{
			cwbvh.Refittable = original.Refittable;
			cwbvh.MayHaveHoles = original.MayHaveHoles;
			cwbvh.BvhOverAabbs = original.BvhOverAabbs;
			cwbvh.BvhOverIndices = original.BvhOverIndices;
			cwbvh.TriCount = original.TriCount;
			cwbvh.IdxCount = original.IdxCount;
			cwbvh.AabbMin = original.AabbMin;
			cwbvh.AabbMax = original.AabbMax;
		}
	}
}
