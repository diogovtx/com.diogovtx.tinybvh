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
	/// Port of tinybvh's BVH4_GPU class: a 4-wide BVH in a flat 16-byte-block blob, ready to be
	/// uploaded to the GPU as is. Node layout, per the C++ comment:
	/// offs 0:  aabbMin (12 bytes), 4x quantized child xmin (4 bytes)
	/// offs 16: aabbMax (12 bytes), 4x quantized child xmax (4 bytes)
	/// offs 32: 4x child ymin, then ymax, zmin, zmax (total 16 bytes)
	/// offs 48: 4x child node info: leaf if MSB set. Leaf: 15 bits tri count, 16 bits offset;
	///          interior: 32 bits for the position of the child node, in float4s.
	/// Triangle data ('by value') immediately follows each leaf node: vert0, vert1 - vert0,
	/// vert2 - vert0, with the original triangle index bit-cast into vert0.w.
	/// </summary>
	public unsafe partial struct Bvh4Gpu : IDisposable
	{
		/// <summary>Value copy of the source Mbvh. Shares its memory; Dispose does not free it.</summary>
		public Mbvh Source;

		/// <summary>Node and triangle data, in 16-byte blocks (owned).</summary>
		[NativeDisableUnsafePtrRestriction] public float4* Data;
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

		// SAH cost parameters, used by Intersect to accumulate the traversal cost.
		public float TraversalCost;
		public float IntersectionCost;

		public Allocator Allocator;

		/// <summary>Traversal stack depth of BVH4_GPU::Intersect (C++: stack[128]).</summary>
		private const int IntersectStackSize = 128;

		public static Bvh4Gpu Create( Allocator allocator )
		{
			return new Bvh4Gpu
			{
				TraversalCost = BvhConstants.DefaultTraversalCost,
				IntersectionCost = BvhConstants.DefaultIntersectionCost,
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
		}

		/// <summary>Port of BVH4_GPU::ConvertFrom.</summary>
		public void ConvertFrom( ref Mbvh bvh4, bool compact = true )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "Bvh4Gpu.ConvertFrom( .. ), bvh4Gpu was not created." );
			}
			if ( bvh4.M != 4 )
			{
				throw new ArgumentException( "Bvh4Gpu.ConvertFrom( .. ), source is not a 4-wide MBVH.", nameof( bvh4 ) );
			}
			if ( bvh4.Nodes == null || bvh4.UsedNodes == 0 )
			{
				throw new ArgumentException( "Bvh4Gpu.ConvertFrom( .. ), source has no nodes.", nameof( bvh4 ) );
			}
			if ( bvh4.Nodes[ 0 ].IsLeaf )
			{
				// The C++ asserts !orig.isLeaf() at the top of the conversion loop; MBVH::ConvertFrom
				// inserts an extra level for a leaf root, so this should never trigger.
				throw new ArgumentException( "Bvh4Gpu.ConvertFrom( .. ), converting a single-node bvh.", nameof( bvh4 ) );
			}
			Bvh4GpuConverter.ConvertFrom( ref this, ref bvh4, compact );
		}

		/// <summary>Port of BVH4_GPU::SAHCost, which forwards to the underlying MBVH&lt;4&gt;.</summary>
		public float SahCost( uint nodeIdx = 0 )
		{
			return Source.SahCost( nodeIdx );
		}

		/// <summary>
		/// Port of BVH4_GPU::Intersect (IntersectAlt4Nodes). For testing the converted data only;
		/// not efficient. This code replicates how traversal on GPU happens.
		/// </summary>
		public int Intersect( ref Ray ray )
		{
			// traverse a blas
			uint offset = 0;
			uint* stack = stackalloc uint[ IntersectStackSize ];
			uint stackPtr = 0, tmp2; // tmp2 is for the SWAP macro
			uint* leaf = stackalloc uint[ 4 ];
			float cost = 0f;
			while ( true )
			{
				cost += TraversalCost;
				// fetch the node
				float4 data0 = Data[ offset + 0 ], data1 = Data[ offset + 1 ];
				float4 data2 = Data[ offset + 2 ], data3 = Data[ offset + 3 ];
				// extract aabb
				float3 bmin = data0.xyz, extent = data1.xyz; // pre-scaled by 1/255
				// reconstruct conservative child aabbs
				float4 d0 = AsUChar4( data0.w ), d1 = AsUChar4( data1.w ), d2 = AsUChar4( data2.x );
				float4 d3 = AsUChar4( data2.y ), d4 = AsUChar4( data2.z ), d5 = AsUChar4( data2.w );
				float3 c0min = bmin + ( extent * new float3( d0.x, d2.x, d4.x ) ), c0max = bmin + ( extent * new float3( d1.x, d3.x, d5.x ) );
				float3 c1min = bmin + ( extent * new float3( d0.y, d2.y, d4.y ) ), c1max = bmin + ( extent * new float3( d1.y, d3.y, d5.y ) );
				float3 c2min = bmin + ( extent * new float3( d0.z, d2.z, d4.z ) ), c2max = bmin + ( extent * new float3( d1.z, d3.z, d5.z ) );
				float3 c3min = bmin + ( extent * new float3( d0.w, d2.w, d4.w ) ), c3max = bmin + ( extent * new float3( d1.w, d3.w, d5.w ) );
				// intersect child aabbs
				float3 t1a = ( c0min - ray.O ) * ray.RD, t2a = ( c0max - ray.O ) * ray.RD;
				float3 t1b = ( c1min - ray.O ) * ray.RD, t2b = ( c1max - ray.O ) * ray.RD;
				float3 t1c = ( c2min - ray.O ) * ray.RD, t2c = ( c2max - ray.O ) * ray.RD;
				float3 t1d = ( c3min - ray.O ) * ray.RD, t2d = ( c3max - ray.O ) * ray.RD;
				float3 minta = math.min( t1a, t2a ), maxta = math.max( t1a, t2a );
				float3 mintb = math.min( t1b, t2b ), maxtb = math.max( t1b, t2b );
				float3 mintc = math.min( t1c, t2c ), maxtc = math.max( t1c, t2c );
				float3 mintd = math.min( t1d, t2d ), maxtd = math.max( t1d, t2d );
				float tmina = math.max( math.max( math.max( minta.x, minta.y ), minta.z ), 0.0f );
				float tminb = math.max( math.max( math.max( mintb.x, mintb.y ), mintb.z ), 0.0f );
				float tminc = math.max( math.max( math.max( mintc.x, mintc.y ), mintc.z ), 0.0f );
				float tmind = math.max( math.max( math.max( mintd.x, mintd.y ), mintd.z ), 0.0f );
				float tmaxa = math.min( math.min( math.min( maxta.x, maxta.y ), maxta.z ), ray.Hit.T );
				float tmaxb = math.min( math.min( math.min( maxtb.x, maxtb.y ), maxtb.z ), ray.Hit.T );
				float tmaxc = math.min( math.min( math.min( maxtc.x, maxtc.y ), maxtc.z ), ray.Hit.T );
				float tmaxd = math.min( math.min( math.min( maxtd.x, maxtd.y ), maxtd.z ), ray.Hit.T );
				float dist0 = tmina > tmaxa ? BvhConstants.Far : tmina, dist1 = tminb > tmaxb ? BvhConstants.Far : tminb;
				float dist2 = tminc > tmaxc ? BvhConstants.Far : tminc, dist3 = tmind > tmaxd ? BvhConstants.Far : tmind, tmp;
				// get child node info fields
				uint c0info = math.asuint( data3.x ), c1info = math.asuint( data3.y );
				uint c2info = math.asuint( data3.z ), c3info = math.asuint( data3.w );
				if ( dist0 < dist2 )
				{
					tmp = dist0;
					dist0 = dist2;
					dist2 = tmp;
					tmp2 = c0info;
					c0info = c2info;
					c2info = tmp2;
				}
				if ( dist1 < dist3 )
				{
					tmp = dist1;
					dist1 = dist3;
					dist3 = tmp;
					tmp2 = c1info;
					c1info = c3info;
					c3info = tmp2;
				}
				if ( dist0 < dist1 )
				{
					tmp = dist0;
					dist0 = dist1;
					dist1 = tmp;
					tmp2 = c0info;
					c0info = c1info;
					c1info = tmp2;
				}
				if ( dist2 < dist3 )
				{
					tmp = dist2;
					dist2 = dist3;
					dist3 = tmp;
					tmp2 = c2info;
					c2info = c3info;
					c3info = tmp2;
				}
				if ( dist1 < dist2 )
				{
					tmp = dist1;
					dist1 = dist2;
					dist2 = tmp;
					tmp2 = c1info;
					c1info = c2info;
					c2info = tmp2;
				}
				// process results, starting with farthest child, so nearest ends on top of stack
				// nextNode is never assigned in the C++ either; the traversal always pops the stack.
				uint nextNode = 0;
				leaf[ 0 ] = 0;
				leaf[ 1 ] = 0;
				leaf[ 2 ] = 0;
				leaf[ 3 ] = 0;
				uint leafs = 0;
				if ( dist0 < BvhConstants.Far )
				{
					if ( ( c0info & 0x80000000 ) != 0 )
					{
						leaf[ leafs++ ] = c0info;
					}
					else if ( c0info != 0 )
					{
						stack[ stackPtr++ ] = c0info;
					}
				}
				if ( dist1 < BvhConstants.Far )
				{
					if ( ( c1info & 0x80000000 ) != 0 )
					{
						leaf[ leafs++ ] = c1info;
					}
					else if ( c1info != 0 )
					{
						stack[ stackPtr++ ] = c1info;
					}
				}
				if ( dist2 < BvhConstants.Far )
				{
					if ( ( c2info & 0x80000000 ) != 0 )
					{
						leaf[ leafs++ ] = c2info;
					}
					else if ( c2info != 0 )
					{
						stack[ stackPtr++ ] = c2info;
					}
				}
				if ( dist3 < BvhConstants.Far )
				{
					if ( ( c3info & 0x80000000 ) != 0 )
					{
						leaf[ leafs++ ] = c3info;
					}
					else if ( c3info != 0 )
					{
						stack[ stackPtr++ ] = c3info;
					}
				}
				// process encountered leafs, if any
				for ( uint i = 0; i < leafs; i++ )
				{
					uint n = ( leaf[ i ] >> 16 ) & 0x7fff;
					uint triStart = offset + ( leaf[ i ] & 0xffff );
					for ( uint j = 0; j < n; j++, triStart += 3 )
					{
						cost += IntersectionCost;
						float3 e2 = Data[ triStart + 2 ].xyz;
						float3 e1 = Data[ triStart + 1 ].xyz;
						float3 v0 = Data[ triStart + 0 ].xyz;
						// MOLLER_TRUMBORE_TEST( ray.hit.t, continue )
						float3 h = math.cross( ray.D, e2 );
						float a = math.dot( e1, h );
						if ( math.abs( a ) < 0.000001f )
						{
							continue;
						}
						float f = 1f / a;
						float3 s = ray.O - v0;
						float u = f * math.dot( s, h );
						float3 q = math.cross( s, e1 );
						float v = f * math.dot( ray.D, q );
						bool miss = u < 0f || v < 0f || ( u + v ) > 1f;
						if ( miss )
						{
							continue;
						}
						float t = f * math.dot( e2, q );
						if ( t < 0f || t > ray.Hit.T )
						{
							continue;
						}
						ray.Hit.T = t;
						ray.Hit.U = u;
						ray.Hit.V = v;
						ray.Hit.Prim = math.asuint( Data[ triStart + 0 ].w );
					}
				}
				// continue with nearest node or first node on the stack
				if ( nextNode != 0 )
				{
					offset = nextNode;
				}
				else
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					offset = stack[ --stackPtr ];
				}
			}
			return ( int )cost; // cast to not break interface.
		}

		/// <summary>Port of BVH4_GPU::IsOccluded, i.e. the FALLBACK_SHADOW_QUERY macro.</summary>
		public bool IsOccluded( in Ray ray )
		{
			Ray r = ray;
			float d = ray.Hit.T;
			Intersect( ref r );
			return r.Hit.T < d;
		}

		/// <summary>Port of the as_uchar4 helper, returning the four bytes as floats.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float4 AsUChar4( float v )
		{
			uint u = math.asuint( v );
			return new float4( u & 255u, ( u >> 8 ) & 255u, ( u >> 16 ) & 255u, ( u >> 24 ) & 255u );
		}

		/// <summary>Ensures the blob can hold count 16-byte blocks. Contents are not preserved when it grows.</summary>
		internal void AllocateBlocks( uint count )
		{
			if ( AllocatedBlocks < count )
			{
				Free( Data );
				Data = ( float4* )Alloc( ( long )count * 16 );
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
	/// Burst-compiled implementation of the BVH4_GPU conversion. Direct calls must be synchronous,
	/// otherwise editor tests silently run the Mono fallback.
	/// </summary>
	[BurstCompile]
	internal static unsafe class Bvh4GpuConverter
	{
		/// <summary>Port of BVH4_GPU::ConvertFrom.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void ConvertFrom( ref Bvh4Gpu gpu, ref Mbvh bvh4, [MarshalAs( UnmanagedType.U1 )] bool compact )
		{
			// get a copy of the original bvh4
			gpu.Source = bvh4;
			uint blocksNeeded = compact ? ( bvh4.UsedNodes * 4 ) : ( bvh4.AllocatedNodes * 4 ); // here, 'block' is 16 bytes.
			blocksNeeded += 6 * bvh4.TriCount; // this layout stores tris in the same buffer.
			gpu.AllocateBlocks( blocksNeeded );
			float4* bvh4Data = gpu.Data;
			UnsafeUtility.MemClear( bvh4Data, ( long )blocksNeeded * 16 );
			CopyBasePropertiesFrom( ref gpu, ref bvh4 );
			// start conversion
			MbvhNode* mbvhNode = bvh4.Nodes;
			uint* stack = stackalloc uint[ 128 ];
			MbvhNode** childNode = stackalloc MbvhNode*[ 4 ];
			uint* childInfo = stackalloc uint[ 4 ];
			uint nodeIdx = 0, newAlt4Ptr = 0, stackPtr = 0, retValPos = 0;
			while ( true )
			{
				MbvhNode* orig = mbvhNode + nodeIdx;
				// convert BVH4 node - must be an interior node.
				float4* nodeBase = bvh4Data + newAlt4Ptr;
				uint baseAlt4Ptr = newAlt4Ptr;
				newAlt4Ptr += 4;
				nodeBase[ 0 ] = new float4( orig->AabbMin, 0f );
				nodeBase[ 1 ] = new float4( ( orig->AabbMax - orig->AabbMin ) * ( 1.0f / 255.0f ), 0f );
				childNode[ 0 ] = mbvhNode + orig->Child[ 0 ];
				childNode[ 1 ] = mbvhNode + orig->Child[ 1 ];
				childNode[ 2 ] = mbvhNode + orig->Child[ 2 ];
				childNode[ 3 ] = mbvhNode + orig->Child[ 3 ];
				// start with leaf child node conversion
				childInfo[ 0 ] = 0;
				childInfo[ 1 ] = 0;
				childInfo[ 2 ] = 0;
				childInfo[ 3 ] = 0; // will store in final fields later
				for ( int i = 0; i < 4; i++ )
				{
					if ( childNode[ i ]->IsLeaf )
					{
						childInfo[ i ] = newAlt4Ptr - baseAlt4Ptr;
						childInfo[ i ] |= childNode[ i ]->TriCount << 16;
						childInfo[ i ] |= 0x80000000;
						for ( uint j = 0; j < childNode[ i ]->TriCount; j++ )
						{
							uint t = bvh4.Source.PrimIdx[ childNode[ i ]->FirstTri + j ];
							bvh4.Source.GetPrimIndices( t, out uint ti0, out uint ti1, out uint ti2 );
							float4 v0 = bvh4.Source.Vertex( ti0 );
							bvh4Data[ newAlt4Ptr + 1 ] = bvh4.Source.Vertex( ti1 ) - v0;
							bvh4Data[ newAlt4Ptr + 2 ] = bvh4.Source.Vertex( ti2 ) - v0;
							v0.w = math.asfloat( t ); // as_float
							bvh4Data[ newAlt4Ptr + 0 ] = v0;
							newAlt4Ptr += 3;
						}
					}
				}
				// process interior nodes
				for ( int i = 0; i < 4; i++ )
				{
					if ( !childNode[ i ]->IsLeaf )
					{
						if ( orig->Child[ i ] == 0 )
						{
							childInfo[ i ] = 0;
						}
						else
						{
							stack[ stackPtr++ ] = ( uint )( ( ( float* )( nodeBase + 3 ) + i ) - ( float* )bvh4Data );
							stack[ stackPtr++ ] = orig->Child[ i ];
						}
					}
				}
				// store child node bounds, quantized
				float3 extent = orig->AabbMax - orig->AabbMin;
				float3 scale;
				scale.x = extent.x > 1e-10f ? ( 254.999f / extent.x ) : 0;
				scale.y = extent.y > 1e-10f ? ( 254.999f / extent.y ) : 0;
				scale.z = extent.z > 1e-10f ? ( 254.999f / extent.z ) : 0;
				byte* slot0 = ( byte* )( nodeBase + 0 ) + 12;    // 4 chars
				byte* slot1 = ( byte* )( nodeBase + 1 ) + 12;    // 4 chars
				byte* slot2 = ( byte* )( nodeBase + 2 );         // 16 chars
				if ( orig->Child[ 0 ] != 0 )
				{
					float3 relBMin = childNode[ 0 ]->AabbMin - orig->AabbMin, relBMax = childNode[ 0 ]->AabbMax - orig->AabbMin;
					slot0[ 0 ] = ToByte( math.floor( relBMin.x * scale.x ) );
					slot1[ 0 ] = ToByte( math.ceil( relBMax.x * scale.x ) );
					slot2[ 0 ] = ToByte( math.floor( relBMin.y * scale.y ) );
					slot2[ 4 ] = ToByte( math.ceil( relBMax.y * scale.y ) );
					slot2[ 8 ] = ToByte( math.floor( relBMin.z * scale.z ) );
					slot2[ 12 ] = ToByte( math.ceil( relBMax.z * scale.z ) );
				}
				if ( orig->Child[ 1 ] != 0 )
				{
					float3 relBMin = childNode[ 1 ]->AabbMin - orig->AabbMin, relBMax = childNode[ 1 ]->AabbMax - orig->AabbMin;
					slot0[ 1 ] = ToByte( math.floor( relBMin.x * scale.x ) );
					slot1[ 1 ] = ToByte( math.ceil( relBMax.x * scale.x ) );
					slot2[ 1 ] = ToByte( math.floor( relBMin.y * scale.y ) );
					slot2[ 5 ] = ToByte( math.ceil( relBMax.y * scale.y ) );
					slot2[ 9 ] = ToByte( math.floor( relBMin.z * scale.z ) );
					slot2[ 13 ] = ToByte( math.ceil( relBMax.z * scale.z ) );
				}
				if ( orig->Child[ 2 ] != 0 )
				{
					float3 relBMin = childNode[ 2 ]->AabbMin - orig->AabbMin, relBMax = childNode[ 2 ]->AabbMax - orig->AabbMin;
					slot0[ 2 ] = ToByte( math.floor( relBMin.x * scale.x ) );
					slot1[ 2 ] = ToByte( math.ceil( relBMax.x * scale.x ) );
					slot2[ 2 ] = ToByte( math.floor( relBMin.y * scale.y ) );
					slot2[ 6 ] = ToByte( math.ceil( relBMax.y * scale.y ) );
					slot2[ 10 ] = ToByte( math.floor( relBMin.z * scale.z ) );
					slot2[ 14 ] = ToByte( math.ceil( relBMax.z * scale.z ) );
				}
				if ( orig->Child[ 3 ] != 0 )
				{
					float3 relBMin = childNode[ 3 ]->AabbMin - orig->AabbMin, relBMax = childNode[ 3 ]->AabbMax - orig->AabbMin;
					slot0[ 3 ] = ToByte( math.floor( relBMin.x * scale.x ) );
					slot1[ 3 ] = ToByte( math.ceil( relBMax.x * scale.x ) );
					slot2[ 3 ] = ToByte( math.floor( relBMin.y * scale.y ) );
					slot2[ 7 ] = ToByte( math.ceil( relBMax.y * scale.y ) );
					slot2[ 11 ] = ToByte( math.floor( relBMin.z * scale.z ) );
					slot2[ 15 ] = ToByte( math.ceil( relBMax.z * scale.z ) );
				}
				// finalize node
				nodeBase[ 3 ] = new float4(
					math.asfloat( childInfo[ 0 ] ), math.asfloat( childInfo[ 1 ] ),
					math.asfloat( childInfo[ 2 ] ), math.asfloat( childInfo[ 3 ] )
				);
				// pop new work from the stack
				if ( retValPos > 0 )
				{
					( ( uint* )bvh4Data )[ retValPos ] = baseAlt4Ptr;
				}
				if ( stackPtr == 0 )
				{
					break;
				}
				nodeIdx = stack[ --stackPtr ];
				retValPos = stack[ --stackPtr ];
			}
			gpu.UsedBlocks = newAlt4Ptr;
		}

		/// <summary>
		/// Port of the C++ (uint8_t)floorf( .. ) / (uint8_t)ceilf( .. ) conversions: the float is
		/// truncated to an integer and the low byte is kept. Inputs are in [0, 255] by construction.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static byte ToByte( float v )
		{
			return ( byte )( int )v;
		}

		/// <summary>Port of BVHBase::CopyBasePropertiesFrom for the Mbvh -&gt; Bvh4Gpu direction.</summary>
		private static void CopyBasePropertiesFrom( ref Bvh4Gpu gpu, ref Mbvh original )
		{
			gpu.Refittable = original.Refittable;
			gpu.MayHaveHoles = original.MayHaveHoles;
			gpu.BvhOverAabbs = original.BvhOverAabbs;
			gpu.BvhOverIndices = original.BvhOverIndices;
			gpu.TriCount = original.TriCount;
			gpu.IdxCount = original.IdxCount;
			gpu.AabbMin = original.AabbMin;
			gpu.AabbMax = original.AabbMax;
		}
	}
}
