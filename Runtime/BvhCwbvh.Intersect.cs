using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Traversal half of tinybvh's BVH8_CWBVH class: the CPU walk of the compressed wide BVH,
	/// which the C++ only compiles under BVH_USEAVX because it needs __lzcnt / __popcnt. The
	/// algorithm itself is the scalar reference implementation the header describes as "for
	/// debugging only, not efficient" - the layout is meant for the GPU kernels - so this is a
	/// straight scalar port, not a SIMD one.
	///
	/// The work runs in a Burst direct call rather than in the struct methods themselves: the
	/// quantised slab test and the Moeller-Trumbore test have to match a compiled C++ build bit
	/// for bit, and Mono evaluates float expressions in double.
	/// </summary>
	public unsafe partial struct BvhCwbvh
	{
		/// <summary>
		/// Port of BVH8_CWBVH::Intersect( Ray&amp; ). The hit, if any, is written to ray.Hit.
		/// Returns 0, like the C++, which has no step counter for this layout.
		/// </summary>
		public int Intersect( ref Ray ray )
		{
			if ( Data == null || Tris == null )
			{
				return 0;
			}
			BvhCwbvhTraversal.Intersect( ref this, ref ray );
			return 0;
		}

		/// <summary>
		/// Port of BVH8_CWBVH::IsOccluded, which is the FALLBACK_SHADOW_QUERY macro: the ray is
		/// copied, intersected, and the query reports whether anything was found closer than the
		/// original ray length. There is no early-out shadow walk for this layout.
		/// </summary>
		public bool IsOccluded( in Ray ray )
		{
			if ( Data == null || Tris == null )
			{
				return false;
			}
			Ray copy = ray;
			float d = ray.Hit.T;
			BvhCwbvhTraversal.Intersect( ref this, ref copy );
			return copy.Hit.T < d;
		}
	}

	/// <summary>
	/// Burst-compiled implementation of the CWBVH traversal. Direct calls must be synchronous,
	/// otherwise editor tests silently run the Mono fallback and drift in the last bits.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhCwbvhTraversal
	{
		/// <summary>Traversal stack depth; the C++ uses 128 node groups.</summary>
		private const int StackSize = 128;

		/// <summary>Port of tinybvh_max for floats: a plain ternary, without the NaN handling math.max adds.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float Max( float a, float b )
		{
			return a > b ? a : b;
		}

		/// <summary>Port of tinybvh_min for floats; see <see cref="Max"/>.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float Min( float a, float b )
		{
			return a < b ? a : b;
		}

		/// <summary>Port of extract_byte: byte n of i, zero-extended.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static uint ExtractByte( uint i, int n )
		{
			return ( i >> ( n * 8 ) ) & 0xFF;
		}

		/// <summary>Port of sign_extend_s8x4: spreads the sign bit of each byte of i over that byte.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static uint SignExtendS8x4( uint i )
		{
			uint b0 = ( i & 0b10000000000000000000000000000000u ) != 0 ? 0xff000000u : 0u;
			uint b1 = ( i & 0b00000000100000000000000000000000u ) != 0 ? 0x00ff0000u : 0u;
			uint b2 = ( i & 0b00000000000000001000000000000000u ) != 0 ? 0x0000ff00u : 0u;
			uint b3 = ( i & 0b00000000000000000000000010000000u ) != 0 ? 0x000000ffu : 0u;
			return b0 + b1 + b2 + b3;
		}

		/// <summary>Port of __bfind: the index of the highest set bit.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static uint Bfind( uint x )
		{
			return ( uint )( 31 - math.lzcnt( x ) );
		}

		/// <summary>
		/// One half of the eight-child slab test. The C++ writes the two halves out in full, twice
		/// over; they differ only in which words of the node they read, so they are factored into
		/// one helper here. The arithmetic and its order are unchanged: the C++ fills six
		/// four-element arrays and then loops over the lanes, which is the same value per lane as
		/// computing each lane's six terms inside the loop.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static uint HitMask4( uint meta4, uint octinv,
			uint swizzledLox, uint swizzledHix, uint swizzledLoy, uint swizzledHiy, uint swizzledLoz, uint swizzledHiz,
			float adjustedIdirx, float adjustedIdiry, float adjustedIdirz,
			float origx, float origy, float origz, float tmin, float tmax )
		{
			uint isInner4 = ( meta4 & ( meta4 << 1 ) ) & 0x10101010;
			uint innerMask4 = SignExtendS8x4( isInner4 << 3 );
			uint bitIndex4 = ( meta4 ^ ( octinv & innerMask4 ) ) & 0x1F1F1F1F;
			uint childBits4 = ( meta4 >> 5 ) & 0x07070707;
			uint hitmask = 0;
			for ( int i = 0; i < 4; i++ )
			{
				int shift = i * 8;
				float tminx = ( ( swizzledLox >> shift ) & 0xFF ) * adjustedIdirx + origx;
				float tminy = ( ( swizzledLoy >> shift ) & 0xFF ) * adjustedIdiry + origy;
				float tminz = ( ( swizzledLoz >> shift ) & 0xFF ) * adjustedIdirz + origz;
				float tmaxx = ( ( swizzledHix >> shift ) & 0xFF ) * adjustedIdirx + origx;
				float tmaxy = ( ( swizzledHiy >> shift ) & 0xFF ) * adjustedIdiry + origy;
				float tmaxz = ( ( swizzledHiz >> shift ) & 0xFF ) * adjustedIdirz + origz;
				// Use VMIN, VMAX to compute the slabs
				float cmin = Max( Max( Max( tminx, tminy ), tminz ), tmin );
				float cmax = Min( Min( Min( tmaxx, tmaxy ), tmaxz ), tmax );
				if ( cmin <= cmax )
				{
					hitmask |= ExtractByte( childBits4, i ) << ( int )ExtractByte( bitIndex4, i );
				}
			}
			return hitmask;
		}

		/// <summary>Port of BVH8_CWBVH::Intersect( Ray&amp; ).</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void Intersect( ref BvhCwbvh cwbvh, ref Ray ray )
		{
			uint2* traversalStack = stackalloc uint2[ StackSize ];
			uint hitAddr = 0, stackPtr = 0;
			float2 triangleuv = new float2( 0f, 0f );
			float4* blasNodes = cwbvh.Data;
			float4* blasTris = cwbvh.Tris;
			float tmin = 0f, tmax = ray.Hit.T;
			uint octinv = ( uint )( 7 - ( ( ray.D.x < 0f ? 4 : 0 ) | ( ray.D.y < 0f ? 2 : 0 ) | ( ray.D.z < 0f ? 1 : 0 ) ) ) * 0x1010101u;
			uint2 ngroup = new uint2( 0u, 0b10000000000000000000000000000000u );
			uint2 tgroup = new uint2( 0u, 0u );
			while ( true )
			{
				if ( ngroup.y > 0x00FFFFFF )
				{
					uint hits = ngroup.y, imask = ngroup.y;
					uint childBitIndex = Bfind( hits ), childNodeBaseIndex = ngroup.x;
					ngroup.y &= ~( 1u << ( int )childBitIndex );
					if ( ngroup.y > 0x00FFFFFF )
					{
						traversalStack[ stackPtr++ ] = ngroup;
					}
					uint slotIndex = ( childBitIndex - 24 ) ^ ( octinv & 255 );
					uint relativeIndex = ( uint )math.countbits( imask & ~( 0xFFFFFFFFu << ( int )slotIndex ) );
					uint childNodeIndex = childNodeBaseIndex + relativeIndex;
					float4 n0 = blasNodes[ ( childNodeIndex * 5 ) + 0 ], n1 = blasNodes[ ( childNodeIndex * 5 ) + 1 ];
					float4 n2 = blasNodes[ ( childNodeIndex * 5 ) + 2 ], n3 = blasNodes[ ( childNodeIndex * 5 ) + 3 ];
					float4 n4 = blasNodes[ ( childNodeIndex * 5 ) + 4 ], p = n0;
					uint n0w = math.asuint( n0.w );
					int ex = ( sbyte )( n0w & 0xFF );
					int ey = ( sbyte )( ( n0w >> 8 ) & 0xFF );
					int ez = ( sbyte )( ( n0w >> 16 ) & 0xFF );
					ngroup.x = math.asuint( n1.x );
					tgroup.x = math.asuint( n1.y );
					tgroup.y = 0;
					uint hitmask = 0;
					uint vx = ( uint )( ( ex + 127 ) << 23 );
					float adjustedIdirx = math.asfloat( vx ) * ray.RD.x;
					uint vy = ( uint )( ( ey + 127 ) << 23 );
					float adjustedIdiry = math.asfloat( vy ) * ray.RD.y;
					uint vz = ( uint )( ( ez + 127 ) << 23 );
					float adjustedIdirz = math.asfloat( vz ) * ray.RD.z;
					float origx = -( ray.O.x - p.x ) * ray.RD.x;
					float origy = -( ray.O.y - p.y ) * ray.RD.y;
					float origz = -( ray.O.z - p.z ) * ray.RD.z;
					{   // First 4
						uint meta4 = math.asuint( n1.z );
						uint swizzledLox = ray.RD.x < 0f ? math.asuint( n3.z ) : math.asuint( n2.x );
						uint swizzledHix = ray.RD.x < 0f ? math.asuint( n2.x ) : math.asuint( n3.z );
						uint swizzledLoy = ray.RD.y < 0f ? math.asuint( n4.x ) : math.asuint( n2.z );
						uint swizzledHiy = ray.RD.y < 0f ? math.asuint( n2.z ) : math.asuint( n4.x );
						uint swizzledLoz = ray.RD.z < 0f ? math.asuint( n4.z ) : math.asuint( n3.x );
						uint swizzledHiz = ray.RD.z < 0f ? math.asuint( n3.x ) : math.asuint( n4.z );
						hitmask |= HitMask4( meta4, octinv,
							swizzledLox, swizzledHix, swizzledLoy, swizzledHiy, swizzledLoz, swizzledHiz,
							adjustedIdirx, adjustedIdiry, adjustedIdirz, origx, origy, origz, tmin, tmax );
					}
					{   // Second 4
						uint meta4 = math.asuint( n1.w );
						uint swizzledLox = ray.RD.x < 0f ? math.asuint( n3.w ) : math.asuint( n2.y );
						uint swizzledHix = ray.RD.x < 0f ? math.asuint( n2.y ) : math.asuint( n3.w );
						uint swizzledLoy = ray.RD.y < 0f ? math.asuint( n4.y ) : math.asuint( n2.w );
						uint swizzledHiy = ray.RD.y < 0f ? math.asuint( n2.w ) : math.asuint( n4.y );
						uint swizzledLoz = ray.RD.z < 0f ? math.asuint( n4.w ) : math.asuint( n3.y );
						uint swizzledHiz = ray.RD.z < 0f ? math.asuint( n3.y ) : math.asuint( n4.w );
						hitmask |= HitMask4( meta4, octinv,
							swizzledLox, swizzledHix, swizzledLoy, swizzledHiy, swizzledLoz, swizzledHiz,
							adjustedIdirx, adjustedIdiry, adjustedIdirz, origx, origy, origz, tmin, tmax );
					}
					ngroup.y = ( hitmask & 0xFF000000 ) | ( math.asuint( n0.w ) >> 24 );
					tgroup.y = hitmask & 0x00FFFFFF;
				}
				else
				{
					tgroup = ngroup;
					ngroup = new uint2( 0u, 0u );
				}
				while ( tgroup.y != 0 )
				{
					uint triangleIndex = Bfind( tgroup.y );
					tgroup.y -= 1u << ( int )triangleIndex;
					int triAddr = ( int )( tgroup.x + ( triangleIndex * 3 ) );
					float3 e2 = blasTris[ triAddr + 0 ].xyz;
					float3 e1 = blasTris[ triAddr + 1 ].xyz;
					float3 v0 = blasTris[ triAddr + 2 ].xyz;
					// MOLLER_TRUMBORE_TEST( tmax, continue )
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
					if ( t < 0f || t > tmax )
					{
						continue;
					}
					triangleuv = new float2( u, v );
					tmax = t;
					hitAddr = math.asuint( blasTris[ triAddr + 2 ].w );
				}
				if ( ngroup.y > 0x00FFFFFF )
				{
					continue;
				}
				if ( stackPtr > 0 )
				{
					ngroup = traversalStack[ --stackPtr ];
				}
				else
				{
					ray.Hit.T = tmax;
					if ( tmax < BvhConstants.Far )
					{
						ray.Hit.U = triangleuv.x;
						ray.Hit.V = triangleuv.y;
						ray.Hit.Prim = hitAddr;
					}
					break;
				}
			}
		}
	}
}
