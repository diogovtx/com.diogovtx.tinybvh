using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Packet traversal of tinybvh's BVH class: BVH::Intersect256Rays.
	/// </summary>
	public unsafe partial struct Bvh
	{
		/// <summary>The number of rays one call of <see cref="Intersect256Rays( Ray* )"/> consumes.</summary>
		public const int PacketSize = 256;

		/// <summary>
		/// Port of BVH::Intersect256Rays( Ray* packet ): traverse the tree with 256 rays at once,
		/// which amortizes the memory traffic over the bundle.
		/// <para>
		/// The C++ calls this a proof of concept and its constraints are reproduced here: all 256
		/// rays must share packet[ 0 ].O, and they must be laid out as a 4x4 grid of 4x4 tiles, so
		/// that rays 0, 51, 204 and 255 are the corners of the bundle - the four frustum planes are
		/// built from those four rays alone. The leaf test is the packet's own Moller-Trumbore, so
		/// this only handles a non-indexed triangle soup: indexed geometry, custom geometry and
		/// opacity micro maps are all ignored, exactly as in the C++.
		/// </para>
		/// </summary>
		public void Intersect256Rays( Ray* packet )
		{
			if ( packet == null )
			{
				throw new ArgumentNullException( nameof( packet ) );
			}
			if ( UsedNodes == 0 )
			{
				return;
			}
			if ( IsTlas )
			{
				throw new InvalidOperationException( "Bvh.Intersect256Rays( .. ), packet traversal does not support a TLAS." );
			}
			if ( VertIdx != null )
			{
				throw new InvalidOperationException( "Bvh.Intersect256Rays( .. ), packet traversal does not support indexed geometry." );
			}
			BvhPacketTracer.Intersect256Rays( ref this, packet );
		}

		/// <summary>
		/// <see cref="Intersect256Rays( Ray* )"/> over a managed collection. The array must hold at
		/// least <see cref="PacketSize"/> rays; a longer one is traced from its first ray on.
		/// </summary>
		public void Intersect256Rays( NativeArray<Ray> packet )
		{
			if ( packet.Length < PacketSize )
			{
				throw new ArgumentException( $"Bvh.Intersect256Rays( .. ), the packet needs {PacketSize} rays.", nameof( packet ) );
			}
			Intersect256Rays( ( Ray* )NativeArrayUnsafeUtility.GetUnsafePtr( packet ) );
		}

		/// <summary>
		/// <see cref="Intersect256Rays( Ray* )"/> over a slice, so a packet can live inside a larger
		/// ray buffer. The slice must be contiguous.
		/// </summary>
		public void Intersect256Rays( NativeSlice<Ray> packet )
		{
			if ( packet.Length < PacketSize )
			{
				throw new ArgumentException( $"Bvh.Intersect256Rays( .. ), the packet needs {PacketSize} rays.", nameof( packet ) );
			}
			if ( packet.Stride != sizeof( Ray ) )
			{
				throw new ArgumentException( "Bvh.Intersect256Rays( .. ), the packet slice must be contiguous.", nameof( packet ) );
			}
			Intersect256Rays( ( Ray* )NativeSliceUnsafeUtility.GetUnsafePtr( packet ) );
		}
	}

	/// <summary>
	/// Burst-compiled half of the packet traversal. Direct calls must be synchronous, otherwise
	/// editor tests silently run the Mono fallback, which evaluates float math in double.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhPacketTracer
	{
		/// <summary>C++: 'ALIGNED( 64 ) uint32_t stack[64]'. Two entries are pushed per node.</summary>
		private const int PacketStackSize = 64;

		/// <summary>
		/// Port of the CALC_TMIN_TMAX_WITH_SLABTEST_ON_RAY macro: the slab test of one ray of the
		/// packet against a box already expressed relative to the shared origin.
		/// </summary>
		private static void SlabTestOnRay( Ray* packet, int r, float3 o1, float3 o2, out float tmin, out float tmax )
		{
			float3 rD = packet[ r ].RD, t1 = o1 * rD, t2 = o2 * rD;
			tmin = math.max( math.max( math.min( t1.x, t2.x ), math.min( t1.y, t2.y ) ), math.min( t1.z, t2.z ) );
			tmax = math.min( math.min( math.max( t1.x, t2.x ), math.max( t1.y, t2.y ) ), math.max( t1.z, t2.z ) );
		}

		/// <summary>Port of tinybvh_normalize, which leaves a zero-length vector alone.</summary>
		private static float3 Normalize( float3 a )
		{
			float l = math.length( a ), rl = l == 0f ? 0f : ( 1f / l );
			return a * rl;
		}

		/// <summary>
		/// Port of BVH::Intersect256Rays.
		/// Based on Large Ray Packets for Real-time Whitted Ray Tracing, Overbeck et al., 2008,
		/// extended with sorted traversal and reduced stack traffic.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void Intersect256Rays( ref Bvh bvh, Ray* packet )
		{
			// Corner rays are: 0, 51, 204 and 255
			// Construct the bounding planes, with normals pointing outwards
			float3 O = packet[ 0 ].O; // same for all rays in this case
			float3 p0 = packet[ 0 ].O + packet[ 0 ].D; // top-left
			float3 p1 = packet[ 51 ].O + packet[ 51 ].D; // top-right
			float3 p2 = packet[ 204 ].O + packet[ 204 ].D; // bottom-left
			float3 p3 = packet[ 255 ].O + packet[ 255 ].D; // bottom-right
			float3 plane0 = Normalize( math.cross( p0 - O, p0 - p2 ) ); // left plane
			float3 plane1 = Normalize( math.cross( p3 - O, p3 - p1 ) ); // right plane
			float3 plane2 = Normalize( math.cross( p1 - O, p1 - p0 ) ); // top plane
			float3 plane3 = Normalize( math.cross( p2 - O, p2 - p3 ) ); // bottom plane
			// Index of the node float that is furthest along each plane normal. BvhNode has the
			// C++ layout - aabbMin xyz, leftFirst, aabbMax xyz, triCount - so 0..2 pick a component
			// of aabbMin and 4..6 the matching one of aabbMax.
			int sign0x = plane0.x < 0f ? 4 : 0, sign0y = plane0.y < 0f ? 5 : 1, sign0z = plane0.z < 0f ? 6 : 2;
			int sign1x = plane1.x < 0f ? 4 : 0, sign1y = plane1.y < 0f ? 5 : 1, sign1z = plane1.z < 0f ? 6 : 2;
			int sign2x = plane2.x < 0f ? 4 : 0, sign2y = plane2.y < 0f ? 5 : 1, sign2z = plane2.z < 0f ? 6 : 2;
			int sign3x = plane3.x < 0f ? 4 : 0, sign3y = plane3.y < 0f ? 5 : 1, sign3z = plane3.z < 0f ? 6 : 2;
			float d0 = math.dot( O, plane0 ), d1 = math.dot( O, plane1 );
			float d2 = math.dot( O, plane2 ), d3 = math.dot( O, plane3 );
			// Traverse the tree with the packet
			int first = 0, last = 255; // first and last active ray in the packet
			BvhNode* node = bvh.Nodes;
			uint* stack = stackalloc uint[ PacketStackSize ];
			uint stackPtr = 0;
			while ( true )
			{
				if ( node->IsLeaf )
				{
					// handle leaf node
					for ( uint j = 0; j < node->TriCount; j++ )
					{
						uint idx = bvh.PrimIdx[ node->LeftFirst + j ], vid = idx * 3;
						float4 v0_ = bvh.Vertex( vid );
						float3 e1 = ( bvh.Vertex( vid + 1 ) - v0_ ).xyz, e2 = ( bvh.Vertex( vid + 2 ) - v0_ ).xyz;
						float3 s = O - v0_.xyz;
						for ( int i = first; i <= last; i++ )
						{
							Ray* ray = packet + i;
							float3 h = math.cross( ray->D, e2 );
							float a = math.dot( e1, h );
							if ( math.abs( a ) < 0.0000001f )
							{
								continue; // ray parallel to triangle
							}
							float f = 1f / a, u = f * math.dot( s, h );
							float3 q = math.cross( s, e1 );
							float v = f * math.dot( ray->D, q );
							if ( u < 0f || v < 0f || ( u + v ) > 1f )
							{
								continue;
							}
							float t = f * math.dot( e2, q );
							if ( t <= 0f || t >= ray->Hit.T )
							{
								continue;
							}
							ray->Hit.T = t;
							ray->Hit.U = u;
							ray->Hit.V = v;
							// INST_IDX_BITS == 32: the instance index lives in its own field.
							ray->Hit.Prim = idx;
							ray->Hit.Inst = ray->InstIdx;
						}
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					else // pop
					{
						last = ( int )stack[ --stackPtr ];
						node = bvh.Nodes + stack[ --stackPtr ];
						first = last >> 8;
						last &= 255;
					}
				}
				else
				{
					// fetch pointers to child nodes
					BvhNode* left = bvh.Nodes + node->LeftFirst;
					BvhNode* right = bvh.Nodes + node->LeftFirst + 1;
					bool visitLeft = true, visitRight = true;
					int leftFirst = first, leftLast = last, rightFirst = first, rightLast = last;
					float distLeft, distRight;
					{
						// see if we want to intersect the left child
						float3 o1 = new float3( left->AabbMin.x - O.x, left->AabbMin.y - O.y, left->AabbMin.z - O.z );
						float3 o2 = new float3( left->AabbMax.x - O.x, left->AabbMax.y - O.y, left->AabbMax.z - O.z );
						// 1. Early-in test: if first ray hits the node, the packet visits the node
						bool earlyHit;
						{
							SlabTestOnRay( packet, first, o1, o2, out float tmin, out float tmax );
							earlyHit = tmax >= tmin && tmin < packet[ first ].Hit.T && tmax >= 0f;
							distLeft = tmin;
						}
						if ( !earlyHit ) // 2. Early-out test: if the node aabb is outside the four planes, we skip the node
						{
							float* minmax = ( float* )left;
							float3 c0 = new float3( minmax[ sign0x ], minmax[ sign0y ], minmax[ sign0z ] );
							float3 c1 = new float3( minmax[ sign1x ], minmax[ sign1y ], minmax[ sign1z ] );
							float3 c2 = new float3( minmax[ sign2x ], minmax[ sign2y ], minmax[ sign2z ] );
							float3 c3 = new float3( minmax[ sign3x ], minmax[ sign3y ], minmax[ sign3z ] );
							if ( math.dot( c0, plane0 ) > d0 || math.dot( c1, plane1 ) > d1 ||
								math.dot( c2, plane2 ) > d2 || math.dot( c3, plane3 ) > d3 )
							{
								visitLeft = false;
							}
							else // 3. Last resort: update first and last, stay in node if first > last
							{
								for ( ; leftFirst <= leftLast; leftFirst++ )
								{
									SlabTestOnRay( packet, leftFirst, o1, o2, out float tmin, out float tmax );
									if ( tmax >= tmin && tmin < packet[ leftFirst ].Hit.T && tmax >= 0f )
									{
										distLeft = tmin;
										break;
									}
								}
								for ( ; leftLast >= leftFirst; leftLast-- )
								{
									SlabTestOnRay( packet, leftLast, o1, o2, out float tmin, out float tmax );
									if ( tmax >= tmin && tmin < packet[ leftLast ].Hit.T && tmax >= 0f )
									{
										break;
									}
								}
								visitLeft = leftLast >= leftFirst;
							}
						}
					}
					{
						// see if we want to intersect the right child
						float3 o1 = new float3( right->AabbMin.x - O.x, right->AabbMin.y - O.y, right->AabbMin.z - O.z );
						float3 o2 = new float3( right->AabbMax.x - O.x, right->AabbMax.y - O.y, right->AabbMax.z - O.z );
						// 1. Early-in test: if first ray hits the node, the packet visits the node
						bool earlyHit;
						{
							SlabTestOnRay( packet, first, o1, o2, out float tmin, out float tmax );
							earlyHit = tmax >= tmin && tmin < packet[ first ].Hit.T && tmax >= 0f;
							distRight = tmin;
						}
						if ( !earlyHit ) // 2. Early-out test: if the node aabb is outside the four planes, we skip the node
						{
							float* minmax = ( float* )right;
							float3 c0 = new float3( minmax[ sign0x ], minmax[ sign0y ], minmax[ sign0z ] );
							float3 c1 = new float3( minmax[ sign1x ], minmax[ sign1y ], minmax[ sign1z ] );
							float3 c2 = new float3( minmax[ sign2x ], minmax[ sign2y ], minmax[ sign2z ] );
							float3 c3 = new float3( minmax[ sign3x ], minmax[ sign3y ], minmax[ sign3z ] );
							if ( math.dot( c0, plane0 ) > d0 || math.dot( c1, plane1 ) > d1 ||
								math.dot( c2, plane2 ) > d2 || math.dot( c3, plane3 ) > d3 )
							{
								visitRight = false;
							}
							else // 3. Last resort: update first and last, stay in node if first > last
							{
								for ( ; rightFirst <= rightLast; rightFirst++ )
								{
									SlabTestOnRay( packet, rightFirst, o1, o2, out float tmin, out float tmax );
									if ( tmax >= tmin && tmin < packet[ rightFirst ].Hit.T && tmax >= 0f )
									{
										distRight = tmin;
										break;
									}
								}
								// Upstream asymmetry, reproduced: the left child walks back to
								// leftFirst here, the right child walks back to first.
								for ( ; rightLast >= first; rightLast-- )
								{
									SlabTestOnRay( packet, rightLast, o1, o2, out float tmin, out float tmax );
									if ( tmax >= tmin && tmin < packet[ rightLast ].Hit.T && tmax >= 0f )
									{
										break;
									}
								}
								visitRight = rightLast >= rightFirst;
							}
						}
					}
					// process intersection result
					if ( visitLeft && visitRight )
					{
						if ( distLeft < distRight ) // push right, continue with left
						{
							stack[ stackPtr++ ] = node->LeftFirst + 1;
							stack[ stackPtr++ ] = ( uint )( ( rightFirst << 8 ) + rightLast );
							node = left;
							first = leftFirst;
							last = leftLast;
						}
						else // push left, continue with right
						{
							stack[ stackPtr++ ] = node->LeftFirst;
							stack[ stackPtr++ ] = ( uint )( ( leftFirst << 8 ) + leftLast );
							node = right;
							first = rightFirst;
							last = rightLast;
						}
					}
					else if ( visitLeft ) // continue with left
					{
						node = left;
						first = leftFirst;
						last = leftLast;
					}
					else if ( visitRight ) // continue with right
					{
						node = right;
						first = rightFirst;
						last = rightLast;
					}
					else if ( stackPtr == 0 )
					{
						break;
					}
					else // pop
					{
						last = ( int )stack[ --stackPtr ];
						node = bvh.Nodes + stack[ --stackPtr ];
						first = last >> 8;
						last &= 255;
					}
				}
			}
		}
	}
}
