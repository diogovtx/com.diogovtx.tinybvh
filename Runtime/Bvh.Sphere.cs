using Unity.Burst;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Sphere overlap query of tinybvh's BVH class: BVH::IntersectSphere.
	/// </summary>
	public unsafe partial struct Bvh
	{
		/// <summary>
		/// Port of BVH::IntersectSphere( const bvhvec3&amp; pos, const float r ): returns true as
		/// soon as one primitive of the tree overlaps the sphere. Works on triangles only, indexed
		/// or not; custom geometry and opacity maps are ignored, exactly as in the C++.
		/// </summary>
		public bool IntersectSphere( float3 pos, float r )
		{
			if ( UsedNodes == 0 )
			{
				return false;
			}
			BvhSphereQuery.IntersectSphere( ref this, in pos, r, out int hit );
			return hit != 0;
		}
	}

	/// <summary>
	/// Burst-compiled half of the sphere query. Direct calls must be synchronous, otherwise editor
	/// tests silently run the Mono fallback, which evaluates float math in double.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhSphereQuery
	{
		/// <summary>Traversal stack depth of BVH::IntersectSphere (C++: 'BVHNode* stack[64]').</summary>
		private const int SphereStackSize = 64;

		/// <summary>Port of BVH::IntersectSphere. Writes 1 to hit on an overlap, 0 otherwise.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void IntersectSphere( ref Bvh bvh, in float3 pos, float r, out int hit )
		{
			hit = 0;
			float3 bmin = pos - new float3( r ), bmax = pos + new float3( r );
			BvhNode* node = bvh.Nodes;
			BvhNode** stack = stackalloc BvhNode*[ SphereStackSize ];
			uint stackPtr = 0;
			float r2 = r * r;
			while ( true )
			{
				if ( node->IsLeaf )
				{
					// check if the leaf aabb overlaps the sphere: https://gamedev.stackexchange.com/a/156877
					float dist2 = 0f;
					if ( pos.x < node->AabbMin.x )
					{
						dist2 += ( node->AabbMin.x - pos.x ) * ( node->AabbMin.x - pos.x );
					}
					if ( pos.x > node->AabbMax.x )
					{
						dist2 += ( pos.x - node->AabbMax.x ) * ( pos.x - node->AabbMax.x );
					}
					if ( pos.y < node->AabbMin.y )
					{
						dist2 += ( node->AabbMin.y - pos.y ) * ( node->AabbMin.y - pos.y );
					}
					if ( pos.y > node->AabbMax.y )
					{
						dist2 += ( pos.y - node->AabbMax.y ) * ( pos.y - node->AabbMax.y );
					}
					if ( pos.z < node->AabbMin.z )
					{
						dist2 += ( node->AabbMin.z - pos.z ) * ( node->AabbMin.z - pos.z );
					}
					if ( pos.z > node->AabbMax.z )
					{
						dist2 += ( pos.z - node->AabbMax.z ) * ( pos.z - node->AabbMax.z );
					}
					if ( dist2 <= r2 )
					{
						// tri/sphere test: https://gist.github.com/yomotsu/d845f21e2e1eb49f647f#file-gistfile1-js-L223
						for ( uint i = 0; i < node->TriCount; i++ )
						{
							uint idx = bvh.PrimIdx[ node->LeftFirst + i ];
							// The C++ inlines the indexed and non-indexed vertex addressing here;
							// GetPrimIndices is the same mapping, so it covers both branches.
							bvh.GetPrimIndices( idx, out uint i0, out uint i1, out uint i2 );
							float3 a = bvh.Vertex( i0 ).xyz, b = bvh.Vertex( i1 ).xyz, c = bvh.Vertex( i2 ).xyz;
							float3 A = a - pos, B = b - pos, C = c - pos;
							float rr = r * r;
							float3 V = math.cross( B - A, C - A );
							float d = math.dot( A, V ), e = math.dot( V, V );
							if ( d * d > rr * e )
							{
								continue;
							}
							float aa = math.dot( A, A ), ab = math.dot( A, B ), ac = math.dot( A, C );
							float bb = math.dot( B, B ), bc = math.dot( B, C ), cc = math.dot( C, C );
							if ( ( aa > rr && ab > aa && ac > aa ) || ( bb > rr && ab > bb && bc > bb ) ||
								( cc > rr && ac > cc && bc > cc ) )
							{
								continue;
							}
							float3 AB = B - A, BC = C - B, CA = A - C;
							float d1 = ab - aa, d2 = bc - bb, d3 = ac - cc;
							float e1 = math.dot( AB, AB ), e2 = math.dot( BC, BC ), e3 = math.dot( CA, CA );
							float3 Q1 = ( A * e1 ) - ( AB * d1 ), Q2 = ( B * e2 ) - ( BC * d2 ), Q3 = ( C * e3 ) - ( CA * d3 );
							float3 QC = ( C * e1 ) - Q1, QA = ( A * e2 ) - Q2, QB = ( B * e3 ) - Q3;
							if ( ( math.dot( Q1, Q1 ) > rr * e1 * e1 && math.dot( Q1, QC ) >= 0f ) ||
								( math.dot( Q2, Q2 ) > rr * e2 * e2 && math.dot( Q2, QA ) >= 0f ) ||
								( math.dot( Q3, Q3 ) > rr * e3 * e3 && math.dot( Q3, QB ) >= 0f ) )
							{
								continue;
							}
							// const float dist = sqrtf( d * d / e ) - r; // we're not using this.
							hit = 1;
							return;
						}
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
					continue;
				}
				BvhNode* child1 = bvh.Nodes + node->LeftFirst, child2 = bvh.Nodes + node->LeftFirst + 1;
				bool hit1 = child1->Intersect( bmin, bmax ), hit2 = child2->Intersect( bmin, bmax );
				if ( hit1 && hit2 )
				{
					stack[ stackPtr++ ] = child2;
					node = child1;
				}
				else if ( hit1 )
				{
					node = child1;
				}
				else if ( hit2 )
				{
					node = child2;
				}
				else
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
				}
			}
		}
	}
}
