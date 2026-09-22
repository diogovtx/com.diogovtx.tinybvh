using System.Runtime.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Traversal half of tinybvh's BVH class. The C++ compiles eight template variants of each
	/// traversal function, keyed on the sign of the ray direction (posX, posY, posZ); here the
	/// same predicates are evaluated once at the top of the traversal and passed down, so the
	/// arithmetic - and therefore the resulting t values - stays identical.
	/// </summary>
	public unsafe partial struct Bvh
	{
		/// <summary>Traversal stack depth of BVH::Intersect (C++: BVHNode* stack[256]).</summary>
		private const int IntersectStackSize = 256;
		/// <summary>Traversal stack depth of IntersectTLAS, IsOccluded and IsOccludedTLAS (C++: stack[64]).</summary>
		private const int SmallStackSize = 64;

		/// <summary>
		/// Port of BVH::Intersect( Ray& ), including its octant dispatcher. Returns the traversal
		/// cost; the hit, if any, is written to ray.Hit.
		/// </summary>
		public int Intersect( ref Ray ray )
		{
			if ( UsedNodes == 0 )
			{
				return 0;
			}
			bool posX = ray.D.x >= 0f;
			bool posY = ray.D.y >= 0f;
			bool posZ = ray.D.z >= 0f;
			if ( IsTlas )
			{
				return IntersectTlas( ref ray, posX, posY, posZ );
			}
			return IntersectBlas( ref ray, posX, posY, posZ );
		}

		/// <summary>
		/// Port of BVH::IsOccluded( const Ray& ), including its octant dispatcher. Returns true as
		/// soon as any primitive is hit within ray.Hit.T; the ray itself is left untouched.
		/// </summary>
		public bool IsOccluded( in Ray ray )
		{
			if ( UsedNodes == 0 )
			{
				return false;
			}
			bool posX = ray.D.x >= 0f;
			bool posY = ray.D.y >= 0f;
			bool posZ = ray.D.z >= 0f;
			if ( IsTlas )
			{
				return IsOccludedTlas( ray, posX, posY, posZ );
			}
			return IsOccludedBlas( ray, posX, posY, posZ );
		}

		/// <summary>Port of the templated BVH::Intersect body: traversal over triangles.</summary>
		private int IntersectBlas( ref Ray ray, bool posX, bool posY, bool posZ )
		{
			BvhNode* node = Nodes;
			BvhNode** stack = stackalloc BvhNode*[ IntersectStackSize ];
			uint stackPtr = 0;
			float cost = 0f;
			float rox = ray.O.x * ray.RD.x;
			float roy = ray.O.y * ray.RD.y;
			float roz = ray.O.z * ray.RD.z;
			while ( true )
			{
				cost += TraversalCost;
				if ( node->IsLeaf )
				{
					// Performance note: if indexed primitives (ENABLE_INDEXED_GEOMETRY) and custom
					// geometry (ENABLE_CUSTOM_GEOMETRY) are both disabled, this leaf code reduces
					// to a regular loop over triangles. Otherwise, the extra flexibility comes at
					// a small performance cost.
					// The C++ picks between three leaf loops: indexed, custom and plain triangles.
					// GetPrimIndices already covers indexed and plain, so one test of the callback
					// per leaf is left. Unlike the C++, which tries the indexed loop first, a BVH
					// that has both VertIdx and a callback set uses the callback here.
					if ( CustomIntersect.IsCreated )
					{
						Ray* rayPtr = ( Ray* )UnsafeUtility.AddressOf( ref ray );
						for ( uint i = 0; i < node->TriCount; i++ )
						{
							if ( CustomIntersect.Invoke( rayPtr, PrimIdx[ node->LeftFirst + i ] ) != 0 )
							{
								// INST_IDX_BITS == 32: the instance index lives in its own field.
								ray.Hit.Inst = ray.InstIdx;
							}
							cost += IntersectionCost;
						}
					}
					else
					{
						for ( uint i = 0; i < node->TriCount; i++ )
						{
							uint pi = PrimIdx[ node->LeftFirst + i ];
							GetPrimIndices( pi, out uint i0, out uint i1, out uint i2 );
							IntersectTri( ref ray, pi, i0, i1, i2 );
							cost += IntersectionCost;
						}
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
					continue;
				}
				BvhNode* child1 = Nodes + node->LeftFirst;
				BvhNode* child2 = Nodes + node->LeftFirst + 1;
				SlabTestTwoNodes( child1, child2, ray, rox, roy, roz, posX, posY, posZ, out float dist1, out float dist2 );
				if ( dist1 > dist2 )
				{
					float td = dist1;
					dist1 = dist2;
					dist2 = td;
					BvhNode* tn = child1;
					child1 = child2;
					child2 = tn;
				}
				if ( dist1 == BvhConstants.Far /* missed both child nodes */ )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
				}
				else /* hit at least one node */
				{
					node = child1; /* continue with the nearest */
					if ( dist2 != BvhConstants.Far )
					{
						stack[ stackPtr++ ] = child2; /* push far child */
					}
				}
			}
			return ( int )cost; // cast to not break interface.
		}

		/// <summary>Port of the templated BVH::IntersectTLAS body: traversal over BLAS instances.</summary>
		private int IntersectTlas( ref Ray ray, bool posX, bool posY, bool posZ )
		{
			BvhNode* node = Nodes;
			BvhNode** stack = stackalloc BvhNode*[ SmallStackSize ];
			uint stackPtr = 0;
			float cost = 0f;
			float rox = ray.O.x * ray.RD.x;
			float roy = ray.O.y * ray.RD.y;
			float roz = ray.O.z * ray.RD.z;
			while ( true )
			{
				cost += TraversalCost;
				if ( node->IsLeaf )
				{
					Ray tmpRay = default;
					for ( uint i = 0; i < node->TriCount; i++ )
					{
						// BLAS traversal
						uint instIdx = PrimIdx[ node->LeftFirst + i ];
						BlasInstance* inst = Instances + instIdx;
						// Check if the ray should intersect this BLAS Instance, otherwise skip it
						if ( ( inst->Mask & ray.Mask ) == 0 )
						{
							continue;
						}
						// 1. Transform ray with the inverse of the instance transform
						tmpRay.O = inst->InvTransform.TransformPoint( ray.O );
						tmpRay.D = inst->InvTransform.TransformVector( ray.D );
						tmpRay.InstIdx = instIdx; // INST_IDX_BITS == 32, so the C++ shift by (32 - 32) is a no-op.
						tmpRay.Hit = ray.Hit;
						tmpRay.RD = BvhMath.Rcp( tmpRay.D );
						// 2. Traverse BLAS with the transformed ray.
						// A TLAS built over a single-layout Bvh* list keeps the direct call the C++
						// layout dispatch collapses to; one built over a mixed BlasRef list takes the
						// layout chain in IntersectBlasRef. Either way the octant predicates are recomputed
						// for the transformed direction, exactly as the BVH::Intersect dispatcher would.
						if ( Blasses != null )
						{
							Bvh* blas = Blasses + inst->BlasIdx;
							cost += blas->IntersectBlas( ref tmpRay, tmpRay.D.x >= 0f, tmpRay.D.y >= 0f, tmpRay.D.z >= 0f );
						}
						else
						{
							cost += IntersectBlasRef( BlasRefs[ inst->BlasIdx ], ref tmpRay );
						}
						// 3. Restore ray
						ray.Hit = tmpRay.Hit;
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
					continue;
				}
				BvhNode* child1 = Nodes + node->LeftFirst;
				BvhNode* child2 = Nodes + node->LeftFirst + 1;
				SlabTestTwoNodes( child1, child2, ray, rox, roy, roz, posX, posY, posZ, out float dist1, out float dist2 );
				if ( dist1 > dist2 )
				{
					float td = dist1;
					dist1 = dist2;
					dist2 = td;
					BvhNode* tn = child1;
					child1 = child2;
					child2 = tn;
				}
				if ( dist1 == BvhConstants.Far /* missed both child nodes */ )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
				}
				else /* hit at least one node */
				{
					node = child1; /* continue with the nearest */
					if ( dist2 != BvhConstants.Far )
					{
						stack[ stackPtr++ ] = child2; /* push far child */
					}
				}
			}
			return ( int )cost;
		}

		/// <summary>Port of the templated BVH::IsOccluded body.</summary>
		private bool IsOccludedBlas( in Ray ray, bool posX, bool posY, bool posZ )
		{
			BvhNode* node = Nodes;
			BvhNode** stack = stackalloc BvhNode*[ SmallStackSize ];
			uint stackPtr = 0;
			float rox = ray.O.x * ray.RD.x;
			float roy = ray.O.y * ray.RD.y;
			float roz = ray.O.z * ray.RD.z;
			while ( true )
			{
				if ( node->IsLeaf )
				{
					// See the note in IntersectBlas: one test of the callback per leaf.
					if ( CustomIsOccluded.IsCreated )
					{
						// the callback signature takes a pointer where the C++ takes a const Ray&,
						// so it is handed the address of a copy; ray itself must not change.
						Ray tmpRay = ray;
						Ray* rayPtr = &tmpRay;
						for ( uint i = 0; i < node->TriCount; i++ )
						{
							if ( CustomIsOccluded.Invoke( rayPtr, PrimIdx[ node->LeftFirst + i ] ) != 0 )
							{
								return true;
							}
						}
					}
					else
					{
						for ( uint i = 0; i < node->TriCount; i++ )
						{
							uint pi = PrimIdx[ node->LeftFirst + i ];
							GetPrimIndices( pi, out uint i0, out uint i1, out uint i2 );
							if ( TriOccludes( ray, pi, i0, i1, i2 ) )
							{
								return true;
							}
						}
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
					continue;
				}
				BvhNode* child1 = Nodes + node->LeftFirst;
				BvhNode* child2 = Nodes + node->LeftFirst + 1;
				SlabTestTwoNodes( child1, child2, ray, rox, roy, roz, posX, posY, posZ, out float dist1, out float dist2 );
				if ( dist1 > dist2 )
				{
					float td = dist1;
					dist1 = dist2;
					dist2 = td;
					BvhNode* tn = child1;
					child1 = child2;
					child2 = tn;
				}
				if ( dist1 == BvhConstants.Far /* missed both child nodes */ )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
				}
				else /* hit at least one node */
				{
					node = child1; /* continue with the nearest */
					if ( dist2 != BvhConstants.Far )
					{
						stack[ stackPtr++ ] = child2; /* push far child */
					}
				}
			}
			return false;
		}

		/// <summary>Port of the templated BVH::IsOccludedTLAS body.</summary>
		private bool IsOccludedTlas( in Ray ray, bool posX, bool posY, bool posZ )
		{
			BvhNode* node = Nodes;
			BvhNode** stack = stackalloc BvhNode*[ SmallStackSize ];
			uint stackPtr = 0;
			Ray tmpRay = default;
			float rox = ray.O.x * ray.RD.x;
			float roy = ray.O.y * ray.RD.y;
			float roz = ray.O.z * ray.RD.z;
			while ( true )
			{
				if ( node->IsLeaf )
				{
					for ( uint i = 0; i < node->TriCount; i++ )
					{
						// BLAS traversal
						BlasInstance* inst = Instances + PrimIdx[ node->LeftFirst + i ];
						// Check if the ray should intersect this BLAS Instance, otherwise skip it
						if ( ( inst->Mask & ray.Mask ) == 0 )
						{
							continue;
						}
						// 1. Transform ray with the inverse of the instance transform
						tmpRay.O = inst->InvTransform.TransformPoint( ray.O );
						tmpRay.D = inst->InvTransform.TransformVector( ray.D );
						tmpRay.Hit.T = ray.Hit.T;
						tmpRay.RD = BvhMath.Rcp( tmpRay.D );
						// 2. Traverse BLAS with the transformed ray.
						// A TLAS built over a single-layout Bvh* list keeps the direct call the C++
						// layout dispatch collapses to; one built over a mixed BlasRef list takes the
						// layout chain in IsOccludedBlasRef. Either way the octant predicates are recomputed
						// for the transformed direction, exactly as the BVH::IsOccluded dispatcher would.
						if ( Blasses != null )
						{
							Bvh* blas = Blasses + inst->BlasIdx;
							if ( blas->IsOccludedBlas( tmpRay, tmpRay.D.x >= 0f, tmpRay.D.y >= 0f, tmpRay.D.z >= 0f ) )
							{
								return true;
							}
						}
						else if ( IsOccludedBlasRef( BlasRefs[ inst->BlasIdx ], tmpRay ) )
						{
							return true;
						}
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
					continue;
				}
				BvhNode* child1 = Nodes + node->LeftFirst;
				BvhNode* child2 = Nodes + node->LeftFirst + 1;
				SlabTestTwoNodes( child1, child2, ray, rox, roy, roz, posX, posY, posZ, out float dist1, out float dist2 );
				if ( dist1 > dist2 )
				{
					float td = dist1;
					dist1 = dist2;
					dist2 = td;
					BvhNode* tn = child1;
					child1 = child2;
					child2 = tn;
				}
				if ( dist1 == BvhConstants.Far /* missed both child nodes */ )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
				}
				else /* hit at least one node */
				{
					node = child1; /* continue with the nearest */
					if ( dist2 != BvhConstants.Far )
					{
						stack[ stackPtr++ ] = child2; /* push far child */
					}
				}
			}
			return false;
		}

		/// <summary>
		/// Port of the SLAB_TEST_TWO_NODES macro: slab test against both children of a node at once.
		/// The octant flags select which corner of each box yields the near plane, so no min/max
		/// per axis is needed. Outputs BvhConstants.Far for a child that was missed.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static void SlabTestTwoNodes( BvhNode* child1, BvhNode* child2, in Ray ray,
			float rox, float roy, float roz, bool posX, bool posY, bool posZ,
			out float dist1, out float dist2 )
		{
			float tx1a = ( ( posX ? child1->AabbMin.x : child1->AabbMax.x ) * ray.RD.x ) - rox; /* expect fma. */
			float ty1a = ( ( posY ? child1->AabbMin.y : child1->AabbMax.y ) * ray.RD.y ) - roy;
			float tz1a = ( ( posZ ? child1->AabbMin.z : child1->AabbMax.z ) * ray.RD.z ) - roz;
			float tx1b = ( ( posX ? child2->AabbMin.x : child2->AabbMax.x ) * ray.RD.x ) - rox;
			float ty1b = ( ( posY ? child2->AabbMin.y : child2->AabbMax.y ) * ray.RD.y ) - roy;
			float tz1b = ( ( posZ ? child2->AabbMin.z : child2->AabbMax.z ) * ray.RD.z ) - roz;
			float tx2a = ( ( posX ? child1->AabbMax.x : child1->AabbMin.x ) * ray.RD.x ) - rox;
			float ty2a = ( ( posY ? child1->AabbMax.y : child1->AabbMin.y ) * ray.RD.y ) - roy;
			float tz2a = ( ( posZ ? child1->AabbMax.z : child1->AabbMin.z ) * ray.RD.z ) - roz;
			float tx2b = ( ( posX ? child2->AabbMax.x : child2->AabbMin.x ) * ray.RD.x ) - rox;
			float ty2b = ( ( posY ? child2->AabbMax.y : child2->AabbMin.y ) * ray.RD.y ) - roy;
			float tz2b = ( ( posZ ? child2->AabbMax.z : child2->AabbMin.z ) * ray.RD.z ) - roz;
			float tmina = math.max( math.max( tx1a, ty1a ), math.max( tz1a, 0.0f ) );
			float tminb = math.max( math.max( tx1b, ty1b ), math.max( tz1b, 0.0f ) );
			float tmaxa = math.min( math.min( tx2a, ty2a ), math.min( tz2a, ray.Hit.T ) );
			float tmaxb = math.min( math.min( tx2b, ty2b ), math.min( tz2b, ray.Hit.T ) );
			dist1 = BvhConstants.Far;
			dist2 = BvhConstants.Far;
			if ( tmaxa >= tmina )
			{
				dist1 = tmina;
			}
			if ( tmaxb >= tminb )
			{
				dist2 = tminb;
			}
		}

		/// <summary>
		/// Port of the opacity-map evaluation shared by BVHBase::IntersectTri and
		/// BVHBase::TriOccludes: maps the barycentrics onto one of the OpMapN^2 micro-triangles
		/// and returns its bit. Only called when OpMap is set.
		///
		/// The three truncations are written as explicit ( int ) casts of a float product so that
		/// Mono, which is free to evaluate float expressions at higher precision, and Burst agree.
		/// Like the C++, the index is not clamped: a hit with u + v == 1 exactly indexes past the
		/// primitive's own words, so a map buffer needs a couple of slack words at the end.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		internal bool OpacityOpaque( uint triIdx, float u, float v )
		{
			float fN = OpMapN;
			int row = ( int )( ( u + v ) * fN );
			int diag = ( int )( ( 1f - u ) * fN );
			int idx = ( row * row ) + ( int )( v * fN ) + ( diag - ( ( int )OpMapN - 1 - row ) );
			uint* om = OpMap + ( triIdx * ( ( ( OpMapN * OpMapN ) + 31 ) >> 5 ) );
			return ( om[ idx >> 5 ] & ( 1u << ( idx & 31 ) ) ) != 0;
		}

		/// <summary>
		/// Port of BVHBase::IntersectTri, Moeller-Trumbore path including the opacity map test.
		/// Registers a hit only when it is closer than the current ray.Hit.T.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		internal void IntersectTri( ref Ray ray, uint triIdx, uint i0, uint i1, uint i2 )
		{
			// Moeller-Trumbore ray/triangle intersection algorithm.
			float4 v0_ = Vertex( i0 );
			float3 v0 = v0_.xyz;
			float3 e1 = ( Vertex( i1 ) - v0_ ).xyz;
			float3 e2 = ( Vertex( i2 ) - v0_ ).xyz;
			float3 h = math.cross( ray.D, e2 );
			float a = math.dot( e1, h );
			if ( math.abs( a ) < 0.000001f )
			{
				return;
			}
			float f = 1f / a;
			float3 s = ray.O - v0;
			float u = f * math.dot( s, h );
			float3 q = math.cross( s, e1 );
			float v = f * math.dot( ray.D, q );
			bool miss = u < 0f || v < 0f || ( u + v ) > 1f;
			if ( miss )
			{
				return;
			}
			float t = f * math.dot( e2, q );
			if ( t < 0f || t > ray.Hit.T )
			{
				return;
			}
			// evaluate opacity map, if present.
			if ( OpMap != null && !OpacityOpaque( triIdx, u, v ) )
			{
				return;
			}
			// register a hit: ray is shortened to t.
			ray.Hit.T = t;
			ray.Hit.U = u;
			ray.Hit.V = v;
			// INST_IDX_BITS == 32: the instance index lives in its own field.
			ray.Hit.Prim = triIdx;
			ray.Hit.Inst = ray.InstIdx;
		}

		/// <summary>
		/// Port of BVHBase::TriOccludes, Moeller-Trumbore path including the opacity map test.
		/// ray.Hit.T is the maximum distance; the ray is not modified.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		internal bool TriOccludes( in Ray ray, uint triIdx, uint i0, uint i1, uint i2 )
		{
			// Moeller-Trumbore ray/triangle intersection algorithm
			float4 v0_ = Vertex( i0 );
			float3 v0 = v0_.xyz;
			float3 e1 = ( Vertex( i1 ) - v0_ ).xyz;
			float3 e2 = ( Vertex( i2 ) - v0_ ).xyz;
			float3 h = math.cross( ray.D, e2 );
			float a = math.dot( e1, h );
			if ( math.abs( a ) < 0.000001f )
			{
				return false;
			}
			float f = 1f / a;
			float3 s = ray.O - v0;
			float u = f * math.dot( s, h );
			float3 q = math.cross( s, e1 );
			float v = f * math.dot( ray.D, q );
			bool miss = u < 0f || v < 0f || ( u + v ) > 1f;
			if ( miss )
			{
				return false;
			}
			float t = f * math.dot( e2, q );
			if ( t < 0f || t > ray.Hit.T )
			{
				return false;
			}
			// evaluate opacity map, if present.
			if ( OpMap != null && !OpacityOpaque( triIdx, u, v ) )
			{
				return false;
			}
			// occluded.
			return true;
		}
	}
}
