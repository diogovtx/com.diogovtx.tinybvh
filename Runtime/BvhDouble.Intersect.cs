using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Port of tinybvh's customIntersect callback for BVH_Double. Intersects one custom primitive
	/// and updates ray-&gt;Hit when the hit is closer than ray-&gt;Hit.T; returns 1 when it registered
	/// a hit. Returns a byte rather than a bool: bool is not in the list of types Burst accepts as
	/// a function pointer return type.
	/// </summary>
	[UnmanagedFunctionPointer( CallingConvention.Cdecl )]
	public unsafe delegate byte CustomIntersectDoubleDelegate( RayDouble* ray, ulong prim );

	/// <summary>
	/// Port of tinybvh's customIsOccluded callback for BVH_Double. Returns 1 when the primitive
	/// blocks the ray within ray-&gt;Hit.T. The ray is passed by pointer for symmetry with
	/// CustomIntersectDoubleDelegate, but, as in the C++ (which takes a const RayEx&amp;), it must
	/// not be modified.
	/// </summary>
	[UnmanagedFunctionPointer( CallingConvention.Cdecl )]
	public unsafe delegate byte CustomOccludedDoubleDelegate( RayDouble* ray, ulong prim );

	/// <summary>
	/// Traversal half of the BVH_Double port. Unlike the single-precision BVH, the C++ compiles
	/// only one variant of each traversal function, so there is no octant dispatch here.
	/// </summary>
	public unsafe partial struct BvhDouble
	{
		/// <summary>Traversal stack depth of every double-precision traversal (C++: BVHNode* stack[64]).</summary>
		private const int StackSize = 64;

		/// <summary>
		/// Port of BVH_Double::Intersect( RayEx&amp; ). Returns the traversal cost; the hit, if any,
		/// is written to ray.Hit.
		/// </summary>
		public int Intersect( ref RayDouble ray )
		{
			if ( UsedNodes == 0 )
			{
				return 0;
			}
			if ( IsTlas )
			{
				return IntersectTlas( ref ray );
			}
			return IntersectBlas( ref ray );
		}

		/// <summary>
		/// Port of BVH_Double::IsOccluded( const RayEx&amp; ). Returns true as soon as any primitive
		/// is hit within ray.Hit.T; the ray itself is left untouched.
		/// </summary>
		public bool IsOccluded( in RayDouble ray )
		{
			if ( UsedNodes == 0 )
			{
				return false;
			}
			if ( IsTlas )
			{
				return IsOccludedTlas( ray );
			}
			return IsOccludedBlas( ray );
		}

		/// <summary>Port of the BVH_Double::Intersect body: traversal over triangles.</summary>
		private int IntersectBlas( ref RayDouble ray )
		{
			BvhDoubleNode* node = Nodes;
			BvhDoubleNode** stack = stackalloc BvhDoubleNode*[ StackSize ];
			uint stackPtr = 0;
			// the C++ accumulates the traversal cost in a float, not a double; kept as is.
			float cost = 0f;
			while ( true )
			{
				cost += TraversalCost;
				if ( node->IsLeaf )
				{
					if ( CustomIntersect.IsCreated )
					{
						RayDouble* rayPtr = ( RayDouble* )UnsafeUtility.AddressOf( ref ray );
						for ( ulong i = 0; i < node->TriCount; i++ )
						{
							if ( CustomIntersect.Invoke( rayPtr, PrimIdx[ node->LeftFirst + i ] ) != 0 )
							{
								ray.Hit.Inst = ray.InstIdx;
							}
							cost += IntersectionCost;
						}
					}
					else
					{
						for ( ulong i = 0; i < node->TriCount; i++ )
						{
							ulong idx = PrimIdx[ node->LeftFirst + i ];
							GetPrimIndices( idx, out ulong i0, out ulong i1, out ulong i2 );
							IntersectTri( ref ray, idx, i0, i1, i2 );
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
				BvhDoubleNode* child1 = Nodes + node->LeftFirst;
				BvhDoubleNode* child2 = Nodes + node->LeftFirst + 1;
				double dist1 = child1->Intersect( ray ), dist2 = child2->Intersect( ray );
				if ( dist1 > dist2 )
				{
					double td = dist1;
					dist1 = dist2;
					dist2 = td;
					BvhDoubleNode* tn = child1;
					child1 = child2;
					child2 = tn;
				}
				if ( dist1 == BvhDoubleConstants.Far /* missed both child nodes */ )
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
					if ( dist2 != BvhDoubleConstants.Far )
					{
						stack[ stackPtr++ ] = child2; /* push far child */
					}
				}
			}
			return ( int )cost;
		}

		/// <summary>Port of BVH_Double::IntersectTLAS.</summary>
		private int IntersectTlas( ref RayDouble ray )
		{
			BvhDoubleNode* node = Nodes;
			BvhDoubleNode** stack = stackalloc BvhDoubleNode*[ StackSize ];
			uint stackPtr = 0;
			float cost = 0f;
			while ( true )
			{
				cost += TraversalCost;
				if ( node->IsLeaf )
				{
					RayDouble tmp = default;
					for ( ulong i = 0; i < node->TriCount; i++ )
					{
						// BLAS traversal
						ulong instIdx = PrimIdx[ node->LeftFirst + i ];
						BlasInstanceDouble* inst = Instances + instIdx;
						if ( ( inst->Mask & ray.Mask ) == 0 )
						{
							continue;
						}
						BvhDouble* blas = Blasses + inst->BlasIdx;
						// 1. Transform ray with the inverse of the instance transform
						tmp.O = inst->InvTransform.TransformPoint( ray.O );
						tmp.D = inst->InvTransform.TransformVector( ray.D );
						// note that, unlike IsOccludedTLAS, this is a plain reciprocal in the C++.
						tmp.RD = new double3( 1.0 / tmp.D.x, 1.0 / tmp.D.y, 1.0 / tmp.D.z );
						tmp.Hit = ray.Hit;
						// 2. Traverse BLAS with the transformed ray
						tmp.InstIdx = instIdx;
						cost += blas->Intersect( ref tmp );
						// 3. Restore ray
						ray.Hit = tmp.Hit;
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
					continue;
				}
				BvhDoubleNode* child1 = Nodes + node->LeftFirst;
				BvhDoubleNode* child2 = Nodes + node->LeftFirst + 1;
				double dist1 = child1->Intersect( ray ), dist2 = child2->Intersect( ray );
				if ( dist1 > dist2 )
				{
					double td = dist1;
					dist1 = dist2;
					dist2 = td;
					BvhDoubleNode* tn = child1;
					child1 = child2;
					child2 = tn;
				}
				if ( dist1 == BvhDoubleConstants.Far /* missed both child nodes */ )
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
					if ( dist2 != BvhDoubleConstants.Far )
					{
						stack[ stackPtr++ ] = child2; /* push far child */
					}
				}
			}
			return ( int )cost;
		}

		/// <summary>Port of the BVH_Double::IsOccluded body.</summary>
		private bool IsOccludedBlas( in RayDouble ray )
		{
			BvhDoubleNode* node = Nodes;
			BvhDoubleNode** stack = stackalloc BvhDoubleNode*[ StackSize ];
			uint stackPtr = 0;
			while ( true )
			{
				if ( node->IsLeaf )
				{
					if ( CustomIsOccluded.IsCreated )
					{
						// the callback signature takes a pointer where the C++ takes a const RayEx&,
						// so it is handed the address of a copy; ray itself must not change.
						RayDouble tmpRay = ray;
						RayDouble* rayPtr = &tmpRay;
						for ( ulong i = 0; i < node->TriCount; i++ )
						{
							if ( CustomIsOccluded.Invoke( rayPtr, PrimIdx[ node->LeftFirst + i ] ) != 0 )
							{
								return true;
							}
						}
					}
					else
					{
						for ( ulong i = 0; i < node->TriCount; i++ )
						{
							ulong idx = PrimIdx[ node->LeftFirst + i ];
							GetPrimIndices( idx, out ulong i0, out ulong i1, out ulong i2 );
							if ( TriOccludes( ray, i0, i1, i2 ) )
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
				BvhDoubleNode* child1 = Nodes + node->LeftFirst;
				BvhDoubleNode* child2 = Nodes + node->LeftFirst + 1;
				double dist1 = child1->Intersect( ray ), dist2 = child2->Intersect( ray );
				if ( dist1 > dist2 )
				{
					double td = dist1;
					dist1 = dist2;
					dist2 = td;
					BvhDoubleNode* tn = child1;
					child1 = child2;
					child2 = tn;
				}
				if ( dist1 == BvhDoubleConstants.Far /* missed both child nodes */ )
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
					if ( dist2 != BvhDoubleConstants.Far )
					{
						stack[ stackPtr++ ] = child2; /* push far child */
					}
				}
			}
			return false;
		}

		/// <summary>
		/// Port of BVH_Double::IsOccludedTLAS.
		/// Upstream bug, reproduced here on purpose: the C++ declares its transformed ray 'tmp'
		/// outside the traversal loop and never assigns tmp.hit, so the BLAS is queried with an
		/// uninitialised ray length - unlike BVH::IsOccludedTLAS, which does tmpRay.hit.t =
		/// ray.hit.t. The zero-initialised copy used here is the defined equivalent, and it gives
		/// a ray of length zero, so no BLAS ever reports an occlusion. The reference dump agrees:
		/// occludedFull is 0 for all 65536 TLAS rays of all three scenes, because the stack
		/// garbage MSVC left in tmp.hit.t was non-positive as well.
		/// </summary>
		private bool IsOccludedTlas( in RayDouble ray )
		{
			BvhDoubleNode* node = Nodes;
			BvhDoubleNode** stack = stackalloc BvhDoubleNode*[ StackSize ];
			uint stackPtr = 0;
			RayDouble tmp = default;
			while ( true )
			{
				if ( node->IsLeaf )
				{
					for ( ulong i = 0; i < node->TriCount; i++ )
					{
						// BLAS traversal
						BlasInstanceDouble* inst = Instances + PrimIdx[ node->LeftFirst + i ];
						if ( ( inst->Mask & ray.Mask ) == 0 )
						{
							continue;
						}
						BvhDouble* blas = Blasses + inst->BlasIdx;
						// 1. Transform ray with the inverse of the instance transform
						tmp.O = inst->InvTransform.TransformPoint( ray.O );
						tmp.D = inst->InvTransform.TransformVector( ray.D );
						tmp.RD.x = tmp.D.x > 1e-24 ? ( 1.0 / tmp.D.x ) : ( tmp.D.x < -1e-24 ? ( 1.0 / tmp.D.x ) : BvhDoubleConstants.Far );
						tmp.RD.y = tmp.D.y > 1e-24 ? ( 1.0 / tmp.D.y ) : ( tmp.D.y < -1e-24 ? ( 1.0 / tmp.D.y ) : BvhDoubleConstants.Far );
						tmp.RD.z = tmp.D.z > 1e-24 ? ( 1.0 / tmp.D.z ) : ( tmp.D.z < -1e-24 ? ( 1.0 / tmp.D.z ) : BvhDoubleConstants.Far );
						// 2. Traverse BLAS with the transformed ray
						if ( blas->IsOccluded( tmp ) )
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
				BvhDoubleNode* child1 = Nodes + node->LeftFirst;
				BvhDoubleNode* child2 = Nodes + node->LeftFirst + 1;
				double dist1 = child1->Intersect( ray ), dist2 = child2->Intersect( ray );
				if ( dist1 > dist2 )
				{
					double td = dist1;
					dist1 = dist2;
					dist2 = td;
					BvhDoubleNode* tn = child1;
					child1 = child2;
					child2 = tn;
				}
				if ( dist1 == BvhDoubleConstants.Far /* missed both child nodes */ )
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
					if ( dist2 != BvhDoubleConstants.Far )
					{
						stack[ stackPtr++ ] = child2; /* push far child */
					}
				}
			}
			return false;
		}

		/// <summary>
		/// Moeller-Trumbore triangle test of the BVH_Double::Intersect leaf loop, in double
		/// precision. The C++ 'continue's out of the leaf loop; here that is a return.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private void IntersectTri( ref RayDouble ray, ulong idx, ulong i0, ulong i1, ulong i2 )
		{
			double3 e1 = Verts[ i1 ] - Verts[ i0 ];
			double3 e2 = Verts[ i2 ] - Verts[ i0 ];
			double3 h = BvhDoubleMath.Cross( ray.D, e2 );
			double a = BvhDoubleMath.Dot( e1, h );
			if ( math.abs( a ) < 0.0000001 )
			{
				return; // ray parallel to triangle
			}
			double f = 1 / a;
			double3 s = ray.O - Verts[ i0 ];
			double u = f * BvhDoubleMath.Dot( s, h );
			double3 q = BvhDoubleMath.Cross( s, e1 );
			double v = f * BvhDoubleMath.Dot( ray.D, q );
			if ( u < 0 || v < 0 || u + v > 1 )
			{
				return;
			}
			double t = f * BvhDoubleMath.Dot( e2, q );
			if ( t > 0 && t < ray.Hit.T )
			{
				// register a hit: ray is shortened to t
				ray.Hit.T = t;
				ray.Hit.U = u;
				ray.Hit.V = v;
				ray.Hit.Prim = idx;
				ray.Hit.Inst = ray.InstIdx;
			}
		}

		/// <summary>Triangle test of the BVH_Double::IsOccluded leaf loop.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private bool TriOccludes( in RayDouble ray, ulong i0, ulong i1, ulong i2 )
		{
			double3 e1 = Verts[ i1 ] - Verts[ i0 ];
			double3 e2 = Verts[ i2 ] - Verts[ i0 ];
			double3 h = BvhDoubleMath.Cross( ray.D, e2 );
			double a = BvhDoubleMath.Dot( e1, h );
			if ( math.abs( a ) < 0.0000001 )
			{
				return false; // ray parallel to triangle
			}
			double f = 1 / a;
			double3 s = ray.O - Verts[ i0 ];
			double u = f * BvhDoubleMath.Dot( s, h );
			double3 q = BvhDoubleMath.Cross( s, e1 );
			double v = f * BvhDoubleMath.Dot( ray.D, q );
			if ( u < 0 || v < 0 || u + v > 1 )
			{
				return false;
			}
			double t = f * BvhDoubleMath.Dot( e2, q );
			return t > 0 && t < ray.Hit.T;
		}
	}
}
