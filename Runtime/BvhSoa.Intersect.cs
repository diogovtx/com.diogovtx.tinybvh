using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using static Unity.Burst.Intrinsics.X86;

namespace TinyBVH
{
	/// <summary>
	/// Traversal half of tinybvh's BVH_SoA class, which the C++ only compiles under BVH_USEAVX
	/// even though the slab test itself is 128-bit: every node operation is an _mm_ intrinsic on
	/// the xxxx/yyyy/zzzz lanes, so SSE is all the algorithm needs.
	///
	/// The work runs in a Burst direct call rather than in the struct methods themselves: the slab
	/// test and the Moeller-Trumbore test have to match a compiled C++ build bit for bit, and Mono
	/// evaluates float expressions in double. Both a SIMD path and a scalar fallback are provided,
	/// as in Bvh4Cpu.Intersect.cs; unlike there, the two agree bit for bit, because the transpose
	/// the C++ performs only reorders lanes and min/max are exact.
	/// </summary>
	public unsafe partial struct BvhSoa
	{
		/// <summary>Port of BVH_SoA::Intersect( Ray&amp; ). Returns the traversal cost; the hit, if any, is written to ray.Hit.</summary>
		public int Intersect( ref Ray ray )
		{
			if ( Nodes == null || UsedNodes == 0 )
			{
				return 0;
			}
			float cost = 0f;
			BvhSoaTraversal.Intersect( ref this, ref ray, ref cost );
			return ( int )cost; // cast to not break interface.
		}

		/// <summary>
		/// Port of BVH_SoA::IsOccluded( const Ray&amp; ). Returns true as soon as any primitive is
		/// hit within ray.Hit.T; the ray itself is left untouched.
		/// </summary>
		public bool IsOccluded( in Ray ray )
		{
			if ( Nodes == null || UsedNodes == 0 )
			{
				return false;
			}
			// the Burst entry point takes the ray by ref, so it is handed the address of a copy.
			Ray copy = ray;
			int occluded = 0;
			BvhSoaTraversal.IsOccluded( ref this, ref copy, ref occluded );
			return occluded != 0;
		}
	}

	/// <summary>
	/// Burst-compiled implementation of the BVH_SoA traversal. Direct calls must be synchronous,
	/// otherwise editor tests silently run the Mono fallback and drift in the last bits.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhSoaTraversal
	{
		/// <summary>Traversal stack depth, the C++ 'BVHNode* stack[64]'.</summary>
		private const int StackSize = 64;

		/// <summary>_MM_SHUFFLE( 1, 0, 1, 0 ): the low two lanes of each operand.</summary>
		private const int ShuffleLow = ( 1 << 6 ) | ( 0 << 4 ) | ( 1 << 2 ) | 0;
		/// <summary>_MM_SHUFFLE( 3, 2, 3, 2 ): the high two lanes of each operand.</summary>
		private const int ShuffleHigh = ( 3 << 6 ) | ( 2 << 4 ) | ( 3 << 2 ) | 2;

		/// <summary>Port of _mm_min_ps for one lane: the second operand wins a tie, unlike math.min.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float Min( float a, float b )
		{
			return a < b ? a : b;
		}

		/// <summary>Port of _mm_max_ps for one lane; see <see cref="Min"/>.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float Max( float a, float b )
		{
			return a > b ? a : b;
		}

		/// <summary>Port of BVH_SoA::Intersect( Ray&amp; ).</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void Intersect( ref BvhSoa soa, ref Ray ray, ref float cost )
		{
			if ( Sse.IsSseSupported )
			{
				IntersectSimd( ref soa, ref ray, ref cost );
			}
			else
			{
				IntersectScalar( ref soa, ref ray, ref cost );
			}
		}

		/// <summary>Port of BVH_SoA::IsOccluded( const Ray&amp; ). Writes 1 to occluded on a hit.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void IsOccluded( ref BvhSoa soa, ref Ray ray, ref int occluded )
		{
			if ( Sse.IsSseSupported )
			{
				occluded = IsOccludedSimd( ref soa, ref ray ) ? 1 : 0;
			}
			else
			{
				occluded = IsOccludedScalar( ref soa, ref ray ) ? 1 : 0;
			}
		}

		/// <summary>
		/// Port of the BVH_SoA::Intersect leaf loop. The C++ picks between an indexed and a plain
		/// triangle loop; GetPrimIndices covers both, so one loop is left. The triangle test is
		/// BVHBase::IntersectTri, which the base BVH already has.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static void IntersectLeaf( ref BvhSoa soa, ref Ray ray, BvhSoaNode* node, ref float cost )
		{
			for ( uint i = 0; i < node->TriCount; i++, cost += soa.IntersectionCost )
			{
				uint pi = soa.Source.PrimIdx[ node->FirstTri + i ];
				soa.Source.GetPrimIndices( pi, out uint i0, out uint i1, out uint i2 );
				soa.Source.IntersectTri( ref ray, pi, i0, i1, i2 );
			}
		}

		/// <summary>Port of the BVH_SoA::IsOccluded leaf loop; see <see cref="IntersectLeaf"/>.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static bool OccludedLeaf( ref BvhSoa soa, ref Ray ray, BvhSoaNode* node )
		{
			for ( uint i = 0; i < node->TriCount; i++ )
			{
				uint pi = soa.Source.PrimIdx[ node->FirstTri + i ];
				soa.Source.GetPrimIndices( pi, out uint i0, out uint i1, out uint i2 );
				if ( soa.Source.TriOccludes( ray, pi, i0, i1, i2 ) )
				{
					return true;
				}
			}
			return false;
		}

		/// <summary>Port of the BVH_SoA::Intersect traversal, SSE path.</summary>
		private static void IntersectSimd( ref BvhSoa soa, ref Ray ray, ref float cost )
		{
			BvhSoaNode* node = soa.Nodes;
			BvhSoaNode** stack = stackalloc BvhSoaNode*[ StackSize ];
			uint stackPtr = 0;
			v128 ox4 = Sse.set1_ps( ray.O.x ), rdx4 = Sse.set1_ps( ray.RD.x );
			v128 oy4 = Sse.set1_ps( ray.O.y ), rdy4 = Sse.set1_ps( ray.RD.y );
			v128 oz4 = Sse.set1_ps( ray.O.z ), rdz4 = Sse.set1_ps( ray.RD.z );
			while ( true )
			{
				cost += soa.TraversalCost;
				if ( node->IsLeaf )
				{
					IntersectLeaf( ref soa, ref ray, node, ref cost );
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
					continue;
				}
				v128 x4 = Sse.mul_ps( Sse.sub_ps( Sse.load_ps( &node->Xxxx ), ox4 ), rdx4 );
				v128 y4 = Sse.mul_ps( Sse.sub_ps( Sse.load_ps( &node->Yyyy ), oy4 ), rdy4 );
				v128 z4 = Sse.mul_ps( Sse.sub_ps( Sse.load_ps( &node->Zzzz ), oz4 ), rdz4 );
				// transpose
				v128 t0 = Sse.unpacklo_ps( x4, y4 ), t2 = Sse.unpacklo_ps( z4, z4 );
				v128 t1 = Sse.unpackhi_ps( x4, y4 ), t3 = Sse.unpackhi_ps( z4, z4 );
				v128 xyzw1a = Sse.shuffle_ps( t0, t2, ShuffleLow );
				v128 xyzw2a = Sse.shuffle_ps( t0, t2, ShuffleHigh );
				v128 xyzw1b = Sse.shuffle_ps( t1, t3, ShuffleLow );
				v128 xyzw2b = Sse.shuffle_ps( t1, t3, ShuffleHigh );
				// process
				v128 tmina4 = Sse.min_ps( xyzw1a, xyzw2a ), tmaxa4 = Sse.max_ps( xyzw1a, xyzw2a );
				v128 tminb4 = Sse.min_ps( xyzw1b, xyzw2b ), tmaxb4 = Sse.max_ps( xyzw1b, xyzw2b );
				// transpose back
				t0 = Sse.unpacklo_ps( tmina4, tmaxa4 );
				t2 = Sse.unpacklo_ps( tminb4, tmaxb4 );
				t1 = Sse.unpackhi_ps( tmina4, tmaxa4 );
				t3 = Sse.unpackhi_ps( tminb4, tmaxb4 );
				x4 = Sse.shuffle_ps( t0, t2, ShuffleLow );
				y4 = Sse.shuffle_ps( t0, t2, ShuffleHigh );
				z4 = Sse.shuffle_ps( t1, t3, ShuffleLow );
				uint lidx = node->Left, ridx = node->Right;
				v128 min4 = Sse.max_ps( Sse.max_ps( Sse.max_ps( x4, y4 ), z4 ), Sse.setzero_ps() );
				v128 max4 = Sse.min_ps( Sse.min_ps( Sse.min_ps( x4, y4 ), z4 ), Sse.set1_ps( ray.Hit.T ) );
				float tmina_0 = min4.Float0, tmaxa_1 = max4.Float1;
				float tminb_2 = min4.Float2, tmaxb_3 = max4.Float3;
				float dist1 = tmaxa_1 >= tmina_0 ? tmina_0 : BvhConstants.Far;
				float dist2 = tmaxb_3 >= tminb_2 ? tminb_2 : BvhConstants.Far;
				if ( dist1 > dist2 )
				{
					float t = dist1;
					dist1 = dist2;
					dist2 = t;
					uint i = lidx;
					lidx = ridx;
					ridx = i;
				}
				if ( dist1 == BvhConstants.Far )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
				}
				else
				{
					node = soa.Nodes + lidx;
					if ( dist2 != BvhConstants.Far )
					{
						stack[ stackPtr++ ] = soa.Nodes + ridx;
					}
				}
			}
		}

		/// <summary>
		/// Scalar fallback for <see cref="IntersectSimd"/>. The C++ transpose only moves lanes
		/// around, so the four distances reduce to a plain per-axis slab test; min and max are
		/// exact, so this produces the same bits as the SSE path.
		/// </summary>
		private static void IntersectScalar( ref BvhSoa soa, ref Ray ray, ref float cost )
		{
			BvhSoaNode* node = soa.Nodes;
			BvhSoaNode** stack = stackalloc BvhSoaNode*[ StackSize ];
			uint stackPtr = 0;
			while ( true )
			{
				cost += soa.TraversalCost;
				if ( node->IsLeaf )
				{
					IntersectLeaf( ref soa, ref ray, node, ref cost );
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
					continue;
				}
				uint lidx = node->Left, ridx = node->Right;
				SlabTestTwoChildren( node, ray, out float dist1, out float dist2 );
				if ( dist1 > dist2 )
				{
					float t = dist1;
					dist1 = dist2;
					dist2 = t;
					uint i = lidx;
					lidx = ridx;
					ridx = i;
				}
				if ( dist1 == BvhConstants.Far )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
				}
				else
				{
					node = soa.Nodes + lidx;
					if ( dist2 != BvhConstants.Far )
					{
						stack[ stackPtr++ ] = soa.Nodes + ridx;
					}
				}
			}
		}

		/// <summary>Port of the BVH_SoA::IsOccluded traversal, SSE path.</summary>
		private static bool IsOccludedSimd( ref BvhSoa soa, ref Ray ray )
		{
			BvhSoaNode* node = soa.Nodes;
			BvhSoaNode** stack = stackalloc BvhSoaNode*[ StackSize ];
			uint stackPtr = 0;
			v128 ox4 = Sse.set1_ps( ray.O.x ), rdx4 = Sse.set1_ps( ray.RD.x );
			v128 oy4 = Sse.set1_ps( ray.O.y ), rdy4 = Sse.set1_ps( ray.RD.y );
			v128 oz4 = Sse.set1_ps( ray.O.z ), rdz4 = Sse.set1_ps( ray.RD.z );
			while ( true )
			{
				if ( node->IsLeaf )
				{
					if ( OccludedLeaf( ref soa, ref ray, node ) )
					{
						return true;
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
					continue;
				}
				v128 x4 = Sse.mul_ps( Sse.sub_ps( Sse.load_ps( &node->Xxxx ), ox4 ), rdx4 );
				v128 y4 = Sse.mul_ps( Sse.sub_ps( Sse.load_ps( &node->Yyyy ), oy4 ), rdy4 );
				v128 z4 = Sse.mul_ps( Sse.sub_ps( Sse.load_ps( &node->Zzzz ), oz4 ), rdz4 );
				// transpose
				v128 t0 = Sse.unpacklo_ps( x4, y4 ), t2 = Sse.unpacklo_ps( z4, z4 );
				v128 t1 = Sse.unpackhi_ps( x4, y4 ), t3 = Sse.unpackhi_ps( z4, z4 );
				v128 xyzw1a = Sse.shuffle_ps( t0, t2, ShuffleLow );
				v128 xyzw2a = Sse.shuffle_ps( t0, t2, ShuffleHigh );
				v128 xyzw1b = Sse.shuffle_ps( t1, t3, ShuffleLow );
				v128 xyzw2b = Sse.shuffle_ps( t1, t3, ShuffleHigh );
				// process
				v128 tmina4 = Sse.min_ps( xyzw1a, xyzw2a ), tmaxa4 = Sse.max_ps( xyzw1a, xyzw2a );
				v128 tminb4 = Sse.min_ps( xyzw1b, xyzw2b ), tmaxb4 = Sse.max_ps( xyzw1b, xyzw2b );
				// transpose back
				t0 = Sse.unpacklo_ps( tmina4, tmaxa4 );
				t2 = Sse.unpacklo_ps( tminb4, tmaxb4 );
				t1 = Sse.unpackhi_ps( tmina4, tmaxa4 );
				t3 = Sse.unpackhi_ps( tminb4, tmaxb4 );
				x4 = Sse.shuffle_ps( t0, t2, ShuffleLow );
				y4 = Sse.shuffle_ps( t0, t2, ShuffleHigh );
				z4 = Sse.shuffle_ps( t1, t3, ShuffleLow );
				uint lidx = node->Left, ridx = node->Right;
				v128 min4 = Sse.max_ps( Sse.max_ps( Sse.max_ps( x4, y4 ), z4 ), Sse.setzero_ps() );
				v128 max4 = Sse.min_ps( Sse.min_ps( Sse.min_ps( x4, y4 ), z4 ), Sse.set1_ps( ray.Hit.T ) );
				float tmina_0 = min4.Float0, tmaxa_1 = max4.Float1;
				float tminb_2 = min4.Float2, tmaxb_3 = max4.Float3;
				float dist1 = tmaxa_1 >= tmina_0 ? tmina_0 : BvhConstants.Far;
				float dist2 = tmaxb_3 >= tminb_2 ? tminb_2 : BvhConstants.Far;
				if ( dist1 > dist2 )
				{
					float t = dist1;
					dist1 = dist2;
					dist2 = t;
					uint i = lidx;
					lidx = ridx;
					ridx = i;
				}
				if ( dist1 == BvhConstants.Far )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
				}
				else
				{
					node = soa.Nodes + lidx;
					if ( dist2 != BvhConstants.Far )
					{
						stack[ stackPtr++ ] = soa.Nodes + ridx;
					}
				}
			}
			return false;
		}

		/// <summary>Scalar fallback for <see cref="IsOccludedSimd"/>; see <see cref="IntersectScalar"/>.</summary>
		private static bool IsOccludedScalar( ref BvhSoa soa, ref Ray ray )
		{
			BvhSoaNode* node = soa.Nodes;
			BvhSoaNode** stack = stackalloc BvhSoaNode*[ StackSize ];
			uint stackPtr = 0;
			while ( true )
			{
				if ( node->IsLeaf )
				{
					if ( OccludedLeaf( ref soa, ref ray, node ) )
					{
						return true;
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
					continue;
				}
				uint lidx = node->Left, ridx = node->Right;
				SlabTestTwoChildren( node, ray, out float dist1, out float dist2 );
				if ( dist1 > dist2 )
				{
					float t = dist1;
					dist1 = dist2;
					dist2 = t;
					uint i = lidx;
					lidx = ridx;
					ridx = i;
				}
				if ( dist1 == BvhConstants.Far )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
				}
				else
				{
					node = soa.Nodes + lidx;
					if ( dist2 != BvhConstants.Far )
					{
						stack[ stackPtr++ ] = soa.Nodes + ridx;
					}
				}
			}
			return false;
		}

		/// <summary>
		/// Scalar form of the SoA slab test: the C++ scales the four lanes of each axis, transposes
		/// them into per-child min/max vectors, reduces over the axes and reads lanes 0 and 1 for
		/// the left child and 2 and 3 for the right one. Written out per child here.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static void SlabTestTwoChildren( BvhSoaNode* node, in Ray ray, out float dist1, out float dist2 )
		{
			float* xxxx = ( float* )&node->Xxxx;
			float* yyyy = ( float* )&node->Yyyy;
			float* zzzz = ( float* )&node->Zzzz;
			float x0 = ( xxxx[ 0 ] - ray.O.x ) * ray.RD.x, x1 = ( xxxx[ 1 ] - ray.O.x ) * ray.RD.x;
			float x2 = ( xxxx[ 2 ] - ray.O.x ) * ray.RD.x, x3 = ( xxxx[ 3 ] - ray.O.x ) * ray.RD.x;
			float y0 = ( yyyy[ 0 ] - ray.O.y ) * ray.RD.y, y1 = ( yyyy[ 1 ] - ray.O.y ) * ray.RD.y;
			float y2 = ( yyyy[ 2 ] - ray.O.y ) * ray.RD.y, y3 = ( yyyy[ 3 ] - ray.O.y ) * ray.RD.y;
			float z0 = ( zzzz[ 0 ] - ray.O.z ) * ray.RD.z, z1 = ( zzzz[ 1 ] - ray.O.z ) * ray.RD.z;
			float z2 = ( zzzz[ 2 ] - ray.O.z ) * ray.RD.z, z3 = ( zzzz[ 3 ] - ray.O.z ) * ray.RD.z;
			float tmina_0 = Max( Max( Max( Min( x0, x1 ), Min( y0, y1 ) ), Min( z0, z1 ) ), 0f );
			float tmaxa_1 = Min( Min( Min( Max( x0, x1 ), Max( y0, y1 ) ), Max( z0, z1 ) ), ray.Hit.T );
			float tminb_2 = Max( Max( Max( Min( x2, x3 ), Min( y2, y3 ) ), Min( z2, z3 ) ), 0f );
			float tmaxb_3 = Min( Min( Min( Max( x2, x3 ), Max( y2, y3 ) ), Max( z2, z3 ) ), ray.Hit.T );
			dist1 = tmaxa_1 >= tmina_0 ? tmina_0 : BvhConstants.Far;
			dist2 = tmaxb_3 >= tminb_2 ? tminb_2 : BvhConstants.Far;
		}
	}
}
