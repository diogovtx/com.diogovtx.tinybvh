using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Constants of tinybvh's double-precision BVH (DOUBLE_PRECISION_SUPPORT). These live here
	/// rather than in BvhConstants so the double-precision port stays confined to its own files.
	/// </summary>
	public static class BvhDoubleConstants
	{
		/// <summary>Miss distance of the double-precision traversal, tinybvh's BVH_DBL_FAR.</summary>
		public const double Far = 1e300;

		/// <summary>
		/// tinybvh's single-precision BVH_FAR widened to double. The C++ uses this - not
		/// BVH_DBL_FAR - to seed the bounds in BLASInstanceEx::Update and in
		/// BVH_Double::Build( customGetAABB, .. ), so the two are kept apart here as well.
		/// </summary>
		public const double WidenedFar = BvhConstants.Far;
	}

	/// <summary>Double-precision counterparts of the tinybvh_* scalar helpers, in the C++ operation order.</summary>
	public static class BvhDoubleMath
	{
		/// <summary>Port of tinybvh_min( double, double ).</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static double Min( double a, double b )
		{
			return a < b ? a : b;
		}

		/// <summary>Port of tinybvh_max( double, double ).</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static double Max( double a, double b )
		{
			return a > b ? a : b;
		}

		/// <summary>
		/// Port of tinybvh_min( bvhdbl3, bvhdbl3 ). Written out rather than calling math.min, which
		/// carries a NaN guard the C++ ternary does not have.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static double3 Min( double3 a, double3 b )
		{
			return new double3( Min( a.x, b.x ), Min( a.y, b.y ), Min( a.z, b.z ) );
		}

		/// <summary>Port of tinybvh_max( bvhdbl3, bvhdbl3 ).</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static double3 Max( double3 a, double3 b )
		{
			return new double3( Max( a.x, b.x ), Max( a.y, b.y ), Max( a.z, b.z ) );
		}

		/// <summary>Port of tinybvh_dot( bvhdbl3, bvhdbl3 ).</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static double Dot( double3 a, double3 b )
		{
			return ( a.x * b.x ) + ( a.y * b.y ) + ( a.z * b.z );
		}

		/// <summary>Port of tinybvh_cross( bvhdbl3, bvhdbl3 ).</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static double3 Cross( double3 a, double3 b )
		{
			return new double3(
				( a.y * b.z ) - ( a.z * b.y ),
				( a.z * b.x ) - ( a.x * b.z ),
				( a.x * b.y ) - ( a.y * b.x ) );
		}

		/// <summary>Port of tinybvh_halfarea( bvhdbl3 ), used by the SAH evaluation.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static double HalfArea( double3 v )
		{
			return v.x < -BvhDoubleConstants.Far ? 0.0 : ( ( v.x * v.y ) + ( v.y * v.z ) + ( v.z * v.x ) );
		}

		/// <summary>Port of tinybvh_clamp for integers; the double builder uses it on bin indices.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static int Clamp( int x, int a, int b )
		{
			return x > a ? ( x < b ? x : b ) : a;
		}
	}

	/// <summary>
	/// Port of BVH_Double::BVHNode: the double-precision 'traditional' BVH node. Child and
	/// primitive indices are 64-bit, so the node is exactly 64 bytes, as in the C++.
	/// </summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct BvhDoubleNode
	{
		public double3 AabbMin;
		public double3 AabbMax;
		/// <summary>Index of the left child for interior nodes, of the first primitive index for leaves.</summary>
		public ulong LeftFirst;
		/// <summary>Primitive count; zero for interior nodes.</summary>
		public ulong TriCount;

		public bool IsLeaf => TriCount > 0;

		/// <summary>Port of BVH_Double::BVHNode::SurfaceArea, same operation order.</summary>
		public double SurfaceArea
		{
			get
			{
				double3 e = AabbMax - AabbMin;
				return ( e.x * e.y ) + ( e.y * e.z ) + ( e.z * e.x );
			}
		}

		/// <summary>
		/// Port of BVH_Double::BVHNode::Intersect: the double-precision slab test. Returns the
		/// entry distance, or BvhDoubleConstants.Far on a miss.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public double Intersect( in RayDouble ray )
		{
			double tx1 = ( AabbMin.x - ray.O.x ) * ray.RD.x, tx2 = ( AabbMax.x - ray.O.x ) * ray.RD.x;
			double tmin = BvhDoubleMath.Min( tx1, tx2 ), tmax = BvhDoubleMath.Max( tx1, tx2 );
			double ty1 = ( AabbMin.y - ray.O.y ) * ray.RD.y, ty2 = ( AabbMax.y - ray.O.y ) * ray.RD.y;
			tmin = BvhDoubleMath.Max( tmin, BvhDoubleMath.Min( ty1, ty2 ) );
			tmax = BvhDoubleMath.Min( tmax, BvhDoubleMath.Max( ty1, ty2 ) );
			double tz1 = ( AabbMin.z - ray.O.z ) * ray.RD.z, tz2 = ( AabbMax.z - ray.O.z ) * ray.RD.z;
			tmin = BvhDoubleMath.Max( tmin, BvhDoubleMath.Min( tz1, tz2 ) );
			tmax = BvhDoubleMath.Min( tmax, BvhDoubleMath.Max( tz1, tz2 ) );
			if ( tmax >= tmin && tmin < ray.Hit.T && tmax >= 0 )
			{
				return tmin;
			}
			return BvhDoubleConstants.Far;
		}
	}

	/// <summary>Port of BVH_Double::Fragment: the bounds of one input primitive, double precision.</summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct FragmentDouble
	{
		public double3 BMin;
		public double3 BMax;
		/// <summary>Index of the original primitive.</summary>
		public ulong PrimIdx;
	}

	/// <summary>Port of IntersectionEx: the double-precision hit record.</summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct IntersectionDouble
	{
		/// <summary>Distance along the ray; BvhDoubleConstants.Far when nothing was hit.</summary>
		public double T;
		public double U;
		public double V;
		/// <summary>Instance index, set by TLAS traversal.</summary>
		public ulong Inst;
		/// <summary>Primitive index.</summary>
		public ulong Prim;
	}

	/// <summary>Port of RayEx: the double-precision ray definition.</summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct RayDouble
	{
		public double3 O;
		public double3 D;
		/// <summary>Reciprocal direction. Must be kept in sync with D for single BLAS traversal.</summary>
		public double3 RD;
		public IntersectionDouble Hit;
		public ulong InstIdx;
		/// <summary>16-bit instance mask; compared against BlasInstanceDouble.Mask during TLAS traversal.</summary>
		public ulong Mask;

		/// <summary>
		/// Port of the RayEx constructor. The direction is normalised with the C++ expression
		/// ( rl = 1 / sqrt( dot ) ) and the reciprocal is a plain 1 / D per component: unlike the
		/// single-precision Ray, the C++ uses no guarded reciprocal here, so neither does this.
		/// </summary>
		public RayDouble( double3 origin, double3 direction, double t = BvhDoubleConstants.Far, uint mask = BvhConstants.RayMaskIntersectAll )
		{
			O = origin;
			D = direction;
			double rl = 1.0 / math.sqrt( ( D.x * D.x ) + ( D.y * D.y ) + ( D.z * D.z ) );
			D.x *= rl;
			D.y *= rl;
			D.z *= rl;
			RD = new double3( 1.0 / D.x, 1.0 / D.y, 1.0 / D.z );
			Hit = default; // the C++ memsets the whole ray before filling it in.
			Hit.U = 0.0;
			Hit.V = 0.0;
			Hit.T = t;
			InstIdx = 0;
			Mask = mask & BvhConstants.RayMaskIntersectAll;
		}
	}

	/// <summary>
	/// Double-precision counterpart of BvhMat4: the 16 doubles of BLASInstanceEx::transform, in
	/// the same row-major cell order, so cell index = row * 4 + column and the translation lives
	/// in Row0.w, Row1.w and Row2.w.
	/// </summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct BvhMat4Double
	{
		public double4 Row0;
		public double4 Row1;
		public double4 Row2;
		public double4 Row3;

		public static BvhMat4Double Identity => new BvhMat4Double
		{
			Row0 = new double4( 1.0, 0.0, 0.0, 0.0 ),
			Row1 = new double4( 0.0, 1.0, 0.0, 0.0 ),
			Row2 = new double4( 0.0, 0.0, 1.0, 0.0 ),
			Row3 = new double4( 0.0, 0.0, 0.0, 1.0 )
		};

		/// <summary>Cell access in tinybvh order: index = row * 4 + column.</summary>
		public unsafe double this[ int i ]
		{
			get
			{
				return ( ( double* )UnsafeUtility.AddressOf( ref Row0 ) )[ i ];
			}
			set
			{
				( ( double* )UnsafeUtility.AddressOf( ref Row0 ) )[ i ] = value;
			}
		}

		/// <summary>Port of tinybvh_transform_point( bvhdbl3, const double* ), same operation order.</summary>
		public double3 TransformPoint( double3 v )
		{
			double3 res = new double3(
				( Row0.x * v.x ) + ( Row0.y * v.y ) + ( Row0.z * v.z ) + Row0.w,
				( Row1.x * v.x ) + ( Row1.y * v.y ) + ( Row1.z * v.z ) + Row1.w,
				( Row2.x * v.x ) + ( Row2.y * v.y ) + ( Row2.z * v.z ) + Row2.w );
			double w = ( Row3.x * v.x ) + ( Row3.y * v.y ) + ( Row3.z * v.z ) + Row3.w;
			if ( w == 1.0 )
			{
				return res;
			}
			return res * ( 1.0 / w );
		}

		/// <summary>Port of tinybvh_transform_vector( bvhdbl3, const double* ), same operation order.</summary>
		public double3 TransformVector( double3 v )
		{
			return new double3(
				( Row0.x * v.x ) + ( Row0.y * v.y ) + ( Row0.z * v.z ),
				( Row1.x * v.x ) + ( Row1.y * v.y ) + ( Row1.z * v.z ),
				( Row2.x * v.x ) + ( Row2.y * v.y ) + ( Row2.z * v.z ) );
		}
	}

	/// <summary>
	/// Port of BLASInstanceEx: a double-precision BLAS placed in a TLAS with a transform.
	/// Update and InvertTransform live in BvhDouble.Build.cs, next to the TLAS builder.
	/// </summary>
	[StructLayout( LayoutKind.Sequential )]
	public partial struct BlasInstanceDouble
	{
		public BvhMat4Double Transform;
		public BvhMat4Double InvTransform;
		/// <summary>World-space bounds, computed by Update.</summary>
		public double3 AabbMin;
		public ulong BlasIdx;
		public double3 AabbMax;
		public ulong Mask;

		/// <summary>Port of BLASInstanceEx( uint64_t idx ), including the C++ member initialisers.</summary>
		public static BlasInstanceDouble Create( ulong blasIdx )
		{
			return new BlasInstanceDouble
			{
				Transform = BvhMat4Double.Identity,
				InvTransform = BvhMat4Double.Identity,
				AabbMin = new double3( BvhDoubleConstants.Far ),
				BlasIdx = blasIdx,
				AabbMax = new double3( -BvhDoubleConstants.Far ),
				Mask = BvhConstants.RayMaskIntersectAll
			};
		}
	}
}
