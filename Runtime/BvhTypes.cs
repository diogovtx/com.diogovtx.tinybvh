using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	public static class BvhConstants
	{
		/// <summary>Miss distance, tinybvh's BVH_FAR.</summary>
		public const float Far = 1e30f;
		/// <summary>Bin count of the binned SAH builder (BVHBINS).</summary>
		public const int Bins = 8;
		/// <summary>Bin count of the SBVH builder (HQBVHBINS).</summary>
		public const int HqBins = 8;
		/// <summary>Threaded builds hand the subtrees rooted at this depth to the job system (MT_SPAWN_DEPTH).</summary>
		public const int MtSpawnDepth = 9;
		/// <summary>Builds of fewer primitives than this stay single-threaded (MT_BUILD_THRESHOLD).</summary>
		public const uint MtBuildThreshold = 50000;
		/// <summary>Largest reciprocal-direction magnitude; see BvhMath.Rcp.</summary>
		public const float RcpMax = 1e30f;
		/// <summary>Default ray mask: intersect every instance (RAY_MASK_INTERSECT_ALL).</summary>
		public const uint RayMaskIntersectAll = 0xFFFF;
		/// <summary>Default SAH cost of a traversal step (C_TRAV).</summary>
		public const float DefaultTraversalCost = 1f;
		/// <summary>Default SAH cost of a primitive intersection (C_INT).</summary>
		public const float DefaultIntersectionCost = 1f;
		/// <summary>Weight of the EPO term in the blended EPO/SAH tree quality metric (W_EPO).</summary>
		public const float EpoWeight = 0.71f;
	}

	/// <summary>Wald 32-byte BVH node. Two fit in a cache line.</summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct BvhNode
	{
		public float3 AabbMin;
		/// <summary>Index of the left child for interior nodes, of the first primitive index for leaves.</summary>
		public uint LeftFirst;
		public float3 AabbMax;
		/// <summary>Primitive count; zero for interior nodes.</summary>
		public uint TriCount;

		public bool IsLeaf => TriCount > 0;
		public float SurfaceArea => BvhMath.SurfaceArea( AabbMin, AabbMax );

		/// <summary>
		/// Port of BVH::BVHNode::Intersect( const bvhvec3&amp; bmin, const bvhvec3&amp; bmax ): true
		/// when the node bounds overlap the given box. Strict on both sides, as in the C++, so
		/// boxes that merely touch do not count as overlapping.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public bool Intersect( float3 bmin, float3 bmax )
		{
			return bmin.x < AabbMax.x && bmax.x > AabbMin.x &&
				bmin.y < AabbMax.y && bmax.y > AabbMin.y &&
				bmin.z < AabbMax.z && bmax.z > AabbMin.z;
		}
	}

	/// <summary>Bounds of an input primitive, as used during construction.</summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct Fragment
	{
		public float3 BMin;
		public uint PrimIdx;
		public float3 BMax;
		/// <summary>Non-zero when the fragment is the result of clipping (spatial splits only).</summary>
		public uint Clipped;

		public void Extend( float3 p )
		{
			BMin = math.min( p, BMin );
			BMax = math.max( p, BMax );
		}
	}

	/// <summary>Intersection record. Fits in five 32-bit values.</summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct Intersection
	{
		/// <summary>Instance index, set by TLAS traversal.</summary>
		public uint Inst;
		/// <summary>Distance along the ray; BvhConstants.Far when nothing was hit.</summary>
		public float T;
		public float U;
		public float V;
		/// <summary>Primitive index.</summary>
		public uint Prim;
	}

	/// <summary>Ray with precomputed reciprocal direction.</summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct Ray
	{
		public float3 O;
		/// <summary>16-bit instance mask; compared against BlasInstance.Mask during TLAS traversal.</summary>
		public uint Mask;
		public float3 D;
		public uint InstIdx;
		/// <summary>Reciprocal direction. Must be kept in sync with D for single BLAS traversal.</summary>
		public float3 RD;
		public Intersection Hit;

		public Ray( float3 origin, float3 direction, float t = BvhConstants.Far, uint mask = BvhConstants.RayMaskIntersectAll )
		{
			O = origin;
			D = math.normalize( direction );
			RD = BvhMath.Rcp( D );
			Mask = mask & BvhConstants.RayMaskIntersectAll;
			InstIdx = 0;
			Hit = default;
			Hit.T = t;
		}
	}

	/// <summary>
	/// Row-major 4x4 matrix, matching tinybvh's bvhmat4 memory layout: Row0 holds cells 0..3,
	/// so the translation lives in Row0.w, Row1.w and Row2.w.
	/// </summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct BvhMat4
	{
		public float4 Row0;
		public float4 Row1;
		public float4 Row2;
		public float4 Row3;

		public static BvhMat4 Identity => new BvhMat4
		{
			Row0 = new float4( 1f, 0f, 0f, 0f ),
			Row1 = new float4( 0f, 1f, 0f, 0f ),
			Row2 = new float4( 0f, 0f, 1f, 0f ),
			Row3 = new float4( 0f, 0f, 0f, 1f )
		};

		/// <summary>Cell access in tinybvh order: index = row * 4 + column.</summary>
		public unsafe float this[ int i ]
		{
			get
			{
				return ( ( float* )UnsafeUtility.AddressOf( ref Row0 ) )[ i ];
			}
			set
			{
				( ( float* )UnsafeUtility.AddressOf( ref Row0 ) )[ i ] = value;
			}
		}

		/// <summary>Port of tinybvh_transform_point, same operation order.</summary>
		public float3 TransformPoint( float3 v )
		{
			float3 res = new float3(
				( Row0.x * v.x ) + ( Row0.y * v.y ) + ( Row0.z * v.z ) + Row0.w,
				( Row1.x * v.x ) + ( Row1.y * v.y ) + ( Row1.z * v.z ) + Row1.w,
				( Row2.x * v.x ) + ( Row2.y * v.y ) + ( Row2.z * v.z ) + Row2.w );
			float w = ( Row3.x * v.x ) + ( Row3.y * v.y ) + ( Row3.z * v.z ) + Row3.w;
			if ( w == 1f )
			{
				return res;
			}
			return res * ( 1f / w );
		}

		/// <summary>Port of tinybvh_transform_vector, same operation order.</summary>
		public float3 TransformVector( float3 v )
		{
			return new float3(
				( Row0.x * v.x ) + ( Row0.y * v.y ) + ( Row0.z * v.z ),
				( Row1.x * v.x ) + ( Row1.y * v.y ) + ( Row1.z * v.z ),
				( Row2.x * v.x ) + ( Row2.y * v.y ) + ( Row2.z * v.z ) );
		}

		/// <summary>Converts from Unity.Mathematics column storage (float4x4.c0 is the first column).</summary>
		public static BvhMat4 FromFloat4x4( float4x4 m )
		{
			float4x4 t = math.transpose( m );
			return new BvhMat4 { Row0 = t.c0, Row1 = t.c1, Row2 = t.c2, Row3 = t.c3 };
		}

		public float4x4 ToFloat4x4()
		{
			return math.transpose( new float4x4( Row0, Row1, Row2, Row3 ) );
		}
	}

	/// <summary>A BLAS placed in a TLAS with a transform. 64-byte aligned in C++; padding is not needed here.</summary>
	[StructLayout( LayoutKind.Sequential )]
	public partial struct BlasInstance
	{
		public BvhMat4 Transform;
		public BvhMat4 InvTransform;
		/// <summary>World-space bounds, computed by Update.</summary>
		public float3 AabbMin;
		public uint BlasIdx;
		public float3 AabbMax;
		public uint Mask;

		public static BlasInstance Create( uint blasIdx )
		{
			return new BlasInstance
			{
				Transform = BvhMat4.Identity,
				InvTransform = BvhMat4.Identity,
				AabbMin = new float3( BvhConstants.Far ),
				BlasIdx = blasIdx,
				AabbMax = new float3( -BvhConstants.Far ),
				Mask = BvhConstants.RayMaskIntersectAll
			};
		}
	}

	public static class BvhMath
	{
		/// <summary>
		/// Port of tinybvh_safercp: 1/x, or a huge value with the sign of x when that is not finite.
		/// Deviation: the C++ returns +-FLT_MAX, which relies on the slab test's origin term overflowing
		/// to infinity and the resulting NaNs falling through min/max. That only works when every
		/// intermediate is rounded to single precision (Mono evaluates float arithmetic in double), so
		/// the magnitude is capped at RcpMax instead: large enough to cull any box the ray is outside
		/// of on that axis, small enough that products with scene coordinates stay finite.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float Rcp( float x )
		{
			float r = 1f / x;
			if ( !( math.abs( r ) < BvhConstants.RcpMax ) )
			{
				r = ( x < 0f || ( x == 0f && ( math.asuint( x ) & 0x80000000u ) != 0 ) ) ? -BvhConstants.RcpMax : BvhConstants.RcpMax;
			}
			return r;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float3 Rcp( float3 a )
		{
			return new float3( Rcp( a.x ), Rcp( a.y ), Rcp( a.z ) );
		}

		/// <summary>Port of BVHBase::SA, same operation order.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float SurfaceArea( float3 aabbMin, float3 aabbMax )
		{
			float3 e = aabbMax - aabbMin;
			return ( e.x * e.y ) + ( e.y * e.z ) + ( e.z * e.x );
		}

		/// <summary>Port of tinybvh_intersect_aabb (slab test). Returns entry distance, or Far on a miss.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float IntersectAabb( in Ray ray, float3 aabbMin, float3 aabbMax )
		{
			float tx1 = ( aabbMin.x - ray.O.x ) * ray.RD.x;
			float tx2 = ( aabbMax.x - ray.O.x ) * ray.RD.x;
			float tmin = math.min( tx1, tx2 );
			float tmax = math.max( tx1, tx2 );
			float ty1 = ( aabbMin.y - ray.O.y ) * ray.RD.y;
			float ty2 = ( aabbMax.y - ray.O.y ) * ray.RD.y;
			tmin = math.max( tmin, math.min( ty1, ty2 ) );
			tmax = math.min( tmax, math.max( ty1, ty2 ) );
			float tz1 = ( aabbMin.z - ray.O.z ) * ray.RD.z;
			float tz2 = ( aabbMax.z - ray.O.z ) * ray.RD.z;
			tmin = math.max( tmin, math.min( tz1, tz2 ) );
			tmax = math.min( tmax, math.max( tz1, tz2 ) );
			if ( tmax >= tmin && tmin < ray.Hit.T && tmax >= 0f )
			{
				return tmin;
			}
			return BvhConstants.Far;
		}
	}
}
