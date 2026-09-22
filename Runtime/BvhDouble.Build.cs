using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Fills the bounds of one custom primitive, double precision. Port of the customGetAABB
	/// callback of BVH_Double::Build( void (*customGetAABB)( .. ), primCount ). The bounds are
	/// written through pointers because Burst function pointers only take blittable arguments.
	/// </summary>
	[UnmanagedFunctionPointer( CallingConvention.Cdecl )]
	public unsafe delegate void GetAabbDoubleDelegate( ulong prim, double3* aabbMin, double3* aabbMax );

	/// <summary>
	/// Construction half of the BVH_Double port: PrepareBuild, the TLAS and custom-geometry
	/// builds and the binned SAH subdivision, all in double precision. The heavy lifting lives in
	/// BvhDoubleBuilder so Burst can compile it.
	/// </summary>
	public unsafe partial struct BvhDouble
	{
		/// <summary>Builds over a triangle soup: three consecutive 24-byte vertices per primitive.</summary>
		public void Build( double3* vertices, ulong primCount )
		{
			Build( vertices, null, primCount );
		}

		/// <summary>Builds over triangles whose vertices are addressed through an index buffer.</summary>
		public void Build( double3* vertices, uint* indices, ulong primCount )
		{
			ValidateBuildInput( primCount );
			if ( vertices == null )
			{
				throw new ArgumentException( "BvhDouble.Build( .. ), vertices == null.", nameof( vertices ) );
			}
			BvhDoubleBuilder.PrepareBuild( ref this, vertices, indices, primCount );
			BvhDoubleBuilder.Build( ref this );
		}

		public void Build( NativeArray<double3> vertices, ulong primCount )
		{
			Build( ( double3* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( vertices ), primCount );
		}

		public void Build( NativeArray<double3> vertices, NativeArray<uint> indices, ulong primCount )
		{
			Build(
				( double3* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( vertices ),
				( uint* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( indices ), primCount );
		}

		/// <summary>
		/// TLAS builder. Port of BVH_Double::Build( BLASInstanceEx*, instCount, BVH_Double**,
		/// blasCount ). Pass a null blas list to keep the instance bounds as they are.
		/// </summary>
		public void BuildTlas( BlasInstanceDouble* instances, ulong instCount, BvhDouble* blasses, ulong blasCount )
		{
			ValidateBuildInput( instCount );
			if ( instances == null )
			{
				throw new ArgumentException( "BvhDouble.BuildTlas( .. ), instances == null.", nameof( instances ) );
			}
			BvhDoubleBuilder.PrepareTlasBuild( ref this, instances, instCount, blasses, blasCount );
			BvhDoubleBuilder.Build( ref this );
		}

		/// <summary>
		/// Builds over custom geometry: the bounds of every primitive are obtained from a
		/// Burst-compiled callback (BurstCompiler.CompileFunctionPointer).
		/// </summary>
		public void Build( FunctionPointer<GetAabbDoubleDelegate> getAabb, ulong primCount )
		{
			ValidateBuildInput( primCount );
			if ( !getAabb.IsCreated )
			{
				throw new ArgumentException( "BvhDouble.Build( .. ), getAabb was not compiled.", nameof( getAabb ) );
			}
			BvhDoubleBuilder.PrepareCustomBuild( ref this, getAabb, primCount );
			BvhDoubleBuilder.Build( ref this );
		}

		/// <summary>
		/// Port of BVH_Double::SAHCost: the SAH cost of the tree, an indication of its quality.
		/// Lower is better.
		/// </summary>
		public double SahCost( ulong nodeIdx = 0 )
		{
			BvhDoubleNode* n = Nodes + nodeIdx;
			if ( n->IsLeaf )
			{
				return IntersectionCost * n->SurfaceArea * n->TriCount;
			}
			double cost = ( TraversalCost * n->SurfaceArea ) + SahCost( n->LeftFirst ) + SahCost( n->LeftFirst + 1 );
			return nodeIdx == 0 ? ( cost / n->SurfaceArea ) : cost;
		}

		/// <summary>Managed-side replacement for the BVH_FATAL_ERROR checks in the C++ builders.</summary>
		private void ValidateBuildInput( ulong primCount )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "BvhDouble.Build( .. ), bvh was not created." );
			}
			if ( primCount == 0 )
			{
				throw new ArgumentException( "BvhDouble.Build( .. ), primCount == 0.", nameof( primCount ) );
			}
		}
	}

	public partial struct BlasInstanceDouble
	{
		/// <summary>Port of BLASInstanceEx::Update.</summary>
		public void Update( ref BvhDouble blas )
		{
			InvertTransform(); // TODO: done unconditionally; for a big TLAS this may be wasteful.
			// transform the eight corners of the root node aabb using the
			// instance transform and calculate the worldspace aabb over those.
			// Note that the C++ seeds these with BVH_FAR, not BVH_DBL_FAR.
			AabbMin = new double3( BvhDoubleConstants.WidenedFar );
			AabbMax = new double3( -BvhDoubleConstants.WidenedFar );
			double3 bmin = blas.AabbMin, bmax = blas.AabbMax;
			for ( int j = 0; j < 8; j++ )
			{
				double3 p = new double3(
					( j & 1 ) != 0 ? bmax.x : bmin.x,
					( j & 2 ) != 0 ? bmax.y : bmin.y,
					( j & 4 ) != 0 ? bmax.z : bmin.z );
				double3 t = Transform.TransformPoint( p );
				AabbMin = BvhDoubleMath.Min( AabbMin, t );
				AabbMax = BvhDoubleMath.Max( AabbMax, t );
			}
		}

		/// <summary>Port of BLASInstanceEx::InvertTransform: the inverse of the matrix in 'Transform'.</summary>
		public unsafe void InvertTransform()
		{
			// math from MESA, via http://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
			double* T = ( double* )UnsafeUtility.AddressOf( ref Transform );
			double* iT = ( double* )UnsafeUtility.AddressOf( ref InvTransform );
			iT[ 0 ] = ( T[ 5 ] * T[ 10 ] * T[ 15 ] ) - ( T[ 5 ] * T[ 11 ] * T[ 14 ] ) - ( T[ 9 ] * T[ 6 ] * T[ 15 ] ) + ( T[ 9 ] * T[ 7 ] * T[ 14 ] ) + ( T[ 13 ] * T[ 6 ] * T[ 11 ] ) - ( T[ 13 ] * T[ 7 ] * T[ 10 ] );
			iT[ 1 ] = -( T[ 1 ] * T[ 10 ] * T[ 15 ] ) + ( T[ 1 ] * T[ 11 ] * T[ 14 ] ) + ( T[ 9 ] * T[ 2 ] * T[ 15 ] ) - ( T[ 9 ] * T[ 3 ] * T[ 14 ] ) - ( T[ 13 ] * T[ 2 ] * T[ 11 ] ) + ( T[ 13 ] * T[ 3 ] * T[ 10 ] );
			iT[ 2 ] = ( T[ 1 ] * T[ 6 ] * T[ 15 ] ) - ( T[ 1 ] * T[ 7 ] * T[ 14 ] ) - ( T[ 5 ] * T[ 2 ] * T[ 15 ] ) + ( T[ 5 ] * T[ 3 ] * T[ 14 ] ) + ( T[ 13 ] * T[ 2 ] * T[ 7 ] ) - ( T[ 13 ] * T[ 3 ] * T[ 6 ] );
			iT[ 3 ] = -( T[ 1 ] * T[ 6 ] * T[ 11 ] ) + ( T[ 1 ] * T[ 7 ] * T[ 10 ] ) + ( T[ 5 ] * T[ 2 ] * T[ 11 ] ) - ( T[ 5 ] * T[ 3 ] * T[ 10 ] ) - ( T[ 9 ] * T[ 2 ] * T[ 7 ] ) + ( T[ 9 ] * T[ 3 ] * T[ 6 ] );
			iT[ 4 ] = -( T[ 4 ] * T[ 10 ] * T[ 15 ] ) + ( T[ 4 ] * T[ 11 ] * T[ 14 ] ) + ( T[ 8 ] * T[ 6 ] * T[ 15 ] ) - ( T[ 8 ] * T[ 7 ] * T[ 14 ] ) - ( T[ 12 ] * T[ 6 ] * T[ 11 ] ) + ( T[ 12 ] * T[ 7 ] * T[ 10 ] );
			iT[ 5 ] = ( T[ 0 ] * T[ 10 ] * T[ 15 ] ) - ( T[ 0 ] * T[ 11 ] * T[ 14 ] ) - ( T[ 8 ] * T[ 2 ] * T[ 15 ] ) + ( T[ 8 ] * T[ 3 ] * T[ 14 ] ) + ( T[ 12 ] * T[ 2 ] * T[ 11 ] ) - ( T[ 12 ] * T[ 3 ] * T[ 10 ] );
			iT[ 6 ] = -( T[ 0 ] * T[ 6 ] * T[ 15 ] ) + ( T[ 0 ] * T[ 7 ] * T[ 14 ] ) + ( T[ 4 ] * T[ 2 ] * T[ 15 ] ) - ( T[ 4 ] * T[ 3 ] * T[ 14 ] ) - ( T[ 12 ] * T[ 2 ] * T[ 7 ] ) + ( T[ 12 ] * T[ 3 ] * T[ 6 ] );
			iT[ 7 ] = ( T[ 0 ] * T[ 6 ] * T[ 11 ] ) - ( T[ 0 ] * T[ 7 ] * T[ 10 ] ) - ( T[ 4 ] * T[ 2 ] * T[ 11 ] ) + ( T[ 4 ] * T[ 3 ] * T[ 10 ] ) + ( T[ 8 ] * T[ 2 ] * T[ 7 ] ) - ( T[ 8 ] * T[ 3 ] * T[ 6 ] );
			iT[ 8 ] = ( T[ 4 ] * T[ 9 ] * T[ 15 ] ) - ( T[ 4 ] * T[ 11 ] * T[ 13 ] ) - ( T[ 8 ] * T[ 5 ] * T[ 15 ] ) + ( T[ 8 ] * T[ 7 ] * T[ 13 ] ) + ( T[ 12 ] * T[ 5 ] * T[ 11 ] ) - ( T[ 12 ] * T[ 7 ] * T[ 9 ] );
			iT[ 9 ] = -( T[ 0 ] * T[ 9 ] * T[ 15 ] ) + ( T[ 0 ] * T[ 11 ] * T[ 13 ] ) + ( T[ 8 ] * T[ 1 ] * T[ 15 ] ) - ( T[ 8 ] * T[ 3 ] * T[ 13 ] ) - ( T[ 12 ] * T[ 1 ] * T[ 11 ] ) + ( T[ 12 ] * T[ 3 ] * T[ 9 ] );
			iT[ 10 ] = ( T[ 0 ] * T[ 5 ] * T[ 15 ] ) - ( T[ 0 ] * T[ 7 ] * T[ 13 ] ) - ( T[ 4 ] * T[ 1 ] * T[ 15 ] ) + ( T[ 4 ] * T[ 3 ] * T[ 13 ] ) + ( T[ 12 ] * T[ 1 ] * T[ 7 ] ) - ( T[ 12 ] * T[ 3 ] * T[ 5 ] );
			iT[ 11 ] = -( T[ 0 ] * T[ 5 ] * T[ 11 ] ) + ( T[ 0 ] * T[ 7 ] * T[ 9 ] ) + ( T[ 4 ] * T[ 1 ] * T[ 11 ] ) - ( T[ 4 ] * T[ 3 ] * T[ 9 ] ) - ( T[ 8 ] * T[ 1 ] * T[ 7 ] ) + ( T[ 8 ] * T[ 3 ] * T[ 5 ] );
			iT[ 12 ] = -( T[ 4 ] * T[ 9 ] * T[ 14 ] ) + ( T[ 4 ] * T[ 10 ] * T[ 13 ] ) + ( T[ 8 ] * T[ 5 ] * T[ 14 ] ) - ( T[ 8 ] * T[ 6 ] * T[ 13 ] ) - ( T[ 12 ] * T[ 5 ] * T[ 10 ] ) + ( T[ 12 ] * T[ 6 ] * T[ 9 ] );
			iT[ 13 ] = ( T[ 0 ] * T[ 9 ] * T[ 14 ] ) - ( T[ 0 ] * T[ 10 ] * T[ 13 ] ) - ( T[ 8 ] * T[ 1 ] * T[ 14 ] ) + ( T[ 8 ] * T[ 2 ] * T[ 13 ] ) + ( T[ 12 ] * T[ 1 ] * T[ 10 ] ) - ( T[ 12 ] * T[ 2 ] * T[ 9 ] );
			iT[ 14 ] = -( T[ 0 ] * T[ 5 ] * T[ 14 ] ) + ( T[ 0 ] * T[ 6 ] * T[ 13 ] ) + ( T[ 4 ] * T[ 1 ] * T[ 14 ] ) - ( T[ 4 ] * T[ 2 ] * T[ 13 ] ) - ( T[ 12 ] * T[ 1 ] * T[ 6 ] ) + ( T[ 12 ] * T[ 2 ] * T[ 5 ] );
			iT[ 15 ] = ( T[ 0 ] * T[ 5 ] * T[ 10 ] ) - ( T[ 0 ] * T[ 6 ] * T[ 9 ] ) - ( T[ 4 ] * T[ 1 ] * T[ 10 ] ) + ( T[ 4 ] * T[ 2 ] * T[ 9 ] ) + ( T[ 8 ] * T[ 1 ] * T[ 6 ] ) - ( T[ 8 ] * T[ 2 ] * T[ 5 ] );
			double det = ( T[ 0 ] * iT[ 0 ] ) + ( T[ 1 ] * iT[ 4 ] ) + ( T[ 2 ] * iT[ 8 ] ) + ( T[ 3 ] * iT[ 12 ] );
			if ( det == 0.0 )
			{
				return; // actually, invert failed. That's bad.
			}
			double invdet = 1.0 / det;
			for ( int i = 0; i < 16; i++ )
			{
				iT[ i ] *= invdet;
			}
		}
	}

	/// <summary>
	/// Burst-compiled implementation of the BVH_Double construction routines. These are plain
	/// static methods over a ref BvhDouble so they can be direct-called from managed code.
	/// Note that tinybvh's threaded double build is not ported: the C++ hands subtrees to its
	/// thread pool but the resulting tree is the same either way, only its node numbering differs,
	/// so only the serial path - the one the reference dump uses - is reproduced here.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhDoubleBuilder
	{
		private const int Bins = BvhConstants.Bins;

		/// <summary>Port of BVH_Double::PrepareBuild: allocate and prepare the fragment list.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void PrepareBuild( ref BvhDouble bvh, double3* vertices, uint* indices, ulong primCount )
		{
			// allocate memory on first build
			bvh.AllocateArrays( primCount );
			bvh.Verts = vertices; // note: we're not copying this data; don't delete.
			bvh.VertIdx = indices; // also not copied; for indexed triangle meshes.
			bvh.IdxCount = primCount;
			bvh.TriCount = primCount;
			// prepare fragments
			BvhDoubleNode* root = bvh.Nodes;
			root->LeftFirst = 0;
			root->TriCount = primCount;
			root->AabbMin = new double3( BvhDoubleConstants.Far );
			root->AabbMax = new double3( -BvhDoubleConstants.Far );
			FragmentDouble* fragment = bvh.Fragments;
			ulong* primIdx = bvh.PrimIdx;
			double3* verts = bvh.Verts;
			if ( indices == null )
			{
				// building a BVH over triangles specified as three 24-byte vertices each.
				for ( ulong i = 0; i < primCount; i++ )
				{
					double3 v0 = verts[ i * 3 ], v1 = verts[ ( i * 3 ) + 1 ], v2 = verts[ ( i * 3 ) + 2 ];
					fragment[ i ].BMin = BvhDoubleMath.Min( BvhDoubleMath.Min( v0, v1 ), v2 );
					fragment[ i ].BMax = BvhDoubleMath.Max( BvhDoubleMath.Max( v0, v1 ), v2 );
					root->AabbMin = BvhDoubleMath.Min( root->AabbMin, fragment[ i ].BMin );
					root->AabbMax = BvhDoubleMath.Max( root->AabbMax, fragment[ i ].BMax );
					fragment[ i ].PrimIdx = i;
					primIdx[ i ] = i;
				}
			}
			else
			{
				// building a BVH over triangles consisting of vertices indexed by 'indices'.
				for ( ulong i = 0; i < primCount; i++ )
				{
					uint i0 = indices[ i * 3 ], i1 = indices[ ( i * 3 ) + 1 ], i2 = indices[ ( i * 3 ) + 2 ];
					double3 v0 = verts[ i0 ], v1 = verts[ i1 ], v2 = verts[ i2 ];
					fragment[ i ].BMin = BvhDoubleMath.Min( BvhDoubleMath.Min( v0, v1 ), v2 );
					fragment[ i ].BMax = BvhDoubleMath.Max( BvhDoubleMath.Max( v0, v1 ), v2 );
					root->AabbMin = BvhDoubleMath.Min( root->AabbMin, fragment[ i ].BMin );
					root->AabbMax = BvhDoubleMath.Max( root->AabbMax, fragment[ i ].BMax );
					fragment[ i ].PrimIdx = i;
					primIdx[ i ] = i;
				}
			}
			// reset node pool; the double builder starts allocating children at node 1.
			bvh.UsedNodes = 1;
			bvh.Instances = null;
			bvh.Blasses = null;
			bvh.BlasCount = 0;
			bvh.BvhOverIndices = indices != null;
			// all set; actual build happens in BvhDoubleBuilder.Build.
		}

		/// <summary>Port of BVH_Double::Build( BLASInstanceEx*, instCount, BVH_Double**, blasCount ).</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void PrepareTlasBuild( ref BvhDouble bvh, BlasInstanceDouble* instances, ulong instCount, BvhDouble* blasses, ulong bCount )
		{
			bvh.TriCount = instCount;
			bvh.IdxCount = instCount;
			bvh.AllocateArrays( instCount );
			bvh.Instances = instances;
			bvh.Blasses = blasses;
			bvh.BlasCount = bCount;
			// the C++ leaves verts untouched here and relies on it being null to derive
			// bvh_over_aabbs; clear it so rebuilding a TLAS over a used BvhDouble stays correct.
			bvh.Verts = null;
			bvh.VertIdx = null;
			bvh.BvhOverIndices = false;
			// copy relevant data from instance array
			BvhDoubleNode* root = bvh.Nodes;
			root->LeftFirst = 0;
			root->TriCount = instCount;
			root->AabbMin = new double3( BvhDoubleConstants.Far );
			root->AabbMax = new double3( -BvhDoubleConstants.Far );
			FragmentDouble* fragment = bvh.Fragments;
			ulong* primIdx = bvh.PrimIdx;
			for ( ulong i = 0; i < instCount; i++ )
			{
				if ( blasses != null ) // if a null pointer is passed, we'll assume the instances have been updated elsewhere.
				{
					instances[ i ].Update( ref blasses[ instances[ i ].BlasIdx ] );
				}
				fragment[ i ].BMin = instances[ i ].AabbMin;
				fragment[ i ].PrimIdx = i;
				fragment[ i ].BMax = instances[ i ].AabbMax;
				root->AabbMin = BvhDoubleMath.Min( root->AabbMin, instances[ i ].AabbMin );
				root->AabbMax = BvhDoubleMath.Max( root->AabbMax, instances[ i ].AabbMax );
				primIdx[ i ] = i;
			}
			// start build
			bvh.UsedNodes = 1;
		}

		/// <summary>Port of BVH_Double::Build( void (*customGetAABB)( .. ), primCount ).</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void PrepareCustomBuild( ref BvhDouble bvh, FunctionPointer<GetAabbDoubleDelegate> getAabb, ulong primCount )
		{
			bvh.TriCount = primCount;
			bvh.IdxCount = primCount;
			bvh.AllocateArrays( primCount );
			// there is no vertex data: leaf primitives are handed to the custom callbacks. The C++
			// leaves verts untouched here and relies on it being null to derive bvh_over_aabbs;
			// clear it so rebuilding over a used BvhDouble stays correct.
			bvh.Verts = null;
			bvh.VertIdx = null;
			bvh.Instances = null;
			bvh.Blasses = null;
			bvh.BlasCount = 0;
			bvh.BvhOverIndices = false;
			// copy relevant data from instance array
			BvhDoubleNode* root = bvh.Nodes;
			root->LeftFirst = 0;
			root->TriCount = primCount;
			// note that the C++ seeds these with BVH_FAR, not BVH_DBL_FAR.
			root->AabbMin = new double3( BvhDoubleConstants.WidenedFar );
			root->AabbMax = new double3( -BvhDoubleConstants.WidenedFar );
			FragmentDouble* fragment = bvh.Fragments;
			ulong* primIdx = bvh.PrimIdx;
			for ( ulong i = 0; i < primCount; i++ )
			{
				double3 bmin, bmax;
				getAabb.Invoke( i, &bmin, &bmax );
				fragment[ i ].BMin = bmin;
				fragment[ i ].BMax = bmax;
				root->AabbMin = BvhDoubleMath.Min( root->AabbMin, fragment[ i ].BMin );
				fragment[ i ].PrimIdx = i;
				root->AabbMax = BvhDoubleMath.Max( root->AabbMax, fragment[ i ].BMax );
				primIdx[ i ] = i;
			}
			// start build
			bvh.UsedNodes = 1;
		}

		/// <summary>Reference builder: binned SAH BVH builder in double precision. Serial, no SIMD.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void Build( ref BvhDouble bvh )
		{
			BuildSubtree( ref bvh, 0, 0 );
		}

		/// <summary>
		/// Port of BVH_Double::Build( nodeIdx, depth ). 'depth' only steers the C++ threaded build,
		/// which is not ported, so it is always zero here and the finalisation below always runs.
		/// </summary>
		internal static void BuildSubtree( ref BvhDouble bvh, ulong nodeIdx, uint depth )
		{
			BvhDoubleNode* bvhNode = bvh.Nodes;
			FragmentDouble* fragment = bvh.Fragments;
			ulong* primIdx = bvh.PrimIdx;
			// the C++ keeps the next free node in a member; the serial path just counts up.
			ulong newNodePtr = bvh.UsedNodes;
			// subdivide root node recursively
			ulong* task = stackalloc ulong[ 512 ];
			ulong taskCount = 0;
			BvhDoubleNode* root = bvhNode;
			double3 minDim = ( root->AabbMax - root->AabbMin ) * 1e-40;
			double3 bestLMin = new double3( 0.0 ), bestLMax = new double3( 0.0 );
			double3 bestRMin = new double3( 0.0 ), bestRMax = new double3( 0.0 );
			// scratch for the bins and the per-split totals; the C++ declares these inside the
			// subdivision loop, so they are reset at the start of each iteration below.
			double3* binMin = stackalloc double3[ 3 * Bins ];
			double3* binMax = stackalloc double3[ 3 * Bins ];
			uint* count = stackalloc uint[ 3 * Bins ];
			double3* lBMin = stackalloc double3[ Bins - 1 ];
			double3* rBMin = stackalloc double3[ Bins - 1 ];
			double3* lBMax = stackalloc double3[ Bins - 1 ];
			double3* rBMax = stackalloc double3[ Bins - 1 ];
			double* ANL = stackalloc double[ Bins - 1 ];
			double* ANR = stackalloc double[ Bins - 1 ];
			while ( true )
			{
				while ( true )
				{
					BvhDoubleNode* node = bvhNode + nodeIdx;
					double SA = node->SurfaceArea;
					// find optimal object split
					for ( int a = 0; a < 3; a++ )
					{
						for ( int i = 0; i < Bins; i++ )
						{
							binMin[ ( a * Bins ) + i ] = new double3( BvhDoubleConstants.Far );
							binMax[ ( a * Bins ) + i ] = new double3( -BvhDoubleConstants.Far );
							count[ ( a * Bins ) + i ] = 0;
						}
					}
					double3 extent = node->AabbMax - node->AabbMin;
					double3 nmin3 = node->AabbMin;
					double3 rpd3 = new double3(
						extent.x > minDim.x ? ( Bins / extent.x ) : 0,
						extent.y > minDim.y ? ( Bins / extent.y ) : 0,
						extent.z > minDim.z ? ( Bins / extent.z ) : 0
					);
					for ( ulong i = 0; i < node->TriCount; i++ ) // process all tris for x,y and z at once
					{
						ulong fi = primIdx[ node->LeftFirst + i ];
						double3 fbi = ( ( ( fragment[ fi ].BMin + fragment[ fi ].BMax ) * 0.5 ) - nmin3 ) * rpd3;
						int3 bi = new int3( ( int )fbi.x, ( int )fbi.y, ( int )fbi.z );
						bi.x = BvhDoubleMath.Clamp( bi.x, 0, Bins - 1 );
						bi.y = BvhDoubleMath.Clamp( bi.y, 0, Bins - 1 );
						bi.z = BvhDoubleMath.Clamp( bi.z, 0, Bins - 1 );
						binMin[ bi.x ] = BvhDoubleMath.Min( binMin[ bi.x ], fragment[ fi ].BMin );
						binMax[ bi.x ] = BvhDoubleMath.Max( binMax[ bi.x ], fragment[ fi ].BMax );
						count[ bi.x ]++;
						binMin[ Bins + bi.y ] = BvhDoubleMath.Min( binMin[ Bins + bi.y ], fragment[ fi ].BMin );
						binMax[ Bins + bi.y ] = BvhDoubleMath.Max( binMax[ Bins + bi.y ], fragment[ fi ].BMax );
						count[ Bins + bi.y ]++;
						binMin[ ( 2 * Bins ) + bi.z ] = BvhDoubleMath.Min( binMin[ ( 2 * Bins ) + bi.z ], fragment[ fi ].BMin );
						binMax[ ( 2 * Bins ) + bi.z ] = BvhDoubleMath.Max( binMax[ ( 2 * Bins ) + bi.z ], fragment[ fi ].BMax );
						count[ ( 2 * Bins ) + bi.z ]++;
					}
					// calculate per-split totals
					double splitCost = BvhDoubleConstants.Far;
					int bestAxis = 0, bestPos = 0;
					for ( int a = 0; a < 3; a++ )
					{
						if ( extent[ a ] > minDim[ a ] )
						{
							double3 l1 = new double3( BvhDoubleConstants.Far ), l2 = new double3( -BvhDoubleConstants.Far );
							double3 r1 = new double3( BvhDoubleConstants.Far ), r2 = new double3( -BvhDoubleConstants.Far );
							uint lN = 0, rN = 0;
							for ( int i = 0; i < Bins - 1; i++ )
							{
								lBMin[ i ] = l1 = BvhDoubleMath.Min( l1, binMin[ ( a * Bins ) + i ] );
								rBMin[ Bins - 2 - i ] = r1 = BvhDoubleMath.Min( r1, binMin[ ( a * Bins ) + ( Bins - 1 - i ) ] );
								lBMax[ i ] = l2 = BvhDoubleMath.Max( l2, binMax[ ( a * Bins ) + i ] );
								rBMax[ Bins - 2 - i ] = r2 = BvhDoubleMath.Max( r2, binMax[ ( a * Bins ) + ( Bins - 1 - i ) ] );
								lN += count[ ( a * Bins ) + i ];
								rN += count[ ( a * Bins ) + ( Bins - 1 - i ) ];
								ANL[ i ] = lN == 0 ? BvhDoubleConstants.Far : ( BvhDoubleMath.HalfArea( l2 - l1 ) * ( double )lN );
								ANR[ Bins - 2 - i ] = rN == 0 ? BvhDoubleConstants.Far : ( BvhDoubleMath.HalfArea( r2 - r1 ) * ( double )rN );
							}
							// evaluate bin totals to find best position for object split
							for ( int i = 0; i < Bins - 1; i++ )
							{
								double C = ANL[ i ] + ANR[ i ];
								if ( C < splitCost )
								{
									splitCost = C;
									bestAxis = a;
									bestPos = i;
									bestLMin = lBMin[ i ];
									bestRMin = rBMin[ i ];
									bestLMax = lBMax[ i ];
									bestRMax = rBMax[ i ];
								}
							}
						}
					}
					splitCost = bvh.TraversalCost + ( bvh.IntersectionCost * splitCost / SA );
					double noSplitCost = ( double )node->TriCount * bvh.IntersectionCost;
					if ( splitCost >= noSplitCost )
					{
						break; // not splitting is better.
					}
					// in-place partition
					ulong j = node->LeftFirst + node->TriCount, src = node->LeftFirst;
					double rpd = rpd3[ bestAxis ], nmin = nmin3[ bestAxis ];
					for ( ulong i = 0; i < node->TriCount; i++ )
					{
						ulong fi = primIdx[ src ];
						// The C++ casts through uint32_t here; the centroid lies inside the node
						// bounds, so the value is never negative and the int cast is equivalent.
						int bi = ( int )( ( ( ( fragment[ fi ].BMin[ bestAxis ] + fragment[ fi ].BMax[ bestAxis ] ) * 0.5 ) - nmin ) * rpd );
						bi = BvhDoubleMath.Clamp( bi, 0, Bins - 1 );
						if ( bi <= bestPos )
						{
							src++;
						}
						else
						{
							--j;
							ulong t = primIdx[ src ];
							primIdx[ src ] = primIdx[ j ];
							primIdx[ j ] = t;
						}
					}
					// create child nodes
					ulong leftCount = src - node->LeftFirst, rightCount = node->TriCount - leftCount;
					if ( leftCount == 0 || rightCount == 0 || taskCount == 512 )
					{
						break; // should not happen.
					}
					ulong n = newNodePtr;
					newNodePtr += 2;
					bvhNode[ n ].AabbMin = bestLMin;
					bvhNode[ n ].AabbMax = bestLMax;
					bvhNode[ n ].LeftFirst = node->LeftFirst;
					bvhNode[ n ].TriCount = leftCount;
					bvhNode[ n + 1 ].AabbMin = bestRMin;
					bvhNode[ n + 1 ].AabbMax = bestRMax;
					bvhNode[ n + 1 ].LeftFirst = j;
					bvhNode[ n + 1 ].TriCount = rightCount;
					node->LeftFirst = n;
					node->TriCount = 0;
					// recurse
					task[ taskCount++ ] = n + 1;
					nodeIdx = n;
				}
				// fetch subdivision task from stack
				if ( taskCount == 0 )
				{
					break;
				}
				nodeIdx = task[ --taskCount ];
			}
			if ( depth == 0 || bvh.TriCount < BvhConstants.MtBuildThreshold )
			{
				bvh.UsedNodes = newNodePtr;
				bvh.AabbMin = bvhNode[ 0 ].AabbMin;
				bvh.AabbMax = bvhNode[ 0 ].AabbMax;
				bvh.Refittable = true; // not using spatial splits: can refit this BVH
				bvh.MayHaveHoles = false; // the reference builder produces a continuous list of nodes
				bvh.BvhOverAabbs = bvh.Verts == null; // bvh over aabbs is suitable as TLAS
			}
		}
	}
}
