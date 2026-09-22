using System;
using System.Threading;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Construction half of the tinybvh BVH port: PrepareBuild plus the binned SAH reference
	/// builder, Refit and the BVH tools. The heavy lifting lives in BvhBuilder so Burst can compile
	/// it; the threaded variant of the subdivision loop lives in Bvh.BuildThreaded.cs.
	/// </summary>
	public unsafe partial struct Bvh
	{
		// BVH builder for triangle geometry.
		// This code uses no SIMD instructions. Faster code, using SSE/AVX, is available for x64 CPUs.

		/// <summary>Builds over a triangle soup: three consecutive 16-byte vertices per primitive.</summary>
		public void Build( float4* vertices, uint primCount )
		{
			Build( ( byte* )vertices, primCount * 3, 16, null, primCount );
		}

		/// <summary>General form: a strided vertex buffer, optionally addressed through indices.</summary>
		public void Build( byte* vertices, uint vertexCount, int vertexStride, uint* indices, uint primCount )
		{
			ValidateBuildInput( vertices, vertexCount, vertexStride, indices, primCount );
			if ( UseSpatialSplits ) // SBVH requested
			{
				BvhHqBuilder.PrepareHqBuild( ref this, vertices, vertexCount, vertexStride, indices, primCount );
				// small inputs are built serially, as in the C++ source.
				if ( UseThreadedBuild && TriCount >= BvhConstants.MtBuildThreshold )
				{
					BvhThreadedBuilder.BuildHq( ref this );
				}
				else
				{
					BvhHqBuilder.BuildHq( ref this );
				}
			}
			else if ( UseFullSweep ) // Full-sweep requested
			{
				BvhBuilder.PrepareBuild( ref this, vertices, vertexCount, vertexStride, indices, primCount );
				BvhFullSweepBuilder.BuildFullSweep( ref this );
			}
			else if ( UseSimdIfAvailable && AvxBuilderSupported ) // No preference: use fast AVX builder
			{
				RunAvxBuild( vertices, vertexCount, vertexStride, indices, primCount );
			}
			else
			{
				// No preference, no AVX: use reference builder.
				BvhBuilder.PrepareBuild( ref this, vertices, vertexCount, vertexStride, indices, primCount );
				RunBinnedBuild();
			}
			if ( PostOptimize )
			{
				Optimize( OptimizeIterations );
			}
		}

		/// <summary>Runs the binned builder over prepared fragments, threaded when requested and the input is large enough; also used by the custom-geometry builds.</summary>
		internal void RunBinnedBuild()
		{
			if ( UseThreadedBuild && TriCount >= BvhConstants.MtBuildThreshold )
			{
				BvhThreadedBuilder.Build( ref this );
			}
			else
			{
				BvhBuilder.Build( ref this );
			}
		}

		public void Build( NativeArray<float4> vertices, uint primCount )
		{
			Build( ( float4* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( vertices ), primCount );
		}

		public void Build( NativeArray<float4> vertices, NativeArray<uint> indices, uint primCount )
		{
			Build(
				( byte* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( vertices ), ( uint )vertices.Length, 16,
				( uint* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( indices ), primCount );
		}

		/// <summary>TLAS builder. Builds a BVH over a list of BLAS instances.</summary>
		public void BuildTlas( BlasInstance* instances, uint instanceCount, Bvh* blasses, uint blasCount )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "Bvh.BuildTlas( .. ), bvh was not created." );
			}
			if ( instances == null )
			{
				throw new ArgumentException( "Bvh.BuildTlas( .. ), instances == null.", nameof( instances ) );
			}
			if ( instanceCount == 0 )
			{
				throw new ArgumentException( "Bvh.BuildTlas( .. ), instanceCount == 0.", nameof( instanceCount ) );
			}
			BvhBuilder.PrepareTlasBuild( ref this, instances, instanceCount, blasses, blasCount );
			BvhBuilder.Build( ref this );
		}

		// Refitting: For animated meshes, where the topology remains intact. This
		// includes trees waving in the wind, or subsequent frames for skinned
		// animations. Repeated refitting tends to lead to deteriorated BVHs and
		// slower ray tracing. Rebuild when this happens.
		public void Refit( uint nodeIdx = 0 )
		{
			if ( !Refittable )
			{
				throw new InvalidOperationException( "Bvh.Refit( .. ), refitting an SBVH or pre-splitted BVH." );
			}
			if ( Nodes == null )
			{
				throw new InvalidOperationException( "Bvh.Refit( .. ), nodes == null." );
			}
			if ( MayHaveHoles )
			{
				throw new InvalidOperationException( "Bvh.Refit( .. ), bvh may have holes." );
			}
			if ( IsTlas )
			{
				throw new InvalidOperationException( "Bvh.Refit( .. ), do not refit a TLAS, use BuildTlas( .. )." );
			}
			BvhBuilder.Refit( ref this );
		}

		/// <summary>
		/// Determine the SAH cost of the tree. This provides an indication
		/// of the quality of the BVH: Lower is better. The summation runs under Burst so the result
		/// is bit-identical to the C++; Mono evaluates the scalar float expression in double.
		/// </summary>
		public float SahCost( uint nodeIdx = 0 )
		{
			BvhBuilder.SahCost( ref this, nodeIdx, out float cost );
			return cost;
		}

		/// <summary>
		/// Determine the number of nodes in the tree. Typically the result should
		/// be usedNodes - 1 (second node is always unused), but some builders may
		/// have unused nodes besides node 1.
		/// </summary>
		public int NodeCount()
		{
			uint retVal = 0, nodeIdx = 0, stackPtr = 0;
			uint* stack = stackalloc uint[ 64 ];
			while ( true )
			{
				BvhNode* n = Nodes + nodeIdx;
				retVal++;
				if ( n->IsLeaf )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
				}
				else
				{
					nodeIdx = n->LeftFirst;
					stack[ stackPtr++ ] = n->LeftFirst + 1;
				}
			}
			return ( int )retVal;
		}

		/// <summary>Determine the total number of primitives / fragments in leaf nodes.</summary>
		public int PrimCount( uint nodeIdx = 0 )
		{
			BvhNode* n = Nodes + nodeIdx;
			return n->IsLeaf ? ( int )n->TriCount : ( PrimCount( n->LeftFirst ) + PrimCount( n->LeftFirst + 1 ) );
		}

		/// <summary>Determine the number of leaf nodes in the tree.</summary>
		public int LeafCount( uint nodeIdx = 0 )
		{
			uint retVal = 0, stackPtr = 0;
			uint* stack = stackalloc uint[ 64 ];
			while ( true )
			{
				BvhNode* n = Nodes + nodeIdx;
				if ( n->IsLeaf )
				{
					retVal++;
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
				}
				else
				{
					nodeIdx = n->LeftFirst;
					stack[ stackPtr++ ] = n->LeftFirst + 1;
				}
			}
			return ( int )retVal;
		}

		/// <summary>Managed-side replacement for the BVH_FATAL_ERROR checks in BVH::PrepareBuild.</summary>
		private void ValidateBuildInput( byte* vertices, uint vertexCount, int vertexStride, uint* indices, uint primCount )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "Bvh.Build( .. ), bvh was not created." );
			}
			if ( vertices == null )
			{
				throw new ArgumentException( "Bvh.Build( .. ), vertices == null.", nameof( vertices ) );
			}
			if ( vertexStride < 16 )
			{
				throw new ArgumentException( "Bvh.Build( .. ), vertexStride < 16.", nameof( vertexStride ) );
			}
			if ( vertexCount == 0 )
			{
				throw new ArgumentException( "Bvh.Build( .. ), empty vertex slice.", nameof( vertexCount ) );
			}
			if ( indices == null && primCount != 0 && ( primCount * 3 ) != vertexCount )
			{
				throw new ArgumentException( "Bvh.Build( .. ), indices == null and primCount does not match vertexCount.", nameof( primCount ) );
			}
			if ( indices != null && primCount == 0 )
			{
				throw new ArgumentException( "Bvh.Build( .. ), primCount == 0.", nameof( primCount ) );
			}
			if ( ( primCount > 0 ? primCount : vertexCount / 3 ) == 0 )
			{
				throw new ArgumentException( "Bvh.Build( .. ), primCount == 0.", nameof( primCount ) );
			}
			// The C++ sizes its bin scratch to MAXHQBINS and does not check hqbvhbins against it.
			uint maxBins = HqBvhOddEven ? ( uint )( BvhHqBuilder.MaxBins - 1 ) : ( uint )BvhHqBuilder.MaxBins;
			if ( UseSpatialSplits && ( HqBvhBins < 2 || HqBvhBins > maxBins ) )
			{
				throw new InvalidOperationException( $"Bvh.Build( .. ), HqBvhBins must be between 2 and {maxBins}." );
			}
		}
	}

	public partial struct BlasInstance
	{
		public void Update( ref Bvh blas )
		{
			// the body lives in UpdateBounds, in BlasInstance.Update.cs, which the overloads for
			// the other BLAS layouts share; the C++ BLASInstance::Update takes a BVHBase and only
			// reads its root bounds.
			UpdateBounds( blas.AabbMin, blas.AabbMax );
		}

		/// <summary>Calculate the inverse of the matrix stored in 'Transform'.</summary>
		public unsafe void InvertTransform()
		{
			// math from MESA, via http://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
			float* T = ( float* )UnsafeUtility.AddressOf( ref Transform );
			float* iT = ( float* )UnsafeUtility.AddressOf( ref InvTransform );
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
			float det = ( T[ 0 ] * iT[ 0 ] ) + ( T[ 1 ] * iT[ 4 ] ) + ( T[ 2 ] * iT[ 8 ] ) + ( T[ 3 ] * iT[ 12 ] );
			if ( det == 0f )
			{
				return; // actually, invert failed. That's bad.
			}
			float invdet = 1f / det;
			for ( int i = 0; i < 16; i++ )
			{
				iT[ i ] *= invdet;
			}
		}
	}

	/// <summary>
	/// Burst-compiled implementation of the BVH construction routines. These are plain static
	/// methods over a ref Bvh so they can be direct-called from managed code and from Burst jobs.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhBuilder
	{
		private const int Bins = BvhConstants.Bins;

		/// <summary>Allocate memory and prepare a list of fragments to build a BVH over.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void PrepareBuild( ref Bvh bvh, byte* vertices, uint vertexCount, int vertexStride, uint* indices, uint prims )
		{
			uint primCount = prims > 0 ? prims : vertexCount / 3;
			uint splitBudget = bvh.UsePresplitting ? ( uint )( int )( primCount * bvh.PresplitFactor ) : 0;
			uint spaceNeeded = ( primCount + splitBudget ) * 2; // upper limit
			// allocate memory on first build
			bvh.AllocateNodes( spaceNeeded );
			bvh.AllocatePrimIdx( primCount + splitBudget );
			bvh.AllocateFragments( primCount + splitBudget );
			bvh.Nodes[ 1 ] = default; // node 1 remains unused, for cache line alignment.
			// set verts, vertIdx
			bvh.TriCount = primCount;
			bvh.Verts = vertices;
			bvh.VertCount = vertexCount;
			bvh.VertStride = vertexStride;
			bvh.VertIdx = indices;
			// prepare root node
			BvhNode* root = bvh.Nodes;
			root->AabbMin = new float3( BvhConstants.Far );
			root->AabbMax = new float3( -BvhConstants.Far );
			// prepare fragments
			Fragment* fragment = bvh.Fragments;
			uint* primIdx = bvh.PrimIdx;
			if ( indices == null )
			{
				// building a BVH over triangles specified as three 16-byte vertices each.
				for ( uint i = 0; i < primCount; i++ )
				{
					float4 v0 = bvh.Vertex( i * 3 ), v1 = bvh.Vertex( ( i * 3 ) + 1 ), v2 = bvh.Vertex( ( i * 3 ) + 2 );
					float4 fmin = math.min( v0, math.min( v1, v2 ) );
					float4 fmax = math.max( v0, math.max( v1, v2 ) );
					fragment[ i ].BMin = fmin.xyz;
					fragment[ i ].BMax = fmax.xyz;
					fragment[ i ].PrimIdx = i;
					fragment[ i ].Clipped = 0;
					root->AabbMin = math.min( root->AabbMin, fragment[ i ].BMin );
					root->AabbMax = math.max( root->AabbMax, fragment[ i ].BMax );
					primIdx[ i ] = i;
				}
			}
			else
			{
				// building a BVH over triangles consisting of vertices indexed by 'indices'.
				for ( uint i = 0; i < primCount; i++ )
				{
					uint i0 = indices[ i * 3 ], i1 = indices[ ( i * 3 ) + 1 ], i2 = indices[ ( i * 3 ) + 2 ];
					float4 v0 = bvh.Vertex( i0 ), v1 = bvh.Vertex( i1 ), v2 = bvh.Vertex( i2 );
					float4 fmin = math.min( v0, math.min( v1, v2 ) );
					float4 fmax = math.max( v0, math.max( v1, v2 ) );
					fragment[ i ].BMin = fmin.xyz;
					fragment[ i ].BMax = fmax.xyz;
					fragment[ i ].PrimIdx = i;
					fragment[ i ].Clipped = 0;
					root->AabbMin = math.min( root->AabbMin, fragment[ i ].BMin );
					root->AabbMax = math.max( root->AabbMax, fragment[ i ].BMax );
					primIdx[ i ] = i;
				}
			}
			// presplitting
			uint fragCount = bvh.UsePresplitting ? BvhPresplitter.Presplit( ref bvh ) : primCount;
			// finalize root node
			root->LeftFirst = 0;
			root->TriCount = fragCount;
			bvh.IdxCount = fragCount;
			bvh.TriCount = fragCount;
			// reset node pool
			bvh.UsedNodes = 2;
			bvh.BvhOverIndices = indices != null;
			// all set; actual build happens in BvhBuilder.Build.
		}

		/// <summary>TLAS builder: prepares a BVH over the world-space bounds of a list of BLAS instances.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void PrepareTlasBuild( ref Bvh bvh, BlasInstance* instances, uint instCount, Bvh* blasses, uint bCount )
		{
			bvh.TriCount = instCount;
			bvh.IdxCount = instCount;
			uint spaceNeeded = instCount * 2; // upper limit
			bvh.AllocateNodes( spaceNeeded );
			bvh.AllocatePrimIdx( instCount );
			bvh.AllocateFragments( instCount );
			bvh.Nodes[ 1 ] = default; // node 1 remains unused, for cache line alignment.
			bvh.Instances = instances;
			bvh.InstanceCount = instCount;
			bvh.Blasses = blasses;
			bvh.BlasCount = bCount;
			// the mixed-layout list is set by the BlasRef overload of BuildTlas, after this call.
			bvh.BlasRefs = null;
			// the C++ leaves verts untouched here and relies on it being null to derive
			// bvh_over_aabbs; clear it so rebuilding a TLAS over a used Bvh stays correct.
			bvh.Verts = null;
			bvh.VertCount = 0;
			bvh.VertIdx = null;
			bvh.BvhOverIndices = false;
			// copy relevant data to the fragment array over which the BVH will be built.
			BvhNode* root = bvh.Nodes;
			root->LeftFirst = 0;
			root->TriCount = instCount;
			root->AabbMin = new float3( BvhConstants.Far );
			root->AabbMax = new float3( -BvhConstants.Far );
			Fragment* fragment = bvh.Fragments;
			uint* primIdx = bvh.PrimIdx;
			for ( uint i = 0; i < instCount; i++ )
			{
				if ( blasses != null ) // if a null pointer is passed, we'll assume the instances have been updated elsewhere.
				{
					instances[ i ].Update( ref blasses[ instances[ i ].BlasIdx ] );
				}
				fragment[ i ].BMin = instances[ i ].AabbMin;
				fragment[ i ].PrimIdx = i;
				fragment[ i ].BMax = instances[ i ].AabbMax;
				fragment[ i ].Clipped = 0;
				root->AabbMin = math.min( root->AabbMin, instances[ i ].AabbMin );
				root->AabbMax = math.max( root->AabbMax, instances[ i ].AabbMax );
				primIdx[ i ] = i;
			}
			// start build
			bvh.UsedNodes = 2;
		}

		/// <summary>Reference builder: binned SAH BVH builder. Not using SIMD, serial.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void Build( ref Bvh bvh )
		{
			int* newNodePtr = stackalloc int[ 1 ];
			newNodePtr[ 0 ] = ( int )bvh.UsedNodes;
			BuildSubtree( ref bvh, 0, 0, newNodePtr, null, null );
			bvh.UsedNodes = ( uint )newNodePtr[ 0 ];
			bvh.ThreadedSubtrees = 0;
			FinishBuild( ref bvh );
		}

		/// <summary>
		/// Threaded build, first phase: subdivides the top of the tree on the calling thread and
		/// collects the roots of the subtrees at BvhConstants.MtSpawnDepth for the parallel phase.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void BuildTop( ref Bvh bvh, int* newNodePtr, uint* pending, int* pendingCount )
		{
			BuildSubtree( ref bvh, 0, 0, newNodePtr, pending, pendingCount );
		}

		/// <summary>Shared tail of the serial and the threaded build: root bounds and the tree flags.</summary>
		internal static void FinishBuild( ref Bvh bvh )
		{
			bvh.AabbMin = bvh.Nodes[ 0 ].AabbMin;
			bvh.AabbMax = bvh.Nodes[ 0 ].AabbMax;
			bvh.Refittable = !bvh.UsePresplitting; // only if not using spatial splits
			bvh.MayHaveHoles = false; // the reference builder produces a continuous list of nodes
			bvh.BvhOverAabbs = bvh.Verts == null; // bvh over aabbs is suitable as TLAS
			BvhPresplitter.FinishPresplit( ref bvh );
		}

		/// <summary>
		/// Port of BVH::Build( nodeIdx, depth ): subdivides nodeIdx and everything below it. Node pairs
		/// are drawn from *newNodePtr with an atomic add, so subtrees can run concurrently. When
		/// 'pending' is non-null the walk stops at BvhConstants.MtSpawnDepth: the two children are
		/// recorded there instead of descended into, for the parallel phase to pick up. The C++ keeps
		/// 'depth' constant inside the loop because it spawns and returns at every level above the
		/// spawn depth; this port descends instead, so the depth is carried on the task stack.
		/// </summary>
		internal static void BuildSubtree( ref Bvh bvh, uint nodeIdx, uint depth, int* newNodePtr, uint* pending, int* pendingCount )
		{
			BvhNode* bvhNode = bvh.Nodes;
			Fragment* fragment = bvh.Fragments;
			uint* primIdx = bvh.PrimIdx;
			// subdivide the subtree root recursively
			uint* task = stackalloc uint[ 512 ];
			uint* taskDepth = stackalloc uint[ 512 ];
			uint taskCount = 0;
			BvhNode* root = bvhNode;
			float3 minDim = ( root->AabbMax - root->AabbMin ) * 1e-20f;
			float3 bestLMin = new float3( 0f ), bestLMax = new float3( 0f );
			float3 bestRMin = new float3( 0f ), bestRMax = new float3( 0f );
			// scratch for the bins and the per-split totals; the C++ declares these inside the
			// subdivision loop, so they are reset at the start of each iteration below.
			float3* binMin = stackalloc float3[ 3 * Bins ];
			float3* binMax = stackalloc float3[ 3 * Bins ];
			uint* count = stackalloc uint[ 3 * Bins ];
			float3* lBMin = stackalloc float3[ Bins - 1 ];
			float3* rBMin = stackalloc float3[ Bins - 1 ];
			float3* lBMax = stackalloc float3[ Bins - 1 ];
			float3* rBMax = stackalloc float3[ Bins - 1 ];
			float* ANL = stackalloc float[ Bins - 1 ];
			float* ANR = stackalloc float[ Bins - 1 ];
			while ( true )
			{
				while ( true )
				{
					BvhNode* node = bvhNode + nodeIdx;
					float SA = node->SurfaceArea;
					if ( SA == 0 )
					{
						break; // can't split an infinitely small node.
					}
					// find optimal object split
					for ( int a = 0; a < 3; a++ )
					{
						for ( int i = 0; i < Bins; i++ )
						{
							binMin[ ( a * Bins ) + i ] = new float3( BvhConstants.Far );
							binMax[ ( a * Bins ) + i ] = new float3( -BvhConstants.Far );
							count[ ( a * Bins ) + i ] = 0;
						}
					}
					float3 extent = node->AabbMax - node->AabbMin;
					float3 nmin3 = node->AabbMin;
					float3 rpd3 = new float3(
						extent.x > minDim.x ? ( Bins / extent.x ) : 0,
						extent.y > minDim.y ? ( Bins / extent.y ) : 0,
						extent.z > minDim.z ? ( Bins / extent.z ) : 0
					);
					for ( uint i = 0; i < node->TriCount; i++ ) // process all tris for x,y and z at once
					{
						uint fi = primIdx[ node->LeftFirst + i ];
						int3 bi = ( int3 )( ( ( ( fragment[ fi ].BMin + fragment[ fi ].BMax ) * 0.5f ) - nmin3 ) * rpd3 );
						bi.x = Clamp( bi.x, 0, Bins - 1 );
						bi.y = Clamp( bi.y, 0, Bins - 1 );
						bi.z = Clamp( bi.z, 0, Bins - 1 );
						binMin[ bi.x ] = math.min( binMin[ bi.x ], fragment[ fi ].BMin );
						binMax[ bi.x ] = math.max( binMax[ bi.x ], fragment[ fi ].BMax );
						count[ bi.x ]++;
						binMin[ Bins + bi.y ] = math.min( binMin[ Bins + bi.y ], fragment[ fi ].BMin );
						binMax[ Bins + bi.y ] = math.max( binMax[ Bins + bi.y ], fragment[ fi ].BMax );
						count[ Bins + bi.y ]++;
						binMin[ ( 2 * Bins ) + bi.z ] = math.min( binMin[ ( 2 * Bins ) + bi.z ], fragment[ fi ].BMin );
						binMax[ ( 2 * Bins ) + bi.z ] = math.max( binMax[ ( 2 * Bins ) + bi.z ], fragment[ fi ].BMax );
						count[ ( 2 * Bins ) + bi.z ]++;
					}
					// calculate per-split totals
					float splitCost = BvhConstants.Far;
					int bestAxis = 0, bestPos = 0;
					for ( int a = 0; a < 3; a++ )
					{
						if ( extent[ a ] > minDim[ a ] )
						{
							float3 l1 = new float3( BvhConstants.Far ), l2 = new float3( -BvhConstants.Far );
							float3 r1 = new float3( BvhConstants.Far ), r2 = new float3( -BvhConstants.Far );
							uint lN = 0, rN = 0;
							for ( int i = 0; i < Bins - 1; i++ )
							{
								lBMin[ i ] = l1 = math.min( l1, binMin[ ( a * Bins ) + i ] );
								rBMin[ Bins - 2 - i ] = r1 = math.min( r1, binMin[ ( a * Bins ) + ( Bins - 1 - i ) ] );
								lBMax[ i ] = l2 = math.max( l2, binMax[ ( a * Bins ) + i ] );
								rBMax[ Bins - 2 - i ] = r2 = math.max( r2, binMax[ ( a * Bins ) + ( Bins - 1 - i ) ] );
								lN += count[ ( a * Bins ) + i ];
								rN += count[ ( a * Bins ) + ( Bins - 1 - i ) ];
								ANL[ i ] = lN == 0 ? BvhConstants.Far : ( HalfArea( l2 - l1 ) * ( float )lN );
								ANR[ Bins - 2 - i ] = rN == 0 ? BvhConstants.Far : ( HalfArea( r2 - r1 ) * ( float )rN );
							}
							// evaluate bin totals to find best position for object split
							for ( int i = 0; i < Bins - 1; i++ )
							{
								float C = ANL[ i ] + ANR[ i ];
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
					float noSplitCost = ( float )node->TriCount * bvh.IntersectionCost;
					if ( splitCost >= noSplitCost )
					{
						// the C++ prints a warning here when a node with more than 512 prims fails to split.
						break; // not splitting is better.
					}
					// in-place partition
					uint j = node->LeftFirst + node->TriCount, src = node->LeftFirst;
					for ( uint i = 0; i < node->TriCount; i++ )
					{
						uint fi = primIdx[ src ];
						// The C++ evaluates a scalar copy of the binning expression here. We reuse the
						// exact float3 expression of the binning pass instead, so both passes agree bit
						// for bit regardless of the runtime's intermediate precision (Mono evaluates
						// scalar float arithmetic in double, which made them disagree at bin edges).
						// The C++ also casts through uint32_t; the centroid lies inside the node bounds,
						// so the value is never negative and the int cast is equivalent.
						int3 bi3 = ( int3 )( ( ( ( fragment[ fi ].BMin + fragment[ fi ].BMax ) * 0.5f ) - nmin3 ) * rpd3 );
						int bi = Clamp( bi3[ bestAxis ], 0, Bins - 1 );
						if ( bi <= bestPos )
						{
							src++;
						}
						else
						{
							--j;
							uint t = primIdx[ src ];
							primIdx[ src ] = primIdx[ j ];
							primIdx[ j ] = t;
						}
					}
					// create child nodes
					uint leftCount = src - node->LeftFirst, rightCount = node->TriCount - leftCount;
					if ( leftCount == 0 || rightCount == 0 || taskCount == 512 )
					{
						break; // should not happen.
					}
					uint n = ( uint )( Interlocked.Add( ref *newNodePtr, 2 ) - 2 );
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
					if ( pending != null && ( depth + 1 ) == BvhConstants.MtSpawnDepth )
					{
						// hand both children to the parallel phase instead of descending into them.
						pending[ ( *pendingCount )++ ] = n;
						pending[ ( *pendingCount )++ ] = n + 1;
						break;
					}
					task[ taskCount ] = n + 1;
					taskDepth[ taskCount++ ] = depth + 1;
					nodeIdx = n;
					depth++;
				}
				// fetch subdivision task from stack
				if ( taskCount == 0 )
				{
					break;
				}
				taskCount--;
				nodeIdx = task[ taskCount ];
				depth = taskDepth[ taskCount ];
			}
		}

		/// <summary>Burst entry point for Bvh.SahCost; see the note there.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void SahCost( ref Bvh bvh, uint nodeIdx, out float cost )
		{
			cost = SahCostRec( ref bvh, nodeIdx );
		}

		/// <summary>Port of BVH::SAHCost, same recursion and summation order.</summary>
		private static float SahCostRec( ref Bvh bvh, uint nodeIdx )
		{
			BvhNode* n = bvh.Nodes + nodeIdx;
			if ( n->IsLeaf )
			{
				return bvh.IntersectionCost * n->SurfaceArea * n->TriCount;
			}
			float cost = ( bvh.TraversalCost * n->SurfaceArea ) + SahCostRec( ref bvh, n->LeftFirst ) + SahCostRec( ref bvh, n->LeftFirst + 1 );
			return nodeIdx == 0 ? ( cost / n->SurfaceArea ) : cost;
		}

		[BurstCompile( CompileSynchronously = true )]
		internal static void Refit( ref Bvh bvh )
		{
			BvhNode* bvhNode = bvh.Nodes;
			uint* primIdx = bvh.PrimIdx;
			uint* vertIdx = bvh.VertIdx;
			for ( int i = ( int )bvh.UsedNodes - 1; i >= 0; i-- )
			{
				if ( i != 1 )
				{
					BvhNode* node = bvhNode + i;
					if ( node->IsLeaf ) // leaf: adjust to current triangle vertex positions
					{
						float4 bmin = new float4( BvhConstants.Far ), bmax = new float4( -BvhConstants.Far );
						if ( vertIdx != null )
						{
							for ( uint first = node->LeftFirst, j = 0; j < node->TriCount; j++ )
							{
								uint vidx = primIdx[ first + j ] * 3;
								uint i0 = vertIdx[ vidx ], i1 = vertIdx[ vidx + 1 ], i2 = vertIdx[ vidx + 2 ];
								float4 v0 = bvh.Vertex( i0 ), v1 = bvh.Vertex( i1 ), v2 = bvh.Vertex( i2 );
								float4 t1 = math.min( v0, bmin ), t2 = math.max( v0, bmax );
								float4 t3 = math.min( v1, v2 ), t4 = math.max( v1, v2 );
								bmin = math.min( t1, t3 );
								bmax = math.max( t2, t4 );
							}
						}
						else
						{
							for ( uint first = node->LeftFirst, j = 0; j < node->TriCount; j++ )
							{
								uint vidx = primIdx[ first + j ] * 3;
								float4 v0 = bvh.Vertex( vidx ), v1 = bvh.Vertex( vidx + 1 ), v2 = bvh.Vertex( vidx + 2 );
								float4 t1 = math.min( v0, bmin ), t2 = math.max( v0, bmax );
								float4 t3 = math.min( v1, v2 ), t4 = math.max( v1, v2 );
								bmin = math.min( t1, t3 );
								bmax = math.max( t2, t4 );
							}
						}
						node->AabbMin = bmin.xyz;
						node->AabbMax = bmax.xyz;
						continue;
					}
					// interior node: adjust to child bounds
					BvhNode* left = bvhNode + node->LeftFirst;
					BvhNode* right = bvhNode + node->LeftFirst + 1;
					node->AabbMin = math.min( left->AabbMin, right->AabbMin );
					node->AabbMax = math.max( left->AabbMax, right->AabbMax );
				}
			}
			bvh.AabbMin = bvhNode[ 0 ].AabbMin;
			bvh.AabbMax = bvhNode[ 0 ].AabbMax;
		}

		/// <summary>Port of tinybvh_halfarea, including the guard for empty (inverted) bins.</summary>
		private static float HalfArea( float3 v )
		{
			return v.x < -BvhConstants.Far ? 0f : ( ( v.x * v.y ) + ( v.y * v.z ) + ( v.z * v.x ) );
		}

		/// <summary>Port of tinybvh_clamp for integers.</summary>
		private static int Clamp( int x, int a, int b )
		{
			return x > a ? ( x < b ? x : b ) : a;
		}
	}
}
