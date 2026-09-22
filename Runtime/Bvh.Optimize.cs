using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;

namespace TinyBVH
{
	/// <summary>
	/// The tree-rotation optimizer of tinybvh's BVH class: BVH::Optimize and the conversion back
	/// from the verbose layout it runs on. See <see cref="BvhVerbose"/> for the optimizer itself.
	/// </summary>
	public unsafe partial struct Bvh
	{
		/// <summary>
		/// Port of BVH::Optimize: convert to BVH_Verbose, optimize, convert back. The C++ passes
		/// the default compact = true on the way back, so the result is a compacted tree, and it
		/// leaves the flags alone: CopyBasePropertiesFrom round-trips refittable / may_have_holes
		/// through the verbose copy, so an optimized binned tree stays refittable and hole-free.
		/// <para>
		/// tinybvh recommends SplitLeafs( 1 ) before and MergeLeafs after for the best result, but
		/// BVH::Optimize does not do that in v1.8.0 and neither does this. Drive a
		/// <see cref="BvhVerbose"/> directly for that pipeline.
		/// </para>
		/// <para>
		/// With stochastic, the optimizer walks its candidate list from a random offset in random
		/// steps; randomSeed seeds the generator of the temporary BvhVerbose this creates. The C++
		/// draws from one process-wide rand() stream that keeps running across calls, so a second
		/// BVH::Optimize( .., true ) there sees different numbers than the first; here every call
		/// restarts from randomSeed. Seed 1 reproduces the first call of a freshly started MSVC
		/// process, which is what the reference tool records. See
		/// <see cref="BvhVerbose.RandomState"/>.
		/// </para>
		/// </summary>
		public void Optimize( uint iterations = 25, bool extreme = false, bool stochastic = false, uint randomSeed = 1 )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "Bvh.Optimize( .. ), bvh was not created." );
			}
			if ( Nodes == null )
			{
				throw new InvalidOperationException( "Bvh.Optimize( .. ), nodes == null." );
			}
			BvhVerbose verbose = BvhVerbose.Create( Allocator );
			verbose.RandomState = randomSeed;
			try
			{
				verbose.ConvertFrom( ref this );
				verbose.Optimize( iterations, extreme, stochastic );
				ConvertFrom( ref verbose );
			}
			finally
			{
				verbose.Dispose();
			}
		}

		/// <summary>
		/// Port of BVH::ConvertFrom( const BVH_Verbose&amp;, bool compact ): rebuilds the Wald
		/// 32-byte node layout from the verbose one, emitting nodes in depth-first order from node
		/// 2 up. Node 1 stays zeroed, as in the C++, which memsets the pool first.
		/// </summary>
		public void ConvertFrom( ref BvhVerbose original, bool compact = true )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "Bvh.ConvertFrom( .. ), bvh was not created." );
			}
			if ( original.Nodes == null )
			{
				throw new ArgumentException( "Bvh.ConvertFrom( .. ), original.Nodes == null.", nameof( original ) );
			}
			if ( original.UsedNodes == 0 )
			{
				throw new ArgumentException( "Bvh.ConvertFrom( .. ), original.UsedNodes == 0.", nameof( original ) );
			}
			BvhOptimizer.ConvertFrom( ref this, ref original, compact );
		}
	}

	/// <summary>
	/// Burst-compiled half of the optimizer entry points. Direct calls must be synchronous,
	/// otherwise editor tests silently run the Mono fallback.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhOptimizer
	{
		/// <summary>C++: 'uint32_t srcStack[1024], dstStack[1024]' in BVH::ConvertFrom.</summary>
		private const int ConvertStackSize = 1024;

		/// <summary>Port of BVH::ConvertFrom( const BVH_Verbose&amp;, bool ).</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void ConvertFrom( ref Bvh bvh, ref BvhVerbose original, [MarshalAs( UnmanagedType.U1 )] bool compact )
		{
			// allocate space
			uint spaceNeeded = compact ? original.UsedNodes : original.AllocatedNodes;
			bvh.AllocateNodes( spaceNeeded );
			BvhNode* bvhNode = bvh.Nodes;
			UnsafeUtility.MemClear( bvhNode, ( long )spaceNeeded * sizeof( BvhNode ) );
			CopyBasePropertiesFrom( ref bvh, ref original );
			bvh.Verts = original.Verts;
			bvh.VertStride = original.VertStride;
			bvh.PrimIdx = original.PrimIdx;
			if ( original.OwnsPrimIdx )
			{
				// MergeLeafs freed the array this Bvh used to own and allocated a fresh one of
				// idxCount entries; the C++ takes the pointer over the same way, silently.
				bvh.AllocatedPrimIdx = original.IdxCount;
				original.OwnsPrimIdx = false;
			}
			// start conversion
			uint srcNodeIdx = 0, dstNodeIdx = 0, stackPtr = 0;
			uint newNodePtr = 2;
			uint* srcStack = stackalloc uint[ ConvertStackSize ];
			uint* dstStack = stackalloc uint[ ConvertStackSize ];
			while ( true )
			{
				BvhVerboseNode* orig = original.Nodes + srcNodeIdx;
				bvhNode[ dstNodeIdx ].AabbMin = orig->AabbMin;
				bvhNode[ dstNodeIdx ].AabbMax = orig->AabbMax;
				if ( orig->IsLeaf )
				{
					bvhNode[ dstNodeIdx ].TriCount = orig->TriCount;
					bvhNode[ dstNodeIdx ].LeftFirst = orig->FirstTri;
					if ( stackPtr == 0 )
					{
						break;
					}
					srcNodeIdx = srcStack[ --stackPtr ];
					dstNodeIdx = dstStack[ stackPtr ];
				}
				else
				{
					bvhNode[ dstNodeIdx ].LeftFirst = newNodePtr;
					uint srcRightIdx = orig->Right;
					srcNodeIdx = orig->Left;
					dstNodeIdx = newNodePtr++;
					srcStack[ stackPtr ] = srcRightIdx;
					dstStack[ stackPtr++ ] = newNodePtr++;
				}
			}
			bvh.UsedNodes = original.UsedNodes;
		}

		/// <summary>Port of BVHBase::CopyBasePropertiesFrom for the BvhVerbose -&gt; Bvh direction.</summary>
		private static void CopyBasePropertiesFrom( ref Bvh bvh, ref BvhVerbose original )
		{
			bvh.Refittable = original.Refittable;
			bvh.MayHaveHoles = original.MayHaveHoles;
			bvh.BvhOverAabbs = original.BvhOverAabbs;
			bvh.BvhOverIndices = original.BvhOverIndices;
			bvh.TriCount = original.TriCount;
			bvh.IdxCount = original.IdxCount;
			bvh.AabbMin = original.AabbMin;
			bvh.AabbMax = original.AabbMax;
		}
	}
}
