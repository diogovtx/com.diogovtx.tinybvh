using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;

namespace TinyBVH
{
	/// <summary>A pending SBVH subtree: the node to subdivide and the index slice it may draw from.</summary>
	internal struct HqSubtree
	{
		public uint Node;
		public uint SliceStart;
		public uint SliceEnd;
	}

	/// <summary>
	/// Threaded construction, the Unity job system standing in for tinybvh's tinybvh_spawn /
	/// tinybvh_barrier hooks. Both builds run in two phases: the calling thread subdivides the top of
	/// the tree down to BvhConstants.MtSpawnDepth and collects the subtrees rooted there, then one
	/// IJobParallelFor builds those subtrees, which touch disjoint primitive index ranges and draw
	/// their nodes (and, for the SBVH, their split fragments) from a shared atomic counter. That
	/// matches the C++, where the same subtrees end up on the thread pool; only the order in which
	/// node pairs are handed out differs, so the node numbering of a threaded build is not
	/// reproducible while the tree shape is. Entered from Bvh.Build once the primitive count reaches
	/// BvhConstants.MtBuildThreshold.
	/// </summary>
	internal static unsafe class BvhThreadedBuilder
	{
		/// <summary>Upper bound on the collected subtrees: a binary tree has at most 2^N nodes at depth N.</summary>
		private const int MaxSubtrees = 1 << BvhConstants.MtSpawnDepth;

		[BurstCompile( CompileSynchronously = true )]
		private struct BuildSubtreeJob : IJobParallelFor
		{
			public Bvh Bvh;
			[NativeDisableUnsafePtrRestriction] public uint* Pending;
			[NativeDisableUnsafePtrRestriction] public int* NewNodePtr;

			public void Execute( int i )
			{
				BvhBuilder.BuildSubtree( ref Bvh, Pending[ i ], BvhConstants.MtSpawnDepth, NewNodePtr, null, null );
			}
		}

		[BurstCompile( CompileSynchronously = true )]
		private struct BuildHqSubtreeJob : IJobParallelFor
		{
			public Bvh Bvh;
			[NativeDisableUnsafePtrRestriction] public HqSubtree* Pending;
			[NativeDisableUnsafePtrRestriction] public uint* IdxTmp;
			[NativeDisableUnsafePtrRestriction] public int* NewNodePtr;
			[NativeDisableUnsafePtrRestriction] public int* NextFrag;

			public void Execute( int i )
			{
				HqSubtree subtree = Pending[ i ];
				BvhHqBuilder.BuildHqTask(
					ref Bvh, subtree.Node, BvhConstants.MtSpawnDepth, subtree.SliceStart, subtree.SliceEnd,
					IdxTmp, NewNodePtr, NextFrag, null, null );
			}
		}

		/// <summary>Threaded binned SAH build; the threaded counterpart of BvhBuilder.Build.</summary>
		internal static void Build( ref Bvh bvh )
		{
			// counters[ 0 ] is the node allocator shared by both phases, counters[ 1 ] the subtree count.
			int* counters = ( int* )bvh.Alloc( 2 * sizeof( int ) );
			uint* pending = ( uint* )bvh.Alloc( MaxSubtrees * sizeof( uint ) );
			counters[ 0 ] = ( int )bvh.UsedNodes;
			counters[ 1 ] = 0;
			BvhBuilder.BuildTop( ref bvh, counters, pending, counters + 1 );
			int subtrees = counters[ 1 ];
			if ( subtrees > 0 )
			{
				BuildSubtreeJob job = new BuildSubtreeJob
				{
					Bvh = bvh,
					Pending = pending,
					NewNodePtr = counters
				};
				job.Schedule( subtrees, 1 ).Complete();
			}
			bvh.UsedNodes = ( uint )counters[ 0 ];
			bvh.ThreadedSubtrees = ( uint )subtrees;
			bvh.Free( pending );
			bvh.Free( counters );
			BvhBuilder.FinishBuild( ref bvh );
		}

		/// <summary>Threaded SBVH build; the threaded counterpart of BvhHqBuilder.BuildHq.</summary>
		internal static void BuildHq( ref Bvh bvh )
		{
			uint slack = bvh.TriCount >> 1; // for split prims
			long idxTmpBytes = ( long )( bvh.TriCount + slack ) * sizeof( uint );
			uint* idxTmp = ( uint* )bvh.Alloc( idxTmpBytes );
			UnsafeUtility.MemClear( idxTmp, idxTmpBytes );
			// counters[ 0 ] is the node allocator, counters[ 1 ] the fragment allocator, counters[ 2 ]
			// the subtree count; the first two are shared by both phases.
			int* counters = ( int* )bvh.Alloc( 3 * sizeof( int ) );
			HqSubtree* pending = ( HqSubtree* )bvh.Alloc( MaxSubtrees * sizeof( HqSubtree ) );
			counters[ 0 ] = 2;
			counters[ 1 ] = ( int )bvh.TriCount;
			counters[ 2 ] = 0;
			BvhHqBuilder.BuildHqTop( ref bvh, idxTmp, counters, counters + 1, pending, counters + 2 );
			int subtrees = counters[ 2 ];
			if ( subtrees > 0 )
			{
				BuildHqSubtreeJob job = new BuildHqSubtreeJob
				{
					Bvh = bvh,
					Pending = pending,
					IdxTmp = idxTmp,
					NewNodePtr = counters,
					NextFrag = counters + 1
				};
				job.Schedule( subtrees, 1 ).Complete();
			}
			bvh.UsedNodes = ( uint )counters[ 0 ];
			bvh.ThreadedSubtrees = ( uint )subtrees;
			bvh.Free( pending );
			bvh.Free( counters );
			bvh.Free( idxTmp );
			BvhHqBuilder.FinishHqBuild( ref bvh );
			BvhLeafTools.Compact( ref bvh );
		}
	}
}
