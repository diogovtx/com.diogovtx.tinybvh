using System.Threading;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Burst-compiled SBVH construction: ports of BVH::PrepareHQBuild, BVH::BuildHQ and
	/// BVH::BuildHQTask, the binned SAH builder with spatial splits. SBVH_UNSPLITTING is enabled and
	/// the bin count comes from Bvh.HqBvhBins / Bvh.HqBvhOddEven, the C++ hqbvhbins and hqbvhoddeven.
	/// ClipFrag and SplitFrag port the scalar branches of the C++ source: the reference tool is
	/// compiled with TINYBVH_NO_SIMD, and the SSE branches can differ in the last bits. The threaded
	/// variant of the subdivision loop lives in Bvh.BuildThreaded.cs.
	/// Enable the builder by setting Bvh.UseSpatialSplits before calling Bvh.Build.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhHqBuilder
	{
		/// <summary>Upper bound on Bvh.HqBvhBins; the bin scratch is a stack buffer of this size (MAXHQBINS).</summary>
		internal const int MaxBins = 256;

		/// <summary>Allocate memory and prepare a list of fragments to build an SBVH over.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void PrepareHqBuild( ref Bvh bvh, byte* vertices, uint vertexCount, int vertexStride, uint* indices, uint prims )
		{
			uint primCount = prims > 0 ? prims : vertexCount / 3;
			uint slack = primCount >> 1; // for split prims
			uint spaceNeeded = primCount * 3;
			// allocate memory on first build
			bvh.AllocateNodes( spaceNeeded );
			bvh.AllocatePrimIdx( primCount + slack );
			bvh.AllocateFragments( primCount + slack );
			bvh.Nodes[ 1 ] = default; // node 1 remains unused, for cache line alignment.
			// set verts, vertIdx
			bvh.Verts = vertices;
			bvh.VertCount = vertexCount;
			bvh.VertStride = vertexStride;
			bvh.VertIdx = indices;
			bvh.IdxCount = primCount + slack;
			bvh.TriCount = primCount;
			// prepare fragments
			BvhNode* root = bvh.Nodes;
			root->LeftFirst = 0;
			root->TriCount = bvh.TriCount;
			root->AabbMin = new float3( BvhConstants.Far );
			root->AabbMax = new float3( -BvhConstants.Far );
			Fragment* fragment = bvh.Fragments;
			uint* primIdx = bvh.PrimIdx;
			if ( indices == null )
			{
				// building a BVH over triangles specified as three 16-byte vertices each.
				for ( uint i = 0; i < bvh.TriCount; i++ )
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
				for ( uint i = 0; i < bvh.TriCount; i++ )
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
			// clear remainder of index array
			UnsafeUtility.MemClear( primIdx + bvh.TriCount, ( long )slack * 4 );
			bvh.BvhOverIndices = indices != null;
			// all set; actual build happens in BvhHqBuilder.BuildHq.
		}

		/// <summary>SBVH builder entry point; port of BVH::BuildHQ.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void BuildHq( ref Bvh bvh )
		{
			uint slack = bvh.TriCount >> 1; // for split prims
			long idxTmpBytes = ( long )( bvh.TriCount + slack ) * sizeof( uint );
			uint* idxTmp = ( uint* )bvh.Alloc( idxTmpBytes );
			UnsafeUtility.MemClear( idxTmp, idxTmpBytes );
			// reset node pool; counters[ 0 ] is this port's equivalent of the C++ member newNodePtr.
			int* counters = stackalloc int[ 2 ];
			counters[ 0 ] = 2;
			counters[ 1 ] = ( int )bvh.TriCount; // nextFrag
			// subdivide recursively
			BuildHqTask( ref bvh, 0, 0, 0, bvh.TriCount + slack, idxTmp, counters, counters + 1, null, null );
			// all done.
			bvh.Free( idxTmp );
			bvh.UsedNodes = ( uint )counters[ 0 ];
			bvh.ThreadedSubtrees = 0;
			FinishHqBuild( ref bvh );
			BvhLeafTools.Compact( ref bvh );
		}

		/// <summary>
		/// Threaded SBVH build, first phase: subdivides the top of the tree on the calling thread and
		/// collects the subtrees rooted at BvhConstants.MtSpawnDepth for the parallel phase.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void BuildHqTop( ref Bvh bvh, uint* idxTmp, int* newNodePtr, int* nextFrag, HqSubtree* pending, int* pendingCount )
		{
			uint slack = bvh.TriCount >> 1; // for split prims
			BuildHqTask( ref bvh, 0, 0, 0, bvh.TriCount + slack, idxTmp, newNodePtr, nextFrag, pending, pendingCount );
		}

		/// <summary>Shared tail of the serial and the threaded SBVH build; the caller ends with Compact.</summary>
		internal static void FinishHqBuild( ref Bvh bvh )
		{
			bvh.AabbMin = bvh.Nodes[ 0 ].AabbMin;
			bvh.AabbMax = bvh.Nodes[ 0 ].AabbMax;
			bvh.Refittable = false; // can't refit an SBVH
			bvh.MayHaveHoles = false; // there may be holes in the index list, but not in the node list
		}

		/// <summary>
		/// Port of BVH::BuildHQTask. Subdivides nodeIdx and everything below it, drawing primitive
		/// index slots from [sliceStart, sliceEnd) of the index array. Node pairs come from
		/// *newNodePtr and split fragments from *nextFrag, both by atomic add, so subtrees with
		/// disjoint slices can run concurrently. When 'pending' is non-null the walk stops at
		/// BvhConstants.MtSpawnDepth: the two children are recorded there, with the slices the C++
		/// hands to its spawned tasks, instead of descended into. The C++ leaves 'depth' unchanged on
		/// its local task stack because it spawns and returns at every level above the spawn depth;
		/// this port descends instead, so the depth is carried on that stack. Note that the odd/even
		/// bin count therefore has to use the depth the task was entered with, not the descending one:
		/// in the C++ 'depth' only ever changes by being handed to a spawned subtree, so a serial
		/// build evaluates hqbvhbins + ( depth &amp; 1 ) with depth == 0 for the whole tree and the flag
		/// does nothing at all. The reference dump was produced without threading, so it is that
		/// behaviour the port has to reproduce.
		/// </summary>
		internal static void BuildHqTask( ref Bvh bvh, uint nodeIdx, uint depth, uint sliceStart, uint sliceEnd, uint* idxTmp, int* newNodePtr, int* nextFrag, HqSubtree* pending, int* pendingCount )
		{
			BvhNode* bvhNode = bvh.Nodes;
			Fragment* fragment = bvh.Fragments;
			uint* primIdx = bvh.PrimIdx;
			// prepare subdivision
			uint* taskNode = stackalloc uint[ 512 ];
			uint* taskDepth = stackalloc uint[ 512 ];
			uint* taskSliceStart = stackalloc uint[ 512 ];
			uint* taskSliceEnd = stackalloc uint[ 512 ];
			uint localTasks = 0;
			float3 bestLMin = new float3( 0f ), bestLMax = new float3( 0f );
			float3 bestRMin = new float3( 0f ), bestRMax = new float3( 0f );
			BvhNode* root = bvhNode;
			float rootArea = HalfArea( root->AabbMax - root->AabbMin );
			float3 minDim = ( root->AabbMax - root->AabbMin ) * 1e-7f; // don't touch, carefully picked
			// Scratch for the bins and the per-split totals. The C++ declares these inside the
			// subdivision loop; every element is written before it is read in each iteration below,
			// and the object-split and spatial-split passes never overlap, so one set is enough.
			float3* binMin = stackalloc float3[ 3 * MaxBins ];
			float3* binMax = stackalloc float3[ 3 * MaxBins ];
			uint* count = stackalloc uint[ 3 * MaxBins ];
			float3* lBMin = stackalloc float3[ MaxBins ];
			float3* rBMin = stackalloc float3[ MaxBins ];
			float3* lBMax = stackalloc float3[ MaxBins ];
			float3* rBMax = stackalloc float3[ MaxBins ];
			float* AL = stackalloc float[ MaxBins ];
			float* AR = stackalloc float[ MaxBins ];
			int* NL = stackalloc int[ MaxBins ];
			int* NR = stackalloc int[ MaxBins ];
			float3* sbinMin = stackalloc float3[ MaxBins ];
			float3* sbinMax = stackalloc float3[ MaxBins ];
			int* countIn = stackalloc int[ MaxBins ];
			int* countOut = stackalloc int[ MaxBins ];
			// alternating bin counts for optimizer; see the note on 'depth' above.
			int bins = ( int )bvh.HqBvhBins;
			int binCount = bvh.HqBvhOddEven ? ( bins + ( int )( depth & 1 ) ) : bins;
			// subdivide
			while ( true )
			{
				while ( true )
				{
					// fetch node to subdivide
					BvhNode* node = bvhNode + nodeIdx;
					// find optimal object split
					for ( int a = 0; a < 3; a++ )
					{
						for ( int i = 0; i < binCount; i++ )
						{
							binMin[ ( a * binCount ) + i ] = new float3( BvhConstants.Far );
							binMax[ ( a * binCount ) + i ] = new float3( -BvhConstants.Far );
							count[ ( a * binCount ) + i ] = 0;
						}
					}
					float3 rpd3 = new float3( ( float )binCount ) / ( node->AabbMax - node->AabbMin );
					float3 nmin3 = node->AabbMin;
					for ( uint i = 0; i < node->TriCount; i++ ) // process all tris for x,y and z at once
					{
						uint fi = primIdx[ node->LeftFirst + i ];
						int3 bi = ( int3 )( ( ( ( fragment[ fi ].BMin + fragment[ fi ].BMax ) * 0.5f ) - nmin3 ) * rpd3 );
						bi.x = Clamp( bi.x, 0, binCount - 1 );
						bi.y = Clamp( bi.y, 0, binCount - 1 );
						bi.z = Clamp( bi.z, 0, binCount - 1 );
						binMin[ bi.x ] = math.min( binMin[ bi.x ], fragment[ fi ].BMin );
						binMax[ bi.x ] = math.max( binMax[ bi.x ], fragment[ fi ].BMax );
						count[ bi.x ]++;
						binMin[ binCount + bi.y ] = math.min( binMin[ binCount + bi.y ], fragment[ fi ].BMin );
						binMax[ binCount + bi.y ] = math.max( binMax[ binCount + bi.y ], fragment[ fi ].BMax );
						count[ binCount + bi.y ]++;
						binMin[ ( 2 * binCount ) + bi.z ] = math.min( binMin[ ( 2 * binCount ) + bi.z ], fragment[ fi ].BMin );
						binMax[ ( 2 * binCount ) + bi.z ] = math.max( binMax[ ( 2 * binCount ) + bi.z ], fragment[ fi ].BMax );
						count[ ( 2 * binCount ) + bi.z ]++;
					}
					// calculate per-split totals
					float noSplitCost = NoSplitCostSAH( ref bvh, ( int )node->TriCount );
					float splitCost = noSplitCost, rSAV = 1f / node->SurfaceArea;
					int bestAxis = 0, bestPos = 0;
					for ( int a = 0; a < 3; a++ )
					{
						if ( ( node->AabbMax[ a ] - node->AabbMin[ a ] ) > minDim[ a ] )
						{
							float3 l1 = new float3( BvhConstants.Far ), l2 = new float3( -BvhConstants.Far );
							float3 r1 = new float3( BvhConstants.Far ), r2 = new float3( -BvhConstants.Far );
							uint lN = 0, rN = 0;
							for ( int i = 0; i < binCount - 1; i++ )
							{
								lBMin[ i ] = l1 = math.min( l1, binMin[ ( a * binCount ) + i ] );
								rBMin[ binCount - 2 - i ] = r1 = math.min( r1, binMin[ ( a * binCount ) + ( binCount - 1 - i ) ] );
								lBMax[ i ] = l2 = math.max( l2, binMax[ ( a * binCount ) + i ] );
								rBMax[ binCount - 2 - i ] = r2 = math.max( r2, binMax[ ( a * binCount ) + ( binCount - 1 - i ) ] );
								lN += count[ ( a * binCount ) + i ];
								rN += count[ ( a * binCount ) + ( binCount - 1 - i ) ];
								NL[ i ] = ( int )lN;
								NR[ binCount - 2 - i ] = ( int )rN;
								AL[ i ] = lN == 0 ? BvhConstants.Far : HalfArea( l2 - l1 );
								AR[ binCount - 2 - i ] = rN == 0 ? BvhConstants.Far : HalfArea( r2 - r1 );
							}
							// evaluate bin totals to find best position for object split
							for ( int i = 0; i < binCount - 1; i++ )
							{
								float C = SplitCostSAH( ref bvh, rSAV, AL[ i ], NL[ i ], AR[ i ], NR[ i ] );
								if ( C >= splitCost )
								{
									continue;
								}
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
					// consider a spatial split
					bool spatial = false;
					int bestNL = 0, bestNR = 0, budget = ( int )( sliceEnd - sliceStart );
					float3 spatialUnion = bestLMax - bestRMin;
					float spatialOverlap = HalfArea( spatialUnion ) / rootArea;
					if ( budget > ( int )node->TriCount && ( spatialOverlap > 1e-4f || splitCost >= noSplitCost ) )
					{
						float minSplitCost = splitCost * 0.985f; // don't accept a spatial split for minimal gain
						for ( int a = 0; a < 3; a++ )
						{
							if ( ( node->AabbMax[ a ] - node->AabbMin[ a ] ) > minDim[ a ] )
							{
								// setup bins
								for ( int i = 0; i < binCount; i++ )
								{
									sbinMin[ i ] = new float3( BvhConstants.Far );
									sbinMax[ i ] = new float3( -BvhConstants.Far );
									countIn[ i ] = 0;
									countOut[ i ] = 0;
								}
								// populate bins with clipped fragments
								float planeDist = ( node->AabbMax[ a ] - node->AabbMin[ a ] ) / ( binCount * 0.9999f );
								float rPlaneDist = 1f / planeDist, nodeMin = node->AabbMin[ a ];
								for ( uint i = 0; i < node->TriCount; i++ )
								{
									uint fi = primIdx[ node->LeftFirst + i ];
									int bin1 = Clamp( ( int )( ( fragment[ fi ].BMin[ a ] - nodeMin ) * rPlaneDist ), 0, binCount - 1 );
									int bin2 = Clamp( ( int )( ( fragment[ fi ].BMax[ a ] - nodeMin ) * rPlaneDist ), 0, binCount - 1 );
									countIn[ bin1 ]++;
									countOut[ bin2 ]++;
									if ( bin2 == bin1 ) // fragment fits in a single bin
									{
										sbinMin[ bin1 ] = math.min( sbinMin[ bin1 ], fragment[ fi ].BMin );
										sbinMax[ bin1 ] = math.max( sbinMax[ bin1 ], fragment[ fi ].BMax );
									}
									else
									{
										for ( int j = bin1; j <= bin2; j++ )
										{
											// clip fragment to each bin it overlaps
											float3 bmin = node->AabbMin, bmax = node->AabbMax;
											bmin[ a ] = nodeMin + ( planeDist * j );
											bmax[ a ] = j == ( binCount - 2 ) ? node->AabbMax[ a ] : ( bmin[ a ] + planeDist );
											Fragment orig = fragment[ fi ];
											if ( !ClipFrag( ref bvh, orig, out Fragment tmpFrag, bmin, bmax, a ) )
											{
												continue;
											}
											sbinMin[ j ] = math.min( sbinMin[ j ], tmpFrag.BMin );
											sbinMax[ j ] = math.max( sbinMax[ j ], tmpFrag.BMax );
										}
									}
								}
								// evaluate split candidates
								float3 l1 = new float3( BvhConstants.Far ), l2 = new float3( -BvhConstants.Far );
								float3 r1 = new float3( BvhConstants.Far ), r2 = new float3( -BvhConstants.Far );
								uint lN = 0, rN = 0;
								for ( int i = 0; i < binCount - 1; i++ )
								{
									lBMin[ i ] = l1 = math.min( l1, sbinMin[ i ] );
									rBMin[ binCount - 2 - i ] = r1 = math.min( r1, sbinMin[ binCount - 1 - i ] );
									lBMax[ i ] = l2 = math.max( l2, sbinMax[ i ] );
									rBMax[ binCount - 2 - i ] = r2 = math.max( r2, sbinMax[ binCount - 1 - i ] );
									lN += ( uint )countIn[ i ];
									rN += ( uint )countOut[ binCount - 1 - i ];
									AL[ i ] = lN == 0 ? BvhConstants.Far : HalfArea( l2 - l1 );
									AR[ binCount - 2 - i ] = rN == 0 ? BvhConstants.Far : HalfArea( r2 - r1 );
									NL[ i ] = ( int )lN;
									NR[ binCount - 2 - i ] = ( int )rN;
								}
								// find best position for spatial split
								for ( int i = 0; i < binCount - 1; i++ )
								{
									float Cspatial = SplitCostSAH( ref bvh, rSAV, AL[ i ], NL[ i ], AR[ i ], NR[ i ] );
									if ( Cspatial < minSplitCost && ( NL[ i ] + NR[ i ] ) < budget && ( NL[ i ] * NR[ i ] ) > 0 )
									{
										spatial = true;
										minSplitCost = splitCost = Cspatial;
										bestAxis = a;
										bestPos = i;
										bestLMin = lBMin[ i ];
										bestLMax = lBMax[ i ];
										bestRMin = rBMin[ i ];
										bestRMax = rBMax[ i ];
										bestNL = NL[ i ];
										bestNR = NR[ i ]; // for unsplitting
										bestLMax[ a ] = bestRMin[ a ]; // accurate
									}
								}
							}
						}
					}
					// evaluate best split cost
					if ( splitCost >= noSplitCost )
					{
						for ( uint i = 0; i < node->TriCount; i++ )
						{
							primIdx[ node->LeftFirst + i ] = fragment[ primIdx[ node->LeftFirst + i ] ].PrimIdx;
						}
						break; // not splitting is better.
					}
					// double-buffered partition
					uint A = sliceStart, B = sliceEnd, src = node->LeftFirst;
					if ( spatial )
					{
						// spatial partitioning
						float planeDist = ( node->AabbMax[ bestAxis ] - node->AabbMin[ bestAxis ] ) / ( binCount * 0.9999f );
						float rPlaneDist = 1f / planeDist, nodeMin = node->AabbMin[ bestAxis ];
						for ( uint i = 0; i < node->TriCount; i++ )
						{
							uint fragIdx = primIdx[ src++ ];
							float f1 = ( fragment[ fragIdx ].BMin[ bestAxis ] - nodeMin ) * rPlaneDist;
							float f2 = ( fragment[ fragIdx ].BMax[ bestAxis ] - nodeMin ) * rPlaneDist;
							uint bin1 = ( uint )( f1 > 0f ? f1 : 0f );
							uint bin2 = ( uint )( f2 > 0f ? f2 : 0f );
							if ( bin2 <= bestPos )
							{
								idxTmp[ A++ ] = fragIdx;
							}
							else if ( bin1 > bestPos )
							{
								idxTmp[ --B ] = fragIdx;
							}
							else
							{
								// unsplitting: 1. Calculate what happens if we add this primitive entirely to the left side
								if ( bestNR > 1 )
								{
									float3 unsplitLMin = math.min( bestLMin, fragment[ fragIdx ].BMin );
									float3 unsplitLMax = math.max( bestLMax, fragment[ fragIdx ].BMax );
									float unsplitAL = HalfArea( unsplitLMax - unsplitLMin );
									float unsplitAR = HalfArea( bestRMax - bestRMin );
									float CunsplitLeft = SplitCostSAH( ref bvh, rSAV, unsplitAL, bestNL, unsplitAR, bestNR - 1 );
									if ( CunsplitLeft <= splitCost )
									{
										bestNR--;
										splitCost = CunsplitLeft;
										idxTmp[ A++ ] = fragIdx;
										bestLMin = unsplitLMin;
										bestLMax = unsplitLMax;
										continue;
									}
								}
								// 2. Calculate what happens if we add this primitive entirely to the right side
								if ( bestNL > 1 )
								{
									float3 unsplitRMin = math.min( bestRMin, fragment[ fragIdx ].BMin );
									float3 unsplitRMax = math.max( bestRMax, fragment[ fragIdx ].BMax );
									float unsplitAL = HalfArea( bestLMax - bestLMin );
									float unsplitAR = HalfArea( unsplitRMax - unsplitRMin );
									float CunsplitRight = SplitCostSAH( ref bvh, rSAV, unsplitAL, bestNL - 1, unsplitAR, bestNR );
									if ( CunsplitRight <= splitCost )
									{
										bestNL--;
										splitCost = CunsplitRight;
										idxTmp[ --B ] = fragIdx;
										bestRMin = unsplitRMin;
										bestRMax = unsplitRMax;
										continue;
									}
								}
								// split straddler
								float splitPos = bestLMax[ bestAxis ];
								if ( SplitFrag( ref bvh, fragment[ fragIdx ], out Fragment part1, out Fragment part2, bestAxis, splitPos ) )
								{
									uint newFragIdx = ( uint )( Interlocked.Add( ref *nextFrag, 1 ) - 1 );
									fragment[ fragIdx ] = part1;
									idxTmp[ A++ ] = fragIdx;
									fragment[ newFragIdx ] = part2;
									idxTmp[ --B ] = newFragIdx;
								}
								else // didn't work out; see what we can do.
								{
									float sahLeft = HalfArea( part1.BMax - part1.BMin );
									if ( sahLeft > 0 )
									{
										idxTmp[ A++ ] = fragIdx;
									}
									else
									{
										idxTmp[ --B ] = fragIdx;
									}
								}
							}
						}
						// for spatial splits, we fully refresh the bounds: clipping is never fully stable..
						bestLMin = bestRMin = new float3( BvhConstants.Far );
						bestLMax = bestRMax = new float3( -BvhConstants.Far );
						for ( uint i = sliceStart; i < A; i++ )
						{
							bestLMin = math.min( bestLMin, fragment[ idxTmp[ i ] ].BMin );
							bestLMax = math.max( bestLMax, fragment[ idxTmp[ i ] ].BMax );
						}
						for ( uint i = B; i < sliceEnd; i++ )
						{
							bestRMin = math.min( bestRMin, fragment[ idxTmp[ i ] ].BMin );
							bestRMax = math.max( bestRMax, fragment[ idxTmp[ i ] ].BMax );
						}
					}
					else
					{
						// object partitioning
						for ( uint i = 0; i < node->TriCount; i++ )
						{
							uint fr = primIdx[ src + i ];
							// The C++ evaluates a scalar copy of the binning expression here. We reuse the
							// exact float3 expression of the binning pass instead, so both passes agree bit
							// for bit regardless of the runtime's intermediate precision (Mono evaluates
							// scalar float arithmetic in double), exactly as the reference builder does.
							int3 bi3 = ( int3 )( ( ( ( fragment[ fr ].BMin + fragment[ fr ].BMax ) * 0.5f ) - nmin3 ) * rpd3 );
							int bi = Clamp( bi3[ bestAxis ], 0, binCount - 1 );
							if ( bi <= bestPos )
							{
								idxTmp[ A++ ] = fr;
							}
							else
							{
								idxTmp[ --B ] = fr;
							}
						}
					}
					// copy back slice data
					UnsafeUtility.MemCpy( primIdx + sliceStart, idxTmp + sliceStart, ( long )( sliceEnd - sliceStart ) * 4 );
					// create child nodes
					uint leftCount = A - sliceStart, rightCount = sliceEnd - B;
					if ( leftCount == 0 || rightCount == 0 )
					{
						// spatial split failed. We shouldn't get here, but we do sometimes..
						for ( uint i = 0; i < node->TriCount; i++ )
						{
							primIdx[ node->LeftFirst + i ] = fragment[ primIdx[ node->LeftFirst + i ] ].PrimIdx;
						}
						node->AabbMin = math.min( bestLMin, bestRMin );
						node->AabbMax = math.max( bestLMax, bestRMax );
						break;
					}
					uint leftChildIdx = ( uint )( Interlocked.Add( ref *newNodePtr, 2 ) - 2 );
					uint rightChildIdx = leftChildIdx + 1;
					bvhNode[ leftChildIdx ].AabbMin = bestLMin;
					bvhNode[ leftChildIdx ].AabbMax = bestLMax;
					bvhNode[ leftChildIdx ].LeftFirst = sliceStart;
					bvhNode[ leftChildIdx ].TriCount = leftCount;
					bvhNode[ rightChildIdx ].AabbMin = bestRMin;
					bvhNode[ rightChildIdx ].AabbMax = bestRMax;
					bvhNode[ rightChildIdx ].LeftFirst = B;
					bvhNode[ rightChildIdx ].TriCount = rightCount;
					node->LeftFirst = leftChildIdx;
					node->TriCount = 0;
					if ( pending != null && ( depth + 1 ) == BvhConstants.MtSpawnDepth )
					{
						// hand both children to the parallel phase instead of descending into them.
						uint mid = ( A + B ) >> 1;
						pending[ *pendingCount ] = new HqSubtree { Node = leftChildIdx, SliceStart = sliceStart, SliceEnd = mid };
						( *pendingCount )++;
						pending[ *pendingCount ] = new HqSubtree { Node = rightChildIdx, SliceStart = mid, SliceEnd = sliceEnd };
						( *pendingCount )++;
						break;
					}
					// proceed with left child, push right child on local stack
					taskNode[ localTasks ] = rightChildIdx;
					taskDepth[ localTasks ] = depth + 1;
					taskSliceStart[ localTasks ] = ( A + B ) >> 1;
					taskSliceEnd[ localTasks++ ] = sliceEnd;
					nodeIdx = leftChildIdx;
					sliceEnd = ( A + B ) >> 1;
					depth++;
				}
				// pop a local task, if any are left
				if ( localTasks == 0 )
				{
					break;
				}
				localTasks--;
				nodeIdx = taskNode[ localTasks ];
				depth = taskDepth[ localTasks ];
				sliceStart = taskSliceStart[ localTasks ];
				sliceEnd = taskSliceEnd[ localTasks ];
			}
		}

		/// <summary>
		/// Port of BVH::SplitCostSAH. The C++ rounds the primitive counts up to a multiple of four
		/// when l_quads is set, which only the BVH4_CPU and BVH8_CPU layouts do; the plain BVH this
		/// builder produces leaves it false, so the counts are used as they are.
		/// </summary>
		private static float SplitCostSAH( ref Bvh bvh, float rAparent, float aLeft, int nLeft, float aRight, int nRight )
		{
			return bvh.TraversalCost + ( ( bvh.IntersectionCost * rAparent ) * ( ( aLeft * ( float )nLeft ) + ( aRight * ( float )nRight ) ) );
		}

		/// <summary>Port of BVH::NoSplitCostSAH; see SplitCostSAH for the l_quads note.</summary>
		private static float NoSplitCostSAH( ref Bvh bvh, int nParent )
		{
			return ( float )nParent * bvh.IntersectionCost;
		}

		/// <summary>
		/// Port of BVH::SplitFrag, non-SSE branch: cut a fragment in two new fragments.
		/// Based on madmann91 code. Also used by the presplitter in Bvh.Presplit.cs.
		/// </summary>
		internal static bool SplitFrag( ref Bvh bvh, in Fragment orig, out Fragment left, out Fragment right, int axis, float pos )
		{
			left.BMin = new float3( BvhConstants.Far );
			left.BMax = new float3( -BvhConstants.Far );
			left.PrimIdx = orig.PrimIdx;
			left.Clipped = 1;
			right = left;
			GetTriangle( ref bvh, orig.PrimIdx, out float3 v0, out float3 v1, out float3 v2 );
			bool l0 = v0[ axis ] <= pos, l1 = v1[ axis ] <= pos, l2 = v2[ axis ] <= pos;
			if ( l0 )
			{
				left.Extend( v0 );
			}
			else
			{
				right.Extend( v0 );
			}
			if ( l1 )
			{
				left.Extend( v1 );
			}
			else
			{
				right.Extend( v1 );
			}
			if ( l2 )
			{
				left.Extend( v2 );
			}
			else
			{
				right.Extend( v2 );
			}
			if ( l0 ^ l1 )
			{
				float3 c = SplitEdge( v0, v1, axis, pos );
				left.Extend( c );
				right.Extend( c );
			}
			if ( l1 ^ l2 )
			{
				float3 c = SplitEdge( v1, v2, axis, pos );
				left.Extend( c );
				right.Extend( c );
			}
			if ( l2 ^ l0 )
			{
				float3 c = SplitEdge( v2, v0, axis, pos );
				left.Extend( c );
				right.Extend( c );
			}
			if ( orig.Clipped != 0 ) // clip against orig box
			{
				left.BMin = math.max( orig.BMin, left.BMin );
				left.BMax = math.min( orig.BMax, left.BMax );
				right.BMin = math.max( orig.BMin, right.BMin );
				right.BMax = math.min( orig.BMax, right.BMax );
			}
			return HalfArea( left.BMax - left.BMin ) > 0 && HalfArea( right.BMax - right.BMin ) > 0;
		}

		/// <summary>Port of BVH::ClipFrag, non-SSE branch: clip a fragment for binning.</summary>
		private static bool ClipFrag( ref Bvh bvh, in Fragment orig, out Fragment newFrag, float3 bmin, float3 bmax, int axis )
		{
			Fragment tmp1;
			tmp1.BMin = new float3( BvhConstants.Far );
			tmp1.BMax = new float3( -BvhConstants.Far );
			tmp1.PrimIdx = orig.PrimIdx;
			tmp1.Clipped = 1;
			Fragment tmp2 = tmp1;
			GetTriangle( ref bvh, orig.PrimIdx, out float3 v0, out float3 v1, out float3 v2 );
			float left = bmin[ axis ], right = bmax[ axis ];
			// clip against min bounds
			bool in0 = v0[ axis ] >= left, in1 = v1[ axis ] >= left, in2 = v2[ axis ] >= left;
			if ( in0 )
			{
				tmp1.Extend( v0 );
			}
			if ( in1 )
			{
				tmp1.Extend( v1 );
			}
			if ( in2 )
			{
				tmp1.Extend( v2 );
			}
			if ( in0 ^ in1 )
			{
				tmp1.Extend( SplitEdge( v0, v1, axis, left ) );
			}
			if ( in1 ^ in2 )
			{
				tmp1.Extend( SplitEdge( v1, v2, axis, left ) );
			}
			if ( in2 ^ in0 )
			{
				tmp1.Extend( SplitEdge( v2, v0, axis, left ) );
			}
			// clip against max bounds
			in0 = v0[ axis ] <= right;
			in1 = v1[ axis ] <= right;
			in2 = v2[ axis ] <= right;
			if ( in0 )
			{
				tmp2.Extend( v0 );
			}
			if ( in1 )
			{
				tmp2.Extend( v1 );
			}
			if ( in2 )
			{
				tmp2.Extend( v2 );
			}
			if ( in0 ^ in1 )
			{
				tmp2.Extend( SplitEdge( v0, v1, axis, right ) );
			}
			if ( in1 ^ in2 )
			{
				tmp2.Extend( SplitEdge( v1, v2, axis, right ) );
			}
			if ( in2 ^ in0 )
			{
				tmp2.Extend( SplitEdge( v2, v0, axis, right ) );
			}
			newFrag.BMin = math.max( tmp1.BMin, tmp2.BMin );
			newFrag.BMax = math.min( tmp1.BMax, tmp2.BMax );
			// the non-SSE branch of the C++ leaves these two fields untouched and the caller only
			// reads the bounds; fill them in like the SSE branch does so the fragment is defined.
			newFrag.PrimIdx = orig.PrimIdx;
			newFrag.Clipped = 1;
			if ( orig.Clipped != 0 ) // clip against orig box
			{
				newFrag.BMin = math.max( orig.BMin, newFrag.BMin );
				newFrag.BMax = math.min( orig.BMax, newFrag.BMax );
			}
			float sa = HalfArea( newFrag.BMax - newFrag.BMin );
			return sa > 0;
		}

		/// <summary>Port of the split_edge lambda shared by SplitFrag and ClipFrag.</summary>
		private static float3 SplitEdge( float3 a, float3 b, int axis, float pos )
		{
			float3 c = a + ( ( pos - a[ axis ] ) / ( b[ axis ] - a[ axis ] ) * ( b - a ) );
			c[ axis ] = pos; // exactly on split position
			return c;
		}

		/// <summary>Fetches the three vertices of a primitive, indexed or not.</summary>
		private static void GetTriangle( ref Bvh bvh, uint prim, out float3 v0, out float3 v1, out float3 v2 )
		{
			uint vidx = prim * 3;
			if ( bvh.VertIdx == null )
			{
				v0 = bvh.Vertex( vidx ).xyz;
				v1 = bvh.Vertex( vidx + 1 ).xyz;
				v2 = bvh.Vertex( vidx + 2 ).xyz;
			}
			else
			{
				v0 = bvh.Vertex( bvh.VertIdx[ vidx ] ).xyz;
				v1 = bvh.Vertex( bvh.VertIdx[ vidx + 1 ] ).xyz;
				v2 = bvh.Vertex( bvh.VertIdx[ vidx + 2 ] ).xyz;
			}
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
