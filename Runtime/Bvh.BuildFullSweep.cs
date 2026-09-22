using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Burst-compiled full-sweep SAH builder: port of BVH::BuildFullSweep plus the FloatToKey /
	/// RadixSort helpers it is built on. Instead of using binning, this builder evaluates all
	/// possible split plane candidates for each axis. Works well with triangle presplitting.
	/// Enable it by setting Bvh.UseFullSweep before calling Bvh.Build; Bvh.UseThreadedBuild has no
	/// effect on it, the C++ spawn path is not ported and this builder always runs serially.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhFullSweepBuilder
	{
		/// <summary>Port of BVH::BuildFullSweep; run over the fragments BvhBuilder.PrepareBuild produced.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void BuildFullSweep( ref Bvh bvh )
		{
			BvhNode* bvhNode = bvh.Nodes;
			Fragment* fragment = bvh.Fragments;
			uint* primIdx = bvh.PrimIdx;
			uint triCount = bvh.TriCount;
			// create 32-bit integer sorting keys from fragment centroids. As in the C++, the still
			// unwritten part of the node pool doubles as the scratch buffer for the keys; they are
			// dead by the time the subdivision starts writing nodes there.
			uint** sortKey = stackalloc uint*[ 3 ];
			for ( uint a = 0; a < 3; a++ )
			{
				sortKey[ a ] = ( uint* )( bvhNode + 2 ) + ( a * triCount );
			}
			for ( uint i = 0; i < triCount; i++ )
			{
				for ( int a = 0; a < 3; a++ )
				{
					sortKey[ a ][ i ] = FloatToKey( fragment[ i ].BMin[ a ] + fragment[ i ].BMax[ a ] );
				}
			}
			// allocate data for O(N) stable partition
			byte* flag = ( byte* )bvh.Alloc( triCount );
			uint** sortedIdx = stackalloc uint*[ 3 ];
			for ( int a = 0; a < 3; a++ )
			{
				sortedIdx[ a ] = ( uint* )bvh.Alloc( ( long )triCount * 4 );
			}
			// note that each sort permutes primIdx as a side effect, which the next one then sorts;
			// the radix sort is stable, so that changes how ties are broken. The C++ does the same.
			for ( uint a = 0; a < 3; a++ )
			{
				RadixSort( primIdx, sortedIdx[ a ], sortKey[ a ], ( int )triCount );
			}
			// allocate space for right sweep
			float* SARs = ( float* )bvh.Alloc( ( long )triCount * sizeof( float ) );
			// subdivide root node recursively
			uint* task = stackalloc uint[ 512 ];
			uint taskCount = 0, nodeIdx = 0;
			int newNodePtr = ( int )bvh.UsedNodes;
			float3 minDim = ( bvhNode->AabbMax - bvhNode->AabbMin ) * 1e-20f;
			while ( true )
			{
				while ( true )
				{
					BvhNode* node = bvhNode + nodeIdx;
					// update node bounds
					node->AabbMin = new float3( BvhConstants.Far );
					node->AabbMax = new float3( -BvhConstants.Far );
					for ( uint i = 0; i < node->TriCount; i++ )
					{
						uint fi = sortedIdx[ 0 ][ node->LeftFirst + i ];
						node->AabbMin = math.min( node->AabbMin, fragment[ fi ].BMin );
						node->AabbMax = math.max( node->AabbMax, fragment[ fi ].BMax );
					}
					if ( node->TriCount == 1 )
					{
						break; // can't split one triangle.
					}
					float3 extent = node->AabbMax - node->AabbMin;
					// iterate over x,y,z
					float splitCost = ( float )node->TriCount * node->SurfaceArea;
					uint splitAxis = 0, splitPos = 0;
					for ( int a = 0; a < 3; a++ )
					{
						if ( extent[ a ] > minDim[ a ] )
						{
							uint firstRightTri = 1;
							float3 rMin = new float3( BvhConstants.Far ), rMax = new float3( -BvhConstants.Far );
							// sweep from right to left
							float if32 = 0f;
							for ( uint i = 0; i < node->TriCount; i++ )
							{
								uint fi = sortedIdx[ a ][ node->LeftFirst + node->TriCount - i - 1 ];
								float SAR = if32 * HalfArea( rMax - rMin );
								SARs[ node->LeftFirst + node->TriCount - i - 1 ] = SAR;
								rMin = math.min( rMin, fragment[ fi ].BMin );
								rMax = math.max( rMax, fragment[ fi ].BMax );
								if ( SAR >= splitCost )
								{
									// right side's cost is already greater than lowest cost and will only increase. Stop early
									firstRightTri = node->TriCount - i;
									break;
								}
								if32 += 1f;
							}
							// sweep from left to right
							float3 lMin = new float3( BvhConstants.Far ), lMax = new float3( -BvhConstants.Far );
							for ( uint i = 0; i < firstRightTri - 1; i++ )
							{
								uint fi = sortedIdx[ a ][ node->LeftFirst + i ];
								lMin = math.min( lMin, fragment[ fi ].BMin );
								lMax = math.max( lMax, fragment[ fi ].BMax );
							}
							if32 = ( float )firstRightTri;
							for ( uint i = firstRightTri - 1; i < node->TriCount - 1; i++ )
							{
								uint fi = sortedIdx[ a ][ node->LeftFirst + i ];
								lMin = math.min( lMin, fragment[ fi ].BMin );
								lMax = math.max( lMax, fragment[ fi ].BMax );
								float SAL = if32 * HalfArea( lMax - lMin );
								float C = SAL + SARs[ node->LeftFirst + i ];
								if ( C < splitCost )
								{
									splitCost = C;
									splitPos = i + 1;
									splitAxis = ( uint )a;
								}
								else if ( SAL >= splitCost )
								{
									break;
								}
								if32 += 1f;
							}
						}
					}
					float noSplitCost = bvh.IntersectionCost * ( float )node->TriCount;
					splitCost = bvh.TraversalCost + ( bvh.IntersectionCost * splitCost / node->SurfaceArea );
					if ( splitCost >= noSplitCost )
					{
						break; // not splitting turns out to be better.
					}
					// partition
					for ( uint i = 0; i < splitPos; i++ )
					{
						flag[ sortedIdx[ splitAxis ][ node->LeftFirst + i ] ] = 0; // "left"
					}
					for ( uint i = splitPos; i < node->TriCount; i++ )
					{
						flag[ sortedIdx[ splitAxis ][ node->LeftFirst + i ] ] = 1; // "right"
					}
					// stable partition needs temp buffer, let's reuse memory
					uint* tmp = ( uint* )SARs;
					for ( uint a = 0; a < 3; a++ )
					{
						if ( a != splitAxis )
						{
							uint p0 = 0, p1 = 0;
							for ( uint i = 0; i < node->TriCount; i++ )
							{
								uint fi = sortedIdx[ a ][ node->LeftFirst + i ];
								if ( flag[ fi ] != 0 )
								{
									tmp[ node->LeftFirst + p1++ ] = fi;
								}
								else
								{
									sortedIdx[ a ][ node->LeftFirst + p0++ ] = fi;
								}
							}
							UnsafeUtility.MemCpy( sortedIdx[ a ] + node->LeftFirst + p0, tmp + node->LeftFirst, ( long )p1 * 4 );
						}
					}
					// create child nodes
					uint leftCount = splitPos, rightCount = node->TriCount - leftCount;
					if ( leftCount >= node->TriCount || rightCount >= node->TriCount || taskCount == 512 )
					{
						break;
					}
					UnsafeUtility.MemCpy( primIdx + node->LeftFirst, sortedIdx[ splitAxis ] + node->LeftFirst, ( long )node->TriCount * 4 );
					uint n = ( uint )newNodePtr;
					newNodePtr += 2;
					bvhNode[ n ].LeftFirst = node->LeftFirst;
					bvhNode[ n ].TriCount = leftCount;
					bvhNode[ n + 1 ].LeftFirst = node->LeftFirst + leftCount;
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
			// cleanup allocated buffers when done
			for ( int a = 0; a < 3; a++ )
			{
				bvh.Free( sortedIdx[ a ] );
			}
			bvh.Free( SARs );
			bvh.Free( flag );
			bvh.AabbMin = bvhNode[ 0 ].AabbMin;
			bvh.AabbMax = bvhNode[ 0 ].AabbMax;
			bvh.Refittable = true; // not using spatial splits: can refit this BVH
			bvh.MayHaveHoles = false; // this builder produces a continuous list of nodes
			bvh.BvhOverAabbs = bvh.Verts == null; // bvh over aabbs is suitable as TLAS
			bvh.UsedNodes = ( uint )newNodePtr;
			bvh.ThreadedSubtrees = 0;
			BvhPresplitter.FinishPresplit( ref bvh );
		}

		/// <summary>Port of FloatToKey: an order-preserving unsigned key for a float.</summary>
		private static uint FloatToKey( float value )
		{
			uint f = math.asuint( value ), mask = ( uint )( ( ( int )f >> 31 ) | ( 1 << 31 ) );
			return f ^ mask;
		}

		/// <summary>
		/// Port of RadixSort, http://stereopsis.com/radix.html. Three 11-bit passes with the input
		/// and output buffers swapped in between, so the caller's 'input' ends up holding the result
		/// of the second pass rather than what it came in with.
		/// </summary>
		private static void RadixSort( uint* input, uint* output, uint* keys, int len )
		{
			const int binSize = 1 << 11;
			const uint mask = binSize - 1;
			int* prefixSum = stackalloc int[ binSize * 3 ];
			UnsafeUtility.MemClear( prefixSum, binSize * 3 * sizeof( int ) );
			for ( int i = 0; i < len; i++ ) // compute histogram for all passes
			{
				uint key = keys[ input[ i ] ];
				prefixSum[ key & mask ]++;
				prefixSum[ ( ( key >> 11 ) & mask ) + binSize ]++;
				prefixSum[ ( ( key >> 22 ) & mask ) + ( 2 * binSize ) ]++;
			}
			// compute prefix sum for all passes
			int sum0 = 0, sum1 = 0, sum2 = 0;
			for ( int i = 0; i < binSize; i++ )
			{
				int temp0 = prefixSum[ i ], temp1 = prefixSum[ i + binSize ], temp2 = prefixSum[ i + ( 2 * binSize ) ];
				prefixSum[ i ] = sum0;
				sum0 += temp0;
				prefixSum[ i + binSize ] = sum1;
				sum1 += temp1;
				prefixSum[ i + ( 2 * binSize ) ] = sum2;
				sum2 += temp2;
			}
			for ( int i = 0; i < 3; i++ ) // sort from LSB to MSB in radix-sized steps
			{
				for ( int j = 0; j < len; j++ )
				{
					uint element = input[ j ], key = keys[ element ];
					output[ prefixSum[ ( ( key >> ( i * 11 ) ) & mask ) + ( uint )( i * binSize ) ]++ ] = element;
				}
				uint* t = input;
				input = output;
				output = t;
			}
		}

		/// <summary>Port of tinybvh_halfarea, including the guard for empty (inverted) boxes.</summary>
		private static float HalfArea( float3 v )
		{
			return v.x < -BvhConstants.Far ? 0f : ( ( v.x * v.y ) + ( v.y * v.z ) + ( v.z * v.x ) );
		}
	}
}
