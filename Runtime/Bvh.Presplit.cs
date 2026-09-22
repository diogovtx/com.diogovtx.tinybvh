using Unity.Burst;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Burst-compiled triangle presplitting: ports of BVH::SplitPriority, BVH::SplitCount,
	/// BVH::GetNodeSize, BVH::Presplit, BVH::PresplitPostPass and BVH::PrimArea. Based on Section 5
	/// of "Fast Parallel Construction of High-Quality Bounding Volume Hierarchies", Karras and Aila,
	/// 2013, and BoyBaykiller's implementation, which is in turn based on code by MadMann91.
	/// Presplit runs from BvhBuilder.PrepareBuild, before the subdivision starts, and turns the
	/// fragment array into a longer list of smaller fragments; FinishPresplit runs at the end of
	/// every binned and full-sweep build and maps the index array back onto the input primitives.
	/// Enable the pass by setting Bvh.UsePresplitting before calling Bvh.Build.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhPresplitter
	{
		/// <summary>
		/// Port of BVH::Presplit. Splits fragments until the split budget is used up and returns the
		/// resulting fragment count, which the caller stores in triCount and idxCount.
		/// </summary>
		internal static uint Presplit( ref Bvh bvh )
		{
			uint fragCount = bvh.TriCount;
			float factor = bvh.PresplitFactor;
			uint splitBudget = ( uint )( int )( bvh.TriCount * bvh.PresplitFactor );
			Fragment* fragment = bvh.Fragments;
			uint* primIdx = bvh.PrimIdx;
			// determine per-triangle split count.
			long scratchCount = ( long )bvh.TriCount + splitBudget;
			float* prio = ( float* )bvh.Alloc( scratchCount * sizeof( float ) );
			int* splits = ( int* )bvh.Alloc( scratchCount * sizeof( int ) );
			while ( true )
			{
				float summedPrio = 0f;
				for ( uint i = 0; i < bvh.TriCount; i++ )
				{
					float p = SplitPriority( ref bvh, fragment[ i ] );
					prio[ i ] = p;
					summedPrio += p;
				}
				uint totalSplits = 0;
				for ( uint i = 0; i < bvh.TriCount; i++ )
				{
					int s = SplitCount( prio[ i ], summedPrio, ( int )bvh.TriCount, factor );
					splits[ i ] = s;
					totalSplits += ( uint )s;
				}
				if ( totalSplits <= bvh.TriCount + splitBudget )
				{
					break;
				}
				factor *= 0.95f; // TODO: will this ever happen? also: tweak based on excess.
			}
			// do actual splitting.
			BvhNode* root = bvh.Nodes;
			float3 rootExtent = root->AabbMax - root->AabbMin;
			for ( uint i = 0; i < fragCount; )
			{
				if ( splits[ i ] == 1 )
				{
					i++;
				}
				else
				{
					Fragment f = fragment[ i ];
					float3 extent = f.BMax - f.BMin;
					int splitAxis = MaxDim( extent );
					float nodeSize = GetNodeSize( extent[ splitAxis ], rootExtent[ splitAxis ] );
					if ( nodeSize >= extent[ splitAxis ] - 0.0001f )
					{
						nodeSize *= 0.5f;
					}
					// snap mid position to nearest split plane
					float midPos = ( f.BMin[ splitAxis ] + f.BMax[ splitAxis ] ) * 0.5f;
					float index = RoundF( ( midPos - root->AabbMin[ splitAxis ] ) / nodeSize );
					float splitPos = root->AabbMin[ splitAxis ] + ( index * nodeSize );
					// actual split
					BvhHqBuilder.SplitFrag( ref bvh, f, out Fragment part1, out Fragment part2, splitAxis, splitPos );
					fragment[ i ] = part1;
					fragment[ fragCount ] = part2;
					// distribute available splits over part1 and part2
					int toDivide = splits[ i ];
					float3 p1Extent = part1.BMax - part1.BMin, p2Extent = part2.BMax - part2.BMin;
					float p1Size = p1Extent[ MaxDim( p1Extent ) ];
					float p2Size = p2Extent[ MaxDim( p2Extent ) ];
					int p1Count = ( int )( ( float )toDivide * p1Size / ( p1Size + p2Size ) );
					splits[ i ] = Clamp( p1Count, 1, toDivide - 1 );
					splits[ fragCount ] = toDivide - splits[ i ];
					primIdx[ fragCount ] = fragCount;
					fragCount++;
				}
			}
			// cleanup
			bvh.Free( prio );
			bvh.Free( splits );
			// all done.
			return fragCount;
		}

		/// <summary>
		/// Presplit half of the build finalisation: maps the index array back onto input primitives
		/// and runs the post pass. Does nothing when presplitting is off. There is no float math here,
		/// so it is safe to run from the managed side of the threaded build as well.
		/// </summary>
		internal static void FinishPresplit( ref Bvh bvh )
		{
			if ( bvh.UsePresplitting )
			{
				// finalize indices in index array
				uint* primIdx = bvh.PrimIdx;
				Fragment* fragment = bvh.Fragments;
				for ( uint i = 0; i < bvh.TriCount; i++ )
				{
					primIdx[ i ] = fragment[ primIdx[ i ] ].PrimIdx;
				}
				if ( bvh.PresplitPostPass )
				{
					PresplitPostPass( ref bvh );
				}
			}
		}

		/// <summary>
		/// Port of BVH::PresplitPostPass: see if we have any leafs that reference the same primitive
		/// multiple times. Duplicates are dropped by shrinking the leaf in place, which leaves the
		/// entries past the new count untouched.
		/// </summary>
		private static void PresplitPostPass( ref Bvh bvh )
		{
			BvhNode* bvhNode = bvh.Nodes;
			uint* primIdx = bvh.PrimIdx;
			for ( uint i = 2; i < bvh.UsedNodes; i++ )
			{
				if ( bvhNode[ i ].IsLeaf )
				{
					uint first = bvhNode[ i ].LeftFirst;
					for ( uint j = 0; j < bvhNode[ i ].TriCount; j++ )
					{
						uint p0 = primIdx[ first + j ];
						for ( uint k = j + 1; k < bvhNode[ i ].TriCount; k++ )
						{
							uint p1 = primIdx[ first + k ];
							if ( p0 == p1 )
							{
								bvhNode[ i ].TriCount--;
								primIdx[ first + k ] = primIdx[ first + bvhNode[ i ].TriCount ];
							}
						}
					}
				}
			}
		}

		/// <summary>Port of BVH::SplitPriority.</summary>
		private static float SplitPriority( ref Bvh bvh, in Fragment f )
		{
			float3 extent = f.BMax - f.BMin;
			float extentPrio = Sqrf( extent[ MaxDim( extent ) ] );
			float boxArea = 2 * HalfArea( extent ); // TODO: half of this seems more appropriate?
			float triArea = PrimArea( ref bvh, f.PrimIdx );
			float emptyAreaPrio = boxArea - triArea;
			return Cbrt( extentPrio * emptyAreaPrio );
		}

		/// <summary>Port of BVH::SplitCount.</summary>
		private static int SplitCount( float prio, float sumPrio, int tris, float factor )
		{
			float shareOfTris = prio / sumPrio * ( float )tris;
			return 1 + ( int )( shareOfTris * factor );
		}

		/// <summary>
		/// Port of BVH::GetNodeSize: the largest power of two not exceeding 'extent', expressed in
		/// units of 'globalSize'. Keeps only the exponent bits of extent / globalSize.
		/// </summary>
		private static float GetNodeSize( float extent, float globalSize )
		{
			// transform into [0.0, 1.0]
			float alpha = extent / globalSize;
			// compute 2^(floor(log2(alpha)))
			uint exponentBits = math.asuint( alpha ) & ( 255u << 23 );
			// transform back into global space
			return math.asfloat( exponentBits ) * globalSize;
		}

		/// <summary>Port of BVH::PrimArea: twice the triangle area, as the split priority uses it.</summary>
		private static float PrimArea( ref Bvh bvh, uint p )
		{
			uint vidx = bvh.PrimIdx[ p ] * 3;
			float3 v0, v1, v2;
			if ( bvh.VertIdx != null )
			{
				v0 = bvh.Vertex( bvh.VertIdx[ vidx ] ).xyz;
				v1 = bvh.Vertex( bvh.VertIdx[ vidx + 1 ] ).xyz;
				v2 = bvh.Vertex( bvh.VertIdx[ vidx + 2 ] ).xyz;
			}
			else
			{
				v0 = bvh.Vertex( vidx ).xyz;
				v1 = bvh.Vertex( vidx + 1 ).xyz;
				v2 = bvh.Vertex( vidx + 2 ).xyz;
			}
			return 0.5f * Length( Cross( v1 - v0, v2 - v0 ) );
		}

		/// <summary>
		/// Port of cbrtf. Burst maps System.Math.Cbrt to an intrinsic; the double result rounded to
		/// float is what the reference tool's cbrtf produces.
		/// </summary>
		private static float Cbrt( float x )
		{
			return ( float )System.Math.Cbrt( ( double )x );
		}

		/// <summary>
		/// Port of roundf, which rounds halves away from zero. Unity.Mathematics' math.round rounds
		/// halves to even, so it cannot be used here. The subtraction is exact: below 2^23 both terms
		/// share an exponent range, above it the fraction is zero.
		/// </summary>
		private static float RoundF( float x )
		{
			float a = math.abs( x );
			float r = math.floor( a );
			if ( ( a - r ) >= 0.5f )
			{
				r += 1f;
			}
			return x < 0f ? -r : r;
		}

		/// <summary>Port of tinybvh_maxdim: index of the component with the largest magnitude.</summary>
		private static int MaxDim( float3 v )
		{
			int r = math.abs( v.x ) > math.abs( v.y ) ? 0 : 1;
			return math.abs( v.z ) > math.abs( v[ r ] ) ? 2 : r;
		}

		/// <summary>Port of tinybvh_cross, same operation order.</summary>
		private static float3 Cross( float3 a, float3 b )
		{
			return new float3(
				( a.y * b.z ) - ( a.z * b.y ),
				( a.z * b.x ) - ( a.x * b.z ),
				( a.x * b.y ) - ( a.y * b.x ) );
		}

		/// <summary>Port of tinybvh_length, same operation order.</summary>
		private static float Length( float3 a )
		{
			return math.sqrt( ( a.x * a.x ) + ( a.y * a.y ) + ( a.z * a.z ) );
		}

		/// <summary>Port of tinybvh_sqrf.</summary>
		private static float Sqrf( float x )
		{
			return x * x;
		}

		/// <summary>Port of tinybvh_halfarea, including the guard for empty (inverted) boxes.</summary>
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
