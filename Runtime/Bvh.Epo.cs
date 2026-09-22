using System;
using Unity.Burst;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// The EPO tree quality metric of tinybvh's BVH class: BVH::EPOCost and its helpers. See
	/// "On Quality Metrics of Bounding Volume Hierarchies", Aila et al., 2013.
	/// </summary>
	public unsafe partial struct Bvh
	{
		/// <summary>
		/// Port of BVH::EPOCost: the blended EPO / SAH cost of the tree. Lower is better. This is
		/// the non-threaded C++ variant; it is O( N * overlap ) and gets slow well before a hundred
		/// thousand primitives.
		/// </summary>
		public float EpoCost( uint nodeIdx = 0 )
		{
			if ( Nodes == null || UsedNodes == 0 )
			{
				throw new InvalidOperationException( "Bvh.EpoCost( .. ), no tree was built." );
			}
			BvhEpo.EpoCost( ref this, nodeIdx, out float cost );
			return cost;
		}
	}

	/// <summary>
	/// Burst-compiled half of the EPO metric. Direct calls must be synchronous, otherwise editor
	/// tests silently run the Mono fallback, which evaluates float math in double and so produces
	/// a different cost than the C++ reference.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhEpo
	{
		/// <summary>C++: 'bvhvec3 vin[10], vout[10]' in BVH::EPOArea.</summary>
		private const int ClipBufferSize = 10;

		/// <summary>Port of BVH::EPOCost( nodeIdx, depth ), the ENABLE_THREADED_BUILDS-off variant.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void EpoCost( ref Bvh bvh, uint nodeIdx, out float result )
		{
			result = EpoCostRec( ref bvh, nodeIdx );
		}

		private static float EpoCostRec( ref Bvh bvh, uint nodeIdx )
		{
			// Determine the EPO cost of the tree. See:
			// "On Quality Metrics of Bounding Volume Hierarchies", Aila et al., 2013.
			BvhNode* n = bvh.Nodes + nodeIdx;
			float cost = ( n->IsLeaf ? ( bvh.IntersectionCost * n->TriCount ) : bvh.TraversalCost ) * EpoArea( ref bvh, nodeIdx, 0 );
			float totalArea = 0f;
			if ( !n->IsLeaf )
			{
				cost += EpoCostRec( ref bvh, n->LeftFirst ) + EpoCostRec( ref bvh, n->LeftFirst + 1 );
			}
			if ( nodeIdx > 0 )
			{
				return cost;
			}
			// recursion ends with node 0: Finalize EPO calculation
			for ( uint i = 0; i < bvh.TriCount; i++ )
			{
				totalArea += PrimArea( ref bvh, i );
			}
			cost /= totalArea;
			return ( ( 1f - BvhConstants.EpoWeight ) * SahCost( ref bvh, 0 ) ) + ( BvhConstants.EpoWeight * cost );
		}

		/// <summary>
		/// Port of BVH::SAHCost, duplicated here because the C++ EPOCost closes with a call to it
		/// and the whole metric has to be evaluated under Burst to match the reference bit for bit.
		/// Kept identical to <see cref="Bvh.SahCost"/>.
		/// </summary>
		private static float SahCost( ref Bvh bvh, uint nodeIdx )
		{
			BvhNode* n = bvh.Nodes + nodeIdx;
			if ( n->IsLeaf )
			{
				return bvh.IntersectionCost * n->SurfaceArea * n->TriCount;
			}
			float cost = ( bvh.TraversalCost * n->SurfaceArea ) + SahCost( ref bvh, n->LeftFirst ) + SahCost( ref bvh, n->LeftFirst + 1 );
			return nodeIdx == 0 ? ( cost / n->SurfaceArea ) : cost;
		}

		/// <summary>Port of BVH::PrimArea: the area of the p'th primitive in primIdx order.</summary>
		private static float PrimArea( ref Bvh bvh, uint p )
		{
			bvh.GetPrimIndices( bvh.PrimIdx[ p ], out uint i0, out uint i1, out uint i2 );
			float3 v0 = bvh.Vertex( i0 ).xyz, v1 = bvh.Vertex( i1 ).xyz, v2 = bvh.Vertex( i2 ).xyz;
			return 0.5f * math.length( math.cross( v1 - v0, v2 - v0 ) );
		}

		/// <summary>Port of tinybvh_aabbs_overlap, scalar branch.</summary>
		private static bool AabbsOverlap( BvhNode* node1, BvhNode* node2 )
		{
			float3 bmin1 = node1->AabbMin, bmin2 = node2->AabbMin;
			float3 bmax1 = node1->AabbMax, bmax2 = node2->AabbMax;
			return bmin1.x <= bmax2.x && bmax1.x >= bmin2.x && bmin1.y <= bmax2.y &&
				bmax1.y >= bmin2.y && bmin1.z <= bmax2.z && bmax1.z >= bmin2.z;
		}

		/// <summary>
		/// Port of BVH::EPOArea: the total primitive area of the tree that falls inside the AABB of
		/// the subtree rooted at subtreeRoot, excluding the subtree's own primitives.
		/// </summary>
		private static float EpoArea( ref Bvh bvh, uint subtreeRoot, uint nodeIdx )
		{
			// abort if we reached the subtree
			if ( nodeIdx == subtreeRoot )
			{
				return 0f;
			}
			BvhNode* n = bvh.Nodes + nodeIdx;
			BvhNode* subtree = bvh.Nodes + subtreeRoot;
			// handle case where n is a leaf node
			float area = 0f;
			if ( n->IsLeaf )
			{
				float3 bmin = subtree->AabbMin, bmax = subtree->AabbMax;
				float3* vin = stackalloc float3[ ClipBufferSize ];
				float3* vout = stackalloc float3[ ClipBufferSize ];
				for ( uint i = 0; i < n->TriCount; i++ )
				{
					// Early out: triangle fully inside the subtree AABB?
					bvh.GetPrimIndices( bvh.PrimIdx[ n->LeftFirst + i ], out uint i0, out uint i1, out uint i2 );
					float3 v0 = bvh.Vertex( i0 ).xyz, v1 = bvh.Vertex( i1 ).xyz, v2 = bvh.Vertex( i2 ).xyz;
					bool allin = v0.x >= bmin.x && v0.x <= bmax.x && v0.y >= bmin.y && v0.y <= bmax.y && v0.z >= bmin.z && v0.z <= bmax.z;
					allin &= v1.x >= bmin.x && v1.x <= bmax.x && v1.y >= bmin.y && v1.y <= bmax.y && v1.z >= bmin.z && v1.z <= bmax.z;
					allin &= v2.x >= bmin.x && v2.x <= bmax.x && v2.y >= bmin.y && v2.y <= bmax.y && v2.z >= bmin.z && v2.z <= bmax.z;
					if ( allin )
					{
						area += 0.5f * math.length( math.cross( v1 - v0, v2 - v0 ) );
						continue;
					}
					// Brute force: clip triangle against six planes of the AABB.
					uint Nin = 3;
					float3 C = default;
					vin[ 0 ] = v0;
					vin[ 1 ] = v1;
					vin[ 2 ] = v2;
					for ( int a = 0; a < 3; a++ )
					{
						uint Nout = 0;
						float l = bmin[ a ], r = bmax[ a ];
						for ( uint v = 0; v < Nin; v++ )
						{
							float3 vert0 = vin[ v ], vert1 = vin[ ( v + 1 ) % Nin ];
							bool v0in = vert0[ a ] >= l, v1in = vert1[ a ] >= l;
							if ( !( v0in || v1in ) )
							{
								continue;
							}
							else if ( v0in ^ v1in )
							{
								C = vert0 + ( ( ( l - vert0[ a ] ) / ( vert1[ a ] - vert0[ a ] ) ) * ( vert1 - vert0 ) );
								C[ a ] = l; // accurate
								vout[ Nout++ ] = C;
							}
							if ( v1in )
							{
								vout[ Nout++ ] = vert1;
							}
						}
						Nin = 0;
						for ( uint v = 0; v < Nout; v++ )
						{
							float3 vert0 = vout[ v ], vert1 = vout[ ( v + 1 ) % Nout ];
							bool v0in = vert0[ a ] <= r, v1in = vert1[ a ] <= r;
							if ( !( v0in || v1in ) )
							{
								continue;
							}
							else if ( v0in ^ v1in )
							{
								C = vert0 + ( ( ( r - vert0[ a ] ) / ( vert1[ a ] - vert0[ a ] ) ) * ( vert1 - vert0 ) );
								C[ a ] = r; // accurate
								vin[ Nin++ ] = C;
							}
							if ( v1in )
							{
								vin[ Nin++ ] = vert1;
							}
						}
					}
					if ( Nin < 3 )
					{
						continue;
					}
					// calculate area of remaining convex shape in vin
					uint tris = Nin - 2;
					float3 p0 = vin[ 0 ];
					for ( uint j = 0; j < tris; j++ )
					{
						float3 p1 = vin[ j + 1 ], p2 = vin[ j + 2 ];
						area += 0.5f * math.length( math.cross( p1 - p0, p2 - p0 ) );
					}
				}
				return area;
			}
			// recurse if n is an inner node
			BvhNode* left = bvh.Nodes + n->LeftFirst, right = bvh.Nodes + n->LeftFirst + 1;
			if ( AabbsOverlap( left, subtree ) )
			{
				area += EpoArea( ref bvh, subtreeRoot, n->LeftFirst );
			}
			if ( AabbsOverlap( right, subtree ) )
			{
				area += EpoArea( ref bvh, subtreeRoot, n->LeftFirst + 1 );
			}
			return area;
		}
	}
}
