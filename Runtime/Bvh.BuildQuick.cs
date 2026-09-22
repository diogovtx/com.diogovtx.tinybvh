using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	public unsafe partial struct Bvh
	{
		/// <summary>
		/// Port of BVH::BuildQuick: a basic single-function BVH builder using mid-point splits, with
		/// no SAH evaluation at all. Much faster than Build and much worse; use it when the tree is
		/// thrown away again immediately. Triangle soup only: three consecutive 16-byte vertices per
		/// primitive, no index buffer. None of the build settings apply.
		/// </summary>
		public void BuildQuick( float4* vertices, uint primCount )
		{
			ValidateBuildInput( ( byte* )vertices, primCount * 3, 16, null, primCount );
			BvhQuickBuilder.BuildQuick( ref this, ( byte* )vertices, primCount * 3, 16 );
		}

		/// <summary>BuildQuick over a NativeArray; the array must outlive the BVH.</summary>
		public void BuildQuick( NativeArray<float4> vertices, uint primCount )
		{
			BuildQuick( ( float4* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( vertices ), primCount );
		}
	}

	/// <summary>Burst-compiled implementation of BVH::BuildQuick.</summary>
	[BurstCompile]
	internal static unsafe class BvhQuickBuilder
	{
		[BurstCompile( CompileSynchronously = true )]
		internal static void BuildQuick( ref Bvh bvh, byte* vertices, uint vertexCount, int vertexStride )
		{
			// Basic single-function BVH builder, using mid-point splits.
			uint primCount = vertexCount / 3;
			uint spaceNeeded = primCount * 2; // upper limit
			// allocate on first build
			bvh.AllocateNodes( spaceNeeded );
			bvh.AllocatePrimIdx( primCount );
			bvh.AllocateFragments( primCount );
			bvh.Nodes[ 1 ] = default; // node 1 remains unused, for cache line alignment.
			bvh.Verts = vertices; // note: we're not copying this data; don't delete.
			bvh.VertCount = vertexCount;
			bvh.VertStride = vertexStride;
			// the C++ only assigns verts here and leaves vertIdx and the two 'bvh over' flags at
			// whatever a previous build left them at; clear them so reusing a Bvh stays correct.
			bvh.VertIdx = null;
			bvh.BvhOverIndices = false;
			bvh.BvhOverAabbs = false;
			bvh.IdxCount = primCount;
			bvh.TriCount = primCount;
			int newNodePtr = 2;
			// assign all triangles to the root node
			BvhNode* bvhNode = bvh.Nodes;
			BvhNode* root = bvhNode;
			root->LeftFirst = 0;
			root->TriCount = bvh.TriCount;
			root->AabbMin = new float3( BvhConstants.Far );
			root->AabbMax = new float3( -BvhConstants.Far );
			// initialize fragments and initialize root node bounds
			Fragment* fragment = bvh.Fragments;
			uint* primIdx = bvh.PrimIdx;
			for ( uint i = 0; i < bvh.TriCount; i++ )
			{
				float4 v0 = bvh.Vertex( i * 3 ), v1 = bvh.Vertex( ( i * 3 ) + 1 ), v2 = bvh.Vertex( ( i * 3 ) + 2 );
				fragment[ i ].BMin = math.min( math.min( v0, v1 ), v2 ).xyz;
				fragment[ i ].BMax = math.max( math.max( v0, v1 ), v2 ).xyz;
				root->AabbMin = math.min( root->AabbMin, fragment[ i ].BMin );
				root->AabbMax = math.max( root->AabbMax, fragment[ i ].BMax );
				primIdx[ i ] = i;
			}
			// subdivide recursively
			uint* task = stackalloc uint[ 512 ];
			uint taskCount = 0, nodeIdx = 0;
			while ( true )
			{
				while ( true )
				{
					BvhNode* node = bvhNode + nodeIdx;
					// in-place partition against midpoint on longest axis
					uint j = node->LeftFirst + node->TriCount, src = node->LeftFirst;
					int axis = 0;
					float3 extent = node->AabbMax - node->AabbMin;
					if ( extent.y > extent.x && extent.y > extent.z )
					{
						axis = 1;
					}
					if ( extent.z > extent.x && extent.z > extent.y )
					{
						axis = 2;
					}
					float splitPos = node->AabbMin[ axis ] + ( extent[ axis ] * 0.5f );
					float3 lbmin = new float3( BvhConstants.Far ), lbmax = new float3( -BvhConstants.Far );
					float3 rbmin = new float3( BvhConstants.Far ), rbmax = new float3( -BvhConstants.Far );
					for ( uint i = 0; i < node->TriCount; i++ )
					{
						uint fi = primIdx[ src ];
						float3 fmin = fragment[ fi ].BMin, fmax = fragment[ fi ].BMax;
						float centroid = ( fmin[ axis ] + fmax[ axis ] ) * 0.5f;
						if ( centroid < splitPos )
						{
							lbmin = math.min( lbmin, fmin );
							lbmax = math.max( lbmax, fmax );
							src++;
							continue;
						}
						rbmin = math.min( rbmin, fmin );
						rbmax = math.max( rbmax, fmax );
						--j;
						uint t = primIdx[ src ];
						primIdx[ src ] = primIdx[ j ];
						primIdx[ j ] = t;
					}
					// create child nodes
					uint leftCount = src - node->LeftFirst, rightCount = node->TriCount - leftCount;
					if ( leftCount == 0 || rightCount == 0 || taskCount == 512 )
					{
						break; // split did not work out.
					}
					uint lci = ( uint )newNodePtr++, rci = ( uint )newNodePtr++;
					bvhNode[ lci ].AabbMin = lbmin;
					bvhNode[ lci ].AabbMax = lbmax;
					bvhNode[ lci ].LeftFirst = node->LeftFirst;
					bvhNode[ lci ].TriCount = leftCount;
					bvhNode[ rci ].AabbMin = rbmin;
					bvhNode[ rci ].AabbMax = rbmax;
					bvhNode[ rci ].LeftFirst = j;
					bvhNode[ rci ].TriCount = rightCount;
					node->LeftFirst = lci;
					node->TriCount = 0;
					task[ taskCount++ ] = rci;
					nodeIdx = lci;
				}
				// fetch subdivision task from stack
				if ( taskCount == 0 )
				{
					break;
				}
				nodeIdx = task[ --taskCount ];
			}
			// all done.
			bvh.AabbMin = bvhNode[ 0 ].AabbMin;
			bvh.AabbMax = bvhNode[ 0 ].AabbMax;
			bvh.UsedNodes = ( uint )newNodePtr;
			bvh.ThreadedSubtrees = 0;
			bvh.Refittable = true; // not using spatial splits: can refit this BVH
			bvh.MayHaveHoles = false; // the reference builder produces a continuous list of nodes
		}
	}
}
