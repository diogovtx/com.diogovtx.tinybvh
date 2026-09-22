using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Alternative 64-byte BVH node layout, which specifies the bounds of the children rather than
	/// the node itself. This layout is used by Aila and Laine in their seminal GPU ray tracing paper.
	/// Port of BVH_GPU::BVHNode.
	/// </summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct BvhGpuNode
	{
		public float3 LMin;
		public uint Left;
		public float3 LMax;
		public uint Right;
		public float3 RMin;
		public uint TriCount;
		public float3 RMax;
		public uint FirstTri; // total: 64 bytes

		public bool IsLeaf => TriCount > 0;
	}

	/// <summary>
	/// Port of tinybvh's BVH_GPU class. The C++ embeds a BVH member and reads bvh.primIdx /
	/// bvh.verts through it; here that member is <see cref="Source"/>, a value copy of the source
	/// Bvh that shares - and does not own - its memory. Keep the source Bvh alive while this is used.
	/// </summary>
	public unsafe partial struct BvhGpu : IDisposable
	{
		/// <summary>Value copy of the source Bvh. Shares its memory; Dispose does not free it.</summary>
		public Bvh Source;

		// Node pool (owned).
		[NativeDisableUnsafePtrRestriction] public BvhGpuNode* Nodes;
		public uint UsedNodes;
		public uint AllocatedNodes;

		// Properties copied from the source by CopyBasePropertiesFrom.
		public uint TriCount;
		public uint IdxCount;
		public float3 AabbMin;
		public float3 AabbMax;
		[MarshalAs( UnmanagedType.U1 )] public bool Refittable;
		[MarshalAs( UnmanagedType.U1 )] public bool MayHaveHoles;
		[MarshalAs( UnmanagedType.U1 )] public bool BvhOverAabbs;
		[MarshalAs( UnmanagedType.U1 )] public bool BvhOverIndices;

		// SAH cost parameters, used by Intersect to accumulate the traversal cost.
		public float TraversalCost;
		public float IntersectionCost;

		public Allocator Allocator;

		/// <summary>Traversal stack depth of BVH_GPU::Intersect (C++: BVHNode* stack[64]).</summary>
		private const int IntersectStackSize = 64;

		public static BvhGpu Create( Allocator allocator )
		{
			return new BvhGpu
			{
				TraversalCost = BvhConstants.DefaultTraversalCost,
				IntersectionCost = BvhConstants.DefaultIntersectionCost,
				Allocator = allocator
			};
		}

		public bool IsCreated => Allocator > Allocator.None;

		public void Dispose()
		{
			Free( Nodes );
			Nodes = null;
			AllocatedNodes = 0;
			UsedNodes = 0;
			TriCount = 0;
			IdxCount = 0;
		}

		/// <summary>
		/// Port of BVH_GPU::ConvertFrom. The default is compact = false because that is what
		/// BVH_GPU::Build passes; 'compact' only selects whether the node pool is sized after the
		/// source's used or allocated node count, the emitted nodes are the same either way.
		/// </summary>
		public void ConvertFrom( ref Bvh original, bool compact = false )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "BvhGpu.ConvertFrom( .. ), bvhGpu was not created." );
			}
			if ( original.Nodes == null )
			{
				throw new ArgumentException( "BvhGpu.ConvertFrom( .. ), original.Nodes == null.", nameof( original ) );
			}
			if ( original.UsedNodes == 0 )
			{
				throw new ArgumentException( "BvhGpu.ConvertFrom( .. ), original.UsedNodes == 0.", nameof( original ) );
			}
			BvhGpuConverter.ConvertFrom( ref this, ref original, compact );
		}

		/// <summary>
		/// Port of BVH_GPU::SAHCost, which forwards to the underlying BVH.
		/// </summary>
		public float SahCost( uint nodeIdx = 0 )
		{
			return Source.SahCost( nodeIdx );
		}

		/// <summary>Port of BVH_GPU::Intersect. Returns the traversal cost; the hit goes to ray.Hit.</summary>
		public int Intersect( ref Ray ray )
		{
			BvhGpuNode* node = Nodes;
			BvhGpuNode** stack = stackalloc BvhGpuNode*[ IntersectStackSize ];
			uint* primIdx = Source.PrimIdx;
			uint stackPtr = 0;
			float cost = 0f;
			while ( true )
			{
				cost += TraversalCost;
				if ( node->IsLeaf )
				{
					for ( uint i = 0; i < node->TriCount; i++ )
					{
						uint pi = primIdx[ node->FirstTri + i ];
						Source.GetPrimIndices( pi, out uint i0, out uint i1, out uint i2 );
						Source.IntersectTri( ref ray, pi, i0, i1, i2 );
						cost += IntersectionCost;
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
					continue;
				}
				float dist1 = BvhConstants.Far, dist2 = BvhConstants.Far;
				float3 t1a = ( node->LMin - ray.O ) * ray.RD, t2a = ( node->LMax - ray.O ) * ray.RD;
				float3 t1b = ( node->RMin - ray.O ) * ray.RD, t2b = ( node->RMax - ray.O ) * ray.RD;
				float tmina = math.max( math.max( math.min( t1a.x, t2a.x ), math.min( t1a.y, t2a.y ) ), math.min( t1a.z, t2a.z ) );
				float tmaxa = math.min( math.min( math.max( t1a.x, t2a.x ), math.max( t1a.y, t2a.y ) ), math.max( t1a.z, t2a.z ) );
				float tminb = math.max( math.max( math.min( t1b.x, t2b.x ), math.min( t1b.y, t2b.y ) ), math.min( t1b.z, t2b.z ) );
				float tmaxb = math.min( math.min( math.max( t1b.x, t2b.x ), math.max( t1b.y, t2b.y ) ), math.max( t1b.z, t2b.z ) );
				if ( tmaxa >= tmina && tmina < ray.Hit.T && tmaxa >= 0f )
				{
					dist1 = tmina;
				}
				if ( tmaxb >= tminb && tminb < ray.Hit.T && tmaxb >= 0f )
				{
					dist2 = tminb;
				}
				uint lidx = node->Left, ridx = node->Right;
				if ( dist1 > dist2 )
				{
					float t = dist1;
					dist1 = dist2;
					dist2 = t;
					uint i = lidx;
					lidx = ridx;
					ridx = i;
				}
				if ( dist1 == BvhConstants.Far )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					node = stack[ --stackPtr ];
				}
				else
				{
					node = Nodes + lidx;
					if ( dist2 != BvhConstants.Far )
					{
						stack[ stackPtr++ ] = Nodes + ridx;
					}
				}
			}
			return ( int )cost; // cast to not break interface.
		}

		/// <summary>Port of BVH_GPU::IsOccluded, i.e. the FALLBACK_SHADOW_QUERY macro.</summary>
		public bool IsOccluded( in Ray ray )
		{
			Ray r = ray;
			float d = ray.Hit.T;
			Intersect( ref r );
			return r.Hit.T < d;
		}

		/// <summary>Ensures the node pool can hold count nodes. Contents are not preserved when it grows.</summary>
		internal void AllocateNodes( uint count )
		{
			if ( AllocatedNodes < count )
			{
				Free( Nodes );
				Nodes = ( BvhGpuNode* )Alloc( ( long )count * sizeof( BvhGpuNode ) );
				AllocatedNodes = count;
			}
		}

		internal void* Alloc( long bytes )
		{
			return UnsafeUtility.Malloc( bytes, 64, Allocator );
		}

		internal void Free( void* ptr )
		{
			if ( ptr != null )
			{
				UnsafeUtility.Free( ptr, Allocator );
			}
		}
	}

	/// <summary>
	/// Burst-compiled implementation of the BVH_GPU conversion. Direct calls must be synchronous,
	/// otherwise editor tests silently run the Mono fallback.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhGpuConverter
	{
		/// <summary>Port of BVH_GPU::ConvertFrom.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void ConvertFrom( ref BvhGpu gpu, ref Bvh original, [MarshalAs( UnmanagedType.U1 )] bool compact )
		{
			// get a copy of the original bvh
			gpu.Source = original;
			// allocate space
			uint spaceNeeded = compact ? original.UsedNodes : original.AllocatedNodes;
			gpu.AllocateNodes( spaceNeeded );
			BvhGpuNode* bvhNode = gpu.Nodes;
			UnsafeUtility.MemClear( bvhNode, ( long )spaceNeeded * sizeof( BvhGpuNode ) );
			CopyBasePropertiesFrom( ref gpu, ref original );
			// recursively convert nodes
			uint* stack = stackalloc uint[ 128 ];
			uint newNodePtr = 0, nodeIdx = 0, stackPtr = 0;
			while ( true )
			{
				BvhNode* orig = original.Nodes + nodeIdx;
				uint idx = newNodePtr++;
				if ( orig->IsLeaf )
				{
					bvhNode[ idx ].TriCount = orig->TriCount;
					bvhNode[ idx ].FirstTri = orig->LeftFirst;
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
					uint newNodeParent = stack[ --stackPtr ];
					bvhNode[ newNodeParent ].Right = newNodePtr;
				}
				else
				{
					BvhNode* left = original.Nodes + orig->LeftFirst;
					BvhNode* right = original.Nodes + orig->LeftFirst + 1;
					float leftArea = HalfArea( left->AabbMax - left->AabbMin );
					float rightArea = HalfArea( right->AabbMax - right->AabbMin );
					// put the larger node to the left to improve cache coherence during traversal
					if ( leftArea > rightArea )
					{
						bvhNode[ idx ].LMin = left->AabbMin;
						bvhNode[ idx ].RMin = right->AabbMin;
						bvhNode[ idx ].LMax = left->AabbMax;
						bvhNode[ idx ].RMax = right->AabbMax;
						bvhNode[ idx ].Left = newNodePtr; // right will be filled when popped
						stack[ stackPtr++ ] = idx;
						stack[ stackPtr++ ] = orig->LeftFirst + 1;
						nodeIdx = orig->LeftFirst;
					}
					else
					{
						bvhNode[ idx ].LMin = right->AabbMin;
						bvhNode[ idx ].RMin = left->AabbMin;
						bvhNode[ idx ].LMax = right->AabbMax;
						bvhNode[ idx ].RMax = left->AabbMax;
						bvhNode[ idx ].Left = newNodePtr; // right will be filled when popped
						stack[ stackPtr++ ] = idx;
						stack[ stackPtr++ ] = orig->LeftFirst;
						nodeIdx = orig->LeftFirst + 1;
					}
				}
			}
			gpu.UsedNodes = newNodePtr;
		}

		/// <summary>Port of tinybvh_halfarea, including the guard for empty (inverted) boxes.</summary>
		private static float HalfArea( float3 v )
		{
			return v.x < -BvhConstants.Far ? 0f : ( ( v.x * v.y ) + ( v.y * v.z ) + ( v.z * v.x ) );
		}

		/// <summary>Port of BVHBase::CopyBasePropertiesFrom for the Bvh -&gt; BvhGpu direction.</summary>
		private static void CopyBasePropertiesFrom( ref BvhGpu gpu, ref Bvh original )
		{
			gpu.Refittable = original.Refittable;
			gpu.MayHaveHoles = original.MayHaveHoles;
			gpu.BvhOverAabbs = original.BvhOverAabbs;
			gpu.BvhOverIndices = original.BvhOverIndices;
			gpu.TriCount = original.TriCount;
			gpu.IdxCount = original.IdxCount;
			gpu.AabbMin = original.AabbMin;
			gpu.AabbMax = original.AabbMax;
		}
	}
}
