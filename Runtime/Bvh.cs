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
	/// Port of tinybvh's BVH class: binary BVH in the Wald 32-byte node layout.
	/// This file holds the data and memory management; construction lives in Bvh.Build.cs
	/// and traversal in Bvh.Intersect.cs. The struct is unmanaged so it can be used from Burst.
	/// Input geometry is referenced, not owned: keep the vertex buffer alive while the BVH is in use.
	/// </summary>
	public unsafe partial struct Bvh : IDisposable
	{
		// Input primitives (not owned): 16-byte float4 vertices with a byte stride, 3 per triangle,
		// optionally addressed through VertIdx (3 indices per primitive).
		[NativeDisableUnsafePtrRestriction] public byte* Verts;
		public uint VertCount;
		public int VertStride;
		[NativeDisableUnsafePtrRestriction] public uint* VertIdx;

		// Node pool (owned). Root is always node 0; node 1 is unused for alignment.
		[NativeDisableUnsafePtrRestriction] public BvhNode* Nodes;
		public uint AllocatedNodes;
		public uint UsedNodes;

		// Primitive index array (owned).
		[NativeDisableUnsafePtrRestriction] public uint* PrimIdx;
		public uint AllocatedPrimIdx;

		// Input primitive bounds (owned).
		[NativeDisableUnsafePtrRestriction] public Fragment* Fragments;
		public uint AllocatedFragments;

		public uint TriCount;
		/// <summary>Number of primitive indices; equals TriCount without spatial splits.</summary>
		public uint IdxCount;

		/// <summary>Bounds of the root node.</summary>
		public float3 AabbMin;
		public float3 AabbMax;

		// Flags maintained by the builders.
		[MarshalAs( UnmanagedType.U1 )] public bool Refittable;
		[MarshalAs( UnmanagedType.U1 )] public bool MayHaveHoles;
		[MarshalAs( UnmanagedType.U1 )] public bool BvhOverAabbs;
		[MarshalAs( UnmanagedType.U1 )] public bool BvhOverIndices;

		/// <summary>
		/// Build settings, tinybvh's BVHSettings. When set, Build uses the SBVH builder
		/// (PrepareHqBuild + BuildHq) instead of the binned SAH reference builder. The resulting
		/// tree is not refittable and its PrimIdx may reference a primitive more than once.
		/// </summary>
		[MarshalAs( UnmanagedType.U1 )] public bool UseSpatialSplits;

		/// <summary>
		/// When set, Build subdivides the tree with the Unity job system, standing in for tinybvh's
		/// thread pool. Node pairs are then handed out by an atomic counter, so the node numbering of
		/// a threaded build is not reproducible; the tree shape is, because the splits themselves do
		/// not depend on threading. Builds of fewer than BvhConstants.MtBuildThreshold primitives stay
		/// serial and produce exactly the tree the serial builder produces, as in the C++ source.
		/// </summary>
		[MarshalAs( UnmanagedType.U1 )] public bool UseThreadedBuild;

		/// <summary>
		/// tinybvh's settings.usePresplitting. When set, Build splits large fragments before the
		/// subdivision starts, which trades index entries for tree quality. TriCount and IdxCount
		/// become the fragment count, which exceeds the input primitive count, and the tree is not
		/// refittable. Ignored by the SBVH builder.
		/// </summary>
		[MarshalAs( UnmanagedType.U1 )] public bool UsePresplitting;

		/// <summary>tinybvh's settings.presplitFactor: presplit budget relative to the input size.</summary>
		public float PresplitFactor;

		/// <summary>tinybvh's settings.presplitPostPass: un-split primitives that ended up in one leaf twice.</summary>
		[MarshalAs( UnmanagedType.U1 )] public bool PresplitPostPass;

		/// <summary>
		/// tinybvh's settings.useFullSweep: build with the full-sweep SAH builder instead of the
		/// binned one. For experiments; it is much slower but pairs well with presplitting.
		/// </summary>
		[MarshalAs( UnmanagedType.U1 )] public bool UseFullSweep;

		/// <summary>
		/// tinybvh's settings.useSIMDifavailable: when set, Build uses the AVX binned builder
		/// (Bvh.BuildAvx) instead of the scalar binned reference builder. The two produce different
		/// trees; see BvhAvxBuilder for what differs.
		/// Deviation: the C++ default is true, but this defaults to false, because every reference
		/// dump except the ".simd.ref" one is produced by the scalar builder and the port's Build
		/// has to keep matching those.
		/// </summary>
		[MarshalAs( UnmanagedType.U1 )] public bool UseSimdIfAvailable;

		/// <summary>tinybvh's settings.postOptimize: run Optimize( OptimizeIterations ) at the end of Build.</summary>
		[MarshalAs( UnmanagedType.U1 )] public bool PostOptimize;

		/// <summary>tinybvh's settings.optimizeIterations: iteration count used by PostOptimize.</summary>
		public uint OptimizeIterations;

		/// <summary>
		/// tinybvh's hqbvhbins: the number of bins the SBVH builder uses. Capped by the port at
		/// BvhHqBuilder.MaxBins, the equivalent of the C++ MAXHQBINS.
		/// </summary>
		public uint HqBvhBins;

		/// <summary>
		/// tinybvh's hqbvhoddeven: odd levels of the SBVH get one extra bin. Note that the C++ only
		/// changes 'depth' when it spawns a subtree, so without threaded builds the whole tree is
		/// built at depth 0 and this flag has no effect; see BvhHqBuilder.BuildHqTask.
		/// </summary>
		[MarshalAs( UnmanagedType.U1 )] public bool HqBvhOddEven;

		/// <summary>Number of subtrees the last build handed to the job system; zero when it ran serially.</summary>
		public uint ThreadedSubtrees;

		// SAH cost parameters.
		public float TraversalCost;
		public float IntersectionCost;

		// Custom geometry callbacks (not owned), tinybvh's customIntersect / customIsOccluded.
		// When set, the leaf loops of Intersect / IsOccluded hand every primitive index in the
		// leaf to these instead of intersecting a triangle. See Bvh.Custom.cs.
		public FunctionPointer<CustomIntersectDelegate> CustomIntersect;
		public FunctionPointer<CustomOccludedDelegate> CustomIsOccluded;

		// Opacity micro maps (not owned), tinybvh's opmap / opmapN. OpMapN^2 bits per primitive,
		// packed low bit first into ( ( OpMapN * OpMapN ) + 31 ) >> 5 uint words per primitive;
		// a zero bit cuts a hole in the corresponding micro-triangle. OpMap null disables the test.
		[NativeDisableUnsafePtrRestriction] public uint* OpMap;
		public uint OpMapN;

		// TLAS data (not owned): set when this BVH was built over BLAS instances.
		[NativeDisableUnsafePtrRestriction] public BlasInstance* Instances;
		public uint InstanceCount;
		[NativeDisableUnsafePtrRestriction] public Bvh* Blasses;
		public uint BlasCount;

		public Allocator Allocator;

		public static Bvh Create( Allocator allocator )
		{
			return new Bvh
			{
				VertStride = 16,
				Refittable = true,
				PresplitFactor = 0.3f,
				PresplitPostPass = true,
				OptimizeIterations = 25,
				HqBvhBins = BvhConstants.HqBins,
				TraversalCost = BvhConstants.DefaultTraversalCost,
				IntersectionCost = BvhConstants.DefaultIntersectionCost,
				Allocator = allocator
			};
		}

		public bool IsCreated => Allocator > Allocator.None;

		public bool IsTlas => Instances != null;

		/// <summary>Port of BVHBase::hasOpacityMicroMaps.</summary>
		public bool HasOpacityMicroMaps => OpMapN > 0;

		/// <summary>
		/// Port of BVHBase::SetOpacityMicroMaps. The map is referenced, not owned, and must hold
		/// ( ( n * n ) + 31 ) >> 5 words per primitive. Pass a null map and n = 0 to remove it.
		/// Note that, exactly as in the C++, converting this BVH to another layout does not carry
		/// the map over: BVHBase::CopyBasePropertiesFrom does not copy opmap or opmapN, so the
		/// setter has to be called on the converted layout as well.
		/// </summary>
		public void SetOpacityMicroMaps( uint* mapData, uint n )
		{
			OpMap = mapData;
			OpMapN = n;
		}

		/// <summary>SetOpacityMicroMaps over a NativeArray; the array must outlive the BVH.</summary>
		public void SetOpacityMicroMaps( NativeArray<uint> mapData, uint n )
		{
			SetOpacityMicroMaps( ( uint* )mapData.GetUnsafePtr(), n );
		}

		/// <summary>Removes a previously set opacity micro map; equivalent to SetOpacityMicroMaps( null, 0 ).</summary>
		public void ClearOpacityMicroMaps()
		{
			SetOpacityMicroMaps( null, 0u );
		}

		public void Dispose()
		{
			Free( Nodes );
			Free( PrimIdx );
			Free( Fragments );
			Nodes = null;
			PrimIdx = null;
			Fragments = null;
			AllocatedNodes = 0;
			AllocatedPrimIdx = 0;
			AllocatedFragments = 0;
			UsedNodes = 0;
			TriCount = 0;
			IdxCount = 0;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public float4 Vertex( uint i )
		{
			return *( float4* )( Verts + ( i * VertStride ) );
		}

		/// <summary>Port of GET_PRIM_INDICES_I0_I1_I2: vertex indices of a primitive, indexed or not.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public void GetPrimIndices( uint prim, out uint i0, out uint i1, out uint i2 )
		{
			if ( VertIdx != null )
			{
				i0 = VertIdx[ prim * 3 ];
				i1 = VertIdx[ ( prim * 3 ) + 1 ];
				i2 = VertIdx[ ( prim * 3 ) + 2 ];
			}
			else
			{
				i0 = prim * 3;
				i1 = ( prim * 3 ) + 1;
				i2 = ( prim * 3 ) + 2;
			}
		}

		/// <summary>Ensures the node pool can hold count nodes. Contents are not preserved when it grows.</summary>
		internal void AllocateNodes( uint count )
		{
			if ( AllocatedNodes < count )
			{
				Free( Nodes );
				Nodes = ( BvhNode* )Alloc( ( long )count * sizeof( BvhNode ) );
				AllocatedNodes = count;
			}
		}

		/// <summary>Ensures the primitive index array can hold count entries. Contents are not preserved when it grows.</summary>
		internal void AllocatePrimIdx( uint count )
		{
			if ( AllocatedPrimIdx < count )
			{
				Free( PrimIdx );
				PrimIdx = ( uint* )Alloc( ( long )count * sizeof( uint ) );
				AllocatedPrimIdx = count;
			}
		}

		/// <summary>Ensures the fragment array can hold count entries. Contents are not preserved when it grows.</summary>
		internal void AllocateFragments( uint count )
		{
			if ( AllocatedFragments < count )
			{
				Free( Fragments );
				Fragments = ( Fragment* )Alloc( ( long )count * sizeof( Fragment ) );
				AllocatedFragments = count;
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
}
