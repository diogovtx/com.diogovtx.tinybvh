using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Port of tinybvh's VoxelSet class (ENABLE_VOXEL_SUPPORT): a 256^3 voxel object with one
	/// 32-bit value per voxel, stored in three levels - a bit per 4x4x4 group of bricks in the top
	/// grid, a brick index per 8x8x8 brick in the grid, and the voxel values in a pool of bricks.
	/// Brick 0 is never handed out, so a zero grid cell means "no brick here". Traversal is a
	/// three-level DDA and lives in VoxelSet.Intersect.cs.
	///
	/// The C++ class derives from BVHBase just so it can be attached to a TLAS; here it is a plain
	/// unmanaged struct that carries the two fields a TLAS needs. A voxel object is always the unit
	/// cube in object space, so the bounds are (0,0,0)-(1,1,1) and never change.
	/// </summary>
	public unsafe partial struct VoxelSet : IDisposable
	{
		/// <summary>Voxels per object edge (C++: objectDim). 64, 128 or 256.</summary>
		public const int ObjectDim = 256;
		/// <summary>Voxel count of the whole object (C++: objectSize).</summary>
		internal const int ObjectSize = ObjectDim * ObjectDim * ObjectDim;
		// grid level: collection of bricks
		internal const int BrickDim = 8;
		internal const int BrickSize = BrickDim * BrickDim * BrickDim;
		internal const int GridDim = ObjectDim / BrickDim;
		internal const int GridSize = GridDim * GridDim * GridDim;
		// topgrid level: 1 bit for each group of bricks
		internal const int GroupDim = 4;
		internal const int GroupSize = GroupDim * GroupDim * GroupDim;
		internal const int TopGridDim = GridDim / GroupDim;
		internal const int TopGridSize = TopGridDim * TopGridDim * TopGridDim;
		// masks
		internal const int TopResMask = GridDim - GroupDim;
		internal const int SuperMask = TopResMask + ( TopResMask * GridDim ) + ( TopResMask * GridDim * GridDim );

		/// <summary>Brick index per grid cell, GridSize entries; zero means "empty". Owned.</summary>
		[NativeDisableUnsafePtrRestriction] public uint* Grid;
		/// <summary>Brick pool, BrickCount * BrickSize voxel values. Owned.</summary>
		[NativeDisableUnsafePtrRestriction] public uint* Brick;
		/// <summary>Capacity of the brick pool in bricks; grows by a quarter when it runs out.</summary>
		public uint BrickCount;
		/// <summary>First free brick; brick 0 is skipped, as 0 denotes an empty brick in the topgrid.</summary>
		public uint FreeBrickPtr;
		/// <summary>One bit per group of GroupDim^3 bricks, TopGridSize / 32 words. Owned.</summary>
		[NativeDisableUnsafePtrRestriction] public uint* TopGrid;

		/// <summary>Bounds of the object; a voxel object is always (1,1,1) in object space.</summary>
		public float3 AabbMin;
		public float3 AabbMax;

		public Allocator Allocator;

		/// <summary>
		/// Port of the VoxelSet constructor. Grid and brick pool start out empty; the top grid is
		/// left uninitialised, exactly as in the C++, because UpdateTopGrid clears it before it
		/// fills it in. Traversing a set that never had UpdateTopGrid called on it reads that
		/// uninitialised memory, in this port as in the original.
		/// </summary>
		public static VoxelSet Create( Allocator allocator )
		{
			VoxelSet set = new VoxelSet
			{
				// will grow as needed; scales roughly quadratically with objectDim
				BrickCount = ( ObjectDim * ObjectDim ) / 16,
				FreeBrickPtr = 1, // first available brick; we'll skip 0
				AabbMin = new float3( 0f ),
				AabbMax = new float3( 1f ), // a voxel object is always (1,1,1) in object space.
				Allocator = allocator
			};
			set.Grid = ( uint* )set.Alloc( GridSize * sizeof( uint ) );
			UnsafeUtility.MemClear( set.Grid, GridSize * sizeof( uint ) );
			set.Brick = ( uint* )set.Alloc( ( long )BrickSize * set.BrickCount * sizeof( uint ) );
			UnsafeUtility.MemClear( set.Brick, ( long )BrickSize * set.BrickCount * sizeof( uint ) );
			set.TopGrid = ( uint* )set.Alloc( TopGridSize / 8 );
			return set;
		}

		public bool IsCreated => Allocator > Allocator.None;

		public void Dispose()
		{
			Free( Grid );
			Free( Brick );
			Free( TopGrid );
			Grid = null;
			Brick = null;
			TopGrid = null;
			BrickCount = 0;
			FreeBrickPtr = 1;
		}

		/// <summary>
		/// Port of VoxelSet::Set: stores a value in a voxel, allocating a brick for the containing
		/// grid cell if there is none yet. Coordinates are not range-checked, as in the C++: a
		/// coordinate of 256 or more wraps into a neighbouring brick or past the grid.
		/// </summary>
		public void Set( uint x, uint y, uint z, uint v )
		{
			// note: not thread-safe.
			uint bx = x / ( uint )BrickDim, by = y / ( uint )BrickDim, bz = z / ( uint )BrickDim;
			uint gridIdx = bx + ( by * ( uint )GridDim ) + ( bz * ( uint )( GridDim * GridDim ) );
			uint brickIdx = Grid[ gridIdx ];
			if ( brickIdx == 0 )
			{
				if ( FreeBrickPtr == BrickCount ) // we ran out; reallocate
				{
					uint newBrickCount = BrickCount + ( BrickCount >> 2 );
					uint* newBrickPool = ( uint* )Alloc( ( long )newBrickCount * BrickSize * sizeof( uint ) );
					UnsafeUtility.MemCpy( newBrickPool, Brick, ( long )BrickCount * BrickSize * sizeof( uint ) );
					UnsafeUtility.MemClear( newBrickPool + ( ( long )BrickCount * BrickSize ), ( long )( newBrickCount - BrickCount ) * BrickSize * sizeof( uint ) );
					Free( Brick );
					Brick = newBrickPool;
					BrickCount = newBrickCount;
				}
				brickIdx = Grid[ gridIdx ] = FreeBrickPtr++;
			}
			uint voxelIdx = ( x & ( uint )( BrickDim - 1 ) ) + ( ( y & ( uint )( BrickDim - 1 ) ) * ( uint )BrickDim ) +
				( ( z & ( uint )( BrickDim - 1 ) ) * ( uint )( BrickDim * BrickDim ) );
			Brick[ ( brickIdx * ( uint )BrickSize ) + voxelIdx ] = v;
		}

		/// <summary>
		/// Port of VoxelSet::UpdateTopGrid: rebuilds the top level, one bit per group of
		/// GroupDim^3 grid cells. Must be called before traversal, and again after any Set.
		/// </summary>
		public void UpdateTopGrid()
		{
			UnsafeUtility.MemClear( TopGrid, TopGridSize / 8 );
			for ( int x = 0; x < TopGridDim; x++ )
			{
				for ( int y = 0; y < TopGridDim; y++ )
				{
					for ( int z = 0; z < TopGridDim; z++ )
					{
						uint* gridBase = Grid + ( x * GroupDim ) + ( y * GroupDim * GridDim ) + ( z * GroupDim * GridDim * GridDim );
						bool hasContent = false;
						for ( int u = 0; u < GroupDim; u++ )
						{
							for ( int v = 0; v < GroupDim; v++ )
							{
								for ( int w = 0; w < GroupDim; w++ )
								{
									if ( gridBase[ u + ( v * GridDim ) + ( w * GridDim * GridDim ) ] != 0 )
									{
										hasContent = true;
										goto break3;
									}
								}
							}
						}
					break3:
						if ( !hasContent )
						{
							continue;
						}
						int topIdx = x + ( y * TopGridDim ) + ( z * TopGridDim * TopGridDim );
						TopGrid[ topIdx >> 5 ] |= 1u << ( topIdx & 31 );
					}
				}
			}
		}

		/// <summary>
		/// Port of VoxelSet::GetNormal: the normal of the voxel face the ray entered through,
		/// derived from how close the hit point is to each of the three pairs of voxel planes.
		/// Only meaningful right after a hit, and computed under Burst because it depends on the
		/// exact float value of ray.Hit.T.
		/// </summary>
		public float3 GetNormal( in Ray ray )
		{
			VoxelSetTraversal.GetNormal( in ray, out float3 normal );
			return normal;
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
