using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Traversal half of tinybvh's VoxelSet: the three-level Amanatides &amp; Woo DDA over the top
	/// grid, the brick grid and the voxels of a brick.
	///
	/// The work runs in a Burst direct call rather than in the struct methods themselves: the DDA
	/// setup and the plane distances have to match a compiled C++ build bit for bit, and Mono
	/// evaluates float expressions in double.
	/// </summary>
	public unsafe partial struct VoxelSet
	{
		/// <summary>
		/// Port of VoxelSet::Intersect( Ray&amp; ). The hit, if any, is written to ray.Hit.
		/// Returns the number of voxels the DDA visited before it found one. Note that the C++
		/// returns 0 both when the ray misses the object bounds and when it leaves the grid without
		/// hitting anything, so the step count is only meaningful for a hit.
		/// </summary>
		public int Intersect( ref Ray ray )
		{
			if ( TopGrid == null )
			{
				return 0;
			}
			VoxelSetTraversal.Intersect( ref this, ref ray, out int steps );
			return steps;
		}

		/// <summary>
		/// Port of VoxelSet::IsOccluded: the same DDA, stopped at ray.Hit.T and without a hit
		/// record. The ray itself is left untouched.
		/// </summary>
		public bool IsOccluded( in Ray ray )
		{
			if ( TopGrid == null )
			{
				return false;
			}
			VoxelSetTraversal.IsOccluded( ref this, in ray, out int occluded );
			return occluded != 0;
		}
	}

	/// <summary>
	/// Burst-compiled implementation of the voxel DDA. Direct calls must be synchronous, otherwise
	/// editor tests silently run the Mono fallback and drift in the last bits.
	/// </summary>
	[BurstCompile]
	internal static unsafe class VoxelSetTraversal
	{
		// Aliases for the VoxelSet constants, so the ported code reads like the C++ member
		// functions it comes from.
		private const int ObjectDim = VoxelSet.ObjectDim;
		private const int BrickDim = VoxelSet.BrickDim;
		private const int BrickSize = VoxelSet.BrickSize;
		private const int GridDim = VoxelSet.GridDim;
		private const int GroupDim = VoxelSet.GroupDim;
		private const int TopGridDim = VoxelSet.TopGridDim;
		private const int SuperMask = VoxelSet.SuperMask;

		/// <summary>
		/// Port of VoxelSet::DDAState. Deviation: the C++ pads the struct with two dummy floats so
		/// that two states fill a cache line; the padding is dropped here. Its X, Y and Z are
		/// uint32 there and int32 here, so the "stepped out of the grid" tests below compare as
		/// unsigned to keep the wrap-around the C++ relies on when a step goes below zero.
		/// </summary>
		private struct DDAState
		{
			public int X, Y, Z;
			public float3 Tmax;
		}

		/// <summary>Port of tinybvh_min for floats: a plain ternary, without the NaN handling math.min adds.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float Min( float a, float b )
		{
			return a < b ? a : b;
		}

		/// <summary>Port of tinybvh_max for floats; see <see cref="Min"/>.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static float Max( float a, float b )
		{
			return a > b ? a : b;
		}

		/// <summary>Port of tinybvh_clamp for int32.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static int Clamp( int x, int a, int b )
		{
			return x > a ? ( x < b ? x : b ) : a; /* NaN safe */
		}

		/// <summary>
		/// Port of VoxelSet::Setup3DDDA. Returns false when the ray misses the unit cube; otherwise
		/// t is the distance at which traversal starts, state holds the first top-grid cell and its
		/// exit distances, and tdelta the per-cell step of the top level.
		///
		/// Upstream bug, reproduced as is. The 0.0000025f below - and the two copies of it in
		/// Intersect and IsOccluded, which nudge the entry point of the grid and brick levels the
		/// same way - is a ray-parameter epsilon, so the distance it moves the entry point is
		/// |ray.D| * 0.0000025f. It only works while that is comfortably larger than the rounding
		/// error of ray.O + ray.D * t, which is about one ulp of the larger of the two. Every level
		/// is entered exactly on a cell plane, so when the nudge loses, ceil() returns the next
		/// plane instead of the current one, the level's first tmax comes out equal to t, and the
		/// DDA steps a cell without advancing t: from there on it is one cell ahead of the ray on
		/// that axis, reads bricks and voxels the ray never enters, and reports their value at a t
		/// that belongs to a different cell - a hit floating in empty space.
		///
		/// A BLAS is reached through BVH::IntersectTLAS, which does not normalise the transformed
		/// direction (see tiny_bvh.h, 'tmpRay.D = tinybvh_transform_vector( ray.D, inst.invTransform )'),
		/// so an instance scaled up by s hands the DDA a direction of length 1/s and shrinks the
		/// nudge by s while leaving the rounding error where it was. At s = 10 it is already at the
		/// noise floor. Callers that place a voxel object with a scale therefore have to give the
		/// world ray a direction of length s, so the object-space ray is unit length; the
		/// CpuRaytracer sample's voxel backend does exactly that.
		///
		/// Second upstream case, from the same nudge. When a ray sits exactly on a cell plane of one
		/// of the three levels on some axis and the nudge cannot move it off, that level's tmax for
		/// the axis comes out as (plane - ray.O) * ray.RD = 0, which is below the current t: the DDA
		/// picks that axis as the nearest crossing, steps it, and sets t back to 0, after which every
		/// distance it reports is meaningless and tmax + tdelta overflows to infinity. A direction
		/// component of exactly zero always hits this, because tinybvh_safercp hands back a huge
		/// reciprocal and the coordinate never leaves the plane - so a ray with a zero component and
		/// an origin coordinate at an exact multiple of 1/8, 1/32 or 1/256 on that axis is lost. A
		/// very small component does the same while the walk is still at t = 0, i.e. for an origin
		/// inside the cube. Neither case has a workaround on the caller's side other than not
		/// building such rays; camera rays and the sample's sun direction never have exact zeros, so
		/// the sample does not run into it. VoxelSampleTests classifies and counts these rays.
		/// </summary>
		private static bool Setup3DDDA( in Ray ray, float3 Dsign, ref DDAState state, int3 step, ref float3 tdelta, ref float t )
		{
			// if ray is not inside the object aabb: advance until it is
			if ( !( ray.O.x >= 0f && ray.O.x <= 1f && ray.O.y >= 0f && ray.O.y <= 1f && ray.O.z >= 0f && ray.O.z <= 1f ) )
			{
				float tx1 = -ray.O.x * ray.RD.x, tx2 = ( 1f - ray.O.x ) * ray.RD.x;
				float tmin = Min( tx1, tx2 ), tmax = Max( tx1, tx2 );
				float ty1 = -ray.O.y * ray.RD.y, ty2 = ( 1f - ray.O.y ) * ray.RD.y;
				tmin = Max( tmin, Min( ty1, ty2 ) );
				tmax = Min( tmax, Max( ty1, ty2 ) );
				float tz1 = -ray.O.z * ray.RD.z, tz2 = ( 1f - ray.O.z ) * ray.RD.z;
				tmin = Max( tmin, Min( tz1, tz2 ) );
				tmax = Min( tmax, Max( tz1, tz2 ) );
				if ( tmax < tmin || tmin > ray.Hit.T || tmax < 0f )
				{
					return false;
				}
				else
				{
					t = tmin;
				}
			}
			// setup amanatides & woo - assume object size is 1x1x1, from (0,0,0) to (1,1,1)
			const float cellSize = 1f / TopGridDim;
			float3 posInGrid = ( ray.O + ( ray.D * ( t + 0.0000025f ) ) ) * ( float )TopGridDim;
			float3 gridPlanes = ( math.ceil( posInGrid ) - Dsign ) * cellSize;
			int3 P = new int3(
				Clamp( ( int )posInGrid.x, 0, TopGridDim - 1 ),
				Clamp( ( int )posInGrid.y, 0, TopGridDim - 1 ),
				Clamp( ( int )posInGrid.z, 0, TopGridDim - 1 )
			);
			state.X = P.x;
			state.Y = P.y;
			state.Z = P.z;
			state.Tmax = ( gridPlanes - ray.O ) * ray.RD;
			tdelta = new float3( ( float )step.x, ( float )step.y, ( float )step.z ) * cellSize * ray.RD;
			// proceed with traversal
			return true;
		}

		/// <summary>Port of VoxelSet::GetNormal. Writes the face normal of the voxel the ray hit.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void GetNormal( in Ray ray, out float3 normal )
		{
			// our object is (1,1,1) in object space, so this scales each voxel to (1,1,1)
			float3 I1 = ( ray.O + ( ray.Hit.T * ray.D ) ) * ( float )ObjectDim;
			float3 fG = new float3( I1.x - math.floor( I1.x ), I1.y - math.floor( I1.y ), I1.z - math.floor( I1.z ) );
			float3 d = new float3( Min( fG.x, 1f - fG.x ), Min( fG.y, 1f - fG.y ), Min( fG.z, 1f - fG.z ) );
			float mind = Min( Min( d.x, d.y ), d.z );
			if ( mind == d.x )
			{
				normal = new float3( ray.D.x > 0f ? -1f : 1f, 0f, 0f );
			}
			else if ( mind == d.y )
			{
				normal = new float3( 0f, ray.D.y > 0f ? -1f : 1f, 0f );
			}
			else
			{
				normal = new float3( 0f, 0f, ray.D.z > 0f ? -1f : 1f );
			}
		}

		/// <summary>Port of VoxelSet::Intersect( Ray&amp; ); steps receives its return value.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void Intersect( ref VoxelSet vox, ref Ray ray, out int steps )
		{
			// setup Amanatides & Woo grid traversal
			DDAState l1_ = default, l2_ = default;
			uint xsign = math.asuint( ray.D.x ) >> 31;
			uint ysign = math.asuint( ray.D.y ) >> 31;
			uint zsign = math.asuint( ray.D.z ) >> 31;
			float3 Dsign = new float3( ( float )xsign, ( float )ysign, ( float )zsign );
			int3 step = new int3( 1 - ( ( int )xsign * 2 ), 1 - ( ( int )ysign * 2 ), 1 - ( ( int )zsign * 2 ) );
			float3 tdelta = default;
			float t = 0f;
			steps = 0;
			if ( !Setup3DDDA( in ray, Dsign, ref l1_, step, ref tdelta, ref t ) )
			{
				return;
			}
			float3 l2tdelta = tdelta * ( 1f / GroupDim );
			float3 l3tdelta = l2tdelta * ( 1f / BrickDim );
			// start stepping:
			while ( true )
			{
				int tidx = l1_.X + ( l1_.Y * TopGridDim ) + ( l1_.Z * TopGridDim * TopGridDim );
				uint cell = vox.TopGrid[ tidx >> 5 ] & ( 1u << ( tidx & 31 ) );
				if ( cell != 0 )
				{
					// setup midlevel traversal
					float3 posInGrid = ( ray.O + ( ( t + 0.0000025f ) * ray.D ) ) * ( float )GridDim;
					float3 gridPlanes = ( math.ceil( posInGrid ) - Dsign ) * ( 1f / GridDim );
					l2_.X = Clamp( ( int )posInGrid.x, l1_.X * GroupDim, ( l1_.X * GroupDim ) + ( GroupDim - 1 ) );
					l2_.Y = Clamp( ( int )posInGrid.y, l1_.Y * GroupDim, ( l1_.Y * GroupDim ) + ( GroupDim - 1 ) );
					l2_.Z = Clamp( ( int )posInGrid.z, l1_.Z * GroupDim, ( l1_.Z * GroupDim ) + ( GroupDim - 1 ) );
					l2_.Tmax = ( gridPlanes - ray.O ) * ray.RD;
					uint* gridBase = vox.Grid + ( ( l2_.X + ( l2_.Y * GridDim ) + ( l2_.Z * GridDim * GridDim ) ) & SuperMask );
					l2_.X &= GroupDim - 1;
					l2_.Y &= GroupDim - 1;
					l2_.Z &= GroupDim - 1;
					// step through midlevel cells
					while ( true )
					{
						uint brickCell = gridBase[ l2_.X + ( l2_.Y * GridDim ) + ( l2_.Z * GridDim * GridDim ) ];
						if ( brickCell != 0 )
						{
							// setup 3DDDA for brick traversal
							uint* brickData = vox.Brick + ( brickCell * ( uint )BrickSize );
							float3 posInBrick = ( ray.O + ( ( t + 0.0000025f ) * ray.D ) ) * ( float )ObjectDim;
							int X = Clamp( ( int )posInBrick.x - ( ( l2_.X + ( l1_.X * GroupDim ) ) * BrickDim ), 0, BrickDim - 1 );
							int Y = Clamp( ( int )posInBrick.y - ( ( l2_.Y + ( l1_.Y * GroupDim ) ) * BrickDim ), 0, BrickDim - 1 );
							int Z = Clamp( ( int )posInBrick.z - ( ( l2_.Z + ( l1_.Z * GroupDim ) ) * BrickDim ), 0, BrickDim - 1 );
							float3 brickPlanes = ( math.ceil( posInBrick ) - Dsign ) * ( 1f / ObjectDim );
							float3 tmax = ( brickPlanes - ray.O ) * ray.RD;
							// step through brick
							while ( true )
							{
								steps++;
								uint v = brickData[ X + ( Y * BrickDim ) + ( Z * BrickDim * BrickDim ) ];
								if ( v != 0 )
								{
									ray.Hit.T = t;
									// INST_IDX_BITS == 32: the instance index lives in its own field.
									ray.Hit.Prim = v;
									ray.Hit.Inst = ray.InstIdx; // store in dedicated field
									return;
								}
								if ( tmax.x < tmax.y )
								{
									if ( tmax.x < tmax.z )
									{
										X += step.x;
										if ( ( uint )X >= ( uint )BrickDim )
										{
											break;
										}
										t = tmax.x;
										tmax.x += l3tdelta.x;
									}
									else
									{
										Z += step.z;
										if ( ( uint )Z >= ( uint )BrickDim )
										{
											break;
										}
										t = tmax.z;
										tmax.z += l3tdelta.z;
									}
								}
								else
								{
									if ( tmax.y < tmax.z )
									{
										Y += step.y;
										if ( ( uint )Y >= ( uint )BrickDim )
										{
											break;
										}
										t = tmax.y;
										tmax.y += l3tdelta.y;
									}
									else
									{
										Z += step.z;
										if ( ( uint )Z >= ( uint )BrickDim )
										{
											break;
										}
										t = tmax.z;
										tmax.z += l3tdelta.z;
									}
								}
							}
						}
						if ( l2_.Tmax.x < l2_.Tmax.y )
						{
							if ( l2_.Tmax.x < l2_.Tmax.z )
							{
								l2_.X += step.x;
								if ( ( uint )l2_.X >= ( uint )GroupDim )
								{
									break;
								}
								t = l2_.Tmax.x;
								l2_.Tmax.x += l2tdelta.x;
							}
							else
							{
								l2_.Z += step.z;
								if ( ( uint )l2_.Z >= ( uint )GroupDim )
								{
									break;
								}
								t = l2_.Tmax.z;
								l2_.Tmax.z += l2tdelta.z;
							}
						}
						else
						{
							if ( l2_.Tmax.y < l2_.Tmax.z )
							{
								l2_.Y += step.y;
								if ( ( uint )l2_.Y >= ( uint )GroupDim )
								{
									break;
								}
								t = l2_.Tmax.y;
								l2_.Tmax.y += l2tdelta.y;
							}
							else
							{
								l2_.Z += step.z;
								if ( ( uint )l2_.Z >= ( uint )GroupDim )
								{
									break;
								}
								t = l2_.Tmax.z;
								l2_.Tmax.z += l2tdelta.z;
							}
						}
					}
				}
				if ( l1_.Tmax.x < l1_.Tmax.y )
				{
					if ( l1_.Tmax.x < l1_.Tmax.z )
					{
						l1_.X += step.x;
						if ( ( uint )l1_.X >= ( uint )TopGridDim )
						{
							break;
						}
						t = l1_.Tmax.x;
						l1_.Tmax.x += tdelta.x;
					}
					else
					{
						l1_.Z += step.z;
						if ( ( uint )l1_.Z >= ( uint )TopGridDim )
						{
							break;
						}
						t = l1_.Tmax.z;
						l1_.Tmax.z += tdelta.z;
					}
				}
				else
				{
					if ( l1_.Tmax.y < l1_.Tmax.z )
					{
						l1_.Y += step.y;
						if ( ( uint )l1_.Y >= ( uint )TopGridDim )
						{
							break;
						}
						t = l1_.Tmax.y;
						l1_.Tmax.y += tdelta.y;
					}
					else
					{
						l1_.Z += step.z;
						if ( ( uint )l1_.Z >= ( uint )TopGridDim )
						{
							break;
						}
						t = l1_.Tmax.z;
						l1_.Tmax.z += tdelta.z;
					}
				}
			}
			// The C++ returns 0 here: a ray that leaves the grid reports no steps, however many
			// voxels it visited on the way out.
			steps = 0;
		}

		/// <summary>Port of VoxelSet::IsOccluded; occluded receives 1 for a hit within ray.Hit.T.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void IsOccluded( ref VoxelSet vox, in Ray ray, out int occluded )
		{
			// setup Amanatides & Woo grid traversal
			DDAState l1_ = default, l2_ = default;
			uint xsign = math.asuint( ray.D.x ) >> 31;
			uint ysign = math.asuint( ray.D.y ) >> 31;
			uint zsign = math.asuint( ray.D.z ) >> 31;
			float3 Dsign = new float3( ( float )xsign, ( float )ysign, ( float )zsign );
			int3 step = new int3( 1 - ( ( int )xsign * 2 ), 1 - ( ( int )ysign * 2 ), 1 - ( ( int )zsign * 2 ) );
			float3 tdelta = default;
			float t = 0f;
			occluded = 0;
			if ( !Setup3DDDA( in ray, Dsign, ref l1_, step, ref tdelta, ref t ) )
			{
				return;
			}
			float3 l2tdelta = tdelta * ( 1f / GroupDim );
			float3 l3tdelta = l2tdelta * ( 1f / BrickDim );
			// start stepping:
			while ( t < ray.Hit.T )
			{
				int tidx = l1_.X + ( l1_.Y * TopGridDim ) + ( l1_.Z * TopGridDim * TopGridDim );
				uint cell = vox.TopGrid[ tidx >> 5 ] & ( 1u << ( tidx & 31 ) );
				if ( cell != 0 )
				{
					// setup midlevel traversal
					float3 posInGrid = ( ray.O + ( ( t + 0.0000025f ) * ray.D ) ) * ( float )GridDim;
					float3 gridPlanes = ( math.ceil( posInGrid ) - Dsign ) * ( 1f / GridDim );
					l2_.X = Clamp( ( int )posInGrid.x, l1_.X * GroupDim, ( l1_.X * GroupDim ) + ( GroupDim - 1 ) );
					l2_.Y = Clamp( ( int )posInGrid.y, l1_.Y * GroupDim, ( l1_.Y * GroupDim ) + ( GroupDim - 1 ) );
					l2_.Z = Clamp( ( int )posInGrid.z, l1_.Z * GroupDim, ( l1_.Z * GroupDim ) + ( GroupDim - 1 ) );
					l2_.Tmax = ( gridPlanes - ray.O ) * ray.RD;
					uint* gridBase = vox.Grid + ( ( l2_.X + ( l2_.Y * GridDim ) + ( l2_.Z * GridDim * GridDim ) ) & SuperMask );
					l2_.X &= GroupDim - 1;
					l2_.Y &= GroupDim - 1;
					l2_.Z &= GroupDim - 1;
					// step through midlevel cells
					while ( true )
					{
						uint brickCell = gridBase[ l2_.X + ( l2_.Y * GridDim ) + ( l2_.Z * GridDim * GridDim ) ];
						if ( brickCell != 0 )
						{
							// setup 3DDDA for brick traversal
							uint* brickData = vox.Brick + ( brickCell * ( uint )BrickSize );
							float3 posInBrick = ( ray.O + ( ( t + 0.0000025f ) * ray.D ) ) * ( float )ObjectDim;
							// The C++ passes 0u as the clamp minimum here and 0 in Intersect; both
							// pick the int32 overload of tinybvh_clamp, so the result is the same.
							int X = Clamp( ( int )posInBrick.x - ( ( l2_.X + ( l1_.X * GroupDim ) ) * BrickDim ), 0, BrickDim - 1 );
							int Y = Clamp( ( int )posInBrick.y - ( ( l2_.Y + ( l1_.Y * GroupDim ) ) * BrickDim ), 0, BrickDim - 1 );
							int Z = Clamp( ( int )posInBrick.z - ( ( l2_.Z + ( l1_.Z * GroupDim ) ) * BrickDim ), 0, BrickDim - 1 );
							float3 brickPlanes = ( math.ceil( posInBrick ) - Dsign ) * ( 1f / ObjectDim );
							float3 tmax = ( brickPlanes - ray.O ) * ray.RD;
							// step through brick
							while ( true )
							{
								uint v = brickData[ X + ( Y * BrickDim ) + ( Z * BrickDim * BrickDim ) ];
								if ( v != 0 )
								{
									occluded = t < ray.Hit.T ? 1 : 0;
									return;
								}
								if ( tmax.x < tmax.y )
								{
									if ( tmax.x < tmax.z )
									{
										X += step.x;
										if ( ( uint )X >= ( uint )BrickDim )
										{
											break;
										}
										t = tmax.x;
										tmax.x += l3tdelta.x;
									}
									else
									{
										Z += step.z;
										if ( ( uint )Z >= ( uint )BrickDim )
										{
											break;
										}
										t = tmax.z;
										tmax.z += l3tdelta.z;
									}
								}
								else
								{
									if ( tmax.y < tmax.z )
									{
										Y += step.y;
										if ( ( uint )Y >= ( uint )BrickDim )
										{
											break;
										}
										t = tmax.y;
										tmax.y += l3tdelta.y;
									}
									else
									{
										Z += step.z;
										if ( ( uint )Z >= ( uint )BrickDim )
										{
											break;
										}
										t = tmax.z;
										tmax.z += l3tdelta.z;
									}
								}
							}
						}
						if ( l2_.Tmax.x < l2_.Tmax.y )
						{
							if ( l2_.Tmax.x < l2_.Tmax.z )
							{
								l2_.X += step.x;
								if ( ( uint )l2_.X >= ( uint )GroupDim )
								{
									break;
								}
								t = l2_.Tmax.x;
								l2_.Tmax.x += l2tdelta.x;
							}
							else
							{
								l2_.Z += step.z;
								if ( ( uint )l2_.Z >= ( uint )GroupDim )
								{
									break;
								}
								t = l2_.Tmax.z;
								l2_.Tmax.z += l2tdelta.z;
							}
						}
						else
						{
							if ( l2_.Tmax.y < l2_.Tmax.z )
							{
								l2_.Y += step.y;
								if ( ( uint )l2_.Y >= ( uint )GroupDim )
								{
									break;
								}
								t = l2_.Tmax.y;
								l2_.Tmax.y += l2tdelta.y;
							}
							else
							{
								l2_.Z += step.z;
								if ( ( uint )l2_.Z >= ( uint )GroupDim )
								{
									break;
								}
								t = l2_.Tmax.z;
								l2_.Tmax.z += l2tdelta.z;
							}
						}
					}
				}
				if ( l1_.Tmax.x < l1_.Tmax.y )
				{
					if ( l1_.Tmax.x < l1_.Tmax.z )
					{
						l1_.X += step.x;
						if ( ( uint )l1_.X >= ( uint )TopGridDim )
						{
							break;
						}
						t = l1_.Tmax.x;
						l1_.Tmax.x += tdelta.x;
					}
					else
					{
						l1_.Z += step.z;
						if ( ( uint )l1_.Z >= ( uint )TopGridDim )
						{
							break;
						}
						t = l1_.Tmax.z;
						l1_.Tmax.z += tdelta.z;
					}
				}
				else
				{
					if ( l1_.Tmax.y < l1_.Tmax.z )
					{
						l1_.Y += step.y;
						if ( ( uint )l1_.Y >= ( uint )TopGridDim )
						{
							break;
						}
						t = l1_.Tmax.y;
						l1_.Tmax.y += tdelta.y;
					}
					else
					{
						l1_.Z += step.z;
						if ( ( uint )l1_.Z >= ( uint )TopGridDim )
						{
							break;
						}
						t = l1_.Tmax.z;
						l1_.Tmax.z += tdelta.z;
					}
				}
			}
			// we shouldn't get here
		}
	}
}
