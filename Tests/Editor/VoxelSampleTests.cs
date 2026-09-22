using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Tests for the voxel path the CpuRaytracer sample drives: the sample's voxelisation of a real
	/// mesh into the unit cube, placed through a TLAS, traced with camera-like rays. The reference
	/// dump tests in VoxelSetTests cover procedural content and object-space rays whose targets are
	/// all inside the cube; these cover what the sample actually does - far more bricks than the
	/// dump's, rays that miss or graze the cube, and rays with zero direction components.
	///
	/// The tests assembly does not reference the samples assembly, so the voxelisation job below is
	/// a copy of CpuVoxelBackend.VoxeliseJob rather than a call into it; the only deliberate
	/// difference is CompileSynchronously, so the test never silently measures the Mono fallback.
	/// </summary>
	public class VoxelSampleTests
	{
		/// <summary>CpuVoxelBackend.CubeMargin.</summary>
		private const float CubeMargin = 0.01f;
		/// <summary>CpuVoxelBackend.SamplesPerVoxel.</summary>
		private const float SamplesPerVoxel = 3f;
		/// <summary>CpuVoxelBackend.MaxSamplesPerEdge.</summary>
		private const int MaxSamplesPerEdge = 1536;

		// VoxelSet's level dimensions. They are internal to the runtime assembly, so the lookup
		// below - which walks the three levels by hand - repeats them here.
		private const int BrickDim = 8;
		private const int BrickSize = BrickDim * BrickDim * BrickDim;
		private const int GridDim = VoxelSet.ObjectDim / BrickDim;

		/// <summary>Camera resolution of the assertive sweep.</summary>
		private const int ProbeRes = 512;
		/// <summary>Camera resolution of the sweep that only measures the upstream bug.</summary>
		private const int DiagnosticRes = 128;
		/// <summary>Number of violating rays listed before a report is truncated.</summary>
		private const int MaxReported = 12;

		private const int FlagHit = 1;
		/// <summary>No voxel is set within one voxel of the hit point: the hit is in thin air.</summary>
		private const int FlagGhost = 2;
		/// <summary>The value the traversal reported is not in the 3x3x3 block around the hit point.</summary>
		private const int FlagValueMissing = 4;
		/// <summary>The world hit point is outside the source mesh bounds, grown by two voxels.</summary>
		private const int FlagOutsideAabb = 8;
		/// <summary>The hit point does not even fall inside the unit cube.</summary>
		private const int FlagCellOutOfRange = 16;
		/// <summary>The ray runs into the DDA's zero-length-step case; see <see cref="IsDegenerate"/>.</summary>
		private const int FlagDegenerate = 32;
		/// <summary>What must never happen; FlagValueMissing is reported but not asserted, see the report.</summary>
		private const int FlagBad = FlagGhost | FlagOutsideAabb | FlagCellOutOfRange;

		/// <summary>One traced camera ray and what the voxel content says about its hit.</summary>
		private struct RayProbe
		{
			public float3 Dir;
			public float3 World;
			/// <summary>Distance at which the object ray enters the unit cube, i.e. the t Setup3DDDA starts from.</summary>
			public float Tmin;
			public float T;
			public uint Prim;
			public uint Inst;
			/// <summary>Value of the cell the hit point itself falls in.</summary>
			public uint CellValue;
			public int3 Cell;
			public int Flags;
		}

		/// <summary>One traced object-space edge-case ray.</summary>
		private struct EdgeProbe
		{
			public float T;
			public uint Prim;
			public uint CellValue;
			public int3 Cell;
			public int OccludedFull;
			public int OccludedHalf;
			public int Flags;
		}

		/// <summary>
		/// Copy of CpuVoxelBackend.VoxeliseJob: every triangle is sampled on a barycentric grid whose
		/// spacing is a third of a voxel along both of its edges.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		private unsafe struct VoxeliseJob : IJob
		{
			[ReadOnly] public NativeArray<float4> Vertices;
			[NativeDisableUnsafePtrRestriction] public VoxelSet* Voxels;
			[WriteOnly] public NativeArray<float4> Mapping;
			public uint TriCount;
			public float Margin;

			public void Execute()
			{
				float3 min = new float3( BvhConstants.Far );
				float3 max = new float3( -BvhConstants.Far );
				int vertexCount = ( int )TriCount * 3;
				for ( int i = 0; i < vertexCount; i++ )
				{
					float3 p = Vertices[ i ].xyz;
					min = math.min( min, p );
					max = math.max( max, p );
				}
				float3 extent = max - min;
				// One scale for all three axes keeps the placement transform a uniform scale.
				float scale = math.max( math.cmax( extent ) * ( 1f + ( 2f * Margin ) ), 1e-6f );
				float3 offset = ( 1f - ( extent / scale ) ) * 0.5f;
				Mapping[ 0 ] = new float4( min, scale );
				Mapping[ 1 ] = new float4( offset, 0f );
				for ( uint t = 0; t < TriCount; t++ )
				{
					int v = ( int )t * 3;
					float3 a = ( ( Vertices[ v ].xyz - min ) / scale ) + offset;
					float3 b = ( ( Vertices[ v + 1 ].xyz - min ) / scale ) + offset;
					float3 c = ( ( Vertices[ v + 2 ].xyz - min ) / scale ) + offset;
					uint value = 1u + ( ( ( t * 2654435761u ) >> 24 ) & 254u );
					FillTriangle( a, b, c, value );
				}
				Voxels->UpdateTopGrid();
			}

			private void FillTriangle( float3 a, float3 b, float3 c, uint value )
			{
				const float dim = VoxelSet.ObjectDim;
				float3 ab = b - a;
				float3 ac = c - a;
				int steps1 = math.clamp( ( int )math.ceil( math.length( ab ) * dim * SamplesPerVoxel ), 1, MaxSamplesPerEdge );
				int steps2 = math.clamp( ( int )math.ceil( math.length( ac ) * dim * SamplesPerVoxel ), 1, MaxSamplesPerEdge );
				for ( int i = 0; i <= steps1; i++ )
				{
					float s = ( float )i / steps1;
					for ( int j = 0; j <= steps2; j++ )
					{
						float u = ( float )j / steps2;
						if ( ( s + u ) > 1f )
						{
							break;
						}
						float3 p = a + ( ab * s ) + ( ac * u );
						uint x = ( uint )math.clamp( ( int )( p.x * dim ), 0, VoxelSet.ObjectDim - 1 );
						uint y = ( uint )math.clamp( ( int )( p.y * dim ), 0, VoxelSet.ObjectDim - 1 );
						uint z = ( uint )math.clamp( ( int )( p.z * dim ), 0, VoxelSet.ObjectDim - 1 );
						Voxels->Set( x, y, z, value );
					}
				}
			}
		}

		/// <summary>Reads one voxel by walking grid and brick pool, the way VoxelSet.Set writes it.</summary>
		private static unsafe uint SampleVoxel( in VoxelSet vox, int3 c )
		{
			int gridIdx = ( c.x / BrickDim ) + ( ( c.y / BrickDim ) * GridDim ) + ( ( c.z / BrickDim ) * GridDim * GridDim );
			uint brickIdx = vox.Grid[ gridIdx ];
			if ( brickIdx == 0 )
			{
				return 0u;
			}
			int voxelIdx = ( c.x & ( BrickDim - 1 ) ) + ( ( c.y & ( BrickDim - 1 ) ) * BrickDim ) +
				( ( c.z & ( BrickDim - 1 ) ) * BrickDim * BrickDim );
			return vox.Brick[ ( brickIdx * BrickSize ) + ( uint )voxelIdx ];
		}

		/// <summary>
		/// The cell a hit point belongs to. The DDA reports the distance at which the ray entered the
		/// voxel, so the point sits exactly on a face and a plain floor would be decided by rounding;
		/// a quarter of a voxel along the ray moves it into the voxel that was reported.
		/// </summary>
		private static int3 CellOfHit( float3 objHit, float3 objDir )
		{
			float3 inside = objHit + ( math.normalize( objDir ) * ( 0.25f / VoxelSet.ObjectDim ) );
			return new int3(
				( int )math.floor( inside.x * VoxelSet.ObjectDim ),
				( int )math.floor( inside.y * VoxelSet.ObjectDim ),
				( int )math.floor( inside.z * VoxelSet.ObjectDim ) );
		}

		/// <summary>
		/// Classifies one hit against the voxel content. The cell the hit point falls in is decided by
		/// last-bit rounding whenever the ray enters a voxel across an edge or a corner, so the block
		/// of 27 cells around it is what the checks look at: a hit is a ghost when not one of them
		/// holds a voxel at all, which no amount of rounding can explain.
		/// </summary>
		private static int ClassifyHit( in VoxelSet vox, int3 cell, uint prim, out uint cellValue )
		{
			cellValue = 0u;
			if ( math.any( cell < 0 ) || math.any( cell > VoxelSet.ObjectDim - 1 ) )
			{
				return FlagCellOutOfRange;
			}
			cellValue = SampleVoxel( vox, cell );
			bool anySet = false;
			bool primFound = false;
			for ( int z = -1; z <= 1; z++ )
			{
				for ( int y = -1; y <= 1; y++ )
				{
					for ( int x = -1; x <= 1; x++ )
					{
						int3 c = cell + new int3( x, y, z );
						if ( math.any( c < 0 ) || math.any( c > VoxelSet.ObjectDim - 1 ) )
						{
							continue;
						}
						uint v = SampleVoxel( vox, c );
						if ( v != 0 )
						{
							anySet = true;
							if ( v == prim )
							{
								primFound = true;
							}
						}
					}
				}
			}
			int flags = 0;
			if ( !anySet )
			{
				flags |= FlagGhost;
			}
			if ( !primFound )
			{
				flags |= FlagValueMissing;
			}
			return flags;
		}

		/// <summary>
		/// True when the ray runs into the second upstream degenerate case of the DDA, documented on
		/// VoxelSet.Setup3DDDA: on one axis the ray sits exactly on a cell plane of one of the three
		/// levels and the entry nudge cannot move it off, so that level's tmax for the axis comes out
		/// as ( plane - O ) * rD = 0 * huge = 0. That is below the current t, the DDA steps the axis
		/// first and sets t back to 0, and every distance it reports afterwards is meaningless. The
		/// classification runs inside Burst, because it turns on exact float comparisons.
		/// </summary>
		private static bool IsDegenerate( float3 o, float3 d )
		{
			bool inside = o.x >= 0f && o.x <= 1f && o.y >= 0f && o.y <= 1f && o.z >= 0f && o.z <= 1f;
			return DegenerateAxis( d.x, o.x, inside ) || DegenerateAxis( d.y, o.y, inside ) ||
				DegenerateAxis( d.z, o.z, inside );
		}

		/// <summary>One axis of <see cref="IsDegenerate"/>.</summary>
		private static bool DegenerateAxis( float d, float o, bool originInside )
		{
			// The plane the level setup picks for this axis, at each of the three levels.
			float dsign = ( math.asuint( d ) >> 31 ) != 0 ? 1f : 0f;
			bool onPlane = ( ( math.ceil( o * 8f ) - dsign ) * ( 1f / 8f ) ) == o
				|| ( ( math.ceil( o * 32f ) - dsign ) * ( 1f / 32f ) ) == o
				|| ( ( math.ceil( o * 256f ) - dsign ) * ( 1f / 256f ) ) == o;
			if ( !onPlane )
			{
				return false;
			}
			// A zero component never leaves the plane, whatever t the walk has reached. A merely small
			// one only fails when the walk starts at t = 0, which is when the origin is inside the
			// cube; for an origin outside it the walk starts at the cube entry instead.
			if ( d == 0f )
			{
				return true;
			}
			return originInside && ( o + ( d * 0.0000025f ) ) == o;
		}

		/// <summary>
		/// Repeats the slab test of VoxelSet.Setup3DDDA, so the report can say where the DDA started.
		/// Returns the entry distance, or 0 for a ray that starts inside the cube.
		/// </summary>
		private static float CubeEntry( float3 o, float3 d )
		{
			if ( o.x >= 0f && o.x <= 1f && o.y >= 0f && o.y <= 1f && o.z >= 0f && o.z <= 1f )
			{
				return 0f;
			}
			float3 rd = BvhMath.Rcp( d );
			return math.cmax( math.min( -o * rd, ( 1f - o ) * rd ) );
		}

		/// <summary>Traces one pinhole view through the TLAS and checks every hit against the voxel content.</summary>
		[BurstCompile( CompileSynchronously = true )]
		private struct ProbeJob : IJobParallelFor
		{
			public Bvh Tlas;
			public VoxelSet Voxels;
			public BvhMat4 InvTransform;
			public float3 Origin;
			public float3 Forward;
			public float3 Right;
			public float3 Up;
			public float HalfX;
			public float HalfY;
			/// <summary>Length the world ray direction is given; see the sample's RayScale.</summary>
			public float RayScale;
			public int Width;
			public int Height;
			public float3 MeshMin;
			public float3 MeshMax;
			public float Grow;
			public int ProbeOffset;
			[NativeDisableParallelForRestriction] public NativeArray<RayProbe> Probes;

			public void Execute( int index )
			{
				int x = index % Width;
				int y = index / Width;
				float sx = ( ( 2f * ( ( x + 0.5f ) / Width ) ) - 1f ) * HalfX;
				float sy = ( 1f - ( 2f * ( ( y + 0.5f ) / Height ) ) ) * HalfY;
				float3 dir = math.normalize( Forward + ( Right * sx ) + ( Up * sy ) ) * RayScale;
				Ray ray = default;
				ray.O = Origin;
				ray.D = dir;
				ray.RD = BvhMath.Rcp( dir );
				ray.Mask = BvhConstants.RayMaskIntersectAll;
				ray.Hit.T = BvhConstants.Far;
				float3 objO = InvTransform.TransformPoint( ray.O );
				float3 objD = InvTransform.TransformVector( ray.D );
				Tlas.Intersect( ref ray );
				RayProbe probe = default;
				probe.Dir = ray.D;
				probe.Tmin = CubeEntry( objO, objD );
				probe.T = ray.Hit.T;
				probe.Prim = ray.Hit.Prim;
				probe.Inst = ray.Hit.Inst;
				if ( ray.Hit.T < BvhConstants.Far )
				{
					probe.Flags |= FlagHit;
					float3 world = ray.O + ( ray.D * ray.Hit.T );
					probe.World = world;
					int3 cell = CellOfHit( InvTransform.TransformPoint( world ), objD );
					probe.Cell = cell;
					probe.Flags |= ClassifyHit( Voxels, cell, ray.Hit.Prim, out uint cellValue );
					probe.CellValue = cellValue;
					if ( math.any( world < ( MeshMin - Grow ) ) || math.any( world > ( MeshMax + Grow ) ) )
					{
						probe.Flags |= FlagOutsideAabb;
					}
				}
				Probes[ ProbeOffset + index ] = probe;
			}
		}

		/// <summary>
		/// Traces the object-space edge-case rays straight through the voxel set: closest hit, the
		/// cell that hit lands in, and the two occlusion queries the hit is checked against.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		private unsafe struct EdgeJob : IJobParallelFor
		{
			[NativeDisableUnsafePtrRestriction] public VoxelSet* Voxels;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<EdgeProbe> Probes;

			public void Execute( int i )
			{
				// Built by hand rather than through the Ray constructor: the directions are already
				// normalised, and the occlusion queries must see the same bits.
				Ray ray = default;
				ray.O = Origins[ i ];
				ray.D = Directions[ i ];
				ray.RD = BvhMath.Rcp( ray.D );
				ray.Mask = BvhConstants.RayMaskIntersectAll;
				ray.Hit.T = BvhConstants.Far;
				Voxels->Intersect( ref ray );
				EdgeProbe probe = default;
				if ( IsDegenerate( ray.O, ray.D ) )
				{
					probe.Flags |= FlagDegenerate;
				}
				probe.T = ray.Hit.T;
				probe.Prim = ray.Hit.Prim;
				if ( ray.Hit.T < BvhConstants.Far )
				{
					probe.Flags |= FlagHit;
					int3 cell = CellOfHit( ray.O + ( ray.D * ray.Hit.T ), ray.D );
					probe.Cell = cell;
					probe.Flags |= ClassifyHit( *Voxels, cell, ray.Hit.Prim, out uint cellValue );
					probe.CellValue = cellValue;
				}
				Ray full = default;
				full.O = Origins[ i ];
				full.D = Directions[ i ];
				full.RD = ray.RD;
				full.Mask = BvhConstants.RayMaskIntersectAll;
				full.Hit.T = BvhConstants.Far;
				probe.OccludedFull = Voxels->IsOccluded( full ) ? 1 : 0;
				Ray half = full;
				half.Hit.T = ray.Hit.T < BvhConstants.Far ? 0.5f * ray.Hit.T : BvhConstants.Far;
				probe.OccludedHalf = Voxels->IsOccluded( half ) ? 1 : 0;
				Probes[ i ] = probe;
			}
		}

		/// <summary>Loads the scene, or ignores the test when it is missing.</summary>
		private static NativeArray<float4> RequireScene( string sceneName, out uint triCount )
		{
			string binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			if ( !File.Exists( binPath ) )
			{
				Assert.Ignore( $"missing {binPath}; run Tools~/fetch.ps1" );
			}
			return BvhSceneFile.Load( binPath, Allocator.Persistent, out triCount );
		}

		/// <summary>
		/// Voxelises a scene the way CpuVoxelBackend does and hands back the unit-cube mapping it
		/// derived, as the sample reads it back for the instance transform.
		/// </summary>
		private static unsafe void Voxelise( NativeArray<float4> vertices, uint triCount, VoxelSet* voxels, out float3 min, out float scale, out float3 offset )
		{
			NativeArray<float4> mapping = new NativeArray<float4>( 2, Allocator.Persistent );
			try
			{
				VoxeliseJob job = new VoxeliseJob
				{
					Vertices = vertices,
					Voxels = voxels,
					Mapping = mapping,
					TriCount = triCount,
					Margin = CubeMargin
				};
				job.Schedule().Complete();
				min = mapping[ 0 ].xyz;
				scale = mapping[ 0 ].w;
				offset = mapping[ 1 ].xyz;
			}
			finally
			{
				mapping.Dispose();
			}
		}

		/// <summary>
		/// The sample's picture, as data: the mesh is voxelised, placed through a TLAS with the
		/// sample's cube-to-mesh transform, and shot at from three directions with a dense pinhole
		/// grid. Every hit has to land within one voxel of a voxel that is actually set, and inside
		/// the mesh bounds grown by two voxels.
		///
		/// The sweep runs twice. The first pass hands the TLAS a direction of length 'scale', which
		/// is what the sample does, so the object-space ray the DDA sees is unit length; that pass is
		/// asserted. The second pass uses a unit world direction - the object ray is then 1/scale
		/// long - and is only counted: it measures the upstream DDA bug documented in
		/// VoxelSet.Intersect.cs, and is what the sample's picture looked like before.
		/// </summary>
		[TestCase( "bunny" )]
		public unsafe void SampleVoxelisation_CameraRays_HitOnlyFilledVoxels( string sceneName )
		{
			NativeArray<float4> verts = RequireScene( sceneName, out uint triCount );
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			const int viewCount = 3;
			NativeArray<VoxelSet> voxArray = new NativeArray<VoxelSet>( 1, Allocator.Persistent );
			NativeArray<BlasRef> blasList = new NativeArray<BlasRef>( 1, Allocator.Persistent );
			NativeArray<BlasInstance> instances = new NativeArray<BlasInstance>( 1, Allocator.Persistent );
			NativeArray<RayProbe> probes = new NativeArray<RayProbe>( viewCount * ProbeRes * ProbeRes, Allocator.Persistent );
			NativeArray<RayProbe> diagnostic = new NativeArray<RayProbe>( viewCount * DiagnosticRes * DiagnosticRes, Allocator.Persistent );
			Bvh tlas = Bvh.Create( Allocator.Persistent );
			VoxelSet vox = VoxelSet.Create( Allocator.Persistent );
			try
			{
				Voxelise( verts, triCount, &vox, out float3 meshMin, out float scale, out float3 offset );
				voxArray[ 0 ] = vox;
				blasList[ 0 ] = BlasRef.From( ( VoxelSet* )voxArray.GetUnsafePtr() );

				// CpuVoxelBackend.Voxelise: the inverse of u = ( p - min ) / scale + offset.
				float4x4 cubeToMesh = float4x4.TRS( meshMin - ( offset * scale ), quaternion.identity, new float3( scale ) );
				BlasInstance instance = BlasInstance.Create( 0 );
				instance.Transform = BvhMat4.FromFloat4x4( cubeToMesh );
				instances[ 0 ] = instance;
				tlas.BuildTlas( ( BlasInstance* )instances.GetUnsafePtr(), 1,
					( BlasRef* )blasList.GetUnsafePtr(), 1 );

				float3 meshMax = meshMin;
				for ( int i = 0; i < ( int )triCount * 3; i++ )
				{
					meshMax = math.max( meshMax, verts[ i ].xyz );
				}
				float voxelSize = scale / VoxelSet.ObjectDim;
				float3 centre = ( meshMin + meshMax ) * 0.5f;
				float radius = math.length( meshMax - meshMin ) * 0.5f;
				float distance = 2.5f * radius;
				// The frustum takes in the whole unit cube and a wide margin around it, so rays that
				// miss the instance box entirely are part of the sweep.
				float half = ( scale * 0.8660254f * 1.3f ) / distance;
				float3[] views =
				{
					math.normalize( new float3( 0.6f, 0.35f, 1f ) ),
					math.normalize( new float3( -1f, 0.2f, -0.35f ) ),
					math.normalize( new float3( 0.15f, 1f, 0.3f ) )
				};
				float3[] eyes = new float3[ viewCount ];

				for ( int pass = 0; pass < 2; pass++ )
				{
					int res = pass == 0 ? ProbeRes : DiagnosticRes;
					for ( int v = 0; v < viewCount; v++ )
					{
						float3 eye = centre + ( views[ v ] * distance );
						eyes[ v ] = eye;
						float3 forward = math.normalize( centre - eye );
						float3 worldUp = math.abs( forward.y ) > 0.95f ? new float3( 0f, 0f, 1f ) : new float3( 0f, 1f, 0f );
						float3 right = math.normalize( math.cross( forward, worldUp ) );
						float3 up = math.cross( right, forward );
						ProbeJob job = new ProbeJob
						{
							Tlas = tlas,
							Voxels = vox,
							InvTransform = instances[ 0 ].InvTransform,
							Origin = eye,
							Forward = forward,
							Right = right,
							Up = up,
							HalfX = half,
							HalfY = half,
							RayScale = pass == 0 ? scale : 1f,
							Width = res,
							Height = res,
							MeshMin = meshMin,
							MeshMax = meshMax,
							Grow = 2f * voxelSize,
							ProbeOffset = v * res * res,
							Probes = pass == 0 ? probes : diagnostic
						};
						job.Schedule( res * res, 256 ).Complete();
					}
				}

				TestContext.WriteLine( $"{sceneName}: bricks {vox.FreeBrickPtr}, cube scale {scale}, voxel {voxelSize}, eye distance {distance}" );
				Report( diagnostic, DiagnosticRes, viewCount, eyes, "unit world direction (upstream bug, not asserted)", false );
				Report( probes, ProbeRes, viewCount, eyes, "world direction of length scale (what the sample does)", true );
			}
			finally
			{
				tlas.Dispose();
				vox.Dispose();
				verts.Dispose();
				voxArray.Dispose();
				blasList.Dispose();
				instances.Dispose();
				probes.Dispose();
				diagnostic.Dispose();
			}
		}

		/// <summary>Counts and lists the violations of one sweep, and asserts on them when asked to.</summary>
		private static void Report( NativeArray<RayProbe> probes, int res, int viewCount, float3[] eyes,
			string label, bool assert )
		{
			int perView = res * res;
			int hits = 0, ghosts = 0, valueMissing = 0, outside = 0, outOfRange = 0, exact = 0, reported = 0;
			int[] badPerView = new int[ viewCount ];
			int[] minX = new int[ viewCount ];
			int[] maxX = new int[ viewCount ];
			int[] minY = new int[ viewCount ];
			int[] maxY = new int[ viewCount ];
			for ( int v = 0; v < viewCount; v++ )
			{
				minX[ v ] = res;
				maxX[ v ] = -1;
				minY[ v ] = res;
				maxY[ v ] = -1;
			}
			for ( int i = 0; i < viewCount * perView; i++ )
			{
				RayProbe probe = probes[ i ];
				if ( ( probe.Flags & FlagHit ) == 0 )
				{
					continue;
				}
				hits++;
				if ( ( probe.Flags & FlagGhost ) != 0 )
				{
					ghosts++;
				}
				if ( ( probe.Flags & FlagValueMissing ) != 0 )
				{
					valueMissing++;
				}
				if ( ( probe.Flags & FlagOutsideAabb ) != 0 )
				{
					outside++;
				}
				if ( ( probe.Flags & FlagCellOutOfRange ) != 0 )
				{
					outOfRange++;
				}
				if ( probe.CellValue == probe.Prim )
				{
					exact++;
				}
				if ( ( probe.Flags & FlagBad ) == 0 )
				{
					continue;
				}
				int view = i / perView;
				int pixel = i - ( view * perView );
				int px = pixel % res;
				int py = pixel / res;
				badPerView[ view ]++;
				minX[ view ] = math.min( minX[ view ], px );
				maxX[ view ] = math.max( maxX[ view ], px );
				minY[ view ] = math.min( minY[ view ], py );
				maxY[ view ] = math.max( maxY[ view ], py );
				if ( reported < MaxReported )
				{
					reported++;
					TestContext.WriteLine( $"    view {view} pixel ({px},{py}) flags {probe.Flags}: O {eyes[ view ]} D {probe.Dir} t {probe.T} tmin {probe.Tmin} prim {probe.Prim} inst {probe.Inst} world {probe.World} cell {probe.Cell} cellValue {probe.CellValue}" );
				}
			}
			TestContext.WriteLine( $"  {label}: {hits}/{viewCount * perView} hit, ghosts {ghosts}, value not in the 3x3x3 block {valueMissing}, outside grown aabb {outside}, out of range {outOfRange}, exact cell match {exact}" );
			for ( int v = 0; v < viewCount; v++ )
			{
				if ( badPerView[ v ] > 0 )
				{
					TestContext.WriteLine( $"    view {v}: {badPerView[ v ]} violating pixels of {perView}, x [{minX[ v ]}..{maxX[ v ]}] y [{minY[ v ]}..{maxY[ v ]}]" );
				}
			}
			if ( !assert )
			{
				return;
			}
			Assert.Greater( hits, 0, "no camera ray hit the voxel object at all" );
			Assert.AreEqual( 0, outOfRange, "hits whose point is not inside the unit cube" );
			Assert.AreEqual( 0, ghosts, "hits with no voxel set anywhere near them" );
			Assert.AreEqual( 0, outside, "hits outside the mesh bounds grown by two voxels" );
		}

		/// <summary>
		/// Object-space rays of the classes the reference dump never produced: narrow misses of the
		/// unit cube, grazes of a face, an edge and a corner, rays with one or two zero direction
		/// components, and rays that start in the empty space inside the shell. Every hit has to land
		/// next to a voxel that is set, and IsOccluded has to agree with Intersect over the ray length.
		///
		/// Rays that <see cref="IsDegenerate"/> picks out are counted and reported but not asserted:
		/// they run into the DDA's zero-length-step case, an upstream bug the port reproduces on
		/// purpose. Zero direction components as such are asserted; it is only the combination with
		/// an origin exactly on a cell plane of that axis that is excused.
		/// </summary>
		[TestCase( "bunny" )]
		public unsafe void SampleVoxelisation_EdgeRays_AgreeWithContent( string sceneName )
		{
			NativeArray<float4> verts = RequireScene( sceneName, out uint triCount );
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			List<float3> origins = new List<float3>();
			List<float3> directions = new List<float3>();
			List<string> labels = new List<string>();
			VoxelSet vox = VoxelSet.Create( Allocator.Persistent );
			NativeArray<float3> originArray = default;
			NativeArray<float3> directionArray = default;
			NativeArray<EdgeProbe> probes = default;
			try
			{
				Voxelise( verts, triCount, &vox, out float3 _, out float _, out float3 _ );

				void Aim( float3 from, float3 target, string label )
				{
					origins.Add( from );
					directions.Add( math.normalize( target - from ) );
					labels.Add( label );
				}

				float3[] corners =
				{
					new float3( -1.5f, 0.5f, 0.5f ), new float3( 2.5f, 0.5f, 0.5f ),
					new float3( 0.5f, -1.5f, 0.5f ), new float3( 0.5f, 2.5f, 0.5f ),
					new float3( 0.5f, 0.5f, -1.5f ), new float3( 0.5f, 0.5f, 2.5f ),
					new float3( -1.5f, -1.5f, -1.5f ), new float3( 2.5f, 2.5f, 2.5f ),
					new float3( -1.5f, 2.5f, -1.5f ), new float3( 2.5f, -1.5f, 2.5f )
				};
				float[] deltas = { -1e-2f, -1e-3f, -1e-4f, -1e-5f, -1e-6f, 0f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f };
				for ( int c = 0; c < corners.Length; c++ )
				{
					for ( int d = 0; d < deltas.Length; d++ )
					{
						float e = deltas[ d ];
						// (i) and (ii): the target walks from just outside a face, edge or corner of
						// the cube to just inside it.
						Aim( corners[ c ], new float3( -e, 0.5f, 0.5f ), $"face -x {e}" );
						Aim( corners[ c ], new float3( 1f + e, 0.5f, 0.5f ), $"face +x {e}" );
						Aim( corners[ c ], new float3( 0.5f, -e, 0.5f ), $"face -y {e}" );
						Aim( corners[ c ], new float3( 0.5f, 1f + e, 0.5f ), $"face +y {e}" );
						Aim( corners[ c ], new float3( 0.5f, 0.5f, -e ), $"face -z {e}" );
						Aim( corners[ c ], new float3( 0.5f, 0.5f, 1f + e ), $"face +z {e}" );
						Aim( corners[ c ], new float3( -e, -e, 0.5f ), $"edge -x-y {e}" );
						Aim( corners[ c ], new float3( 1f + e, -e, 0.5f ), $"edge +x-y {e}" );
						Aim( corners[ c ], new float3( 0.5f, 1f + e, 1f + e ), $"edge +y+z {e}" );
						Aim( corners[ c ], new float3( -e, -e, -e ), $"corner min {e}" );
						Aim( corners[ c ], new float3( 1f + e, 1f + e, 1f + e ), $"corner max {e}" );
						Aim( corners[ c ], new float3( -e, 1f + e, -e ), $"corner mixed {e}" );
					}
				}

				// (iii): one or two exactly zero direction components, from origins inside and
				// outside the cube and inside and outside each slab.
				float3[] axisDirs =
				{
					new float3( 1f, 0f, 0f ), new float3( -1f, 0f, 0f ),
					new float3( 0f, 1f, 0f ), new float3( 0f, -1f, 0f ),
					new float3( 0f, 0f, 1f ), new float3( 0f, 0f, -1f ),
					math.normalize( new float3( 1f, 1f, 0f ) ), math.normalize( new float3( -1f, 1f, 0f ) ),
					math.normalize( new float3( 0f, 1f, 1f ) ), math.normalize( new float3( 1f, 0f, -1f ) )
				};
				float[] coords = { -2f, -1e-4f, 0f, 1e-4f, 0.25f, 0.5f, 0.75f, 1f - 1e-4f, 1f, 1f + 1e-4f, 3f };
				for ( int a = 0; a < axisDirs.Length; a++ )
				{
					for ( int i = 0; i < coords.Length; i++ )
					{
						for ( int j = 0; j < coords.Length; j++ )
						{
							origins.Add( new float3( coords[ i ], coords[ j ], 0.4f ) );
							directions.Add( axisDirs[ a ] );
							labels.Add( $"axis {a} o ({coords[ i ]},{coords[ j ]},0.4)" );
							origins.Add( new float3( 0.4f, coords[ i ], coords[ j ] ) );
							directions.Add( axisDirs[ a ] );
							labels.Add( $"axis {a} o (0.4,{coords[ i ]},{coords[ j ]})" );
						}
					}
				}

				// (iv): origins in the empty space the shell encloses, fired in every direction of a
				// small sphere. The mesh centre maps to the middle of the cube, which the surface
				// voxeliser leaves empty. The cube centre is on a cell plane of all three levels on
				// all three axes, so a direction with a very small component there runs into the same
				// degenerate case a zero component does; the second origin is off every plane, and
				// carries the real coverage of this class.
				const int sphereSteps = 24;
				for ( int i = 0; i < sphereSteps; i++ )
				{
					for ( int j = 0; j < sphereSteps; j++ )
					{
						float theta = ( ( i + 0.5f ) / sphereSteps ) * math.PI;
						float phi = ( ( j + 0.5f ) / sphereSteps ) * 2f * math.PI;
						float3 d = new float3(
							math.sin( theta ) * math.cos( phi ),
							math.cos( theta ),
							math.sin( theta ) * math.sin( phi ) );
						origins.Add( new float3( 0.5f, 0.5f, 0.5f ) );
						directions.Add( d );
						labels.Add( $"inside ({i},{j})" );
						origins.Add( new float3( 0.5013f, 0.4987f, 0.5029f ) );
						directions.Add( d );
						labels.Add( $"inside off-plane ({i},{j})" );
					}
				}

				int rayCount = origins.Count;
				originArray = new NativeArray<float3>( rayCount, Allocator.Persistent );
				directionArray = new NativeArray<float3>( rayCount, Allocator.Persistent );
				probes = new NativeArray<EdgeProbe>( rayCount, Allocator.Persistent );
				for ( int i = 0; i < rayCount; i++ )
				{
					originArray[ i ] = origins[ i ];
					directionArray[ i ] = directions[ i ];
				}
				EdgeJob job = new EdgeJob
				{
					Voxels = &vox,
					Origins = originArray,
					Directions = directionArray,
					Probes = probes
				};
				job.Schedule( rayCount, 64 ).Complete();

				// Index 0 counts the rays the DDA is well conditioned for, index 1 the ones that run
				// into its zero-length-step case. Only index 0 is asserted; index 1 is an upstream
				// bug the port reproduces on purpose, and is reported so the count stays visible.
				int[] rays = new int[ 2 ];
				int[] hits = new int[ 2 ];
				int[] ghosts = new int[ 2 ];
				int[] valueMissing = new int[ 2 ];
				int[] outOfRange = new int[ 2 ];
				int[] occFull = new int[ 2 ];
				int[] occHalf = new int[ 2 ];
				int reported = 0;
				for ( int i = 0; i < rayCount; i++ )
				{
					EdgeProbe probe = probes[ i ];
					int k = ( probe.Flags & FlagDegenerate ) != 0 ? 1 : 0;
					bool hit = ( probe.Flags & FlagHit ) != 0;
					rays[ k ]++;
					if ( hit )
					{
						hits[ k ]++;
					}
					if ( ( probe.Flags & FlagGhost ) != 0 )
					{
						ghosts[ k ]++;
					}
					if ( ( probe.Flags & FlagValueMissing ) != 0 )
					{
						valueMissing[ k ]++;
					}
					if ( ( probe.Flags & FlagCellOutOfRange ) != 0 )
					{
						outOfRange[ k ]++;
					}
					bool fullBad = ( probe.OccludedFull != 0 ) != hit;
					bool halfBad = hit && probe.OccludedHalf != 0;
					if ( fullBad )
					{
						occFull[ k ]++;
					}
					if ( halfBad )
					{
						occHalf[ k ]++;
					}
					if ( k == 0 && ( ( probe.Flags & FlagBad ) != 0 || fullBad || halfBad ) && reported < MaxReported )
					{
						reported++;
						TestContext.WriteLine( $"  ray {i} [{labels[ i ]}] flags {probe.Flags}: O {origins[ i ]} D {directions[ i ]} t {probe.T} prim {probe.Prim} cell {probe.Cell} cellValue {probe.CellValue} occFull {probe.OccludedFull} occHalf {probe.OccludedHalf}" );
					}
				}
				TestContext.WriteLine( $"{sceneName} voxel edge rays: {rayCount} total" );
				TestContext.WriteLine( $"  well conditioned: {rays[ 0 ]} rays, {hits[ 0 ]} hit, ghosts {ghosts[ 0 ]}, value not in the 3x3x3 block {valueMissing[ 0 ]}, out of range {outOfRange[ 0 ]}, occlusion full {occFull[ 0 ]} half {occHalf[ 0 ]}" );
				TestContext.WriteLine( $"  upstream degenerate (not asserted): {rays[ 1 ]} rays, {hits[ 1 ]} hit, ghosts {ghosts[ 1 ]}, value not in the 3x3x3 block {valueMissing[ 1 ]}, out of range {outOfRange[ 1 ]}, occlusion full {occFull[ 1 ]} half {occHalf[ 1 ]}" );
				Assert.Greater( rays[ 1 ], 0, "the degenerate class went missing; the classification no longer matches the rays" );
				Assert.Greater( hits[ 0 ], 0, "no well conditioned edge-case ray hit the voxel object at all" );
				Assert.AreEqual( 0, outOfRange[ 0 ], "hits whose point is not inside the unit cube" );
				Assert.AreEqual( 0, ghosts[ 0 ], "hits with no voxel set anywhere near them" );
				Assert.AreEqual( 0, occFull[ 0 ], "IsOccluded over the full ray disagrees with Intersect" );
				Assert.AreEqual( 0, occHalf[ 0 ], "IsOccluded shortened to half the hit distance still reports a blocker" );
			}
			finally
			{
				vox.Dispose();
				verts.Dispose();
				if ( originArray.IsCreated )
				{
					originArray.Dispose();
				}
				if ( directionArray.IsCreated )
				{
					directionArray.Dispose();
				}
				if ( probes.IsCreated )
				{
					probes.Dispose();
				}
			}
		}
	}
}
