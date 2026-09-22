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
	/// Tests for the port of tinybvh's VoxelSet against the reference dump of
	/// Tools/RefDump/voxeldump.cpp. The voxel content is procedural and scene-independent, so the
	/// content test does the same work for every scene; the parametrisation is kept because the
	/// TLAS section of the dump - not compared yet - does differ per scene.
	/// Tests are ignored when their reference data is missing.
	/// </summary>
	public class VoxelSetTests
	{
		/// <summary>Number of differing entries listed in an assertion message before it is truncated.</summary>
		const int MaxReportedDiffs = 4;

		/// <summary>Barycentric distance to a triangle edge below which a hit counts as grazing.</summary>
		const float GrazingEps = 1e-3f;

		/// <summary>True when the hit lies within GrazingEps of a triangle edge or vertex, where last-bit rounding decides hit or miss.</summary>
		static bool IsGrazing( float u, float v )
		{
			return u < GrazingEps || v < GrazingEps || ( 1f - u - v ) < GrazingEps;
		}

		static bool BvhNodesEqual( BvhNode a, BvhNode b )
		{
			return BitsEqual( a.AabbMin, b.AabbMin ) && a.LeftFirst == b.LeftFirst
				&& BitsEqual( a.AabbMax, b.AabbMax ) && a.TriCount == b.TriCount;
		}

		static bool BitsEqual( float a, float b )
		{
			return math.asuint( a ) == math.asuint( b );
		}

		static bool BitsEqual( float3 a, float3 b )
		{
			return BitsEqual( a.x, b.x ) && BitsEqual( a.y, b.y ) && BitsEqual( a.z, b.z );
		}

		static string Describe( List<int> diffs )
		{
			if ( diffs.Count == 0 )
			{
				return "none";
			}
			string list = string.Join( ", ", diffs.GetRange( 0, math.min( diffs.Count, MaxReportedDiffs ) ) );
			return $"{diffs.Count} differing, first at [{list}]";
		}

		/// <summary>Loads the reference dump for a scene, or ignores the test when it is missing.</summary>
		static VoxDumpFile RequireData( string sceneName )
		{
			string refPath = BvhSceneFile.TestDataPath( sceneName + ".vox.ref" );
			if ( !File.Exists( refPath ) )
			{
				Assert.Ignore( $"missing {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}
			return VoxDumpFile.Load( refPath );
		}

		/// <summary>
		/// Refills a voxel set with the content voxeldump.cpp describes: a spherical shell, then 64
		/// randomly placed boxes. Runs in Burst, both because there are 256^3 shell candidates and
		/// because the box coordinates come out of a single-precision multiply that Mono would
		/// evaluate in double.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		private unsafe struct FillJob : IJob
		{
			[NativeDisableUnsafePtrRestriction] public VoxelSet* Vox;

			/// <summary>voxeldump.cpp's R(): the reference LCG, seeded 0x12345678 by the caller.</summary>
			private static float R( ref uint s )
			{
				s = ( s * 1664525u ) + 1013904223u;
				return ( s >> 8 ) * ( 1f / 16777216f );
			}

			public void Execute()
			{
				const uint dim = VoxelSet.ObjectDim;
				// 1. a spherical shell.
				for ( uint z = 0; z < dim; z++ )
				{
					for ( uint y = 0; y < dim; y++ )
					{
						for ( uint x = 0; x < dim; x++ )
						{
							float dx = ( float )x - 127.5f, dy = ( float )y - 127.5f, dz = ( float )z - 127.5f;
							float d2 = ( dx * dx ) + ( dy * dy ) + ( dz * dz );
							if ( d2 >= 90f * 90f && d2 < 100f * 100f )
							{
								Vox->Set( x, y, z, 1u + ( ( ( x * 3u ) + ( y * 5u ) + ( z * 7u ) ) & 255u ) );
							}
						}
					}
				}
				// 2. 64 boxes. The generator is seeded here and consumed by the boxes only; the
				// rays of the dump come out of the same stream afterwards, but they are read from
				// the file rather than regenerated.
				uint s = 0x12345678;
				for ( uint b = 0; b < 64; b++ )
				{
					uint x0 = ( uint )( R( ref s ) * 240f );
					uint y0 = ( uint )( R( ref s ) * 240f );
					uint z0 = ( uint )( R( ref s ) * 240f );
					uint sx = 4u + ( uint )( R( ref s ) * 12f );
					uint sy = 4u + ( uint )( R( ref s ) * 12f );
					uint sz = 4u + ( uint )( R( ref s ) * 12f );
					for ( uint z = z0; z < z0 + sz; z++ )
					{
						for ( uint y = y0; y < y0 + sy; y++ )
						{
							for ( uint x = x0; x < x0 + sx; x++ )
							{
								Vox->Set( x, y, z, 300u + b );
							}
						}
					}
				}
			}
		}

		/// <summary>
		/// Traces the reference rays through the voxel set inside Burst, where float expressions are
		/// not widened to double. Mirrors the tracing loop of voxeldump.cpp: the closest hit and its
		/// step count, the normal of that hit, an occlusion query over the full ray and a second one
		/// shortened to half the hit distance.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		private unsafe struct IntersectJob : IJobParallelFor
		{
			[NativeDisableUnsafePtrRestriction] public VoxelSet* Vox;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> Steps;
			public NativeArray<float3> Normals;
			public NativeArray<int> OccludedFull;
			public NativeArray<int> OccludedHalf;

			public void Execute( int i )
			{
				// The dump stores O and D as the Ray constructor left them, so D is already
				// normalised; the ray is built by hand here to keep the direction bit for bit what
				// the C++ traced with, because normalising a normalised vector is not the identity.
				Ray ray = default;
				ray.O = Origins[ i ];
				ray.D = Directions[ i ];
				ray.RD = BvhMath.Rcp( ray.D );
				ray.Mask = BvhConstants.RayMaskIntersectAll;
				ray.Hit.T = BvhConstants.Far;
				Steps[ i ] = Vox->Intersect( ref ray );
				Hits[ i ] = ray.Hit;
				Normals[ i ] = ray.Hit.T < BvhConstants.Far ? Vox->GetNormal( in ray ) : new float3( 0f );
				// The two occlusion rays, on the other hand, are constructed by the C++ from the
				// stored O and D, so they do go through the constructor - renormalisation included.
				OccludedFull[ i ] = Vox->IsOccluded( new Ray( Origins[ i ], Directions[ i ] ) ) ? 1 : 0;
				float halfT = ray.Hit.T < BvhConstants.Far ? 0.5f * ray.Hit.T : BvhConstants.Far;
				OccludedHalf[ i ] = Vox->IsOccluded( new Ray( Origins[ i ], Directions[ i ], halfT ) ) ? 1 : 0;
			}
		}

		/// <summary>
		/// Port of the instance setup of voxeldump.cpp's mixed TLAS: the triangle BVH with an
		/// identity transform, then the voxel set twice - once scaled to the scene extent and
		/// shifted by extent.x * 1.1 along x, once at half that size and shifted by extent.z * 1.1
		/// along z. Runs in Burst because the transform cells come out of single-precision
		/// arithmetic over the scene bounds, which Mono would evaluate in double.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		private unsafe struct SetupInstancesJob : IJob
		{
			public float3 AabbMin;
			public float3 AabbMax;
			[NativeDisableUnsafePtrRestriction] public BlasInstance* Inst;

			public void Execute()
			{
				float3 ext = AabbMax - AabbMin;
				Inst[ 0 ] = BlasInstance.Create( 0 );
				Inst[ 1 ] = BlasInstance.Create( 1 );
				Inst[ 2 ] = BlasInstance.Create( 1 );
				for ( int i = 0; i < 3; i++ )
				{
					Inst[ i ].Mask = 0xFFFF;
				}
				Inst[ 1 ].Transform[ 0 ] = ext.x;
				Inst[ 1 ].Transform[ 5 ] = ext.y;
				Inst[ 1 ].Transform[ 10 ] = ext.z;
				Inst[ 1 ].Transform[ 3 ] = AabbMin.x + ( ext.x * 1.1f );
				Inst[ 1 ].Transform[ 7 ] = AabbMin.y;
				Inst[ 1 ].Transform[ 11 ] = AabbMin.z;
				Inst[ 2 ].Transform[ 0 ] = ext.x * 0.5f;
				Inst[ 2 ].Transform[ 5 ] = ext.y * 0.5f;
				Inst[ 2 ].Transform[ 10 ] = ext.z * 0.5f;
				Inst[ 2 ].Transform[ 3 ] = AabbMin.x;
				Inst[ 2 ].Transform[ 7 ] = AabbMin.y;
				Inst[ 2 ].Transform[ 11 ] = AabbMin.z + ( ext.z * 1.1f );
			}
		}

		/// <summary>
		/// Traces the TLAS rays of the dump, mirroring its loop: one closest-hit query and one
		/// full-length occlusion query per ray.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		private struct TlasIntersectJob : IJobParallelFor
		{
			public Bvh Tlas;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> Hits;
			public NativeArray<int> OccludedFull;

			public void Execute( int i )
			{
				Ray ray = new Ray( Origins[ i ], Directions[ i ] );
				Tlas.Intersect( ref ray );
				Hits[ i ] = ray.Hit;
				OccludedFull[ i ] = Tlas.IsOccluded( new Ray( Origins[ i ], Directions[ i ] ) ) ? 1 : 0;
			}
		}

		static unsafe void BuildContent( VoxelSet* vox )
		{
			FillJob job = new FillJob { Vox = vox };
			job.Schedule().Complete();
			vox->UpdateTopGrid();
		}

		/// <summary>
		/// The three levels of the voxel set after the procedural fill: the brick index per grid
		/// cell, the used part of the brick pool and the top grid bits. Brick allocation order
		/// depends on the order of the Set calls, so this also pins down that the fill runs in the
		/// same order as the C++.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void Content_MatchesReference( string sceneName )
		{
			VoxDumpFile refFile = RequireData( sceneName );
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );
			Assert.AreEqual( 32u, refFile.GridDim, "gridDim: the port is compiled for objectDim 256" );
			Assert.AreEqual( 8u, refFile.BrickDim, "brickDim" );
			Assert.AreEqual( 8u, refFile.TopGridDim, "topGridDim" );

			VoxelSet vox = VoxelSet.Create( Allocator.Persistent );
			try
			{
				BuildContent( &vox );

				Assert.AreEqual( refFile.FreeBrickPtr, vox.FreeBrickPtr, "freeBrickPtr" );
				List<int> gridDiffs = new List<int>();
				for ( int i = 0; i < refFile.Grid.Length; i++ )
				{
					if ( vox.Grid[ i ] != refFile.Grid[ i ] )
					{
						gridDiffs.Add( i );
					}
				}
				List<int> brickDiffs = new List<int>();
				for ( int i = 0; i < refFile.Brick.Length; i++ )
				{
					if ( vox.Brick[ i ] != refFile.Brick[ i ] )
					{
						brickDiffs.Add( i );
					}
				}
				List<int> topDiffs = new List<int>();
				for ( int i = 0; i < refFile.TopGrid.Length; i++ )
				{
					if ( vox.TopGrid[ i ] != refFile.TopGrid[ i ] )
					{
						topDiffs.Add( i );
					}
				}
				TestContext.WriteLine( $"{sceneName} voxel content: bricks {vox.FreeBrickPtr}, grid mismatches {gridDiffs.Count}/{refFile.Grid.Length}, brick mismatches {brickDiffs.Count}/{refFile.Brick.Length}, topgrid mismatches {topDiffs.Count}/{refFile.TopGrid.Length}" );
				Assert.AreEqual( 0, gridDiffs.Count, $"grid mismatch: {Describe( gridDiffs )}" );
				Assert.AreEqual( 0, brickDiffs.Count, $"brick mismatch: {Describe( brickDiffs )}" );
				Assert.AreEqual( 0, topDiffs.Count, $"topgrid mismatch: {Describe( topDiffs )}" );
			}
			finally
			{
				vox.Dispose();
			}
		}

		/// <summary>
		/// The ported three-level DDA over all 65536 object-space reference rays: hit distance,
		/// voxel value, step count, surface normal and the two occlusion queries. Both sides run the
		/// same integer grid walk over identical data, so everything is expected to match exactly;
		/// there is no grazing tolerance to fall back on as there is for triangles.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void Intersect_MatchesReference( string sceneName )
		{
			VoxDumpFile refFile = RequireData( sceneName );
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			int rayCount = refFile.Rays.Length;
			NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> hits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			NativeArray<int> steps = new NativeArray<int>( rayCount, Allocator.Persistent );
			NativeArray<float3> normals = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<int> occludedFull = new NativeArray<int>( rayCount, Allocator.Persistent );
			NativeArray<int> occludedHalf = new NativeArray<int>( rayCount, Allocator.Persistent );
			VoxelSet vox = VoxelSet.Create( Allocator.Persistent );
			try
			{
				BuildContent( &vox );
				for ( int i = 0; i < rayCount; i++ )
				{
					origins[ i ] = refFile.Rays[ i ].O;
					directions[ i ] = refFile.Rays[ i ].D;
				}

				IntersectJob job = new IntersectJob
				{
					Vox = &vox,
					Origins = origins,
					Directions = directions,
					Hits = hits,
					Steps = steps,
					Normals = normals,
					OccludedFull = occludedFull,
					OccludedHalf = occludedHalf
				};
				job.Schedule( rayCount, 64 ).Complete();

				int mismatches = 0, stepMismatches = 0, normalMismatches = 0, fullMismatches = 0, halfMismatches = 0, hitCount = 0;
				for ( int i = 0; i < rayCount; i++ )
				{
					VoxDumpFile.VoxelRay r = refFile.Rays[ i ];
					Intersection hit = hits[ i ];
					if ( r.T < BvhConstants.Far )
					{
						hitCount++;
					}
					if ( !BitsEqual( hit.T, r.T ) || hit.Prim != r.Prim )
					{
						if ( mismatches < 3 )
						{
							TestContext.WriteLine( $"  mismatch ray {i}: O {r.O} D {r.D} ref t {r.T} prim {r.Prim} steps {r.Steps} | got t {hit.T} prim {hit.Prim} steps {steps[ i ]}" );
						}
						mismatches++;
					}
					if ( steps[ i ] != ( int )r.Steps )
					{
						stepMismatches++;
					}
					if ( !BitsEqual( normals[ i ], r.Normal ) )
					{
						normalMismatches++;
					}
					if ( ( occludedFull[ i ] != 0 ) != ( r.OccludedFull != 0 ) )
					{
						fullMismatches++;
					}
					if ( ( occludedHalf[ i ] != 0 ) != ( r.OccludedHalf != 0 ) )
					{
						halfMismatches++;
					}
				}
				TestContext.WriteLine( $"{sceneName} voxel rays: {hitCount}/{rayCount} hit, mismatches {mismatches}, steps {stepMismatches}, normals {normalMismatches}, occlusion full {fullMismatches} half {halfMismatches}" );
				Assert.AreEqual( 0, mismatches, "intersect mismatches" );
				Assert.AreEqual( 0, stepMismatches, "step count mismatches" );
				Assert.AreEqual( 0, normalMismatches, "normal mismatches" );
				Assert.AreEqual( 0, fullMismatches, "full-length occlusion mismatches" );
				Assert.AreEqual( 0, halfMismatches, "half-distance occlusion mismatches" );
			}
			finally
			{
				vox.Dispose();
				origins.Dispose();
				directions.Dispose();
				hits.Dispose();
				steps.Dispose();
				normals.Dispose();
				occludedFull.Dispose();
				occludedHalf.Dispose();
			}
		}

		/// <summary>
		/// The mixed TLAS of the dump: a BLAS list of { triangle BVH over the scene, voxel set } and
		/// three instances, one of which is the triangle BVH and two of which are the voxel set at
		/// different scales. Checks the instance records the build leaves behind, the tree, and all
		/// 65536 rays. The grazing tolerance only applies to hits on the triangle instance; a hit on
		/// a voxel instance is an integer grid walk on both sides and has to match exactly.
		/// </summary>
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public unsafe void VoxelTlas_MatchesReference( string sceneName )
		{
			string binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			if ( !File.Exists( binPath ) )
			{
				Assert.Ignore( $"missing {binPath}; run TestData/fetch.ps1" );
			}
			VoxDumpFile refFile = RequireData( sceneName );
			Assert.IsTrue( BvhBurst.IsActive, "Burst direct calls fell back to Mono; check Logs/test-run.log for Burst errors" );

			int instCount = refFile.TlasInstances.Length;
			int rayCount = refFile.TlasRays.Length;
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			NativeArray<Bvh> bvhArray = new NativeArray<Bvh>( 1, Allocator.Persistent );
			NativeArray<VoxelSet> voxArray = new NativeArray<VoxelSet>( 1, Allocator.Persistent );
			NativeArray<BlasRef> blasList = new NativeArray<BlasRef>( 2, Allocator.Persistent );
			NativeArray<BlasInstance> instances = new NativeArray<BlasInstance>( instCount, Allocator.Persistent );
			NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
			NativeArray<Intersection> hits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
			NativeArray<int> occludedFull = new NativeArray<int>( rayCount, Allocator.Persistent );
			Bvh blas = Bvh.Create( Allocator.Persistent );
			VoxelSet vox = VoxelSet.Create( Allocator.Persistent );
			Bvh tlas = Bvh.Create( Allocator.Persistent );
			try
			{
				Assert.AreEqual( refFile.TriCount, triCount, "triCount" );
				blas.Build( verts, triCount );
				BuildContent( &vox );
				// The BLAS list is { &bvh, &voxels }, and the voxel set is copied into its holder
				// only after the fill, which may have reallocated the brick pool.
				bvhArray[ 0 ] = blas;
				voxArray[ 0 ] = vox;
				blasList[ 0 ] = BlasRef.From( ( Bvh* )bvhArray.GetUnsafePtr() );
				blasList[ 1 ] = BlasRef.From( ( VoxelSet* )voxArray.GetUnsafePtr() );

				SetupInstancesJob setup = new SetupInstancesJob
				{
					AabbMin = blas.AabbMin,
					AabbMax = blas.AabbMax,
					Inst = ( BlasInstance* )instances.GetUnsafePtr()
				};
				setup.Schedule().Complete();

				tlas.BuildTlas( ( BlasInstance* )instances.GetUnsafePtr(), ( uint )instCount,
					( BlasRef* )blasList.GetUnsafePtr(), ( uint )blasList.Length );

				// instance records, as the setup and the build left them.
				for ( int i = 0; i < instCount; i++ )
				{
					BlasInstance built = instances[ i ];
					VoxDumpFile.InstanceRecord expected = refFile.TlasInstances[ i ];
					Assert.AreEqual( expected.BlasIdx, built.BlasIdx, $"instance {i} blasIdx" );
					Assert.AreEqual( expected.Mask, built.Mask, $"instance {i} mask" );
					for ( int c = 0; c < 16; c++ )
					{
						Assert.IsTrue( BitsEqual( built.Transform[ c ], expected.Transform[ c ] ), $"instance {i} transform[{c}]: got {built.Transform[ c ]}, expected {expected.Transform[ c ]}" );
						Assert.IsTrue( BitsEqual( built.InvTransform[ c ], expected.InvTransform[ c ] ), $"instance {i} invTransform[{c}]: got {built.InvTransform[ c ]}, expected {expected.InvTransform[ c ]}" );
					}
					Assert.IsTrue( BitsEqual( built.AabbMin, expected.AabbMin ), $"instance {i} aabbMin: got {built.AabbMin}, expected {expected.AabbMin}" );
					Assert.IsTrue( BitsEqual( built.AabbMax, expected.AabbMax ), $"instance {i} aabbMax: got {built.AabbMax}, expected {expected.AabbMax}" );
				}

				// the TLAS itself.
				Assert.AreEqual( refFile.TlasUsedNodes, tlas.UsedNodes, "TLAS UsedNodes" );
				Assert.IsTrue( BitsEqual( tlas.AabbMin, refFile.TlasAabbMin ), $"TLAS aabbMin: got {tlas.AabbMin}, expected {refFile.TlasAabbMin}" );
				Assert.IsTrue( BitsEqual( tlas.AabbMax, refFile.TlasAabbMax ), $"TLAS aabbMax: got {tlas.AabbMax}, expected {refFile.TlasAabbMax}" );
				List<int> nodeDiffs = new List<int>();
				for ( int i = 0; i < refFile.TlasNodes.Length; i++ )
				{
					if ( !BvhNodesEqual( tlas.Nodes[ i ], refFile.TlasNodes[ i ] ) )
					{
						nodeDiffs.Add( i );
					}
				}
				Assert.AreEqual( 0, nodeDiffs.Count, $"TLAS node mismatch: {Describe( nodeDiffs )}" );
				Assert.AreEqual( refFile.TlasPrimIdx.Length, ( int )tlas.IdxCount, "TLAS IdxCount" );
				for ( int i = 0; i < refFile.TlasPrimIdx.Length; i++ )
				{
					Assert.AreEqual( refFile.TlasPrimIdx[ i ], tlas.PrimIdx[ i ], $"TLAS primIdx[{i}]" );
				}

				// traversal.
				for ( int i = 0; i < rayCount; i++ )
				{
					origins[ i ] = refFile.TlasRays[ i ].O;
					directions[ i ] = refFile.TlasRays[ i ].D;
				}
				TlasIntersectJob job = new TlasIntersectJob
				{
					Tlas = tlas,
					Origins = origins,
					Directions = directions,
					Hits = hits,
					OccludedFull = occludedFull
				};
				job.Schedule( rayCount, 64 ).Complete();

				int mismatches = 0, grazing = 0, occMismatches = 0;
				int[] perInstance = new int[ instCount ];
				for ( int i = 0; i < rayCount; i++ )
				{
					RefDumpFile.TlasRayHit rh = refFile.TlasRays[ i ];
					Intersection hit = hits[ i ];
					bool same = BitsEqual( hit.T, rh.T ) && BitsEqual( hit.U, rh.U ) && BitsEqual( hit.V, rh.V )
						&& hit.Prim == rh.Prim && hit.Inst == rh.Inst;
					if ( !same )
					{
						// Instance 0 is the triangle BVH; instances 1 and 2 are the voxel set, where
						// last-bit rounding cannot flip a hit into a miss the way it can on an edge.
						bool refTriHit = rh.T < BvhConstants.Far && rh.Inst == 0;
						bool gotTriHit = hit.T < BvhConstants.Far && hit.Inst == 0;
						if ( ( refTriHit && IsGrazing( rh.U, rh.V ) ) || ( gotTriHit && IsGrazing( hit.U, hit.V ) ) )
						{
							grazing++;
						}
						else
						{
							if ( mismatches < 3 )
							{
								TestContext.WriteLine( $"  mismatch ray {i}: O {rh.O} D {rh.D} ref t {rh.T} u {rh.U} v {rh.V} prim {rh.Prim} inst {rh.Inst} | got t {hit.T} u {hit.U} v {hit.V} prim {hit.Prim} inst {hit.Inst}" );
							}
							mismatches++;
						}
					}
					if ( ( occludedFull[ i ] != 0 ) != ( rh.OccludedFull != 0 ) )
					{
						occMismatches++;
					}
					if ( hit.T < BvhConstants.Far && hit.Inst < instCount )
					{
						perInstance[ hit.Inst ]++;
					}
				}
				TestContext.WriteLine( $"{sceneName} voxel tlas: nodes {tlas.UsedNodes}, mismatches {mismatches}/{rayCount}, grazing {grazing}, occlusion mismatches {occMismatches}, per instance [{string.Join( ", ", perInstance )}]" );
				Assert.AreEqual( 0, mismatches, "voxel tlas intersect mismatches" );
				Assert.AreEqual( 0, occMismatches, "voxel tlas occlusion mismatches" );
			}
			finally
			{
				tlas.Dispose();
				vox.Dispose();
				blas.Dispose();
				verts.Dispose();
				bvhArray.Dispose();
				voxArray.Dispose();
				blasList.Dispose();
				instances.Dispose();
				origins.Dispose();
				directions.Dispose();
				hits.Dispose();
				occludedFull.Dispose();
			}
		}
	}
}
