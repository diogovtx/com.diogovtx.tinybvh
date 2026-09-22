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
	/// Tests for Bvh4Cpu.Save/Load, Bvh8Cpu.Save/Load and BvhCwbvh.Save/Load, using the bunny test
	/// scene. Each layout is built, saved, loaded into a freshly created struct, checked for
	/// byte-identical raw blocks, and traced with the BLAS rays through both the original and the
	/// loaded struct to confirm bit-exact traversal. The comparison is between the original and the
	/// loaded struct, both traced inside the same Burst job, so it holds regardless of whether Burst
	/// direct calls (used by the builders/converters) fell back to Mono on this machine.
	/// </summary>
	public class BvhWideSaveLoadTests
	{
		const string SceneName = "bunny";

		static bool TryGetPaths( out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( SceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( SceneName + ".ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		static bool BitsEqual( float a, float b )
		{
			return math.asuint( a ) == math.asuint( b );
		}

		[BurstCompile( CompileSynchronously = true )]
		private struct Bvh4CpuCompareJob : IJobParallelFor
		{
			public Bvh4Cpu Original;
			public Bvh4Cpu Loaded;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> OriginalHits;
			public NativeArray<Intersection> LoadedHits;

			public void Execute( int i )
			{
				Ray rayA = new Ray( Origins[ i ], Directions[ i ] );
				Original.Intersect( ref rayA );
				OriginalHits[ i ] = rayA.Hit;
				Ray rayB = new Ray( Origins[ i ], Directions[ i ] );
				Loaded.Intersect( ref rayB );
				LoadedHits[ i ] = rayB.Hit;
			}
		}

		[BurstCompile( CompileSynchronously = true )]
		private struct Bvh8CpuCompareJob : IJobParallelFor
		{
			public Bvh8Cpu Original;
			public Bvh8Cpu Loaded;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> OriginalHits;
			public NativeArray<Intersection> LoadedHits;

			public void Execute( int i )
			{
				Ray rayA = new Ray( Origins[ i ], Directions[ i ] );
				Original.Intersect( ref rayA );
				OriginalHits[ i ] = rayA.Hit;
				Ray rayB = new Ray( Origins[ i ], Directions[ i ] );
				Loaded.Intersect( ref rayB );
				LoadedHits[ i ] = rayB.Hit;
			}
		}

		[BurstCompile( CompileSynchronously = true )]
		private struct CwbvhCompareJob : IJobParallelFor
		{
			public BvhCwbvh Original;
			public BvhCwbvh Loaded;
			[ReadOnly] public NativeArray<float3> Origins;
			[ReadOnly] public NativeArray<float3> Directions;
			public NativeArray<Intersection> OriginalHits;
			public NativeArray<Intersection> LoadedHits;

			public void Execute( int i )
			{
				Ray rayA = new Ray( Origins[ i ], Directions[ i ] );
				Original.Intersect( ref rayA );
				OriginalHits[ i ] = rayA.Hit;
				Ray rayB = new Ray( Origins[ i ], Directions[ i ] );
				Loaded.Intersect( ref rayB );
				LoadedHits[ i ] = rayB.Hit;
			}
		}

		/// <summary>Traces the BLAS rays through both structs in one Burst job and asserts every hit is bit-exact.</summary>
		static void AssertHitsBitExact( Intersection[] originalHits, Intersection[] loadedHits )
		{
			for ( int i = 0; i < originalHits.Length; i++ )
			{
				Intersection a = originalHits[ i ];
				Intersection b = loadedHits[ i ];
				Assert.IsTrue( BitsEqual( a.T, b.T ), $"ray {i} T" );
				Assert.IsTrue( BitsEqual( a.U, b.U ), $"ray {i} U" );
				Assert.IsTrue( BitsEqual( a.V, b.V ), $"ray {i} V" );
				Assert.AreEqual( a.Prim, b.Prim, $"ray {i} Prim" );
			}
		}

		[Test]
		public unsafe void Bvh4Cpu_SaveLoad_RoundTrips()
		{
			if ( !TryGetPaths( out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}
			if ( !BvhBurst.IsActive )
			{
				TestContext.WriteLine( "Burst direct calls are inactive; the round trip still holds since both sides trace the same saved bytes." );
			}

			string savePath = Path.GetTempFileName();
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh4Cpu original = Bvh4Cpu.Create( Allocator.Persistent );
			Bvh4Cpu loaded = Bvh4Cpu.Create( Allocator.Persistent );
			try
			{
				original.Build( verts, triCount );
				original.Save( savePath );

				bool ok = loaded.Load( savePath, triCount );

				Assert.IsTrue( ok, "Load" );
				Assert.AreEqual( original.UsedBlocks, loaded.UsedBlocks, "UsedBlocks" );
				Assert.AreEqual( original.TriCount, loaded.TriCount, "TriCount" );
				Assert.AreEqual( original.IdxCount, loaded.IdxCount, "IdxCount" );
				Assert.AreEqual( original.Refittable, loaded.Refittable, "Refittable" );
				Assert.AreEqual( original.MayHaveHoles, loaded.MayHaveHoles, "MayHaveHoles" );
				Assert.AreEqual( original.BvhOverAabbs, loaded.BvhOverAabbs, "BvhOverAabbs" );
				Assert.AreEqual( original.BvhOverIndices, loaded.BvhOverIndices, "BvhOverIndices" );
				Assert.AreEqual( original.AabbMin, loaded.AabbMin, "AabbMin" );
				Assert.AreEqual( original.AabbMax, loaded.AabbMax, "AabbMax" );

				long dataBytes = ( long )original.UsedBlocks * 64;
				Assert.AreEqual( 0, UnsafeUtility.MemCmp( original.Data, loaded.Data, dataBytes ), "Data bytes differ" );

				RefDumpFile refFile = RefDumpFile.Load( refPath );
				int rayCount = refFile.Rays.Length;
				NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
				NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
				NativeArray<Intersection> originalHits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
				NativeArray<Intersection> loadedHits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
				try
				{
					for ( int i = 0; i < rayCount; i++ )
					{
						origins[ i ] = refFile.Rays[ i ].O;
						directions[ i ] = refFile.Rays[ i ].D;
					}
					Bvh4CpuCompareJob job = new Bvh4CpuCompareJob
					{
						Original = original,
						Loaded = loaded,
						Origins = origins,
						Directions = directions,
						OriginalHits = originalHits,
						LoadedHits = loadedHits
					};
					job.Schedule( rayCount, 64 ).Complete();
					AssertHitsBitExact( originalHits.ToArray(), loadedHits.ToArray() );
				}
				finally
				{
					origins.Dispose();
					directions.Dispose();
					originalHits.Dispose();
					loadedHits.Dispose();
				}
			}
			finally
			{
				original.Dispose();
				loaded.Dispose();
				verts.Dispose();
				if ( File.Exists( savePath ) )
				{
					File.Delete( savePath );
				}
			}
		}

		[Test]
		public unsafe void Bvh4Cpu_Load_RejectsWrongTriangleCount()
		{
			if ( !TryGetPaths( out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			string savePath = Path.GetTempFileName();
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh4Cpu original = Bvh4Cpu.Create( Allocator.Persistent );
			Bvh4Cpu loaded = Bvh4Cpu.Create( Allocator.Persistent );
			try
			{
				original.Build( verts, triCount );
				original.Save( savePath );

				bool ok = loaded.Load( savePath, triCount - 1 );

				Assert.IsFalse( ok, "Load should reject a triangle count mismatch" );
				Assert.IsTrue( loaded.Data == null, "rejected Load must leave Data untouched" );
				Assert.AreEqual( 0u, loaded.UsedBlocks, "rejected Load must leave UsedBlocks untouched" );
			}
			finally
			{
				original.Dispose();
				loaded.Dispose();
				verts.Dispose();
				if ( File.Exists( savePath ) )
				{
					File.Delete( savePath );
				}
			}
		}

		[Test]
		public unsafe void Bvh4Cpu_Load_RejectsCorruptedMagic()
		{
			if ( !TryGetPaths( out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			string savePath = Path.GetTempFileName();
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh4Cpu original = Bvh4Cpu.Create( Allocator.Persistent );
			Bvh4Cpu loaded = Bvh4Cpu.Create( Allocator.Persistent );
			try
			{
				original.Build( verts, triCount );
				original.Save( savePath );

				byte[] bytes = File.ReadAllBytes( savePath );
				bytes[ 0 ] = ( byte )~bytes[ 0 ];
				File.WriteAllBytes( savePath, bytes );

				Assert.IsFalse( loaded.Load( savePath, triCount ), "Load should reject a corrupted magic" );
				Assert.IsTrue( loaded.Data == null, "rejected Load must leave Data untouched" );
				Assert.AreEqual( 0u, loaded.UsedBlocks, "rejected Load must leave UsedBlocks untouched" );
				Assert.IsFalse( loaded.Load( savePath + ".missing", triCount ), "Load should reject a missing file" );
			}
			finally
			{
				original.Dispose();
				loaded.Dispose();
				verts.Dispose();
				if ( File.Exists( savePath ) )
				{
					File.Delete( savePath );
				}
			}
		}

		[Test]
		public unsafe void Bvh8Cpu_SaveLoad_RoundTrips()
		{
			if ( !TryGetPaths( out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}
			if ( !BvhBurst.IsActive )
			{
				TestContext.WriteLine( "Burst direct calls are inactive; the round trip still holds since both sides trace the same saved bytes." );
			}

			string savePath = Path.GetTempFileName();
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh8Cpu original = Bvh8Cpu.Create( Allocator.Persistent );
			Bvh8Cpu loaded = Bvh8Cpu.Create( Allocator.Persistent );
			try
			{
				original.Build( verts, triCount );
				original.Save( savePath );

				bool ok = loaded.Load( savePath, triCount );

				Assert.IsTrue( ok, "Load" );
				Assert.AreEqual( original.UsedBlocks, loaded.UsedBlocks, "UsedBlocks" );
				Assert.AreEqual( original.TriCount, loaded.TriCount, "TriCount" );
				Assert.AreEqual( original.IdxCount, loaded.IdxCount, "IdxCount" );
				Assert.AreEqual( original.Refittable, loaded.Refittable, "Refittable" );
				Assert.AreEqual( original.MayHaveHoles, loaded.MayHaveHoles, "MayHaveHoles" );
				Assert.AreEqual( original.BvhOverAabbs, loaded.BvhOverAabbs, "BvhOverAabbs" );
				Assert.AreEqual( original.BvhOverIndices, loaded.BvhOverIndices, "BvhOverIndices" );
				Assert.AreEqual( original.AabbMin, loaded.AabbMin, "AabbMin" );
				Assert.AreEqual( original.AabbMax, loaded.AabbMax, "AabbMax" );

				long dataBytes = ( long )original.UsedBlocks * 64;
				Assert.AreEqual( 0, UnsafeUtility.MemCmp( original.Data, loaded.Data, dataBytes ), "Data bytes differ" );

				RefDumpFile refFile = RefDumpFile.Load( refPath );
				int rayCount = refFile.Rays.Length;
				NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
				NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
				NativeArray<Intersection> originalHits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
				NativeArray<Intersection> loadedHits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
				try
				{
					for ( int i = 0; i < rayCount; i++ )
					{
						origins[ i ] = refFile.Rays[ i ].O;
						directions[ i ] = refFile.Rays[ i ].D;
					}
					Bvh8CpuCompareJob job = new Bvh8CpuCompareJob
					{
						Original = original,
						Loaded = loaded,
						Origins = origins,
						Directions = directions,
						OriginalHits = originalHits,
						LoadedHits = loadedHits
					};
					job.Schedule( rayCount, 64 ).Complete();
					AssertHitsBitExact( originalHits.ToArray(), loadedHits.ToArray() );
				}
				finally
				{
					origins.Dispose();
					directions.Dispose();
					originalHits.Dispose();
					loadedHits.Dispose();
				}
			}
			finally
			{
				original.Dispose();
				loaded.Dispose();
				verts.Dispose();
				if ( File.Exists( savePath ) )
				{
					File.Delete( savePath );
				}
			}
		}

		[Test]
		public unsafe void Bvh8Cpu_Load_RejectsWrongTriangleCount()
		{
			if ( !TryGetPaths( out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			string savePath = Path.GetTempFileName();
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh8Cpu original = Bvh8Cpu.Create( Allocator.Persistent );
			Bvh8Cpu loaded = Bvh8Cpu.Create( Allocator.Persistent );
			try
			{
				original.Build( verts, triCount );
				original.Save( savePath );

				bool ok = loaded.Load( savePath, triCount - 1 );

				Assert.IsFalse( ok, "Load should reject a triangle count mismatch" );
				Assert.IsTrue( loaded.Data == null, "rejected Load must leave Data untouched" );
				Assert.AreEqual( 0u, loaded.UsedBlocks, "rejected Load must leave UsedBlocks untouched" );
			}
			finally
			{
				original.Dispose();
				loaded.Dispose();
				verts.Dispose();
				if ( File.Exists( savePath ) )
				{
					File.Delete( savePath );
				}
			}
		}

		[Test]
		public unsafe void Bvh8Cpu_Load_RejectsCorruptedMagic()
		{
			if ( !TryGetPaths( out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			string savePath = Path.GetTempFileName();
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh8Cpu original = Bvh8Cpu.Create( Allocator.Persistent );
			Bvh8Cpu loaded = Bvh8Cpu.Create( Allocator.Persistent );
			try
			{
				original.Build( verts, triCount );
				original.Save( savePath );

				byte[] bytes = File.ReadAllBytes( savePath );
				bytes[ 0 ] = ( byte )~bytes[ 0 ];
				File.WriteAllBytes( savePath, bytes );

				Assert.IsFalse( loaded.Load( savePath, triCount ), "Load should reject a corrupted magic" );
				Assert.IsTrue( loaded.Data == null, "rejected Load must leave Data untouched" );
				Assert.AreEqual( 0u, loaded.UsedBlocks, "rejected Load must leave UsedBlocks untouched" );
				Assert.IsFalse( loaded.Load( savePath + ".missing", triCount ), "Load should reject a missing file" );
			}
			finally
			{
				original.Dispose();
				loaded.Dispose();
				verts.Dispose();
				if ( File.Exists( savePath ) )
				{
					File.Delete( savePath );
				}
			}
		}

		[Test]
		public unsafe void Cwbvh_SaveLoad_RoundTrips()
		{
			if ( !TryGetPaths( out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}
			if ( !BvhBurst.IsActive )
			{
				TestContext.WriteLine( "Burst direct calls are inactive; the round trip still holds since both sides trace the same saved bytes." );
			}

			string savePath = Path.GetTempFileName();
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh8 = Mbvh.Create( 8, Allocator.Persistent );
			BvhCwbvh original = BvhCwbvh.Create( Allocator.Persistent );
			BvhCwbvh loaded = BvhCwbvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				bvh.Compact();
				bvh.SplitLeafs( 3 );
				mbvh8.ConvertFrom( ref bvh, true );
				original.ConvertFrom( ref mbvh8, true );
				original.Save( savePath );

				bool ok = loaded.Load( savePath, triCount );

				Assert.IsTrue( ok, "Load" );
				Assert.AreEqual( original.UsedBlocks, loaded.UsedBlocks, "UsedBlocks" );
				Assert.AreEqual( original.TriBlocks, loaded.TriBlocks, "TriBlocks" );
				Assert.AreEqual( original.TriCount, loaded.TriCount, "TriCount" );
				Assert.AreEqual( original.IdxCount, loaded.IdxCount, "IdxCount" );
				Assert.AreEqual( original.Refittable, loaded.Refittable, "Refittable" );
				Assert.AreEqual( original.MayHaveHoles, loaded.MayHaveHoles, "MayHaveHoles" );
				Assert.AreEqual( original.BvhOverAabbs, loaded.BvhOverAabbs, "BvhOverAabbs" );
				Assert.AreEqual( original.BvhOverIndices, loaded.BvhOverIndices, "BvhOverIndices" );
				Assert.AreEqual( original.AabbMin, loaded.AabbMin, "AabbMin" );
				Assert.AreEqual( original.AabbMax, loaded.AabbMax, "AabbMax" );

				long nodeBytes = ( long )original.UsedBlocks * 16;
				Assert.AreEqual( 0, UnsafeUtility.MemCmp( original.Data, loaded.Data, nodeBytes ), "Data bytes differ" );
				long triBytes = ( long )original.TriBlocks * 16;
				Assert.AreEqual( 0, UnsafeUtility.MemCmp( original.Tris, loaded.Tris, triBytes ), "Tris bytes differ" );

				RefDumpFile refFile = RefDumpFile.Load( refPath );
				int rayCount = refFile.Rays.Length;
				NativeArray<float3> origins = new NativeArray<float3>( rayCount, Allocator.Persistent );
				NativeArray<float3> directions = new NativeArray<float3>( rayCount, Allocator.Persistent );
				NativeArray<Intersection> originalHits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
				NativeArray<Intersection> loadedHits = new NativeArray<Intersection>( rayCount, Allocator.Persistent );
				try
				{
					for ( int i = 0; i < rayCount; i++ )
					{
						origins[ i ] = refFile.Rays[ i ].O;
						directions[ i ] = refFile.Rays[ i ].D;
					}
					CwbvhCompareJob job = new CwbvhCompareJob
					{
						Original = original,
						Loaded = loaded,
						Origins = origins,
						Directions = directions,
						OriginalHits = originalHits,
						LoadedHits = loadedHits
					};
					job.Schedule( rayCount, 64 ).Complete();
					AssertHitsBitExact( originalHits.ToArray(), loadedHits.ToArray() );
				}
				finally
				{
					origins.Dispose();
					directions.Dispose();
					originalHits.Dispose();
					loadedHits.Dispose();
				}
			}
			finally
			{
				original.Dispose();
				loaded.Dispose();
				mbvh8.Dispose();
				bvh.Dispose();
				verts.Dispose();
				if ( File.Exists( savePath ) )
				{
					File.Delete( savePath );
				}
			}
		}

		[Test]
		public unsafe void Cwbvh_Load_RejectsWrongTriangleCount()
		{
			if ( !TryGetPaths( out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			string savePath = Path.GetTempFileName();
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh8 = Mbvh.Create( 8, Allocator.Persistent );
			BvhCwbvh original = BvhCwbvh.Create( Allocator.Persistent );
			BvhCwbvh loaded = BvhCwbvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				bvh.Compact();
				bvh.SplitLeafs( 3 );
				mbvh8.ConvertFrom( ref bvh, true );
				original.ConvertFrom( ref mbvh8, true );
				original.Save( savePath );

				bool ok = loaded.Load( savePath, triCount - 1 );

				Assert.IsFalse( ok, "Load should reject a triangle count mismatch" );
				Assert.IsTrue( loaded.Data == null, "rejected Load must leave Data untouched" );
				Assert.AreEqual( 0u, loaded.UsedBlocks, "rejected Load must leave UsedBlocks untouched" );
			}
			finally
			{
				original.Dispose();
				loaded.Dispose();
				mbvh8.Dispose();
				bvh.Dispose();
				verts.Dispose();
				if ( File.Exists( savePath ) )
				{
					File.Delete( savePath );
				}
			}
		}

		[Test]
		public unsafe void Cwbvh_Load_RejectsCorruptedMagic()
		{
			if ( !TryGetPaths( out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			string savePath = Path.GetTempFileName();
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			Mbvh mbvh8 = Mbvh.Create( 8, Allocator.Persistent );
			BvhCwbvh original = BvhCwbvh.Create( Allocator.Persistent );
			BvhCwbvh loaded = BvhCwbvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				bvh.Compact();
				bvh.SplitLeafs( 3 );
				mbvh8.ConvertFrom( ref bvh, true );
				original.ConvertFrom( ref mbvh8, true );
				original.Save( savePath );

				byte[] bytes = File.ReadAllBytes( savePath );
				bytes[ 0 ] = ( byte )~bytes[ 0 ];
				File.WriteAllBytes( savePath, bytes );

				Assert.IsFalse( loaded.Load( savePath, triCount ), "Load should reject a corrupted magic" );
				Assert.IsTrue( loaded.Data == null, "rejected Load must leave Data untouched" );
				Assert.AreEqual( 0u, loaded.UsedBlocks, "rejected Load must leave UsedBlocks untouched" );
				Assert.IsFalse( loaded.Load( savePath + ".missing", triCount ), "Load should reject a missing file" );
			}
			finally
			{
				original.Dispose();
				loaded.Dispose();
				mbvh8.Dispose();
				bvh.Dispose();
				verts.Dispose();
				if ( File.Exists( savePath ) )
				{
					File.Delete( savePath );
				}
			}
		}
	}
}
