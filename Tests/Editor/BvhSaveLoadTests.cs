using System.IO;
using NUnit.Framework;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>Tests for Bvh.Save / Bvh.Load, using the suzanne test scene only.</summary>
	public class BvhSaveLoadTests
	{
		const int TracedRayCount = 1000;

		static bool TryGetPaths( out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( "suzanne.bin" );
			refPath = BvhSceneFile.TestDataPath( "suzanne.ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		/// <summary>Compares two built BVHs node-for-node and index-for-index, bit exact.</summary>
		static unsafe void AssertBvhsByteIdentical( Bvh a, Bvh b )
		{
			Assert.AreEqual( a.UsedNodes, b.UsedNodes, "UsedNodes" );
			Assert.AreEqual( a.IdxCount, b.IdxCount, "IdxCount" );
			long nodeBytes = ( long )a.UsedNodes * sizeof( BvhNode );
			Assert.AreEqual( 0, UnsafeUtility.MemCmp( a.Nodes, b.Nodes, nodeBytes ), "node bytes differ" );
			long idxBytes = ( long )a.IdxCount * sizeof( uint );
			Assert.AreEqual( 0, UnsafeUtility.MemCmp( a.PrimIdx, b.PrimIdx, idxBytes ), "PrimIdx bytes differ" );
		}

		[Test]
		public void SaveLoad_RoundTripsNodesAndIndices()
		{
			if ( !TryGetPaths( out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			string savePath = Path.GetTempFileName();
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh original = Bvh.Create( Allocator.Persistent );
			Bvh loaded = Bvh.Create( Allocator.Persistent );
			try
			{
				original.Build( verts, triCount );
				original.Save( savePath );

				bool ok = loaded.Load( savePath, verts, triCount );

				Assert.IsTrue( ok, "Load" );
				Assert.AreEqual( original.UsedNodes, loaded.UsedNodes, "UsedNodes" );
				Assert.AreEqual( original.TriCount, loaded.TriCount, "TriCount" );
				Assert.AreEqual( original.IdxCount, loaded.IdxCount, "IdxCount" );
				Assert.AreEqual( original.Refittable, loaded.Refittable, "Refittable" );
				Assert.AreEqual( original.MayHaveHoles, loaded.MayHaveHoles, "MayHaveHoles" );
				Assert.AreEqual( original.BvhOverAabbs, loaded.BvhOverAabbs, "BvhOverAabbs" );
				Assert.AreEqual( original.BvhOverIndices, loaded.BvhOverIndices, "BvhOverIndices" );
				Assert.AreEqual( original.AabbMin, loaded.AabbMin, "AabbMin" );
				Assert.AreEqual( original.AabbMax, loaded.AabbMax, "AabbMax" );
				AssertBvhsByteIdentical( original, loaded );

				RefDumpFile refFile = RefDumpFile.Load( refPath );
				for ( int i = 0; i < TracedRayCount; i++ )
				{
					RefDumpFile.RayHit rh = refFile.Rays[ i ];

					Ray originalRay = new Ray( rh.O, rh.D );
					original.Intersect( ref originalRay );
					Ray loadedRay = new Ray( rh.O, rh.D );
					loaded.Intersect( ref loadedRay );

					Assert.AreEqual( originalRay.Hit.T, loadedRay.Hit.T, $"ray {i} T" );
					Assert.AreEqual( originalRay.Hit.U, loadedRay.Hit.U, $"ray {i} U" );
					Assert.AreEqual( originalRay.Hit.V, loadedRay.Hit.V, $"ray {i} V" );
					Assert.AreEqual( originalRay.Hit.Prim, loadedRay.Hit.Prim, $"ray {i} Prim" );
					Assert.AreEqual( originalRay.Hit.Inst, loadedRay.Hit.Inst, $"ray {i} Inst" );

					bool originalOccluded = original.IsOccluded( new Ray( rh.O, rh.D ) );
					bool loadedOccluded = loaded.IsOccluded( new Ray( rh.O, rh.D ) );
					Assert.AreEqual( originalOccluded, loadedOccluded, $"ray {i} IsOccluded" );
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
		public void Load_RejectsWrongTriangleCount()
		{
			if ( !TryGetPaths( out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			string savePath = Path.GetTempFileName();
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh original = Bvh.Create( Allocator.Persistent );
			Bvh loaded = Bvh.Create( Allocator.Persistent );
			try
			{
				original.Build( verts, triCount );
				original.Save( savePath );

				NativeArray<float4> shortVerts = verts.GetSubArray( 0, ( int )( triCount - 1 ) * 3 );
				bool ok = loaded.Load( savePath, shortVerts, triCount - 1 );

				Assert.IsFalse( ok, "Load should reject a triangle count mismatch" );
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
		public void Load_RejectsCorruptedMagic()
		{
			if ( !TryGetPaths( out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			string savePath = Path.GetTempFileName();
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh original = Bvh.Create( Allocator.Persistent );
			Bvh loaded = Bvh.Create( Allocator.Persistent );
			try
			{
				original.Build( verts, triCount );
				original.Save( savePath );

				byte[] bytes = File.ReadAllBytes( savePath );
				bytes[ 0 ] = ( byte )~bytes[ 0 ];
				File.WriteAllBytes( savePath, bytes );

				Assert.IsFalse( loaded.Load( savePath, verts, triCount ), "Load should reject a corrupted magic" );
				Assert.IsFalse( loaded.Load( savePath + ".missing", verts, triCount ), "Load should reject a missing file" );
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
		public void Build_AfterLoad_MatchesFreshBuild()
		{
			if ( !TryGetPaths( out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; run TestData/fetch.ps1 and Tools/RefDump/run_all.bat" );
			}

			string savePath = Path.GetTempFileName();
			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh original = Bvh.Create( Allocator.Persistent );
			Bvh loaded = Bvh.Create( Allocator.Persistent );
			Bvh freshBuild = Bvh.Create( Allocator.Persistent );
			try
			{
				original.Build( verts, triCount );
				original.Save( savePath );
				Assert.IsTrue( loaded.Load( savePath, verts, triCount ), "Load" );

				loaded.Build( verts, triCount );
				freshBuild.Build( verts, triCount );

				Assert.AreEqual( freshBuild.UsedNodes, loaded.UsedNodes, "UsedNodes" );
				AssertBvhsByteIdentical( freshBuild, loaded );
			}
			finally
			{
				original.Dispose();
				loaded.Dispose();
				freshBuild.Dispose();
				verts.Dispose();
				if ( File.Exists( savePath ) )
				{
					File.Delete( savePath );
				}
			}
		}
	}
}
