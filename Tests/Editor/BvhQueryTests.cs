using System.IO;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Tests for the query functions ported after the ".ref" format was frozen - sphere overlap,
	/// the EPO cost metric and 256-ray packet traversal - against the ".feat.ref" dump produced by
	/// Tools/RefDump/featdump.cpp. Every reference tree in this file is the plain binned build,
	/// which the port already reproduces bit for bit, so any difference found here belongs to the
	/// query itself. Tests are ignored when reference data is missing.
	/// </summary>
	public class BvhQueryTests
	{
		/// <summary>Barycentric distance to a triangle edge below which a hit counts as grazing.</summary>
		const float GrazingEps = 1e-3f;

		/// <summary>True when the hit lies within GrazingEps of a triangle edge or vertex, where last-bit rounding decides hit or miss.</summary>
		static bool IsGrazing( float u, float v )
		{
			return u < GrazingEps || v < GrazingEps || ( 1f - u - v ) < GrazingEps;
		}

		static bool BitsEqual( float a, float b )
		{
			return math.asuint( a ) == math.asuint( b );
		}

		static bool TryGetPaths( string sceneName, out string binPath, out string featPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			featPath = BvhSceneFile.TestDataPath( sceneName + ".feat.ref" );
			return File.Exists( binPath ) && File.Exists( featPath );
		}

		static void RequireData( string sceneName, out string binPath, out string featPath )
		{
			if ( !TryGetPaths( sceneName, out binPath, out featPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {featPath}; run TestData/fetch.ps1 to fetch scenes and Tools/RefDump/run_all.bat to generate reference data" );
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void IntersectSphere_MatchesReference( string sceneName )
		{
			RequireData( sceneName, out string binPath, out string featPath );

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				FeatDumpFile featFile = FeatDumpFile.Load( featPath );
				bvh.Build( verts, triCount );

				int mismatches = 0, hits = 0;
				for ( int i = 0; i < featFile.Spheres.Length; i++ )
				{
					FeatDumpFile.SphereQuery query = featFile.Spheres[ i ];
					bool hit = bvh.IntersectSphere( query.Pos, query.R );
					if ( hit )
					{
						hits++;
					}
					if ( hit != ( query.Hit != 0 ) )
					{
						if ( mismatches < 3 )
						{
							TestContext.WriteLine( $"  mismatch sphere {i}: pos {query.Pos} r {query.R}, ref {query.Hit != 0}, got {hit}" );
						}
						mismatches++;
					}
				}

				TestContext.WriteLine( $"{sceneName} spheres: {hits}/{featFile.Spheres.Length} overlap, mismatches {mismatches}" );
				Assert.AreEqual( 0, mismatches, "sphere query mismatches" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void EpoCost_MatchesReference( string sceneName )
		{
			RequireData( sceneName, out string binPath, out string featPath );

			FeatDumpFile featFile = FeatDumpFile.Load( featPath );
			if ( !featFile.HasEpo )
			{
				Assert.Ignore( $"{sceneName}: featdump.cpp skips EPOCost above 100k triangles ({featFile.TriCount} tris)" );
			}

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			try
			{
				bvh.Build( verts, triCount );
				float sah = bvh.SahCost();
				float epo = bvh.EpoCost();

				TestContext.WriteLine( $"{sceneName}: SAH {sah} (ref {featFile.SahBinned}), EPO {epo} (ref {featFile.EpoBinned})" );
				TestContext.WriteLine( $"{sceneName}: EPO bits {math.asuint( epo ):x8} vs ref {math.asuint( featFile.EpoBinned ):x8}, relative delta {( epo - featFile.EpoBinned ) / featFile.EpoBinned:E3}" );

				Assert.IsTrue( BitsEqual( sah, featFile.SahBinned ), $"SAH cost of the binned build: got {sah}, expected {featFile.SahBinned}" );
				Assert.IsTrue( BitsEqual( epo, featFile.EpoBinned ), $"EPO cost of the binned build: got {epo}, expected {featFile.EpoBinned}" );

				// The same metric over the BVH::BuildQuick tree.
				bvh.BuildQuick( verts, triCount );
				float epoQuick = bvh.EpoCost();
				TestContext.WriteLine( $"{sceneName}: EPO of the quick build {epoQuick} (ref {featFile.EpoQuick})" );
				Assert.IsTrue( BitsEqual( epoQuick, featFile.EpoQuick ), $"EPO cost of the quick build: got {epoQuick}, expected {featFile.EpoQuick}" );
			}
			finally
			{
				bvh.Dispose();
				verts.Dispose();
			}
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Intersect256Rays_MatchesReference( string sceneName )
		{
			RequireData( sceneName, out string binPath, out string featPath );

			NativeArray<float4> verts = BvhSceneFile.Load( binPath, Allocator.Persistent, out uint triCount );
			Bvh bvh = Bvh.Create( Allocator.Persistent );
			NativeArray<Ray> packet = new NativeArray<Ray>( Bvh.PacketSize, Allocator.Persistent );
			try
			{
				FeatDumpFile featFile = FeatDumpFile.Load( featPath );
				bvh.Build( verts, triCount );

				int rayCount = 0, hits = 0, mismatches = 0, grazing = 0, ties = 0;
				int exactT = 0, exactU = 0, exactV = 0;
				for ( int p = 0; p < featFile.Packets.Length; p++ )
				{
					FeatDumpFile.Packet refPacket = featFile.Packets[ p ];
					for ( int i = 0; i < Bvh.PacketSize; i++ )
					{
						// The dump holds packet[ i ].D, which featdump.cpp already normalized; the
						// Ray constructor would normalize it a second time and move it by an ulp,
						// so overwrite D and the reciprocal derived from it with the reference
						// direction, to trace exactly what the C++ traced.
						Ray ray = new Ray( refPacket.O, refPacket.Rays[ i ].D );
						ray.D = refPacket.Rays[ i ].D;
						ray.RD = BvhMath.Rcp( refPacket.Rays[ i ].D );
						packet[ i ] = ray;
					}
					bvh.Intersect256Rays( packet );

					for ( int i = 0; i < Bvh.PacketSize; i++ )
					{
						FeatDumpFile.PacketRay refRay = refPacket.Rays[ i ];
						Ray ray = packet[ i ];
						rayCount++;

						bool refHit = refRay.T < BvhConstants.Far;
						bool gotHit = ray.Hit.T < BvhConstants.Far;
						bool sameT = refHit && gotHit && math.abs( ray.Hit.T - refRay.T ) <= 1e-4f * math.max( math.abs( refRay.T ), 1e-12f );
						if ( refHit != gotHit || ( refHit && !sameT ) )
						{
							// A hit within a hair of a triangle edge or vertex is decided by
							// last-bit rounding; the two sides can legitimately disagree.
							if ( ( refHit && IsGrazing( refRay.U, refRay.V ) ) || ( gotHit && IsGrazing( ray.Hit.U, ray.Hit.V ) ) )
							{
								grazing++;
							}
							else
							{
								if ( mismatches < 3 )
								{
									TestContext.WriteLine( $"  mismatch packet {p} ray {i}: O {refPacket.O} D {refRay.D} ref t {refRay.T} u {refRay.U} v {refRay.V} prim {refRay.Prim} | got t {ray.Hit.T} u {ray.Hit.U} v {ray.Hit.V} prim {ray.Hit.Prim}" );
								}
								mismatches++;
							}
						}
						else if ( refHit && ray.Hit.Prim != refRay.Prim )
						{
							ties++;
						}

						if ( refHit && gotHit )
						{
							hits++;
							if ( BitsEqual( ray.Hit.T, refRay.T ) )
							{
								exactT++;
							}
							if ( BitsEqual( ray.Hit.U, refRay.U ) )
							{
								exactU++;
							}
							if ( BitsEqual( ray.Hit.V, refRay.V ) )
							{
								exactV++;
							}
						}
					}
				}

				TestContext.WriteLine( $"{sceneName} packets: {featFile.Packets.Length} packets, {rayCount} rays, {hits} hits, mismatches {mismatches}, grazing {grazing}, same-distance ties {ties}" );
				TestContext.WriteLine( $"{sceneName} packets: bit-exact over hits: t {exactT}/{hits}, u {exactU}/{hits}, v {exactV}/{hits}" );
				Assert.AreEqual( 0, mismatches, "packet traversal mismatches" );
				Assert.AreEqual( hits, exactT, "packet traversal bit-exact hit distances" );
				Assert.AreEqual( hits, exactU, "packet traversal bit-exact barycentric u" );
				Assert.AreEqual( hits, exactV, "packet traversal bit-exact barycentric v" );
				Assert.AreEqual( 0, ties, "packet traversal primitive index mismatches" );
			}
			finally
			{
				packet.Dispose();
				bvh.Dispose();
				verts.Dispose();
			}
		}
	}
}
