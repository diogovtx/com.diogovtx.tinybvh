using System.IO;
using System.Text;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Reads the ".feat.ref" reference file produced by Tools/RefDump/featdump.cpp (format
	/// "TBVHFEA1"): the alternative builders, presplitting, SBVH bin settings, EPO cost, sphere
	/// queries, 256-ray packets and the stochastic optimizer. See that file's header comment for
	/// the authoritative layout and for how each tree was built. Every tree block traces the same
	/// 65536 rays as the BLAS section of the ".ref" file.
	/// </summary>
	public class FeatDumpFile
	{
		/// <summary>One built tree plus the rays traced against it.</summary>
		public class TreeBlock
		{
			public uint UsedNodes;
			public float SahCost;
			public float3 AabbMin;
			public float3 AabbMax;
			public BvhNode[] Nodes;
			/// <summary>What the tool wrote for this block: bvh.idxCount for the builders, PrimCount() for the SBVH and optimizer blocks.</summary>
			public uint[] PrimIdx;
			public RefDumpFile.RayHit[] Rays;
		}

		public struct SphereQuery
		{
			public float3 Pos;
			public float R;
			public uint Hit;
		}

		public struct PacketRay
		{
			public float3 D;
			public float T;
			public float U;
			public float V;
			public uint Prim;
		}

		/// <summary>256 rays with a shared origin, in the 4x4-of-4x4 tile order Intersect256Rays expects.</summary>
		public class Packet
		{
			public float3 O;
			public PacketRay[] Rays;
		}

		public uint TriCount;

		/// <summary>BVH::BuildQuick.</summary>
		public TreeBlock Quick;
		/// <summary>settings.useFullSweep.</summary>
		public TreeBlock FullSweep;
		/// <summary>settings.usePresplitting, binned, post pass on. PrimIdx holds bvh.idxCount entries, which exceeds TriCount.</summary>
		public TreeBlock Presplit;
		/// <summary>settings.usePresplitting with settings.useFullSweep.</summary>
		public TreeBlock PresplitFullSweep;
		/// <summary>settings.usePresplitting with presplitPostPass = false, binned.</summary>
		public TreeBlock PresplitNoPostPass;
		/// <summary>SBVH with hqbvhbins = 32 and hqbvhoddeven = true. PrimIdx holds PrimCount() entries.</summary>
		public TreeBlock HqBins;

		/// <summary>The iteration count the stochastic optimizer used; pass the same one.</summary>
		public uint StochasticIterations;
		public float StochasticSahBefore;
		/// <summary>Binned build followed by BVH::Optimize( iterations, false, true ), with MSVC rand() from its initial state; see featdump.cpp.</summary>
		public TreeBlock Stochastic;

		public float SahBinned;
		/// <summary>False for scenes where the tool skipped EPOCost (over 100k triangles).</summary>
		public bool HasEpo;
		public float EpoBinned;
		public float EpoQuick;

		/// <summary>BVH::IntersectSphere on the binned build.</summary>
		public SphereQuery[] Spheres;
		/// <summary>BVH::Intersect256Rays on the binned build.</summary>
		public Packet[] Packets;

		public static FeatDumpFile Load( string path )
		{
			using ( FileStream stream = File.OpenRead( path ) )
			using ( BinaryReader reader = new BinaryReader( stream ) )
			{
				string magic = Encoding.ASCII.GetString( reader.ReadBytes( 8 ) );
				if ( magic != "TBVHFEA1" )
				{
					throw new InvalidDataException( "unexpected magic: " + magic );
				}
				FeatDumpFile file = new FeatDumpFile();
				file.TriCount = reader.ReadUInt32();
				file.Quick = ReadTreeBlock( reader );
				file.FullSweep = ReadTreeBlock( reader );
				file.Presplit = ReadTreeBlock( reader );
				file.PresplitFullSweep = ReadTreeBlock( reader );
				file.PresplitNoPostPass = ReadTreeBlock( reader );
				file.HqBins = ReadTreeBlock( reader );
				file.StochasticIterations = reader.ReadUInt32();
				file.StochasticSahBefore = reader.ReadSingle();
				file.Stochastic = ReadTreeBlock( reader );

				file.SahBinned = reader.ReadSingle();
				file.HasEpo = reader.ReadUInt32() != 0;
				file.EpoBinned = reader.ReadSingle();
				file.EpoQuick = reader.ReadSingle();

				uint sphereCount = reader.ReadUInt32();
				file.Spheres = new SphereQuery[ sphereCount ];
				for ( uint i = 0; i < sphereCount; i++ )
				{
					SphereQuery q;
					q.Pos = RefDumpFile.ReadFloat3( reader );
					q.R = reader.ReadSingle();
					q.Hit = reader.ReadUInt32();
					file.Spheres[ i ] = q;
				}

				uint packetCount = reader.ReadUInt32();
				file.Packets = new Packet[ packetCount ];
				for ( uint p = 0; p < packetCount; p++ )
				{
					Packet packet = new Packet();
					packet.O = RefDumpFile.ReadFloat3( reader );
					packet.Rays = new PacketRay[ 256 ];
					for ( int i = 0; i < 256; i++ )
					{
						PacketRay ray;
						ray.D = RefDumpFile.ReadFloat3( reader );
						ray.T = reader.ReadSingle();
						ray.U = reader.ReadSingle();
						ray.V = reader.ReadSingle();
						ray.Prim = reader.ReadUInt32();
						packet.Rays[ i ] = ray;
					}
					file.Packets[ p ] = packet;
				}
				return file;
			}
		}

		static TreeBlock ReadTreeBlock( BinaryReader reader )
		{
			TreeBlock block = new TreeBlock();
			block.UsedNodes = reader.ReadUInt32();
			block.SahCost = reader.ReadSingle();
			block.AabbMin = RefDumpFile.ReadFloat3( reader );
			block.AabbMax = RefDumpFile.ReadFloat3( reader );
			uint nodeCount = reader.ReadUInt32();
			block.Nodes = RefDumpFile.ReadNodes( reader, nodeCount );
			uint idxCount = reader.ReadUInt32();
			block.PrimIdx = RefDumpFile.ReadUInts( reader, idxCount );
			block.Rays = RefDumpFile.ReadRays( reader );
			return block;
		}
	}
}
