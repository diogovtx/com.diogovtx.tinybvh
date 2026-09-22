using System.IO;
using System.Text;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Reads the ".dbl.ref" reference file produced by Tools/RefDump/dbldump.cpp (format
	/// "TBVHDBL1"): the double-precision BVH (BVH_Double, RayEx, BLASInstanceEx) built over the
	/// scene triangles, over the same triangles welded into an indexed mesh, and a TLAS over three
	/// instances of the first tree. See that file's header comment for the authoritative layout.
	///
	/// One deviation from that comment: it calls the TLAS ray records 96 bytes, but the tool
	/// writes them field by field with no padding, so they are 92 bytes (48 + 24 + 8 + 8 + 4).
	/// </summary>
	public class DblDumpFile
	{
		/// <summary>An 88-byte BLAS / indexed ray record.</summary>
		public struct RayHit
		{
			public double3 O;
			public double3 D;
			public double T;
			public double U;
			public double V;
			public ulong Prim;
			public uint OccludedFull;
			public uint OccludedHalf;
		}

		/// <summary>A 92-byte TLAS ray record.</summary>
		public struct TlasRayHit
		{
			public double3 O;
			public double3 D;
			public double T;
			public double U;
			public double V;
			public ulong Prim;
			public ulong Inst;
			public uint OccludedFull;
		}

		/// <summary>One BLASInstanceEx, as it stood after the TLAS build updated it.</summary>
		public struct InstanceRecord
		{
			public BvhMat4Double Transform;
			public BvhMat4Double InvTransform;
			public double3 AabbMin;
			public double3 AabbMax;
			public ulong BlasIdx;
			public ulong Mask;
		}

		public uint TriCount;

		// BLAS section.
		public ulong UsedNodes;
		public double SahCost;
		public double3 AabbMin;
		public double3 AabbMax;
		public BvhDoubleNode[] Nodes;
		public ulong[] PrimIdx;
		public RayHit[] Rays;

		// Indexed section: the same triangles welded into an indexed mesh. Read these arrays
		// rather than re-welding, so the two sides cannot drift apart over the welding rule.
		public double3[] WeldedVertices;
		/// <summary>Three vertex indices per triangle, in triangle order; length is TriCount * 3.</summary>
		public uint[] Indices;
		public ulong IndexedUsedNodes;
		public BvhDoubleNode[] IndexedNodes;
		public ulong[] IndexedPrimIdx;
		public RayHit[] IndexedRays;

		// TLAS section.
		public InstanceRecord[] Instances;
		public ulong TlasUsedNodes;
		public double3 TlasAabbMin;
		public double3 TlasAabbMax;
		public BvhDoubleNode[] TlasNodes;
		public ulong[] TlasPrimIdx;
		public TlasRayHit[] TlasRays;

		public static DblDumpFile Load( string path )
		{
			using ( FileStream stream = File.OpenRead( path ) )
			using ( BinaryReader reader = new BinaryReader( stream ) )
			{
				string magic = Encoding.ASCII.GetString( reader.ReadBytes( 8 ) );
				if ( magic != "TBVHDBL1" )
				{
					throw new InvalidDataException( "unexpected magic: " + magic );
				}
				DblDumpFile file = new DblDumpFile();
				file.TriCount = reader.ReadUInt32();

				file.UsedNodes = reader.ReadUInt64();
				file.SahCost = reader.ReadDouble();
				file.AabbMin = ReadDouble3( reader );
				file.AabbMax = ReadDouble3( reader );
				file.Nodes = ReadNodes( reader, reader.ReadUInt64() );
				file.PrimIdx = ReadULongs( reader, reader.ReadUInt64() );
				file.Rays = ReadRays( reader );

				uint weldedCount = reader.ReadUInt32();
				file.WeldedVertices = new double3[ weldedCount ];
				for ( uint i = 0; i < weldedCount; i++ )
				{
					file.WeldedVertices[ i ] = ReadDouble3( reader );
				}
				uint indexCount = reader.ReadUInt32();
				file.Indices = new uint[ indexCount ];
				for ( uint i = 0; i < indexCount; i++ )
				{
					file.Indices[ i ] = reader.ReadUInt32();
				}
				file.IndexedUsedNodes = reader.ReadUInt64();
				file.IndexedNodes = ReadNodes( reader, reader.ReadUInt64() );
				file.IndexedPrimIdx = ReadULongs( reader, reader.ReadUInt64() );
				file.IndexedRays = ReadRays( reader );

				uint instCount = reader.ReadUInt32();
				file.Instances = new InstanceRecord[ instCount ];
				for ( uint i = 0; i < instCount; i++ )
				{
					InstanceRecord record;
					record.Transform = ReadMat4( reader );
					record.InvTransform = ReadMat4( reader );
					record.AabbMin = ReadDouble3( reader );
					record.AabbMax = ReadDouble3( reader );
					record.BlasIdx = reader.ReadUInt64();
					record.Mask = reader.ReadUInt64();
					file.Instances[ i ] = record;
				}
				file.TlasUsedNodes = reader.ReadUInt64();
				file.TlasAabbMin = ReadDouble3( reader );
				file.TlasAabbMax = ReadDouble3( reader );
				file.TlasNodes = ReadNodes( reader, reader.ReadUInt64() );
				file.TlasPrimIdx = ReadULongs( reader, reader.ReadUInt64() );
				uint tlasRayCount = reader.ReadUInt32();
				file.TlasRays = new TlasRayHit[ tlasRayCount ];
				for ( uint i = 0; i < tlasRayCount; i++ )
				{
					TlasRayHit hit;
					hit.O = ReadDouble3( reader );
					hit.D = ReadDouble3( reader );
					hit.T = reader.ReadDouble();
					hit.U = reader.ReadDouble();
					hit.V = reader.ReadDouble();
					hit.Prim = reader.ReadUInt64();
					hit.Inst = reader.ReadUInt64();
					hit.OccludedFull = reader.ReadUInt32();
					file.TlasRays[ i ] = hit;
				}
				return file;
			}
		}

		internal static double3 ReadDouble3( BinaryReader reader )
		{
			double x = reader.ReadDouble();
			double y = reader.ReadDouble();
			double z = reader.ReadDouble();
			return new double3( x, y, z );
		}

		static BvhMat4Double ReadMat4( BinaryReader reader )
		{
			BvhMat4Double m = default;
			for ( int i = 0; i < 16; i++ )
			{
				m[ i ] = reader.ReadDouble();
			}
			return m;
		}

		static BvhDoubleNode ReadNode( BinaryReader reader )
		{
			BvhDoubleNode node;
			node.AabbMin = ReadDouble3( reader );
			node.AabbMax = ReadDouble3( reader );
			node.LeftFirst = reader.ReadUInt64();
			node.TriCount = reader.ReadUInt64();
			return node;
		}

		internal static BvhDoubleNode[] ReadNodes( BinaryReader reader, ulong count )
		{
			BvhDoubleNode[] nodes = new BvhDoubleNode[ count ];
			for ( ulong i = 0; i < count; i++ )
			{
				nodes[ i ] = ReadNode( reader );
			}
			return nodes;
		}

		internal static ulong[] ReadULongs( BinaryReader reader, ulong count )
		{
			ulong[] values = new ulong[ count ];
			for ( ulong i = 0; i < count; i++ )
			{
				values[ i ] = reader.ReadUInt64();
			}
			return values;
		}

		internal static RayHit[] ReadRays( BinaryReader reader )
		{
			uint rayCount = reader.ReadUInt32();
			RayHit[] rays = new RayHit[ rayCount ];
			for ( uint i = 0; i < rayCount; i++ )
			{
				RayHit hit;
				hit.O = ReadDouble3( reader );
				hit.D = ReadDouble3( reader );
				hit.T = reader.ReadDouble();
				hit.U = reader.ReadDouble();
				hit.V = reader.ReadDouble();
				hit.Prim = reader.ReadUInt64();
				hit.OccludedFull = reader.ReadUInt32();
				hit.OccludedHalf = reader.ReadUInt32();
				rays[ i ] = hit;
			}
			return rays;
		}
	}
}
