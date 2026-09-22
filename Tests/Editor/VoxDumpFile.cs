using System;
using System.IO;
using System.Text;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Reads the ".vox.ref" reference file produced by Tools/RefDump/voxeldump.cpp (format
	/// "TBVHVOX1"): the contents of a procedurally filled 256^3 VoxelSet, 65536 object-space rays
	/// with their VoxelSet::Intersect, GetNormal and IsOccluded results, and a mixed TLAS that has
	/// the voxel set attached as a BLAS next to a triangle BVH.
	///
	/// Note that a ray record is 56 bytes, not the 60 the header comment of voxeldump.cpp claims;
	/// the field list in that comment is right, only the total is off.
	///
	/// The voxel content itself is scene-independent - the scene file only feeds the TLAS - so the
	/// three ".vox.ref" files differ only in their TLAS section. That section is parsed here so the
	/// file is read to the end, but nothing compares it yet: attaching a voxel BLAS to a TLAS is a
	/// later phase of the port.
	/// </summary>
	public class VoxDumpFile
	{
		/// <summary>One of the 65536 object-space rays and everything the C++ recorded for it.</summary>
		public struct VoxelRay
		{
			public float3 O;
			public float3 D;
			public float T;
			/// <summary>The voxel value that was hit, VoxelSet::Intersect's ray.hit.prim.</summary>
			public uint Prim;
			/// <summary>VoxelSet::Intersect's return value; zero for a miss, as the C++ reports it.</summary>
			public uint Steps;
			/// <summary>GetNormal after the hit; zero for a miss.</summary>
			public float3 Normal;
			public uint OccludedFull;
			public uint OccludedHalf;
		}

		/// <summary>A BLAS instance record as the C++ left it after the TLAS build.</summary>
		public struct InstanceRecord
		{
			public BvhMat4 Transform;
			public BvhMat4 InvTransform;
			public float3 AabbMin;
			public float3 AabbMax;
			public uint BlasIdx;
			public uint Mask;
		}

		public uint TriCount;

		// Voxel set contents, as they were after UpdateTopGrid.
		public uint GridDim;
		public uint BrickDim;
		public uint TopGridDim;
		/// <summary>Brick index per grid cell, GridDim^3 entries.</summary>
		public uint[] Grid;
		public uint FreeBrickPtr;
		/// <summary>The used part of the brick pool: FreeBrickPtr * BrickDim^3 voxel values.</summary>
		public uint[] Brick;
		/// <summary>Top grid bits, TopGridDim^3 / 32 words.</summary>
		public uint[] TopGrid;

		public VoxelRay[] Rays;

		// Mixed TLAS section; parsed but not compared yet.
		public InstanceRecord[] TlasInstances;
		public uint TlasUsedNodes;
		public float3 TlasAabbMin;
		public float3 TlasAabbMax;
		public BvhNode[] TlasNodes;
		public uint[] TlasPrimIdx;
		public RefDumpFile.TlasRayHit[] TlasRays;

		public static VoxDumpFile Load( string path )
		{
			using ( FileStream stream = File.OpenRead( path ) )
			using ( BinaryReader reader = new BinaryReader( stream ) )
			{
				string magic = Encoding.ASCII.GetString( reader.ReadBytes( 8 ) );
				if ( magic != "TBVHVOX1" )
				{
					throw new InvalidDataException( "unexpected magic: " + magic );
				}
				VoxDumpFile file = new VoxDumpFile();
				file.TriCount = reader.ReadUInt32();

				file.GridDim = reader.ReadUInt32();
				file.BrickDim = reader.ReadUInt32();
				file.TopGridDim = reader.ReadUInt32();
				uint gridSize = reader.ReadUInt32();
				file.Grid = ReadUIntBlock( reader, gridSize );
				file.FreeBrickPtr = reader.ReadUInt32();
				uint brickWords = file.FreeBrickPtr * file.BrickDim * file.BrickDim * file.BrickDim;
				file.Brick = ReadUIntBlock( reader, brickWords );
				uint topGridWords = reader.ReadUInt32();
				file.TopGrid = ReadUIntBlock( reader, topGridWords );

				uint rayCount = reader.ReadUInt32();
				file.Rays = new VoxelRay[ rayCount ];
				for ( uint i = 0; i < rayCount; i++ )
				{
					VoxelRay ray;
					ray.O = RefDumpFile.ReadFloat3( reader );
					ray.D = RefDumpFile.ReadFloat3( reader );
					ray.T = reader.ReadSingle();
					ray.Prim = reader.ReadUInt32();
					ray.Steps = reader.ReadUInt32();
					ray.Normal = RefDumpFile.ReadFloat3( reader );
					ray.OccludedFull = reader.ReadUInt32();
					ray.OccludedHalf = reader.ReadUInt32();
					file.Rays[ i ] = ray;
				}

				uint instCount = reader.ReadUInt32();
				file.TlasInstances = new InstanceRecord[ instCount ];
				for ( uint i = 0; i < instCount; i++ )
				{
					InstanceRecord inst;
					inst.Transform = RefDumpFile.ReadMat4( reader );
					inst.InvTransform = RefDumpFile.ReadMat4( reader );
					inst.AabbMin = RefDumpFile.ReadFloat3( reader );
					inst.AabbMax = RefDumpFile.ReadFloat3( reader );
					inst.BlasIdx = reader.ReadUInt32();
					inst.Mask = reader.ReadUInt32();
					file.TlasInstances[ i ] = inst;
				}
				file.TlasUsedNodes = reader.ReadUInt32();
				file.TlasAabbMin = RefDumpFile.ReadFloat3( reader );
				file.TlasAabbMax = RefDumpFile.ReadFloat3( reader );
				uint tlasNodeCount = reader.ReadUInt32();
				file.TlasNodes = RefDumpFile.ReadNodes( reader, tlasNodeCount );
				uint tlasIdxCount = reader.ReadUInt32();
				file.TlasPrimIdx = RefDumpFile.ReadUInts( reader, tlasIdxCount );
				uint tlasRayCount = reader.ReadUInt32();
				file.TlasRays = new RefDumpFile.TlasRayHit[ tlasRayCount ];
				for ( uint i = 0; i < tlasRayCount; i++ )
				{
					RefDumpFile.TlasRayHit ray;
					ray.O = RefDumpFile.ReadFloat3( reader );
					ray.D = RefDumpFile.ReadFloat3( reader );
					ray.T = reader.ReadSingle();
					ray.U = reader.ReadSingle();
					ray.V = reader.ReadSingle();
					ray.Prim = reader.ReadUInt32();
					ray.Inst = reader.ReadUInt32();
					ray.OccludedFull = reader.ReadUInt32();
					file.TlasRays[ i ] = ray;
				}
				if ( stream.Position != stream.Length )
				{
					throw new InvalidDataException( "trailing data in " + path );
				}
				return file;
			}
		}

		/// <summary>
		/// Bulk read of a run of little-endian uints. The brick pool alone is ten megabytes, which
		/// is why this does not go through RefDumpFile.ReadUInts.
		/// </summary>
		static uint[] ReadUIntBlock( BinaryReader reader, uint count )
		{
			byte[] bytes = reader.ReadBytes( ( int )count * 4 );
			if ( bytes.Length != ( int )count * 4 )
			{
				throw new InvalidDataException( "unexpected end of file" );
			}
			uint[] values = new uint[ count ];
			Buffer.BlockCopy( bytes, 0, values, 0, bytes.Length );
			return values;
		}
	}
}
