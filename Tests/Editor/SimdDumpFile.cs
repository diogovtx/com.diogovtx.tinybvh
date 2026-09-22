using System.IO;
using System.Text;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Reads the ".simd.ref" reference file produced by Tools/RefDump/simddump.cpp (format
	/// "TBVHSIM2"), the only dump tool compiled with SIMD enabled: the CPU traversal of the CWBVH
	/// layout and the BVH_SoA layout, which tinybvh compiles only under BVH_USEAVX, the tree of the
	/// AVX binned builder (BVH::BuildAVX) and a TLAS whose BLASes use mixed layouts. The base tree
	/// is included so a consumer can check that the SIMD build produced the same binned tree as
	/// the scalar tools before trusting the traversal results. The rays are the BLAS rays of the
	/// ".ref" file.
	/// </summary>
	public class SimdDumpFile
	{
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

		public struct RayOrigin
		{
			public float3 O;
			public float3 D;
		}

		/// <summary>The 24-byte hit record; the ray itself is in <see cref="Rays"/> at the same index.</summary>
		public struct Hit
		{
			public float T;
			public float U;
			public float V;
			public uint Prim;
			public uint OccludedFull;
			public uint OccludedHalf;
		}

		/// <summary>Raw BVH_SoA::BVHNode: child slabs as xxxx/yyyy/zzzz (left min, left max, right min, right max), then the four indices.</summary>
		public struct SoaNodeRecord
		{
			public float4 X;
			public float4 Y;
			public float4 Z;
			public uint Left;
			public uint Right;
			public uint TriCount;
			public uint FirstTri;
		}

		public uint TriCount;
		public BvhNode[] BaseNodes;
		public uint[] BasePrimIdx;
		public RayOrigin[] Rays;

		public uint CwbvhUsedBlocks;
		/// <summary>FNV-1a 64 over CwbvhUsedBlocks * 16 bytes of node data.</summary>
		public ulong CwbvhNodeHash;
		/// <summary>FNV-1a 64 over bvh8.idxCount * 64 bytes of triangle data.</summary>
		public ulong CwbvhTriHash;
		public Hit[] CwbvhHits;

		public uint SoaUsedNodes;
		public SoaNodeRecord[] SoaNodes;
		public Hit[] SoaHits;

		// AVX binned builder (BVH::BuildAVX, serial): a fresh BVH with the default
		// settings.useSIMDifavailable = true. Its tree differs from the scalar binned one.
		public uint AvxUsedNodes;
		public float AvxSahCost;
		public float3 AvxAabbMin;
		public float3 AvxAabbMax;
		public BvhNode[] AvxNodes;
		public uint[] AvxPrimIdx;
		/// <summary>BVH::Intersect / IsOccluded on the AVX-built tree, for <see cref="Rays"/>.</summary>
		public Hit[] AvxHits;

		// Mixed-layout TLAS: BLAS list { BVH, BVH4_CPU, BVH8_CPU, BVH_SoA }, five instances; see
		// simddump.cpp for the transforms and masks.
		public InstanceRecord[] TlasInstances;
		public uint TlasUsedNodes;
		public float3 TlasAabbMin;
		public float3 TlasAabbMax;
		public BvhNode[] TlasNodes;
		public uint[] TlasPrimIdx;
		/// <summary>Rays generated over the TLAS bounds, with BVH::Intersect and BVH::IsOccluded results.</summary>
		public RefDumpFile.TlasRayHit[] TlasRays;

		public static SimdDumpFile Load( string path )
		{
			using ( FileStream stream = File.OpenRead( path ) )
			using ( BinaryReader reader = new BinaryReader( stream ) )
			{
				string magic = Encoding.ASCII.GetString( reader.ReadBytes( 8 ) );
				if ( magic != "TBVHSIM2" )
				{
					throw new InvalidDataException( "unexpected magic: " + magic );
				}
				SimdDumpFile file = new SimdDumpFile();
				file.TriCount = reader.ReadUInt32();
				uint nodeCount = reader.ReadUInt32();
				file.BaseNodes = RefDumpFile.ReadNodes( reader, nodeCount );
				uint idxCount = reader.ReadUInt32();
				file.BasePrimIdx = RefDumpFile.ReadUInts( reader, idxCount );
				uint rayCount = reader.ReadUInt32();
				file.Rays = new RayOrigin[ rayCount ];
				for ( uint i = 0; i < rayCount; i++ )
				{
					RayOrigin ray;
					ray.O = RefDumpFile.ReadFloat3( reader );
					ray.D = RefDumpFile.ReadFloat3( reader );
					file.Rays[ i ] = ray;
				}

				file.CwbvhUsedBlocks = reader.ReadUInt32();
				file.CwbvhNodeHash = reader.ReadUInt64();
				file.CwbvhTriHash = reader.ReadUInt64();
				file.CwbvhHits = ReadHits( reader );

				file.SoaUsedNodes = reader.ReadUInt32();
				file.SoaNodes = new SoaNodeRecord[ file.SoaUsedNodes ];
				for ( uint i = 0; i < file.SoaUsedNodes; i++ )
				{
					SoaNodeRecord node;
					node.X = ReadFloat4( reader );
					node.Y = ReadFloat4( reader );
					node.Z = ReadFloat4( reader );
					node.Left = reader.ReadUInt32();
					node.Right = reader.ReadUInt32();
					node.TriCount = reader.ReadUInt32();
					node.FirstTri = reader.ReadUInt32();
					file.SoaNodes[ i ] = node;
				}
				file.SoaHits = ReadHits( reader );

				file.AvxUsedNodes = reader.ReadUInt32();
				file.AvxSahCost = reader.ReadSingle();
				file.AvxAabbMin = RefDumpFile.ReadFloat3( reader );
				file.AvxAabbMax = RefDumpFile.ReadFloat3( reader );
				uint avxNodeCount = reader.ReadUInt32();
				file.AvxNodes = RefDumpFile.ReadNodes( reader, avxNodeCount );
				uint avxIdxCount = reader.ReadUInt32();
				file.AvxPrimIdx = RefDumpFile.ReadUInts( reader, avxIdxCount );
				file.AvxHits = ReadHits( reader );

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

		static Hit[] ReadHits( BinaryReader reader )
		{
			uint count = reader.ReadUInt32();
			Hit[] hits = new Hit[ count ];
			for ( uint i = 0; i < count; i++ )
			{
				Hit hit;
				hit.T = reader.ReadSingle();
				hit.U = reader.ReadSingle();
				hit.V = reader.ReadSingle();
				hit.Prim = reader.ReadUInt32();
				hit.OccludedFull = reader.ReadUInt32();
				hit.OccludedHalf = reader.ReadUInt32();
				hits[ i ] = hit;
			}
			return hits;
		}

		static float4 ReadFloat4( BinaryReader reader )
		{
			float x = reader.ReadSingle();
			float y = reader.ReadSingle();
			float z = reader.ReadSingle();
			float w = reader.ReadSingle();
			return new float4( x, y, z, w );
		}
	}
}
