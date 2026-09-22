using System.IO;
using System.Text;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Reads the "&lt;scene&gt;.layouts.ref" reference file produced by Tools~/RefDump/layoutdump.cpp
	/// (format "TBVHLAY4", or the older "TBVHLAY3" without the trailing BVH8_CPU section and
	/// "TBVHLAY2" without the BVH4_CPU one either). That file's header comment is authoritative;
	/// this reader matches it:
	///
	/// char[8] "TBVHLAY4"; u32 triCount
	/// -- BVH_GPU, ConvertFrom( bvh, false ) --
	///    u32 usedNodes; usedNodes * 64 bytes raw BVH_GPU::BVHNode
	///    u32 rayCount;  rayCount * 40 bytes: f32[3] O, f32[3] D, f32 t, f32 u, f32 v, u32 prim
	/// -- MBVH&lt;4&gt;, fresh ConvertFrom( bvh, true ) --
	///    u32 usedNodes; per node 52 bytes: aabbMin, firstTri, aabbMax, triCount, child[4], childCount
	/// -- BVH4_GPU, ConvertFrom( that mbvh4, true ) --
	///    u32 usedBlocks; usedBlocks * 16 bytes raw bvh4Data
	///    u32 rayCount;   rayCount * 40 bytes
	///    u32 usedNodes;  MBVH&lt;4&gt; nodes (52 bytes each) again, AFTER the BVH4_GPU conversion
	/// -- prepared base for CWBVH: fresh BVH, Compact(), SplitLeafs( 3 ) --
	///    u32 usedNodes; usedNodes * 32 bytes raw BVH::BVHNode
	///    u32 idxCount;  idxCount * u32 primIdx
	/// -- MBVH&lt;8&gt;, ConvertFrom( prepared base, true ) --
	///    u32 usedNodes; per node 68 bytes: aabbMin, firstTri, aabbMax, triCount, child[8], childCount
	/// -- CWBVH, ConvertFrom( that mbvh8, true ) --
	///    u32 usedBlocks; usedBlocks * 16 bytes raw bvh8Data
	///    u32 triBlocks;  triBlocks * 16 bytes raw bvh8Tris
	///    u32 usedNodes;  MBVH&lt;8&gt; nodes (68 bytes each) again, AFTER the CWBVH conversion
	/// -- BVH4_CPU, b4.Build( verts, triCount ) - "TBVHLAY3" and up --
	///    u32 baseUsedNodes; baseUsedNodes * 32 bytes raw BVH::BVHNode, after CombineLeafs( 4, .. ) + SplitLeafs( 4 )
	///    u32 baseIdxCount;  baseIdxCount * u32 primIdx
	///    u32 m4UsedNodes;   MBVH&lt;4&gt; nodes (52 bytes each), re-converted from that reshaped base
	///    u32 usedBlocks;    usedBlocks * 64 bytes raw bvh4Data (64-byte cache lines, not 16-byte blocks)
	/// -- BVH8_CPU, b8.Build( verts, triCount ) - "TBVHLAY4" only --
	///    u32 baseUsedNodes; baseUsedNodes * 32 bytes raw BVH::BVHNode, after Compact() + CombineLeafs( 4, .. ) + SplitLeafs( 4 )
	///    u32 baseIdxCount;  baseIdxCount * u32 primIdx
	///    u32 m8UsedNodes;   MBVH&lt;8&gt; nodes (68 bytes each), re-converted from that reshaped base
	///    u32 usedBlocks;    usedBlocks * 64 bytes raw bvh8Data (64-byte cache lines)
	/// </summary>
	public class LayoutDumpFile
	{
		public struct RayHit
		{
			public float3 O;
			public float3 D;
			public float T;
			public float U;
			public float V;
			public uint Prim;
		}

		/// <summary>One MBVH node as stored in the file. Child has exactly M entries.</summary>
		public struct MbvhNodeRecord
		{
			public float3 AabbMin;
			public uint FirstTri;
			public float3 AabbMax;
			public uint TriCount;
			public uint[] Child;
			public uint ChildCount;
		}

		public uint TriCount;

		// BVH_GPU section.
		public BvhGpuNode[] GpuNodes;
		public RayHit[] GpuRays;

		// MBVH<4> section.
		public MbvhNodeRecord[] Mbvh4Nodes;

		// BVH4_GPU section.
		public uint4[] Bvh4GpuBlocks;
		public RayHit[] Bvh4GpuRays;
		public MbvhNodeRecord[] Mbvh4NodesAfter;

		// Prepared base BVH for the CWBVH path: Compact() followed by SplitLeafs( 3 ).
		public BvhNode[] PreparedNodes;
		public uint[] PreparedPrimIdx;

		// MBVH<8> section.
		public MbvhNodeRecord[] Mbvh8Nodes;

		// CWBVH section.
		public uint4[] CwbvhBlocks;
		public uint4[] CwbvhTriBlocks;
		public MbvhNodeRecord[] Mbvh8NodesAfter;

		// BVH4_CPU section; absent in a "TBVHLAY2" file.
		public bool HasBvh4Cpu;
		public BvhNode[] Bvh4CpuBaseNodes;
		public uint[] Bvh4CpuBasePrimIdx;
		public MbvhNodeRecord[] Bvh4CpuMbvhNodes;
		/// <summary>Number of 64-byte cache-line blocks; Bvh4CpuBlocks holds four uint4 per block.</summary>
		public uint Bvh4CpuUsedBlocks;
		public uint4[] Bvh4CpuBlocks;

		// BVH8_CPU section; absent in a "TBVHLAY2" or "TBVHLAY3" file.
		public bool HasBvh8Cpu;
		public BvhNode[] Bvh8CpuBaseNodes;
		public uint[] Bvh8CpuBasePrimIdx;
		public MbvhNodeRecord[] Bvh8CpuMbvhNodes;
		/// <summary>Number of 64-byte cache-line blocks; Bvh8CpuBlocks holds four uint4 per block.</summary>
		public uint Bvh8CpuUsedBlocks;
		public uint4[] Bvh8CpuBlocks;

		public static LayoutDumpFile Load( string path )
		{
			using ( FileStream stream = File.OpenRead( path ) )
			using ( BinaryReader reader = new BinaryReader( stream ) )
			{
				byte[] magic = reader.ReadBytes( 8 );
				string magicStr = Encoding.ASCII.GetString( magic );
				if ( magicStr != "TBVHLAY2" && magicStr != "TBVHLAY3" && magicStr != "TBVHLAY4" )
				{
					throw new InvalidDataException( "unexpected magic: " + magicStr );
				}

				LayoutDumpFile file = new LayoutDumpFile();
				file.HasBvh4Cpu = magicStr == "TBVHLAY3" || magicStr == "TBVHLAY4";
				file.HasBvh8Cpu = magicStr == "TBVHLAY4";
				file.TriCount = reader.ReadUInt32();

				// BVH_GPU section.
				uint gpuNodeCount = reader.ReadUInt32();
				file.GpuNodes = new BvhGpuNode[ gpuNodeCount ];
				for ( uint i = 0; i < gpuNodeCount; i++ )
				{
					BvhGpuNode node;
					node.LMin = ReadFloat3( reader );
					node.Left = reader.ReadUInt32();
					node.LMax = ReadFloat3( reader );
					node.Right = reader.ReadUInt32();
					node.RMin = ReadFloat3( reader );
					node.TriCount = reader.ReadUInt32();
					node.RMax = ReadFloat3( reader );
					node.FirstTri = reader.ReadUInt32();
					file.GpuNodes[ i ] = node;
				}
				file.GpuRays = ReadRays( reader );

				// MBVH<4> section.
				file.Mbvh4Nodes = ReadMbvhNodes( reader, 4 );

				// BVH4_GPU section.
				file.Bvh4GpuBlocks = ReadBlocks( reader );
				file.Bvh4GpuRays = ReadRays( reader );
				file.Mbvh4NodesAfter = ReadMbvhNodes( reader, 4 );

				// Prepared base BVH.
				file.PreparedNodes = ReadBvhNodes( reader );
				file.PreparedPrimIdx = ReadPrimIdx( reader );

				// MBVH<8> section.
				file.Mbvh8Nodes = ReadMbvhNodes( reader, 8 );

				// CWBVH section.
				file.CwbvhBlocks = ReadBlocks( reader );
				file.CwbvhTriBlocks = ReadBlocks( reader );
				file.Mbvh8NodesAfter = ReadMbvhNodes( reader, 8 );

				// BVH4_CPU section.
				if ( file.HasBvh4Cpu )
				{
					file.Bvh4CpuBaseNodes = ReadBvhNodes( reader );
					file.Bvh4CpuBasePrimIdx = ReadPrimIdx( reader );
					file.Bvh4CpuMbvhNodes = ReadMbvhNodes( reader, 4 );
					// Unlike the other layouts, this one counts 64-byte cache lines, so every block
					// is read as four uint4.
					file.Bvh4CpuUsedBlocks = reader.ReadUInt32();
					file.Bvh4CpuBlocks = new uint4[ file.Bvh4CpuUsedBlocks * 4 ];
					for ( int i = 0; i < file.Bvh4CpuBlocks.Length; i++ )
					{
						file.Bvh4CpuBlocks[ i ] = ReadUInt4( reader );
					}
				}

				// BVH8_CPU section; 64-byte cache lines again, so four uint4 per block.
				if ( file.HasBvh8Cpu )
				{
					file.Bvh8CpuBaseNodes = ReadBvhNodes( reader );
					file.Bvh8CpuBasePrimIdx = ReadPrimIdx( reader );
					file.Bvh8CpuMbvhNodes = ReadMbvhNodes( reader, 8 );
					file.Bvh8CpuUsedBlocks = reader.ReadUInt32();
					file.Bvh8CpuBlocks = new uint4[ file.Bvh8CpuUsedBlocks * 4 ];
					for ( int i = 0; i < file.Bvh8CpuBlocks.Length; i++ )
					{
						file.Bvh8CpuBlocks[ i ] = ReadUInt4( reader );
					}
				}

				return file;
			}
		}

		static float3 ReadFloat3( BinaryReader reader )
		{
			float x = reader.ReadSingle();
			float y = reader.ReadSingle();
			float z = reader.ReadSingle();
			return new float3( x, y, z );
		}

		static RayHit[] ReadRays( BinaryReader reader )
		{
			uint rayCount = reader.ReadUInt32();
			RayHit[] rays = new RayHit[ rayCount ];
			for ( uint i = 0; i < rayCount; i++ )
			{
				RayHit hit;
				hit.O = ReadFloat3( reader );
				hit.D = ReadFloat3( reader );
				hit.T = reader.ReadSingle();
				hit.U = reader.ReadSingle();
				hit.V = reader.ReadSingle();
				hit.Prim = reader.ReadUInt32();
				rays[ i ] = hit;
			}
			return rays;
		}

		static uint4 ReadUInt4( BinaryReader reader )
		{
			uint x = reader.ReadUInt32();
			uint y = reader.ReadUInt32();
			uint z = reader.ReadUInt32();
			uint w = reader.ReadUInt32();
			return new uint4( x, y, z, w );
		}

		static uint4[] ReadBlocks( BinaryReader reader )
		{
			uint blockCount = reader.ReadUInt32();
			uint4[] blocks = new uint4[ blockCount ];
			for ( uint i = 0; i < blockCount; i++ )
			{
				blocks[ i ] = ReadUInt4( reader );
			}
			return blocks;
		}

		/// <summary>Reads a u32 count followed by that many raw 32-byte BVH::BVHNode records.</summary>
		static BvhNode[] ReadBvhNodes( BinaryReader reader )
		{
			uint nodeCount = reader.ReadUInt32();
			BvhNode[] nodes = new BvhNode[ nodeCount ];
			for ( uint i = 0; i < nodeCount; i++ )
			{
				BvhNode node;
				node.AabbMin = ReadFloat3( reader );
				node.LeftFirst = reader.ReadUInt32();
				node.AabbMax = ReadFloat3( reader );
				node.TriCount = reader.ReadUInt32();
				nodes[ i ] = node;
			}
			return nodes;
		}

		/// <summary>Reads a u32 count followed by that many primitive indices.</summary>
		static uint[] ReadPrimIdx( BinaryReader reader )
		{
			uint idxCount = reader.ReadUInt32();
			uint[] primIdx = new uint[ idxCount ];
			for ( uint i = 0; i < idxCount; i++ )
			{
				primIdx[ i ] = reader.ReadUInt32();
			}
			return primIdx;
		}

		static MbvhNodeRecord[] ReadMbvhNodes( BinaryReader reader, int m )
		{
			uint nodeCount = reader.ReadUInt32();
			MbvhNodeRecord[] nodes = new MbvhNodeRecord[ nodeCount ];
			for ( uint i = 0; i < nodeCount; i++ )
			{
				MbvhNodeRecord node;
				node.AabbMin = ReadFloat3( reader );
				node.FirstTri = reader.ReadUInt32();
				node.AabbMax = ReadFloat3( reader );
				node.TriCount = reader.ReadUInt32();
				node.Child = new uint[ m ];
				for ( int c = 0; c < m; c++ )
				{
					node.Child[ c ] = reader.ReadUInt32();
				}
				node.ChildCount = reader.ReadUInt32();
				nodes[ i ] = node;
			}
			return nodes;
		}
	}
}
