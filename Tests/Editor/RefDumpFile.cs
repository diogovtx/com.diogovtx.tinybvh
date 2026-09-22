using System.IO;
using System.Text;
using Unity.Mathematics;
using TinyBVH;

namespace TinyBVH.Tests
{
	/// <summary>
	/// Reads the ".ref" reference file produced by Tools/RefDump/refdump.cpp (format "TBVHREF5").
	/// See that file's header comment for the authoritative layout; this reader matches it exactly,
	/// including the 48-byte (not 40-byte) per-ray records for the BLAS and TLAS ray sections.
	/// </summary>
	public class RefDumpFile
	{
		public struct RayHit
		{
			public float3 O;
			public float3 D;
			public float T;
			public float U;
			public float V;
			public uint Prim;
			public uint OccludedFull;
			public uint OccludedHalf;
		}

		public struct RefitHit
		{
			public float T;
			public float U;
			public float V;
			public uint Prim;
		}

		public struct TlasRayHit
		{
			public float3 O;
			public float3 D;
			public float T;
			public float U;
			public float V;
			public uint Prim;
			public uint Inst;
			public uint OccludedFull;
		}

		public struct InstanceRecord
		{
			public BvhMat4 Transform;
			public BvhMat4 InvTransform;
			public float3 AabbMin;
			public float3 AabbMax;
			public uint Mask;
		}

		public uint TriCount;

		// BLAS section.
		public uint UsedNodes;
		public float SahCost;
		public float3 AabbMin;
		public float3 AabbMax;
		public BvhNode[] Nodes;
		public uint[] PrimIdx;

		// BLAS rays.
		public RayHit[] Rays;

		// Refit section.
		public float3 RefitAabbMin;
		public float3 RefitAabbMax;
		public RefitHit[] RefitHits;

		// TLAS section.
		public InstanceRecord[] Instances;
		public uint TlasUsedNodes;
		public float3 TlasAabbMin;
		public float3 TlasAabbMax;
		public BvhNode[] TlasNodes;
		public uint[] TlasPrimIdx;
		public TlasRayHit[] TlasRays;

		// SBVH section: a spatial-split build over the same geometry, traced with the BLAS rays.
		public uint SbvhUsedNodes;
		public float SbvhSahCost;
		public float3 SbvhAabbMin;
		public float3 SbvhAabbMax;
		public BvhNode[] SbvhNodes;
		/// <summary>Only the PrimCount() entries Compact wrote; see the refdump.cpp format note.</summary>
		public uint[] SbvhPrimIdx;
		public RayHit[] SbvhRays;

		/// <summary>
		/// One optimizer reference section. <see cref="OptPlain"/> is a plain BVH::Optimize on a
		/// fresh binned build; <see cref="OptMerged"/> is the SplitLeafs( 1 ) / Optimize /
		/// MergeLeafs pipeline driven through BVH_Verbose. Both use extreme = stochastic = false.
		/// </summary>
		public class OptimizeSection
		{
			/// <summary>The iteration count the reference used; pass the same one.</summary>
			public uint Iterations;
			public uint UsedNodes;
			public float SahCost;
			public float SahCostBefore;
			public float3 AabbMin;
			public float3 AabbMax;
			public BvhNode[] Nodes;
			/// <summary>PrimCount() entries, not idxCount; see the refdump.cpp format note.</summary>
			public uint[] PrimIdx;
			public RayHit[] Rays;
		}

		// Optimizer sections, added in TBVHREF3.
		public OptimizeSection OptPlain;
		public OptimizeSection OptMerged;

		// Indexed-geometry sections, added in TBVHREF4: the same triangles welded into an indexed
		// mesh, plus a binned and a spatial-split build over it. The welding was done by the dump
		// tool; read these arrays rather than re-welding, so the two sides cannot drift apart.
		public float4[] WeldedVertices;
		/// <summary>Three vertex indices per triangle, in triangle order; length is TriCount * 3.</summary>
		public uint[] Indices;
		public uint IndexedUsedNodes;
		public BvhNode[] IndexedNodes;
		public uint[] IndexedPrimIdx;
		public RayHit[] IndexedRays;
		public uint IndexedSbvhUsedNodes;
		public BvhNode[] IndexedSbvhNodes;
		/// <summary>PrimCount() entries, not idxCount; see the refdump.cpp format note.</summary>
		public uint[] IndexedSbvhPrimIdx;
		public RayHit[] IndexedSbvhRays;

		// Opacity micro map section, added in TBVHREF5.
		/// <summary>Opacity micro map subdivision the reference used; bits per triangle is OpMapN * OpMapN.</summary>
		public uint OpMapN;
		/// <summary>uint words per triangle in the reference map, ( OpMapN * OpMapN + 31 ) / 32.</summary>
		public uint OpMapWords;
		/// <summary>The BLAS rays traced against a BVH carrying the procedural opacity map; see the refdump.cpp header.</summary>
		public RayHit[] OpacityRays;

		public static RefDumpFile Load( string path )
		{
			using ( FileStream stream = File.OpenRead( path ) )
			using ( BinaryReader reader = new BinaryReader( stream ) )
			{
				byte[] magic = reader.ReadBytes( 8 );
				string magicStr = Encoding.ASCII.GetString( magic );
				if ( magicStr != "TBVHREF5" )
				{
					throw new InvalidDataException( "unexpected magic: " + magicStr );
				}

				RefDumpFile file = new RefDumpFile();
				file.TriCount = reader.ReadUInt32();

				// BLAS section.
				file.UsedNodes = reader.ReadUInt32();
				file.SahCost = reader.ReadSingle();
				file.AabbMin = ReadFloat3( reader );
				file.AabbMax = ReadFloat3( reader );
				uint nodeCount = reader.ReadUInt32();
				file.Nodes = ReadNodes( reader, nodeCount );
				uint idxCount = reader.ReadUInt32();
				file.PrimIdx = ReadUInts( reader, idxCount );

				// BLAS rays.
				uint rayCount = reader.ReadUInt32();
				file.Rays = new RayHit[ rayCount ];
				for ( uint i = 0; i < rayCount; i++ )
				{
					RayHit hit;
					hit.O = ReadFloat3( reader );
					hit.D = ReadFloat3( reader );
					hit.T = reader.ReadSingle();
					hit.U = reader.ReadSingle();
					hit.V = reader.ReadSingle();
					hit.Prim = reader.ReadUInt32();
					hit.OccludedFull = reader.ReadUInt32();
					hit.OccludedHalf = reader.ReadUInt32();
					file.Rays[ i ] = hit;
				}

				// Refit section: root bounds, then rayCount hits in the same order as the BLAS rays.
				file.RefitAabbMin = ReadFloat3( reader );
				file.RefitAabbMax = ReadFloat3( reader );
				file.RefitHits = new RefitHit[ rayCount ];
				for ( uint i = 0; i < rayCount; i++ )
				{
					RefitHit hit;
					hit.T = reader.ReadSingle();
					hit.U = reader.ReadSingle();
					hit.V = reader.ReadSingle();
					hit.Prim = reader.ReadUInt32();
					file.RefitHits[ i ] = hit;
				}

				// TLAS section.
				uint instCount = reader.ReadUInt32();
				file.Instances = new InstanceRecord[ instCount ];
				for ( uint i = 0; i < instCount; i++ )
				{
					InstanceRecord inst;
					inst.Transform = ReadMat4( reader );
					inst.InvTransform = ReadMat4( reader );
					inst.AabbMin = ReadFloat3( reader );
					inst.AabbMax = ReadFloat3( reader );
					inst.Mask = reader.ReadUInt32();
					file.Instances[ i ] = inst;
				}

				file.TlasUsedNodes = reader.ReadUInt32();
				file.TlasAabbMin = ReadFloat3( reader );
				file.TlasAabbMax = ReadFloat3( reader );
				uint tlasNodeCount = reader.ReadUInt32();
				file.TlasNodes = ReadNodes( reader, tlasNodeCount );
				uint tlasIdxCount = reader.ReadUInt32();
				file.TlasPrimIdx = ReadUInts( reader, tlasIdxCount );

				uint tlasRayCount = reader.ReadUInt32();
				file.TlasRays = new TlasRayHit[ tlasRayCount ];
				for ( uint i = 0; i < tlasRayCount; i++ )
				{
					TlasRayHit hit;
					hit.O = ReadFloat3( reader );
					hit.D = ReadFloat3( reader );
					hit.T = reader.ReadSingle();
					hit.U = reader.ReadSingle();
					hit.V = reader.ReadSingle();
					hit.Prim = reader.ReadUInt32();
					hit.Inst = reader.ReadUInt32();
					hit.OccludedFull = reader.ReadUInt32();
					file.TlasRays[ i ] = hit;
				}

				// SBVH section.
				file.SbvhUsedNodes = reader.ReadUInt32();
				file.SbvhSahCost = reader.ReadSingle();
				file.SbvhAabbMin = ReadFloat3( reader );
				file.SbvhAabbMax = ReadFloat3( reader );
				uint sbvhNodeCount = reader.ReadUInt32();
				file.SbvhNodes = ReadNodes( reader, sbvhNodeCount );
				uint sbvhIdxCount = reader.ReadUInt32();
				file.SbvhPrimIdx = ReadUInts( reader, sbvhIdxCount );

				uint sbvhRayCount = reader.ReadUInt32();
				file.SbvhRays = new RayHit[ sbvhRayCount ];
				for ( uint i = 0; i < sbvhRayCount; i++ )
				{
					RayHit hit;
					hit.O = ReadFloat3( reader );
					hit.D = ReadFloat3( reader );
					hit.T = reader.ReadSingle();
					hit.U = reader.ReadSingle();
					hit.V = reader.ReadSingle();
					hit.Prim = reader.ReadUInt32();
					hit.OccludedFull = reader.ReadUInt32();
					hit.OccludedHalf = reader.ReadUInt32();
					file.SbvhRays[ i ] = hit;
				}

				// Optimizer sections.
				file.OptPlain = ReadOptimizeSection( reader );
				file.OptMerged = ReadOptimizeSection( reader );

				// Indexed-geometry sections: the welded mesh, then a binned and a spatial-split
				// build over it, each followed by the BLAS rays traced against it.
				uint weldedCount = reader.ReadUInt32();
				file.WeldedVertices = new float4[ weldedCount ];
				for ( uint i = 0; i < weldedCount; i++ )
				{
					float x = reader.ReadSingle();
					float y = reader.ReadSingle();
					float z = reader.ReadSingle();
					float w = reader.ReadSingle();
					file.WeldedVertices[ i ] = new float4( x, y, z, w );
				}
				uint indexCount = reader.ReadUInt32();
				file.Indices = ReadUInts( reader, indexCount );

				file.IndexedUsedNodes = reader.ReadUInt32();
				uint indexedNodeCount = reader.ReadUInt32();
				file.IndexedNodes = ReadNodes( reader, indexedNodeCount );
				uint indexedPrimCount = reader.ReadUInt32();
				file.IndexedPrimIdx = ReadUInts( reader, indexedPrimCount );
				file.IndexedRays = ReadRays( reader );

				file.IndexedSbvhUsedNodes = reader.ReadUInt32();
				uint indexedSbvhNodeCount = reader.ReadUInt32();
				file.IndexedSbvhNodes = ReadNodes( reader, indexedSbvhNodeCount );
				uint indexedSbvhPrimCount = reader.ReadUInt32();
				file.IndexedSbvhPrimIdx = ReadUInts( reader, indexedSbvhPrimCount );
				file.IndexedSbvhRays = ReadRays( reader );

				// Opacity micro map section: procedural map applied to a fresh binned BVH, traced
				// with the same BLAS rays; see the refdump.cpp header for the opacity rule.
				file.OpMapN = reader.ReadUInt32();
				file.OpMapWords = reader.ReadUInt32();
				file.OpacityRays = ReadRays( reader );

				return file;
			}
		}

		static OptimizeSection ReadOptimizeSection( BinaryReader reader )
		{
			OptimizeSection section = new OptimizeSection();
			section.Iterations = reader.ReadUInt32();
			section.UsedNodes = reader.ReadUInt32();
			section.SahCost = reader.ReadSingle();
			section.SahCostBefore = reader.ReadSingle();
			section.AabbMin = ReadFloat3( reader );
			section.AabbMax = ReadFloat3( reader );
			uint nodeCount = reader.ReadUInt32();
			section.Nodes = ReadNodes( reader, nodeCount );
			uint primCount = reader.ReadUInt32();
			section.PrimIdx = ReadUInts( reader, primCount );
			uint rayCount = reader.ReadUInt32();
			section.Rays = new RayHit[ rayCount ];
			for ( uint i = 0; i < rayCount; i++ )
			{
				RayHit hit;
				hit.O = ReadFloat3( reader );
				hit.D = ReadFloat3( reader );
				hit.T = reader.ReadSingle();
				hit.U = reader.ReadSingle();
				hit.V = reader.ReadSingle();
				hit.Prim = reader.ReadUInt32();
				hit.OccludedFull = reader.ReadUInt32();
				hit.OccludedHalf = reader.ReadUInt32();
				section.Rays[ i ] = hit;
			}
			return section;
		}

		/// <summary>Reads one count-prefixed block of 48-byte ray records. Shared with the other dump readers.</summary>
		internal static RayHit[] ReadRays( BinaryReader reader )
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
				hit.OccludedFull = reader.ReadUInt32();
				hit.OccludedHalf = reader.ReadUInt32();
				rays[ i ] = hit;
			}
			return rays;
		}

		internal static float3 ReadFloat3( BinaryReader reader )
		{
			float x = reader.ReadSingle();
			float y = reader.ReadSingle();
			float z = reader.ReadSingle();
			return new float3( x, y, z );
		}

		internal static BvhMat4 ReadMat4( BinaryReader reader )
		{
			BvhMat4 m = default;
			for ( int i = 0; i < 16; i++ )
			{
				m[ i ] = reader.ReadSingle();
			}
			return m;
		}

		static BvhNode ReadNode( BinaryReader reader )
		{
			BvhNode node;
			node.AabbMin = ReadFloat3( reader );
			node.LeftFirst = reader.ReadUInt32();
			node.AabbMax = ReadFloat3( reader );
			node.TriCount = reader.ReadUInt32();
			return node;
		}

		internal static BvhNode[] ReadNodes( BinaryReader reader, uint count )
		{
			BvhNode[] nodes = new BvhNode[ count ];
			for ( uint i = 0; i < count; i++ )
			{
				nodes[ i ] = ReadNode( reader );
			}
			return nodes;
		}

		internal static uint[] ReadUInts( BinaryReader reader, uint count )
		{
			uint[] values = new uint[ count ];
			for ( uint i = 0; i < count; i++ )
			{
				values[ i ] = reader.ReadUInt32();
			}
			return values;
		}
	}
}
