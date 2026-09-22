using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace TinyBVH
{
	/// <summary>The BVH layouts the compute shader can traverse, one kernel pair each.</summary>
	public enum GpuLayout
	{
		Bvh2,
		BvhGpu,
		Bvh4Gpu,
		Cwbvh,
		/// <summary>Aila-Laine TLAS over BLAS instances; see GpuTracer.UploadTlas.</summary>
		Tlas
	}

	/// <summary>
	/// The BLAS layouts the TLAS kernels can enter, mirroring TINYBVH_BLAS_* in the HLSL and
	/// the blasType field of the OpenCL BLASDesc.
	/// </summary>
	public enum GpuBlasType : uint
	{
		BvhGpu = 0,
		Cwbvh = 1
	}

	/// <summary>
	/// Where one BLAS lives inside the shared GPU buffers. Mirrors TinyBlasDesc in
	/// TinyBvhTraversal.hlsl; the offsets into the float4 buffers are in float4 units.
	/// </summary>
	[StructLayout( LayoutKind.Sequential )]
	public struct GpuBlasDesc
	{
		public uint NodeOffset;
		public uint IndexOffset;
		public uint VertOffset;
		public uint BlasType;
		/// <summary>
		/// Word index of this BLAS's opacity micro map inside the shared map buffer, or
		/// <see cref="GpuTracer.NoOpMap"/> when it has none; the .cl's blasDesc.opmapOffset.
		/// </summary>
		public uint OpMapOffset;
	}

	/// <summary>
	/// Uploads a converted BVH to GPU buffers and dispatches the traversal kernels in
	/// Resources/TinyBvhTrace.compute. One instance holds one uploaded acceleration
	/// structure; call the Upload overload matching the layout you want to trace.
	/// </summary>
	public sealed unsafe class GpuTracer : IDisposable
	{
		private const int LayoutCount = 5;
		private const int RenderGroupSize = 8;
		private const int BatchGroupSize = 64;
		private const int Float4Stride = 16;
		private const int UintStride = 4;
		private const int BlasDescStride = 20;

		/// <summary>
		/// Value of <see cref="GpuBlasDesc.OpMapOffset"/> meaning "this BLAS has no opacity micro
		/// map"; TINYBVH_NO_OPMAP in the HLSL and the .cl's 0x99999999 sentinel.
		/// </summary>
		public const uint NoOpMap = 0x99999999u;

		private static readonly string[] renderKernelNames =
		{
			"Render_Bvh2", "Render_BvhGpu", "Render_Bvh4Gpu", "Render_Cwbvh", "Render_Tlas"
		};

		private static readonly string[] batchKernelNames =
		{
			"Batch_Bvh2", "Batch_BvhGpu", "Batch_Bvh4Gpu", "Batch_Cwbvh", "Batch_Tlas"
		};

		private static readonly int bvh2NodesId = Shader.PropertyToID( "Bvh2Nodes" );
		private static readonly int bvhGpuNodesId = Shader.PropertyToID( "BvhGpuNodes" );
		private static readonly int primIdxId = Shader.PropertyToID( "PrimIdx" );
		private static readonly int vertsId = Shader.PropertyToID( "Verts" );
		private static readonly int vertIdxId = Shader.PropertyToID( "VertIdx" );
		private static readonly int indexedId = Shader.PropertyToID( "Indexed" );
		private static readonly int opMapId = Shader.PropertyToID( "OpMap" );
		private static readonly int opMapNId = Shader.PropertyToID( "OpMapN" );
		private static readonly int bvh4DataId = Shader.PropertyToID( "Bvh4Data" );
		private static readonly int cwbvhNodesId = Shader.PropertyToID( "CwbvhNodes" );
		private static readonly int cwbvhTrisId = Shader.PropertyToID( "CwbvhTris" );
		private static readonly int tlasNodesId = Shader.PropertyToID( "TlasNodes" );
		private static readonly int tlasIdxId = Shader.PropertyToID( "TlasIdx" );
		private static readonly int instancesId = Shader.PropertyToID( "Instances" );
		private static readonly int blasDescId = Shader.PropertyToID( "BlasDesc" );
		private static readonly int hitInstancesId = Shader.PropertyToID( "HitInstances" );
		private static readonly int targetId = Shader.PropertyToID( "Target" );
		private static readonly int rayOriginsId = Shader.PropertyToID( "RayOrigins" );
		private static readonly int rayDirectionsId = Shader.PropertyToID( "RayDirections" );
		private static readonly int hitsId = Shader.PropertyToID( "Hits" );
		private static readonly int occludedId = Shader.PropertyToID( "Occluded" );
		private static readonly int originId = Shader.PropertyToID( "Origin" );
		private static readonly int topLeftDirId = Shader.PropertyToID( "TopLeftDir" );
		private static readonly int horizontalId = Shader.PropertyToID( "Horizontal" );
		private static readonly int verticalId = Shader.PropertyToID( "Vertical" );
		private static readonly int sunDirId = Shader.PropertyToID( "SunDir" );
		private static readonly int sceneDiagonalId = Shader.PropertyToID( "SceneDiagonal" );
		private static readonly int modeId = Shader.PropertyToID( "Mode" );
		private static readonly int shadowsId = Shader.PropertyToID( "Shadows" );
		private static readonly int widthId = Shader.PropertyToID( "Width" );
		private static readonly int heightId = Shader.PropertyToID( "Height" );
		private static readonly int rayCountId = Shader.PropertyToID( "RayCount" );

		private readonly ComputeShader shader;
		private readonly int[] renderKernels = new int[ LayoutCount ];
		private readonly int[] batchKernels = new int[ LayoutCount ];

		// Layout buffers; only the ones the current layout needs are populated.
		private GraphicsBuffer bvh2Nodes;
		private GraphicsBuffer bvhGpuNodes;
		private GraphicsBuffer primIdx;
		private GraphicsBuffer verts;
		private GraphicsBuffer vertIdx;
		private GraphicsBuffer opMap;
		private GraphicsBuffer bvh4Data;
		private GraphicsBuffer cwbvhNodes;
		private GraphicsBuffer cwbvhTris;

		// TLAS buffers; the BLASses share the layout buffers above.
		private GraphicsBuffer tlasNodes;
		private GraphicsBuffer tlasIdx;
		private GraphicsBuffer instances;
		private GraphicsBuffer blasDesc;

		// Batch I/O buffers.
		private GraphicsBuffer rayOrigins;
		private GraphicsBuffer rayDirections;
		private GraphicsBuffer hits;
		private GraphicsBuffer occluded;
		private GraphicsBuffer hitInstances;

		/// <summary>Aila-Laine conversion of the uploaded TLAS; owned, reused by UpdateTlas.</summary>
		private BvhGpu tlasGpu;
		/// <summary>Node count of the uploaded BLASses, added to the TLAS nodes for NodeCount.</summary>
		private long blasNodeCount;

		private bool uploaded;

		/// <summary>
		/// Set when the uploaded structure was built over indexed geometry, i.e. its source BVH has
		/// a VertIdx. It is pushed to the shader at dispatch time rather than at upload time because
		/// Resources.Load hands every GpuTracer the same ComputeShader asset, so its uniforms are
		/// shared between tracers while its per-kernel buffer bindings are not.
		/// </summary>
		private bool indexed;

		/// <summary>
		/// Opacity micro map subdivision of the uploaded structure, zero when it has no maps. Like
		/// <see cref="indexed"/> it is pushed to the shader at dispatch time, because the uniform is
		/// shared between every GpuTracer holding the same ComputeShader asset.
		/// </summary>
		private uint opMapN;

		public static bool IsSupported => SystemInfo.supportsComputeShaders;

		/// <summary>Layout of the last Upload call.</summary>
		public GpuLayout Layout { get; private set; }

		/// <summary>Node count of the uploaded structure, for the layouts that have nodes.</summary>
		public long NodeCount { get; private set; }

		/// <summary>Used 16-byte blocks of the uploaded structure, for the blob layouts.</summary>
		public long BlockCount { get; private set; }

		public GpuTracer()
		{
			shader = Resources.Load<ComputeShader>( "TinyBvhTrace" );
			if ( shader == null )
			{
				throw new InvalidOperationException( "GpuTracer: could not load Resources/TinyBvhTrace.compute." );
			}
			for ( int i = 0; i < LayoutCount; i++ )
			{
				renderKernels[ i ] = shader.FindKernel( renderKernelNames[ i ] );
				batchKernels[ i ] = shader.FindKernel( batchKernelNames[ i ] );
			}
		}

		/// <summary>Uploads the base Wald-layout BVH: 32-byte nodes, primitive indices and vertices.</summary>
		public void Upload( ref Bvh bvh )
		{
			if ( bvh.UsedNodes == 0 )
			{
				throw new InvalidOperationException( "GpuTracer.Upload( ref Bvh ), bvh was not built." );
			}
			Layout = GpuLayout.Bvh2;
			NodeCount = bvh.UsedNodes;
			BlockCount = 0;
			int float4Count = ( int )bvh.UsedNodes * 2;
			EnsureBuffer( ref bvh2Nodes, float4Count, Float4Stride );
			Fill<float4>( bvh2Nodes, bvh.Nodes, float4Count );
			UploadTriangles( ref bvh );
			int render = renderKernels[ ( int )GpuLayout.Bvh2 ];
			int batch = batchKernels[ ( int )GpuLayout.Bvh2 ];
			SetBuffer( render, batch, bvh2NodesId, bvh2Nodes );
			SetBuffer( render, batch, primIdxId, primIdx );
			SetBuffer( render, batch, vertsId, verts );
			SetBuffer( render, batch, vertIdxId, vertIdx );
			SetBuffer( render, batch, opMapId, opMap );
			uploaded = true;
		}

		/// <summary>Uploads the Aila-Laine 64-byte layout; triangles still come from the base BVH.</summary>
		public void Upload( ref BvhGpu bvh )
		{
			if ( bvh.UsedNodes == 0 )
			{
				throw new InvalidOperationException( "GpuTracer.Upload( ref BvhGpu ), bvh was not converted." );
			}
			Layout = GpuLayout.BvhGpu;
			NodeCount = bvh.UsedNodes;
			BlockCount = 0;
			int float4Count = ( int )bvh.UsedNodes * 4;
			EnsureBuffer( ref bvhGpuNodes, float4Count, Float4Stride );
			Fill<float4>( bvhGpuNodes, bvh.Nodes, float4Count );
			UploadTriangles( ref bvh.Source );
			int render = renderKernels[ ( int )GpuLayout.BvhGpu ];
			int batch = batchKernels[ ( int )GpuLayout.BvhGpu ];
			SetBuffer( render, batch, bvhGpuNodesId, bvhGpuNodes );
			SetBuffer( render, batch, primIdxId, primIdx );
			SetBuffer( render, batch, vertsId, verts );
			SetBuffer( render, batch, vertIdxId, vertIdx );
			SetBuffer( render, batch, opMapId, opMap );
			uploaded = true;
		}

		/// <summary>Uploads the BVH4_GPU blob; it carries its own triangle data.</summary>
		public void Upload( ref Bvh4Gpu bvh )
		{
			if ( bvh.UsedBlocks == 0 )
			{
				throw new InvalidOperationException( "GpuTracer.Upload( ref Bvh4Gpu ), bvh was not converted." );
			}
			Layout = GpuLayout.Bvh4Gpu;
			indexed = false; // the blob carries its own triangles, so the kernel never reads VertIdx.
			opMapN = 0; // no opacity micro map support in this layout, as in traverse_bvh4.cl.
			NodeCount = bvh.Source.UsedNodes;
			BlockCount = bvh.UsedBlocks;
			int float4Count = ( int )bvh.UsedBlocks;
			EnsureBuffer( ref bvh4Data, float4Count, Float4Stride );
			Fill<float4>( bvh4Data, bvh.Data, float4Count );
			int render = renderKernels[ ( int )GpuLayout.Bvh4Gpu ];
			int batch = batchKernels[ ( int )GpuLayout.Bvh4Gpu ];
			SetBuffer( render, batch, bvh4DataId, bvh4Data );
			uploaded = true;
		}

		/// <summary>Uploads the CWBVH node blob and its separate triangle blob.</summary>
		public void Upload( ref BvhCwbvh bvh )
		{
			if ( bvh.UsedBlocks == 0 )
			{
				throw new InvalidOperationException( "GpuTracer.Upload( ref BvhCwbvh ), bvh was not converted." );
			}
			Layout = GpuLayout.Cwbvh;
			indexed = false; // the blob carries its own triangles, so the kernel never reads VertIdx.
			opMapN = 0; // no opacity micro map support in this layout, as in traverse_cwbvh.cl.
			NodeCount = bvh.Source.UsedNodes;
			BlockCount = bvh.UsedBlocks;
			int nodeFloat4Count = ( int )bvh.UsedBlocks;
			int triFloat4Count = ( int )bvh.TriBlocks;
			EnsureBuffer( ref cwbvhNodes, nodeFloat4Count, Float4Stride );
			Fill<float4>( cwbvhNodes, bvh.Data, nodeFloat4Count );
			EnsureBuffer( ref cwbvhTris, triFloat4Count, Float4Stride );
			Fill<float4>( cwbvhTris, bvh.Tris, triFloat4Count );
			int render = renderKernels[ ( int )GpuLayout.Cwbvh ];
			int batch = batchKernels[ ( int )GpuLayout.Cwbvh ];
			SetBuffer( render, batch, cwbvhNodesId, cwbvhNodes );
			SetBuffer( render, batch, cwbvhTrisId, cwbvhTris );
			uploaded = true;
		}

		/// <summary>
		/// Uploads a TLAS plus its BLASses in the Aila-Laine layout. The TLAS itself is converted
		/// to that layout here and kept, so UpdateTlas can refresh it cheaply. The BLASses are
		/// concatenated into the shared node, index and vertex buffers, one descriptor each.
		/// </summary>
		public void UploadTlas( ref Bvh tlas, ReadOnlySpan<BvhGpu> blasses )
		{
			ValidateTlas( ref tlas );
			ValidateBlasCount( blasses.Length );
			int nodeTotal = 0;
			int idxTotal = 0;
			int vertTotal = 0;
			int mappedTris = 0;
			opMapN = 0;
			for ( int i = 0; i < blasses.Length; i++ )
			{
				if ( blasses[ i ].Source.VertStride != 16 )
				{
					throw new NotSupportedException( "GpuTracer: only a 16-byte vertex stride is supported." );
				}
				if ( blasses[ i ].Source.VertIdx != null )
				{
					throw new NotSupportedException( "GpuTracer.UploadTlas( .. ), a BLAS over indexed geometry is not supported; only the single-BVH Bvh2 and BvhGpu uploads read VertIdx." );
				}
				// The subdivision is a single shader uniform, so every mapped BLAS has to share it.
				uint blasMapN = BlasOpMapN( blasses[ i ] );
				if ( blasMapN > 0 )
				{
					if ( opMapN > 0 && blasMapN != opMapN )
					{
						throw new NotSupportedException( "GpuTracer.UploadTlas( .. ), all BLASses with an opacity micro map must use the same subdivision." );
					}
					opMapN = blasMapN;
					mappedTris += ( int )blasses[ i ].Source.TriCount;
				}
				nodeTotal += ( int )blasses[ i ].UsedNodes;
				idxTotal += ( int )blasses[ i ].Source.IdxCount;
				vertTotal += ( int )blasses[ i ].Source.VertCount;
			}
			int opWords = ( int )OpMapWords( opMapN );
			EnsureBuffer( ref bvhGpuNodes, nodeTotal * 4, Float4Stride );
			EnsureBuffer( ref primIdx, idxTotal, UintStride );
			EnsureBuffer( ref verts, vertTotal, Float4Stride );
			EnsureBuffer( ref opMap, mappedTris * opWords, UintStride );
			GpuBlasDesc[] descs = new GpuBlasDesc[ blasses.Length ];
			int nodeAt = 0;
			int idxAt = 0;
			int vertAt = 0;
			int mapAt = 0;
			for ( int i = 0; i < blasses.Length; i++ )
			{
				int nodeCount = ( int )blasses[ i ].UsedNodes;
				int idxCount = ( int )blasses[ i ].Source.IdxCount;
				int vertCount = ( int )blasses[ i ].Source.VertCount;
				// A BLAS without a map gets the sentinel, as blasDesc.opmapOffset does in the .cl.
				uint mapOffset = NoOpMap;
				if ( BlasOpMapN( blasses[ i ] ) > 0 )
				{
					int mapCount = ( int )blasses[ i ].Source.TriCount * opWords;
					mapOffset = ( uint )mapAt;
					Fill<uint>( opMap, blasses[ i ].Source.OpMap, mapCount, mapAt );
					mapAt += mapCount;
				}
				descs[ i ] = new GpuBlasDesc
				{
					NodeOffset = ( uint )( nodeAt * 4 ),
					IndexOffset = ( uint )idxAt,
					VertOffset = ( uint )vertAt,
					BlasType = ( uint )GpuBlasType.BvhGpu,
					OpMapOffset = mapOffset
				};
				Fill<float4>( bvhGpuNodes, blasses[ i ].Nodes, nodeCount * 4, nodeAt * 4 );
				Fill<uint>( primIdx, blasses[ i ].Source.PrimIdx, idxCount, idxAt );
				Fill<float4>( verts, blasses[ i ].Source.Verts, vertCount, vertAt );
				nodeAt += nodeCount;
				idxAt += idxCount;
				vertAt += vertCount;
			}
			BlockCount = 0;
			// The kernel compiles both BLAS branches, so the CWBVH buffers have to be bound too.
			EnsureBuffer( ref cwbvhNodes, 0, Float4Stride );
			EnsureBuffer( ref cwbvhTris, 0, Float4Stride );
			FinishTlasUpload( ref tlas, descs, nodeTotal );
		}

		/// <summary>Uploads a TLAS whose BLASses are all in the CWBVH layout.</summary>
		public void UploadTlas( ref Bvh tlas, ReadOnlySpan<BvhCwbvh> blasses )
		{
			ValidateTlas( ref tlas );
			ValidateBlasCount( blasses.Length );
			int nodeTotal = 0;
			int blockTotal = 0;
			int triTotal = 0;
			for ( int i = 0; i < blasses.Length; i++ )
			{
				nodeTotal += ( int )blasses[ i ].Source.UsedNodes;
				blockTotal += ( int )blasses[ i ].UsedBlocks;
				triTotal += ( int )blasses[ i ].TriBlocks;
			}
			EnsureBuffer( ref cwbvhNodes, blockTotal, Float4Stride );
			EnsureBuffer( ref cwbvhTris, triTotal, Float4Stride );
			GpuBlasDesc[] descs = new GpuBlasDesc[ blasses.Length ];
			int blockAt = 0;
			int triAt = 0;
			for ( int i = 0; i < blasses.Length; i++ )
			{
				int blockCount = ( int )blasses[ i ].UsedBlocks;
				int triCount = ( int )blasses[ i ].TriBlocks;
				descs[ i ] = new GpuBlasDesc
				{
					NodeOffset = ( uint )blockAt,
					IndexOffset = 0,
					VertOffset = ( uint )triAt,
					BlasType = ( uint )GpuBlasType.Cwbvh,
					// No opacity micro map support in this layout, as in traverse_cwbvh.cl.
					OpMapOffset = NoOpMap
				};
				Fill<float4>( cwbvhNodes, blasses[ i ].Data, blockCount, blockAt );
				Fill<float4>( cwbvhTris, blasses[ i ].Tris, triCount, triAt );
				blockAt += blockCount;
				triAt += triCount;
			}
			BlockCount = blockTotal;
			opMapN = 0;
			// The kernel compiles both BLAS branches, so the Aila-Laine buffers have to be bound too.
			EnsureBuffer( ref bvhGpuNodes, 0, Float4Stride );
			EnsureBuffer( ref primIdx, 0, UintStride );
			EnsureBuffer( ref verts, 0, Float4Stride );
			EnsureBuffer( ref opMap, 0, UintStride );
			FinishTlasUpload( ref tlas, descs, nodeTotal );
		}

		/// <summary>
		/// Re-uploads only the TLAS nodes, its instance indices and the instance records, for
		/// animated instances. The BLASses stay as the last UploadTlas left them, so the TLAS
		/// must still be built over the same BLAS list.
		/// </summary>
		public void UpdateTlas( ref Bvh tlas )
		{
			if ( Layout != GpuLayout.Tlas || !uploaded )
			{
				throw new InvalidOperationException( "GpuTracer.UpdateTlas( .. ), no TLAS was uploaded." );
			}
			ValidateTlas( ref tlas );
			UploadTopLevel( ref tlas );
			int render = renderKernels[ ( int )GpuLayout.Tlas ];
			int batch = batchKernels[ ( int )GpuLayout.Tlas ];
			SetBuffer( render, batch, tlasNodesId, tlasNodes );
			SetBuffer( render, batch, tlasIdxId, tlasIdx );
			SetBuffer( render, batch, instancesId, instances );
		}

		private static void ValidateTlas( ref Bvh tlas )
		{
			if ( tlas.UsedNodes == 0 )
			{
				throw new InvalidOperationException( "GpuTracer: the TLAS was not built." );
			}
			if ( !tlas.IsTlas )
			{
				throw new ArgumentException( "GpuTracer: the bvh is not a TLAS.", nameof( tlas ) );
			}
		}

		private static void ValidateBlasCount( int blasCount )
		{
			if ( blasCount == 0 )
			{
				throw new ArgumentException( "GpuTracer.UploadTlas( .. ), no BLASses were given." );
			}
		}

		/// <summary>Uploads the descriptors and the top level, then binds everything to the TLAS kernels.</summary>
		private void FinishTlasUpload( ref Bvh tlas, GpuBlasDesc[] descs, int blasNodes )
		{
			Layout = GpuLayout.Tlas;
			// UploadTlas rejects indexed BLASses, so the TLAS kernels always use plain addressing;
			// the buffer still has to be bound because they compile TinyPrimIndices.
			indexed = false;
			EnsureBuffer( ref vertIdx, 0, UintStride );
			blasNodeCount = blasNodes;
			EnsureBuffer( ref blasDesc, descs.Length, BlasDescStride );
			blasDesc.SetData( descs );
			UploadTopLevel( ref tlas );
			int render = renderKernels[ ( int )GpuLayout.Tlas ];
			int batch = batchKernels[ ( int )GpuLayout.Tlas ];
			SetBuffer( render, batch, bvhGpuNodesId, bvhGpuNodes );
			SetBuffer( render, batch, primIdxId, primIdx );
			SetBuffer( render, batch, vertsId, verts );
			SetBuffer( render, batch, vertIdxId, vertIdx );
			SetBuffer( render, batch, opMapId, opMap );
			SetBuffer( render, batch, cwbvhNodesId, cwbvhNodes );
			SetBuffer( render, batch, cwbvhTrisId, cwbvhTris );
			SetBuffer( render, batch, tlasNodesId, tlasNodes );
			SetBuffer( render, batch, tlasIdxId, tlasIdx );
			SetBuffer( render, batch, instancesId, instances );
			SetBuffer( render, batch, blasDescId, blasDesc );
			uploaded = true;
		}

		/// <summary>Converts the TLAS to the Aila-Laine layout and uploads it with its instances.</summary>
		private void UploadTopLevel( ref Bvh tlas )
		{
			if ( !tlasGpu.IsCreated )
			{
				tlasGpu = BvhGpu.Create( Allocator.Persistent );
			}
			tlasGpu.ConvertFrom( ref tlas );
			// Node count of the whole structure: the top level plus every BLAS.
			NodeCount = tlasGpu.UsedNodes + blasNodeCount;
			int nodeFloat4Count = ( int )tlasGpu.UsedNodes * 4;
			EnsureBuffer( ref tlasNodes, nodeFloat4Count, Float4Stride );
			Fill<float4>( tlasNodes, tlasGpu.Nodes, nodeFloat4Count );
			EnsureBuffer( ref tlasIdx, ( int )tlas.IdxCount, UintStride );
			Fill<uint>( tlasIdx, tlas.PrimIdx, ( int )tlas.IdxCount );
			// BlasInstance is the OpenCL "struct Instance" without its dummy[8] padding: the two
			// 64-byte matrices, then aabbMin with blasIdx in w and aabbMax with mask in w.
			EnsureBuffer( ref instances, ( int )tlas.InstanceCount, sizeof( BlasInstance ) );
			Fill<BlasInstance>( instances, tlas.Instances, ( int )tlas.InstanceCount );
		}

		/// <summary>
		/// Traces one primary ray per pixel of target and shades it. target must have
		/// enableRandomWrite set and be created.
		/// </summary>
		public void Render( RenderTexture target, float3 origin, float3 topLeftDir, float3 horizontal,
			float3 vertical, int mode, bool shadows, float3 sunDir, float sceneDiagonal )
		{
			if ( !uploaded )
			{
				throw new InvalidOperationException( "GpuTracer.Render( .. ), nothing was uploaded." );
			}
			int kernel = renderKernels[ ( int )Layout ];
			shader.SetVector( originId, new Vector4( origin.x, origin.y, origin.z, 0f ) );
			shader.SetVector( topLeftDirId, new Vector4( topLeftDir.x, topLeftDir.y, topLeftDir.z, 0f ) );
			shader.SetVector( horizontalId, new Vector4( horizontal.x, horizontal.y, horizontal.z, 0f ) );
			shader.SetVector( verticalId, new Vector4( vertical.x, vertical.y, vertical.z, 0f ) );
			shader.SetVector( sunDirId, new Vector4( sunDir.x, sunDir.y, sunDir.z, 0f ) );
			shader.SetFloat( sceneDiagonalId, sceneDiagonal );
			shader.SetInt( modeId, mode );
			shader.SetInt( indexedId, indexed ? 1 : 0 );
			shader.SetInt( opMapNId, ( int )opMapN );
			shader.SetInt( shadowsId, shadows ? 1 : 0 );
			shader.SetInt( widthId, target.width );
			shader.SetInt( heightId, target.height );
			shader.SetTexture( kernel, targetId, target );
			int groupsX = ( target.width + RenderGroupSize - 1 ) / RenderGroupSize;
			int groupsY = ( target.height + RenderGroupSize - 1 ) / RenderGroupSize;
			shader.Dispatch( kernel, groupsX, groupsY, 1 );
		}

		/// <summary>
		/// Traces origins.Length rays and reads the results back synchronously. The w of each
		/// origin is the ray's tmax, used both for the closest-hit and the any-hit query.
		/// </summary>
		public void TraceBatch( NativeArray<float4> origins, NativeArray<float4> directions,
			NativeArray<float4> hitResults, NativeArray<uint> occludedResults )
		{
			TraceBatch( origins, directions, hitResults, occludedResults, default );
		}

		/// <summary>
		/// TraceBatch with the fifth result the TLAS kernel produces: the index of the hit
		/// instance, written to its own buffer so the Hits record stays four values wide. Pass a
		/// default (uncreated) array for the layouts that do not have instances.
		/// </summary>
		public void TraceBatch( NativeArray<float4> origins, NativeArray<float4> directions,
			NativeArray<float4> hitResults, NativeArray<uint> occludedResults, NativeArray<uint> instanceResults )
		{
			if ( !uploaded )
			{
				throw new InvalidOperationException( "GpuTracer.TraceBatch( .. ), nothing was uploaded." );
			}
			int count = origins.Length;
			if ( directions.Length != count || hitResults.Length != count || occludedResults.Length != count )
			{
				throw new ArgumentException( "GpuTracer.TraceBatch( .. ), all four arrays must have the same length." );
			}
			if ( instanceResults.IsCreated && instanceResults.Length != count )
			{
				throw new ArgumentException( "GpuTracer.TraceBatch( .. ), instanceResults must have the same length." );
			}
			EnsureBuffer( ref rayOrigins, count, Float4Stride );
			EnsureBuffer( ref rayDirections, count, Float4Stride );
			EnsureBuffer( ref hits, count, Float4Stride );
			EnsureBuffer( ref occluded, count, UintStride );
			rayOrigins.SetData( origins );
			rayDirections.SetData( directions );
			int kernel = batchKernels[ ( int )Layout ];
			shader.SetInt( rayCountId, count );
			shader.SetInt( indexedId, indexed ? 1 : 0 );
			shader.SetInt( opMapNId, ( int )opMapN );
			shader.SetBuffer( kernel, rayOriginsId, rayOrigins );
			shader.SetBuffer( kernel, rayDirectionsId, rayDirections );
			shader.SetBuffer( kernel, hitsId, hits );
			shader.SetBuffer( kernel, occludedId, occluded );
			if ( Layout == GpuLayout.Tlas )
			{
				EnsureBuffer( ref hitInstances, count, UintStride );
				shader.SetBuffer( kernel, hitInstancesId, hitInstances );
			}
			shader.Dispatch( kernel, ( count + BatchGroupSize - 1 ) / BatchGroupSize, 1, 1 );
			AsyncGPUReadback.RequestIntoNativeArray( ref hitResults, hits ).WaitForCompletion();
			AsyncGPUReadback.RequestIntoNativeArray( ref occludedResults, occluded ).WaitForCompletion();
			if ( instanceResults.IsCreated && Layout == GpuLayout.Tlas )
			{
				AsyncGPUReadback.RequestIntoNativeArray( ref instanceResults, hitInstances ).WaitForCompletion();
			}
		}

		public void Dispose()
		{
			Release( ref bvh2Nodes );
			Release( ref bvhGpuNodes );
			Release( ref primIdx );
			Release( ref verts );
			Release( ref vertIdx );
			Release( ref opMap );
			Release( ref bvh4Data );
			Release( ref cwbvhNodes );
			Release( ref cwbvhTris );
			Release( ref tlasNodes );
			Release( ref tlasIdx );
			Release( ref instances );
			Release( ref blasDesc );
			Release( ref rayOrigins );
			Release( ref rayDirections );
			Release( ref hits );
			Release( ref occluded );
			Release( ref hitInstances );
			if ( tlasGpu.IsCreated )
			{
				tlasGpu.Dispose();
				tlasGpu = default;
			}
			uploaded = false;
		}

		/// <summary>Uploads the primitive index array and the vertex buffer the BVH2 layouts index into.</summary>
		private void UploadTriangles( ref Bvh bvh )
		{
			int idxCount = ( int )bvh.IdxCount;
			EnsureBuffer( ref primIdx, idxCount, UintStride );
			Fill<uint>( primIdx, bvh.PrimIdx, idxCount );
			if ( bvh.VertStride != 16 )
			{
				throw new NotSupportedException( "GpuTracer: only a 16-byte vertex stride is supported." );
			}
			int vertCount = ( int )bvh.VertCount;
			EnsureBuffer( ref verts, vertCount, Float4Stride );
			Fill<float4>( verts, bvh.Verts, vertCount );
			// A BVH over indexed geometry needs its vertex indices too, three per primitive; the
			// buffer still has to be bound when there are none, since the kernels reference it.
			indexed = bvh.VertIdx != null;
			int vertIdxCount = indexed ? ( int )bvh.TriCount * 3 : 0;
			EnsureBuffer( ref vertIdx, vertIdxCount, UintStride );
			Fill<uint>( vertIdx, bvh.VertIdx, vertIdxCount );
			// The opacity micro map, when the BVH carries one. Only the primitives' own words are
			// uploaded: the unclamped index the C++ computes for a hit with u + v == 1 exactly can
			// reach past the last of them, and a StructuredBuffer read past the end returns zero,
			// which is what the slack words at the end of a CPU-side map buffer hold too.
			opMapN = bvh.OpMap != null ? bvh.OpMapN : 0u;
			int opMapCount = ( int )( bvh.TriCount * OpMapWords( opMapN ) );
			EnsureBuffer( ref opMap, opMapCount, UintStride );
			Fill<uint>( opMap, bvh.OpMap, opMapCount );
		}

		/// <summary>uint words one primitive's opacity micro map takes, tinybvh's ( N * N + 31 ) >> 5.</summary>
		private static uint OpMapWords( uint n )
		{
			return ( ( n * n ) + 31 ) >> 5;
		}

		/// <summary>Opacity micro map subdivision of a BLAS's source BVH, zero when it has no map.</summary>
		private static uint BlasOpMapN( in BvhGpu blas )
		{
			return blas.Source.OpMap != null ? blas.Source.OpMapN : 0u;
		}

		private void SetBuffer( int renderKernel, int batchKernel, int nameId, GraphicsBuffer buffer )
		{
			shader.SetBuffer( renderKernel, nameId, buffer );
			shader.SetBuffer( batchKernel, nameId, buffer );
		}

		private static void EnsureBuffer( ref GraphicsBuffer buffer, int count, int stride )
		{
			int wanted = math.max( count, 1 );
			if ( buffer != null && buffer.count == wanted && buffer.stride == stride )
			{
				return;
			}
			buffer?.Release();
			buffer = new GraphicsBuffer( GraphicsBuffer.Target.Structured, wanted, stride );
		}

		private static void Release( ref GraphicsBuffer buffer )
		{
			if ( buffer != null )
			{
				buffer.Release();
				buffer = null;
			}
		}

		/// <summary>
		/// Copies count elements from an unmanaged pointer into buffer without an intermediate
		/// managed array, by wrapping the memory in an alias NativeArray.
		/// </summary>
		private static void Fill<T>( GraphicsBuffer buffer, void* source, int count, int destOffset = 0 ) where T : struct
		{
			if ( count == 0 )
			{
				return;
			}
			NativeArray<T> alias = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<T>( source, count, Allocator.None );
#if ENABLE_UNITY_COLLECTIONS_CHECKS
			AtomicSafetyHandle safety = AtomicSafetyHandle.Create();
			NativeArrayUnsafeUtility.SetAtomicSafetyHandle( ref alias, safety );
#endif
			buffer.SetData( alias, 0, destOffset, count );
#if ENABLE_UNITY_COLLECTIONS_CHECKS
			AtomicSafetyHandle.Release( safety );
#endif
		}
	}
}
