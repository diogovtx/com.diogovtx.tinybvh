using System.Collections.Generic;
using Stopwatch = System.Diagnostics.Stopwatch;
using System.Text;
using TinyBVH;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.InputSystem;

namespace TinyBVH.Samples
{
	public enum SceneSource
	{
		BinFile,
		SceneMeshes
	}

	/// <summary>How the active backend keeps up with the deforming mesh.</summary>
	public enum DeformUpdate
	{
		Refit,
		Rebuild
	}

	/// <summary>
	/// Hosts a set of IRaytraceBackend instances against one scene, lets the user switch between
	/// them and their display modes from an on-screen UI, orbit the camera, and benchmark them.
	/// </summary>
	[RequireComponent( typeof( Camera ) )]
	public class RaytracerSample : MonoBehaviour
	{
		private const int BenchmarkWarmupFrames = 5;
		private const int BenchmarkTimedFrames = 30;
		private const float OrbitSpeed = 0.25f;
		private const float PanSpeed = 0.0015f;
		private const float ZoomSpeed = 0.001f;
		/// <summary>Turn rate of the animated instances, degrees per second.</summary>
		private const float AnimateSpeed = 40f;
		/// <summary>Peak displacement of the deformation, as a fraction of the source mesh diagonal.</summary>
		private const float DeformAmplitude = 0.02f;
		/// <summary>Number of full periods of the deformation wave across the source mesh.</summary>
		private const float DeformWaves = 3f;
		/// <summary>Travel rate of the deformation wave, periods per second.</summary>
		private const float DeformSpeed = 0.5f;
		/// <summary>Instance spacing as a multiple of the largest extent of the source mesh.</summary>
		private const float InstanceSpacing = 1.15f;
		/// <summary>Subdivision of the procedural opacity map the "Opacity map" toggle builds.</summary>
		private const uint OpacityMapSubdivision = 8;

		[SerializeField] private SceneSource sceneSource = SceneSource.BinFile;
		[SerializeField] private int sceneIndex;
		[SerializeField, Min( 1 )] private int renderScale = 2;
		[SerializeField] private DisplayMode displayMode = DisplayMode.Shaded;
		[SerializeField] private bool shadows = true;
		[SerializeField] private BvhBuilder builder = BvhBuilder.Binned;
		[SerializeField] private bool presplit;
		[SerializeField] private bool packets;
		[SerializeField] private bool threadedBuild;
		[SerializeField] private bool optimize;
		[SerializeField] private bool opacityMap;
		[SerializeField] private Light sun;
		[SerializeField] private bool autoFrame = true;
		[SerializeField] private int instanceGridIndex;
		[SerializeField] private bool animateInstances;
		[SerializeField] private bool deform;
		[SerializeField] private DeformUpdate deformUpdate = DeformUpdate.Refit;

		private readonly string[] binScenes = { "bunny.bin", "suzanne.bin", "cryteksponza.bin" };
		/// <summary>Panel labels of BvhBuilder, in enum order.</summary>
		private readonly string[] builderNames = { "Binned", "SBVH", "FullSweep", "Quick" };
		/// <summary>Side of the instance grid; the first entry is the un-instanced scene.</summary>
		private readonly int[] instanceGrids = { 1, 3, 5 };
		private readonly string[] instanceGridNames = { "1", "3x3", "5x5" };

		private Camera cam;
		private List<IRaytraceBackend> backends;
		private int activeBackend;

		private NativeArray<float4> vertices;
		private uint triCount;
		private float3 baseMin;
		private float3 baseMax;
		private float baseDiagonal = 0.0001f;
		// The undeformed source mesh: the deformation is recomputed from it every frame, into the
		// same 'vertices' array the backends were built over, so it does not accumulate.
		private NativeArray<float4> baseVertices;
		private float deformTime;
		// The source mesh transformed into every instance, for the backends without a TLAS.
		private NativeArray<float4> flatVertices;
		private uint flatTriCount;
		/// <summary>
		/// Procedural per-triangle cut-out pattern for the "Opacity map" toggle, sized for the
		/// flattened triangle count; see EnsureOpacityMapData.
		/// </summary>
		private NativeArray<uint> opacityMapData;
		private NativeArray<BlasInstance> instances;
		private float animAngle;
		private bool wasAnimating;
		private float3 sceneMin;
		private float3 sceneMax;
		private float sceneDiagonal = 0.0001f;

		private float3 pivot;
		private float distance = 1f;
		private float yaw;
		private float pitch;

		private Texture activeTexture;
		private float frameMsEma;
		private float mraysPerSec;
		private int lastWidth;
		private int lastHeight;

		private const int PanelWindowId = 1001;

		private bool showUI = true;
		private Rect panelRect;

		private bool benchmarkRunning;
		private int benchmarkBackendIndex;
		private int benchmarkFrameCount;
		private double benchmarkMsAccum;
		private int benchmarkSavedBackend;
		private List<BenchmarkResult> benchmarkResults;

		// Command-line benchmark mode: benchmarkOutputPath is null unless -benchmark was passed.
		private string benchmarkOutputPath;
		private string benchmarkScenePath;
		private int benchmarkWidth = 1280;
		private int benchmarkHeight = 720;

		private struct BenchmarkResult
		{
			public string Name;
			public float AvgMs;
			public float MraysPerSec;
			public float BuildMs;
			public long Nodes;
		}

		private static float3 ToFloat3( Vector3 v )
		{
			return new float3( v.x, v.y, v.z );
		}

		private static float4 ToFloat4( Vector3 v )
		{
			return new float4( v.x, v.y, v.z, 0f );
		}

		private static Vector3 ToVector3( float3 v )
		{
			return new Vector3( v.x, v.y, v.z );
		}

		private void Awake()
		{
			cam = GetComponent<Camera>();
			backends = BackendList.Create();
		}

		private void Start()
		{
			ParseCommandLine();
			bool cliMode = benchmarkOutputPath != null;
			if ( cliMode )
			{
				renderScale = 1;
				if ( !string.IsNullOrEmpty( benchmarkScenePath ) )
				{
					sceneSource = SceneSource.BinFile;
				}
#if !UNITY_EDITOR
				else
				{
					Debug.LogError( "TinyBVH: -benchmark requires -scene <path-to-.bin> outside the editor (TestData is not shipped in a player)." );
					// Quit is deferred to the end of the frame, so stop Update from rendering
					// through backends that were never built.
					enabled = false;
					Application.Quit();
					return;
				}
#endif
			}
			LoadScene();
			if ( cliMode )
			{
				StartBenchmark();
			}
		}

		/// <summary>
		/// Parses `-benchmark <path> [-scene <path>] [-width <w>] [-height <h>]` from the command line,
		/// enabling headless benchmark mode. Unrecognised arguments (including Unity's own, e.g.
		/// `-screen-width`) are ignored.
		/// </summary>
		private void ParseCommandLine()
		{
			string[] args = System.Environment.GetCommandLineArgs();
			for ( int i = 0; i < args.Length; i++ )
			{
				switch ( args[ i ] )
				{
					case "-benchmark":
						if ( i + 1 < args.Length )
						{
							benchmarkOutputPath = args[ ++i ];
						}
						break;
					case "-scene":
						if ( i + 1 < args.Length )
						{
							benchmarkScenePath = args[ ++i ];
						}
						break;
					case "-width":
						if ( i + 1 < args.Length && int.TryParse( args[ i + 1 ], out int width ) )
						{
							benchmarkWidth = width;
							i++;
						}
						break;
					case "-height":
						if ( i + 1 < args.Length && int.TryParse( args[ i + 1 ], out int height ) )
						{
							benchmarkHeight = height;
							i++;
						}
						break;
				}
			}
		}

		private void Update()
		{
			HandleInput();
			HandleCamera();
			UpdateAnimation();
			float ms = RenderFrame();
			if ( benchmarkRunning )
			{
				AdvanceBenchmark( ms );
			}
		}

		private void OnDestroy()
		{
			if ( vertices.IsCreated )
			{
				vertices.Dispose();
			}
			if ( baseVertices.IsCreated )
			{
				baseVertices.Dispose();
			}
			if ( flatVertices.IsCreated )
			{
				flatVertices.Dispose();
			}
			if ( opacityMapData.IsCreated )
			{
				opacityMapData.Dispose();
			}
			if ( instances.IsCreated )
			{
				instances.Dispose();
			}
			foreach ( IRaytraceBackend backend in backends )
			{
				backend.Dispose();
			}
		}

		/// <summary>Positions the camera to frame the loaded scene, using its vertex bounds.</summary>
		public void FrameScene()
		{
			pivot = ( sceneMin + sceneMax ) * 0.5f;
			float radius = math.length( sceneMax - sceneMin ) * 0.5f;
			distance = math.max( radius * 2.2f, 0.01f );
			Transform t = cam.transform;
			float3 forward = ToFloat3( t.forward );
			if ( math.lengthsq( forward ) < 1e-6f )
			{
				forward = new float3( 0f, 0f, 1f );
			}
			float3 pos = pivot - ( forward * distance );
			Quaternion rot = Quaternion.LookRotation( ToVector3( pivot - pos ), Vector3.up );
			Vector3 euler = rot.eulerAngles;
			pitch = euler.x > 180f ? euler.x - 360f : euler.x;
			yaw = euler.y;
			ApplyCameraTransform();
		}

		private void LoadScene()
		{
			if ( vertices.IsCreated )
			{
				vertices.Dispose();
			}
			if ( sceneSource == SceneSource.BinFile )
			{
				LoadBinFile();
			}
			else
			{
				LoadSceneMeshes();
			}
			ComputeBounds( vertices, out baseMin, out baseMax );
			baseDiagonal = math.max( math.length( baseMax - baseMin ), 0.0001f );
			CaptureBaseVertices();
			ApplyBuildOptions();
			RebuildInstancedScene();
		}

		/// <summary>Keeps a copy of the freshly loaded, undeformed source mesh for the deformation to work from.</summary>
		private void CaptureBaseVertices()
		{
			if ( baseVertices.IsCreated && baseVertices.Length != vertices.Length )
			{
				baseVertices.Dispose();
			}
			if ( !baseVertices.IsCreated )
			{
				baseVertices = new NativeArray<float4>( vertices.Length, Allocator.Persistent );
			}
			baseVertices.CopyFrom( vertices );
			deformTime = 0f;
		}

		/// <summary>Pushes the build-option settings onto every backend, so the next build picks them up.</summary>
		private void ApplyBuildOptions()
		{
			EnsureOpacityMapData();
			foreach ( IRaytraceBackend backend in backends )
			{
				backend.Builder = builder;
				backend.Presplit = presplit;
				backend.ThreadedBuild = threadedBuild;
				backend.Optimize = optimize;
				backend.OpacityMap = opacityMap ? opacityMapData : default;
				backend.OpacityMapN = opacityMap ? OpacityMapSubdivision : 0u;
				// Packet traversal is a render option of the BVH2 backend alone, not a build option,
				// but this is where the serialized settings reach the backends.
				if ( backend is CpuBvhBackend cpuBvh )
				{
					cpuBvh.Packets = packets;
				}
			}
		}

		/// <summary>
		/// Keeps opacityMapData sized for the flattened triangle count of the current scene and
		/// instance grid (triCount * side * side, the same formula FlattenVertices uses), so it is
		/// correctly sized before ApplyBuildOptions hands it to the backends. Regenerated only when
		/// that size changes. Each flattened copy of a triangle shares its source triangle's cut-out
		/// pattern: micro-triangle b of triangle t is opaque when ( ( ( t % triCount ) * 7 + b * 13 )
		/// &amp; 3 ) != 0, one in four holes, N = OpacityMapSubdivision micro-triangles per side.
		/// </summary>
		private void EnsureOpacityMapData()
		{
			int side = instanceGrids[ instanceGridIndex ];
			uint flatCount = triCount * ( uint )( side * side );
			uint words = ( ( OpacityMapSubdivision * OpacityMapSubdivision ) + 31 ) >> 5;
			int wanted = ( int )( ( flatCount * words ) + 2 );
			if ( opacityMapData.IsCreated && opacityMapData.Length == wanted )
			{
				return;
			}
			if ( opacityMapData.IsCreated )
			{
				opacityMapData.Dispose();
			}
			opacityMapData = new NativeArray<uint>( wanted, Allocator.Persistent );
			for ( uint t = 0; t < flatCount; t++ )
			{
				uint srcTri = t % triCount;
				for ( uint b = 0; b < OpacityMapSubdivision * OpacityMapSubdivision; b++ )
				{
					if ( ( ( ( srcTri * 7u ) + ( b * 13u ) ) & 3u ) != 0u )
					{
						int w = ( int )( ( t * words ) + ( b >> 5 ) );
						opacityMapData[ w ] = opacityMapData[ w ] | ( 1u << ( int )( b & 31 ) );
					}
				}
			}
		}

		/// <summary>
		/// Rebuilds the instance list and the flattened geometry for the current instance count,
		/// then every backend: the TLAS backends over the source mesh plus the instances, the
		/// others over the flattened copy.
		/// </summary>
		private void RebuildInstancedScene()
		{
			BuildInstanceList();
			FlattenVertices();
			ComputeSceneBounds();
			BuildBackends();
			if ( autoFrame )
			{
				FrameScene();
			}
		}

		private void BuildBackends()
		{
			foreach ( IRaytraceBackend backend in backends )
			{
				if ( backend is IInstancedBackend instanced )
				{
					instanced.BuildInstanced( vertices, triCount, instances );
				}
				else
				{
					backend.Build( flatVertices, flatTriCount );
				}
				Debug.Log( $"TinyBVH: {backend.Name} built {( backend is IInstancedBackend ? triCount : flatTriCount )} triangles into {backend.NodeCount} nodes in {backend.BuildMs:F2} ms" );
			}
		}

		/// <summary>Creates one instance per grid cell, then gives them their transforms.</summary>
		private void BuildInstanceList()
		{
			int side = instanceGrids[ instanceGridIndex ];
			int count = side * side;
			if ( instances.IsCreated && instances.Length != count )
			{
				instances.Dispose();
			}
			if ( !instances.IsCreated )
			{
				instances = new NativeArray<BlasInstance>( count, Allocator.Persistent );
			}
			for ( int i = 0; i < count; i++ )
			{
				instances[ i ] = BlasInstance.Create( 0 );
			}
			UpdateInstanceTransforms();
		}

		/// <summary>
		/// Lays the instances out on a grid spaced by the source mesh bounds, each rotated about
		/// its own centre and scaled down a little, so the copies are visibly different. With a
		/// single instance and no animation this is the identity, i.e. the un-instanced scene.
		/// </summary>
		private void UpdateInstanceTransforms()
		{
			int side = instanceGrids[ instanceGridIndex ];
			float3 center = ( baseMin + baseMax ) * 0.5f;
			float spacing = math.max( math.cmax( baseMax - baseMin ), 1e-4f ) * InstanceSpacing;
			for ( int z = 0; z < side; z++ )
			{
				for ( int x = 0; x < side; x++ )
				{
					int i = ( z * side ) + x;
					float3 cell = new float3(
						( x - ( ( side - 1 ) * 0.5f ) ) * spacing,
						0f,
						( z - ( ( side - 1 ) * 0.5f ) ) * spacing );
					float angle = math.radians( animAngle + ( i * 37f ) );
					float scale = 1f - ( 0.125f * ( i % 3 ) );
					float4x4 m = math.mul(
						float4x4.Translate( cell + center ),
						math.mul(
							float4x4.TRS( float3.zero, quaternion.RotateY( angle ), new float3( scale ) ),
							float4x4.Translate( -center ) ) );
					BlasInstance inst = instances[ i ];
					inst.Transform = BvhMat4.FromFloat4x4( m );
					instances[ i ] = inst;
				}
			}
		}

		/// <summary>Writes the source mesh transformed by every instance into flatVertices.</summary>
		private void FlattenVertices()
		{
			int count = instances.Length;
			int total = vertices.Length * count;
			if ( flatVertices.IsCreated && flatVertices.Length != math.max( total, 1 ) )
			{
				flatVertices.Dispose();
			}
			if ( !flatVertices.IsCreated )
			{
				flatVertices = new NativeArray<float4>( math.max( total, 1 ), Allocator.Persistent );
			}
			flatTriCount = triCount * ( uint )count;
			if ( total == 0 )
			{
				return;
			}
			FlattenJob job = new FlattenJob
			{
				Source = vertices,
				Instances = instances,
				Dest = flatVertices,
				VertexCount = vertices.Length
			};
			job.Schedule( count, 1 ).Complete();
		}

		/// <summary>
		/// Advances the animation and the deformation and refreshes the active backend only: a TLAS
		/// backend just rebuilds its TLAS, the others rebuild over the re-flattened geometry, and a
		/// deforming mesh is refit instead where that was asked for and the backend can do it. The
		/// backends that were not active are resynchronised once both stop.
		/// </summary>
		private void UpdateAnimation()
		{
			if ( animateInstances || deform )
			{
				if ( deform )
				{
					deformTime += Time.deltaTime;
					DeformVertices();
				}
				if ( animateInstances )
				{
					animAngle += Time.deltaTime * AnimateSpeed;
					UpdateInstanceTransforms();
				}
				RefreshActiveBackend();
				wasAnimating = true;
			}
			else if ( wasAnimating )
			{
				wasAnimating = false;
				RestoreVertices();
				FlattenVertices();
				BuildBackends();
			}
		}

		/// <summary>
		/// Brings the active backend back in sync with the current geometry and instances: a refit
		/// when the mesh is deforming and <see cref="CanRefit"/> allows it, a TLAS rebuild when only
		/// the instances moved, and a full build otherwise.
		/// </summary>
		private void RefreshActiveBackend()
		{
			IRaytraceBackend active = backends[ activeBackend ];
			if ( !( active is IInstancedBackend ) )
			{
				// The non-TLAS backends trace the flattened copy, so it has to follow the source mesh.
				FlattenVertices();
			}
			if ( CanRefit( active ) && active is IRefittableBackend refittable )
			{
				refittable.Refit();
			}
			else if ( active is IInstancedBackend instanced )
			{
				if ( deform )
				{
					instanced.BuildInstanced( vertices, triCount, instances );
				}
				else
				{
					instanced.UpdateInstances( instances );
				}
			}
			else
			{
				active.Build( flatVertices, flatTriCount );
			}
		}

		/// <summary>
		/// True when the backend can follow the deforming mesh with a refit. A spatial-split build
		/// and a presplit build are both not refittable, so the sample falls back to a rebuild and
		/// says so in the stats.
		/// </summary>
		private bool CanRefit( IRaytraceBackend backend )
		{
			return deform && deformUpdate == DeformUpdate.Refit && IsRefittableBuild && backend is IRefittableBackend;
		}

		/// <summary>False for the builds Bvh.Refit throws on: spatial splits and presplitting.</summary>
		private bool IsRefittableBuild => builder != BvhBuilder.Sbvh && !BvhBuilderOptions.PresplitApplies( builder, presplit );

		/// <summary>
		/// Displaces the source mesh with a travelling sine wave, written from the undeformed copy
		/// into the very array the backends were built over, so they see the new positions in place.
		/// </summary>
		private void DeformVertices()
		{
			if ( !baseVertices.IsCreated || baseVertices.Length == 0 )
			{
				return;
			}
			DeformJob job = new DeformJob
			{
				Source = baseVertices,
				Dest = vertices,
				Amplitude = baseDiagonal * DeformAmplitude,
				Frequency = ( DeformWaves * 2f * math.PI ) / baseDiagonal,
				Phase = deformTime * DeformSpeed * 2f * math.PI
			};
			job.Schedule( vertices.Length, 1024 ).Complete();
		}

		/// <summary>Puts the undeformed source mesh back, for when the deformation is switched off.</summary>
		private void RestoreVertices()
		{
			if ( baseVertices.IsCreated && baseVertices.Length == vertices.Length )
			{
				vertices.CopyFrom( baseVertices );
			}
		}

		private void LoadBinFile()
		{
			string path = string.IsNullOrEmpty( benchmarkScenePath ) ? BvhSceneFile.TestDataPath( binScenes[ sceneIndex ] ) : benchmarkScenePath;
			vertices = BvhSceneFile.Load( path, Allocator.Persistent, out triCount );
		}

		private void LoadSceneMeshes()
		{
			MeshFilter[] filters = FindObjectsByType<MeshFilter>( FindObjectsSortMode.None );
			List<float4> verts = new List<float4>();
			foreach ( MeshFilter filter in filters )
			{
				if ( filter == null || !filter.gameObject.activeInHierarchy )
				{
					continue;
				}
				MeshRenderer renderer = filter.GetComponent<MeshRenderer>();
				if ( renderer != null && !renderer.enabled )
				{
					continue;
				}
				Mesh mesh = filter.sharedMesh;
				if ( mesh == null )
				{
					continue;
				}
				if ( !mesh.isReadable )
				{
					Debug.LogWarning( $"TinyBVH: skipping non-readable mesh '{mesh.name}' on '{filter.name}'. Enable Read/Write on the mesh import settings." );
					continue;
				}
				Matrix4x4 localToWorld = filter.transform.localToWorldMatrix;
				Vector3[] meshVerts = mesh.vertices;
				for ( int sub = 0; sub < mesh.subMeshCount; sub++ )
				{
					int[] tris = mesh.GetTriangles( sub );
					for ( int i = 0; i < tris.Length; i += 3 )
					{
						verts.Add( ToFloat4( localToWorld.MultiplyPoint3x4( meshVerts[ tris[ i ] ] ) ) );
						verts.Add( ToFloat4( localToWorld.MultiplyPoint3x4( meshVerts[ tris[ i + 1 ] ] ) ) );
						verts.Add( ToFloat4( localToWorld.MultiplyPoint3x4( meshVerts[ tris[ i + 2 ] ] ) ) );
					}
				}
			}
			if ( verts.Count == 0 )
			{
				Debug.LogWarning( "TinyBVH: no readable meshes found for SceneMeshes source." );
			}
			triCount = ( uint )( verts.Count / 3 );
			vertices = new NativeArray<float4>( verts.Count, Allocator.Persistent );
			for ( int i = 0; i < verts.Count; i++ )
			{
				vertices[ i ] = verts[ i ];
			}
		}

		private void ComputeSceneBounds()
		{
			ComputeBounds( flatVertices, out sceneMin, out sceneMax );
			sceneDiagonal = math.max( math.length( sceneMax - sceneMin ), 0.0001f );
		}

		private static void ComputeBounds( NativeArray<float4> verts, out float3 min, out float3 max )
		{
			if ( !verts.IsCreated || verts.Length == 0 )
			{
				min = float3.zero;
				max = float3.zero;
				return;
			}
			min = verts[ 0 ].xyz;
			max = min;
			for ( int i = 1; i < verts.Length; i++ )
			{
				float3 p = verts[ i ].xyz;
				min = math.min( min, p );
				max = math.max( max, p );
			}
		}

		/// <summary>
		/// Displaces one vertex along Y by a sine wave travelling over x + z: smooth, deterministic
		/// in position and time, and always computed from the undeformed vertex.
		/// </summary>
		[BurstCompile]
		private struct DeformJob : IJobParallelFor
		{
			[ReadOnly] public NativeArray<float4> Source;
			public NativeArray<float4> Dest;
			public float Amplitude;
			public float Frequency;
			public float Phase;

			public void Execute( int i )
			{
				float4 v = Source[ i ];
				float wave = math.sin( ( ( v.x + v.z ) * Frequency ) - Phase );
				Dest[ i ] = new float4( v.x, v.y + ( wave * Amplitude ), v.z, v.w );
			}
		}

		/// <summary>Transforms the source mesh by one instance transform per job index.</summary>
		[BurstCompile]
		private struct FlattenJob : IJobParallelFor
		{
			[ReadOnly] public NativeArray<float4> Source;
			[ReadOnly] public NativeArray<BlasInstance> Instances;
			[NativeDisableParallelForRestriction] public NativeArray<float4> Dest;
			public int VertexCount;

			public void Execute( int instance )
			{
				BvhMat4 m = Instances[ instance ].Transform;
				int offset = instance * VertexCount;
				for ( int i = 0; i < VertexCount; i++ )
				{
					Dest[ offset + i ] = new float4( m.TransformPoint( Source[ i ].xyz ), 0f );
				}
			}
		}

		private void HandleInput()
		{
			Keyboard kb = Keyboard.current;
			if ( kb == null )
			{
				return;
			}
			if ( kb.tabKey.wasPressedThisFrame )
			{
				showUI = !showUI;
			}
			for ( int i = 0; i < 5; i++ )
			{
				if ( kb[ ( Key )( ( int )Key.Digit1 + i ) ].wasPressedThisFrame )
				{
					displayMode = ( DisplayMode )i;
				}
			}
			if ( kb.sKey.wasPressedThisFrame )
			{
				shadows = !shadows;
			}
		}

		private bool IsPointerOverPanel()
		{
			if ( !showUI )
			{
				return false;
			}
			Mouse mouse = Mouse.current;
			if ( mouse == null )
			{
				return false;
			}
			Vector2 screenPos = mouse.position.ReadValue();
			Vector2 guiPos = new Vector2( screenPos.x, Screen.height - screenPos.y );
			return panelRect.Contains( guiPos );
		}

		private void HandleCamera()
		{
			Mouse mouse = Mouse.current;
			if ( mouse == null || IsPointerOverPanel() )
			{
				return;
			}
			Vector2 delta = mouse.delta.ReadValue();
			bool changed = false;
			if ( mouse.rightButton.isPressed )
			{
				yaw += delta.x * OrbitSpeed;
				pitch = Mathf.Clamp( pitch - ( delta.y * OrbitSpeed ), -89f, 89f );
				changed = true;
			}
			else if ( mouse.middleButton.isPressed )
			{
				Transform t = cam.transform;
				float panScale = distance * PanSpeed;
				pivot -= ToFloat3( t.right ) * delta.x * panScale;
				pivot += ToFloat3( t.up ) * delta.y * panScale;
				changed = true;
			}
			float scroll = mouse.scroll.ReadValue().y;
			if ( !Mathf.Approximately( scroll, 0f ) )
			{
				distance = Mathf.Max( distance - ( scroll * ZoomSpeed * sceneDiagonal ), sceneDiagonal * 0.001f );
				changed = true;
			}
			if ( changed )
			{
				ApplyCameraTransform();
			}
		}

		private void ApplyCameraTransform()
		{
			Quaternion rot = Quaternion.Euler( pitch, yaw, 0f );
			Vector3 pos = ToVector3( pivot ) - ( rot * Vector3.forward * distance );
			cam.transform.SetPositionAndRotation( pos, rot );
		}

		private RenderSettings BuildRenderSettings( int width, int height )
		{
			Transform t = cam.transform;
			float tanFovY = math.tan( math.radians( cam.fieldOfView ) * 0.5f );
			float tanFovX = tanFovY * ( ( float )width / height );
			float3 forward = ToFloat3( t.forward );
			float3 right = ToFloat3( t.right );
			float3 up = ToFloat3( t.up );
			float3 sunDir = sun != null ? ToFloat3( sun.transform.forward ) : math.normalize( new float3( 0.5f, -1f, 0.3f ) );
			return new RenderSettings
			{
				Width = width,
				Height = height,
				Origin = ToFloat3( t.position ),
				TopLeftDir = forward - ( right * tanFovX ) + ( up * tanFovY ),
				Horizontal = right * tanFovX * 2f,
				Vertical = -up * tanFovY * 2f,
				Mode = displayMode,
				Shadows = shadows,
				SunDir = sunDir,
				SceneDiagonal = sceneDiagonal
			};
		}

		private float RenderFrame()
		{
			int width = benchmarkOutputPath != null ? benchmarkWidth : Mathf.Max( 1, cam.pixelWidth / renderScale );
			int height = benchmarkOutputPath != null ? benchmarkHeight : Mathf.Max( 1, cam.pixelHeight / renderScale );
			RenderSettings settings = BuildRenderSettings( width, height );
			Stopwatch stopwatch = Stopwatch.StartNew();
			activeTexture = backends[ activeBackend ].Render( settings, true );
			stopwatch.Stop();
			float ms = ( float )stopwatch.Elapsed.TotalMilliseconds;
			frameMsEma = frameMsEma <= 0f ? ms : Mathf.Lerp( frameMsEma, ms, 0.1f );
			double seconds = math.max( frameMsEma, 0.001f ) / 1000.0;
			mraysPerSec = ( float )( ( double )width * height * ( shadows ? 2 : 1 ) / seconds / 1e6 );
			lastWidth = width;
			lastHeight = height;
			return ms;
		}

		private void StartBenchmark()
		{
			if ( backends.Count == 0 || benchmarkRunning )
			{
				return;
			}
			benchmarkSavedBackend = activeBackend;
			benchmarkResults = new List<BenchmarkResult>();
			benchmarkBackendIndex = 0;
			benchmarkFrameCount = 0;
			benchmarkMsAccum = 0.0;
			activeBackend = 0;
			benchmarkRunning = true;
		}

		private void AdvanceBenchmark( float ms )
		{
			benchmarkFrameCount++;
			if ( benchmarkFrameCount <= BenchmarkWarmupFrames )
			{
				return;
			}
			benchmarkMsAccum += ms;
			if ( benchmarkFrameCount < BenchmarkWarmupFrames + BenchmarkTimedFrames )
			{
				return;
			}
			IRaytraceBackend backend = backends[ benchmarkBackendIndex ];
			float avgMs = ( float )( benchmarkMsAccum / BenchmarkTimedFrames );
			double seconds = avgMs / 1000.0;
			float mrays = ( float )( ( double )lastWidth * lastHeight * ( shadows ? 2 : 1 ) / seconds / 1e6 );
			benchmarkResults.Add( new BenchmarkResult
			{
				Name = backend.Name,
				AvgMs = avgMs,
				MraysPerSec = mrays,
				BuildMs = backend.BuildMs,
				Nodes = backend.NodeCount
			} );
			benchmarkBackendIndex++;
			benchmarkFrameCount = 0;
			benchmarkMsAccum = 0.0;
			if ( benchmarkBackendIndex >= backends.Count )
			{
				benchmarkRunning = false;
				activeBackend = benchmarkSavedBackend;
				LogBenchmarkResults();
				if ( benchmarkOutputPath != null )
				{
					WriteBenchmarkFile();
#if UNITY_EDITOR
					UnityEditor.EditorApplication.isPlaying = false;
#else
					Application.Quit();
#endif
				}
			}
			else
			{
				activeBackend = benchmarkBackendIndex;
			}
		}

		private void LogBenchmarkResults()
		{
			StringBuilder sb = new StringBuilder( "TinyBVH benchmark results:\n" );
			foreach ( BenchmarkResult r in benchmarkResults )
			{
				sb.AppendLine( $"{r.Name,-24} avg {r.AvgMs,7:F2} ms  {r.MraysPerSec,8:F1} Mrays/s  build {r.BuildMs,7:F2} ms  nodes {r.Nodes}" );
			}
			Debug.Log( sb.ToString() );
		}

		/// <summary>
		/// Writes the benchmark results to benchmarkOutputPath for the command-line benchmark mode:
		/// a header line with the machine/GPU/Unity version, then one tab-separated line per backend.
		/// </summary>
		private void WriteBenchmarkFile()
		{
			// A Burst AOT failure only shows up outside the editor, and it costs the CPU backends
			// an order of magnitude; say so in the log next to the numbers.
			Debug.Log( $"TinyBVH: Burst direct calls active: {BvhBurst.IsActive}" );
			try
			{
				StringBuilder sb = new StringBuilder();
				sb.AppendLine( $"# {SystemInfo.processorType}\t{SystemInfo.graphicsDeviceName}\t{Application.unityVersion}" );
				foreach ( BenchmarkResult r in benchmarkResults )
				{
					sb.AppendLine( string.Format( System.Globalization.CultureInfo.InvariantCulture, "{0}\t{1:F2}\t{2:F2}\t{3:F1}", r.Name, r.BuildMs, r.AvgMs, r.MraysPerSec ) );
				}
				System.IO.File.WriteAllText( benchmarkOutputPath, sb.ToString() );
			}
			catch ( System.Exception e )
			{
				Debug.LogError( $"TinyBVH: failed to write benchmark file '{benchmarkOutputPath}': {e}" );
			}
		}

		private void OnGUI()
		{
			if ( activeTexture != null )
			{
				GUI.DrawTexture( new Rect( 0, 0, Screen.width, Screen.height ), activeTexture, ScaleMode.StretchToFill, false );
			}
			if ( !showUI )
			{
				panelRect = new Rect( -1, -1, 0, 0 );
				return;
			}
			Rect windowRect = new Rect( 10, 10, 320, 0 );
			panelRect = GUILayout.Window( PanelWindowId, windowRect, DrawPanel, "TinyBVH Raytracer" );
		}

		private void DrawPanel( int windowId )
		{
			GUILayout.Label( "Backend" );
			string[] names = new string[ backends.Count ];
			for ( int i = 0; i < backends.Count; i++ )
			{
				names[ i ] = backends[ i ].IsGpu ? $"{backends[ i ].Name} (GPU)" : backends[ i ].Name;
			}
			int newBackend = GUILayout.SelectionGrid( activeBackend, names, 1 );
			if ( newBackend != activeBackend && !benchmarkRunning )
			{
				activeBackend = newBackend;
			}

			GUILayout.Space( 6 );
			GUILayout.Label( "Display mode" );
			displayMode = ( DisplayMode )GUILayout.SelectionGrid( ( int )displayMode, System.Enum.GetNames( typeof( DisplayMode ) ), 3 );
			shadows = GUILayout.Toggle( shadows, "Shadows" );
			if ( backends[ activeBackend ] is CpuBvhBackend packetBackend )
			{
				bool newPackets = GUILayout.Toggle( packets, "Packets (256 rays)" );
				if ( newPackets != packets && !benchmarkRunning )
				{
					packets = newPackets;
					packetBackend.Packets = newPackets;
				}
			}

			GUILayout.Space( 6 );
			GUILayout.Label( "Builder" );
			BvhBuilder newBuilder = ( BvhBuilder )GUILayout.SelectionGrid( ( int )builder, builderNames, builderNames.Length );
			if ( newBuilder != builder && !benchmarkRunning )
			{
				builder = newBuilder;
				ApplyBuildOptions();
				RebuildInstancedScene();
			}
			bool newPresplit = GUILayout.Toggle( presplit, "Presplit" );
			if ( newPresplit != presplit && !benchmarkRunning )
			{
				presplit = newPresplit;
				ApplyBuildOptions();
				RebuildInstancedScene();
			}
			if ( presplit && !BvhBuilderOptions.PresplitApplies( builder, presplit ) )
			{
				GUILayout.Label( "Presplit ignored by SBVH and Quick" );
			}
			bool newThreadedBuild = GUILayout.Toggle( threadedBuild, "Threaded build" );
			if ( newThreadedBuild != threadedBuild && !benchmarkRunning )
			{
				threadedBuild = newThreadedBuild;
				ApplyBuildOptions();
				RebuildInstancedScene();
			}
			bool newOptimize = GUILayout.Toggle( optimize, "Optimize (25 iterations)" );
			if ( newOptimize != optimize && !benchmarkRunning )
			{
				optimize = newOptimize;
				ApplyBuildOptions();
				RebuildInstancedScene();
			}
			bool newOpacityMap = GUILayout.Toggle( opacityMap, "Opacity map (N=8)" );
			if ( newOpacityMap != opacityMap && !benchmarkRunning )
			{
				opacityMap = newOpacityMap;
				ApplyBuildOptions();
				RebuildInstancedScene();
			}

			GUILayout.Space( 6 );
			GUILayout.Label( $"Render scale: {renderScale}x" );
			renderScale = Mathf.RoundToInt( GUILayout.HorizontalSlider( renderScale, 1, 8 ) );

			GUILayout.Space( 6 );
			GUILayout.Label( "Scene source" );
			sceneSource = ( SceneSource )GUILayout.SelectionGrid( ( int )sceneSource, System.Enum.GetNames( typeof( SceneSource ) ), 2 );
			if ( sceneSource == SceneSource.BinFile )
			{
				sceneIndex = GUILayout.SelectionGrid( sceneIndex, binScenes, 1 );
			}
			if ( GUILayout.Button( "Reload" ) && !benchmarkRunning )
			{
				LoadScene();
			}

			GUILayout.Space( 6 );
			GUILayout.Label( "Instances" );
			int newGrid = GUILayout.SelectionGrid( instanceGridIndex, instanceGridNames, instanceGridNames.Length );
			if ( newGrid != instanceGridIndex && !benchmarkRunning )
			{
				instanceGridIndex = newGrid;
				// The instance count changes the flattened triangle count the opacity map is sized
				// for, so it has to be re-applied here too.
				ApplyBuildOptions();
				RebuildInstancedScene();
			}
			animateInstances = GUILayout.Toggle( animateInstances, "Animate" );
			deform = GUILayout.Toggle( deform, "Deform" );
			if ( deform )
			{
				GUILayout.Label( "Deform update" );
				deformUpdate = ( DeformUpdate )GUILayout.SelectionGrid( ( int )deformUpdate, System.Enum.GetNames( typeof( DeformUpdate ) ), 2 );
			}

			GUILayout.Space( 6 );
			IRaytraceBackend active = backends[ activeBackend ];
			GUILayout.Label( $"Triangles: {flatTriCount} in {instances.Length} instance(s)" );
			string opacityNote = opacityMap && !active.SupportsOpacityMap ? " (no opacity support)" : "";
			string builderNote = "";
			if ( active is CpuVoxelBackend )
			{
				// There is no triangle BVH under the voxel TLAS, so no build option reaches it.
				builderNote = " (voxels)";
			}
			else if ( active is CpuDoubleBackend && ( builder != BvhBuilder.Binned || presplit ) )
			{
				// BvhDouble has the binned builder only, so the builder choice does not reach it.
				builderNote = " (binned only)";
			}
			GUILayout.Label( $"Nodes: {active.NodeCount}   Build: {active.BuildMs:F2} ms{opacityNote}{builderNote}" );
			if ( active is IRefittableBackend activeRefittable )
			{
				GUILayout.Label( $"Refit: {activeRefittable.RefitMs:F2} ms" );
			}
			if ( deform && deformUpdate == DeformUpdate.Refit && !CanRefit( active ) )
			{
				GUILayout.Label( !IsRefittableBuild
					? "Refit unavailable for this build: rebuilding"
					: "Refit unavailable for this backend: rebuilding" );
			}
			if ( active is IInstancedBackend activeInstanced )
			{
				GUILayout.Label( $"TLAS update: {activeInstanced.UpdateMs:F2} ms" );
			}
			GUILayout.Label( $"Frame: {frameMsEma:F2} ms   {mraysPerSec:F1} Mrays/s" );
			GUILayout.Label( $"Resolution: {lastWidth}x{lastHeight}" );

			GUILayout.Space( 6 );
			if ( GUILayout.Button( benchmarkRunning ? "Benchmarking..." : "Benchmark" ) && !benchmarkRunning )
			{
				StartBenchmark();
			}
			if ( benchmarkResults != null )
			{
				GUILayout.Space( 6 );
				GUILayout.Label( "Benchmark results" );
				foreach ( BenchmarkResult r in benchmarkResults )
				{
					GUILayout.Label( $"{r.Name}: {r.AvgMs:F2} ms, {r.MraysPerSec:F1} Mrays/s, build {r.BuildMs:F2} ms, {r.Nodes} nodes" );
				}
			}

			GUI.DragWindow( new Rect( 0, 0, 10000, 20 ) );
		}
	}
}
