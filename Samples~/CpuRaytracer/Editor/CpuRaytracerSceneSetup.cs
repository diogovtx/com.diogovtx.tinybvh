using TinyBVH.Samples;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace TinyBVH.Samples.Editor
{
	/// <summary>Menu item that sets up a scene ready to play the CPU raytracer sample against bunny.bin.</summary>
	public static class CpuRaytracerSceneSetup
	{
		[MenuItem( "TinyBVH/Create CPU Raytracer Scene" )]
		public static void CreateScene()
		{
			EditorSceneManager.NewScene( NewSceneSetup.EmptyScene, NewSceneMode.Single );

			GameObject lightGo = new GameObject( "Directional Light" );
			Light light = lightGo.AddComponent<Light>();
			light.type = LightType.Directional;
			lightGo.transform.rotation = Quaternion.Euler( 50f, -30f, 0f );

			GameObject camGo = new GameObject( "CPU Raytracer Camera" );
			camGo.AddComponent<Camera>();
			RaytracerSample raytracer = camGo.AddComponent<RaytracerSample>();

			// sceneSource, sceneIndex and autoFrame keep their [SerializeField] defaults
			// (BinFile, 0 -> "bunny.bin", true); only the optional sun reference needs wiring up.
			SerializedObject serialized = new SerializedObject( raytracer );
			serialized.FindProperty( "sun" ).objectReferenceValue = light;
			serialized.ApplyModifiedPropertiesWithoutUndo();

			Selection.activeGameObject = camGo;
		}
	}
}
