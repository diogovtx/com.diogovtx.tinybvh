using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace TinyBVH.Samples.Editor
{
	/// <summary>
	/// Builds a standalone Windows 64-bit player of the CPU raytracer sample, for the headless
	/// benchmark mode of RaytracerSample. Usable from the menu or from the command line with
	/// -executeMethod TinyBVH.Samples.Editor.SamplePlayerBuild.BuildWindows64.
	/// </summary>
	public static class SamplePlayerBuild
	{
		private const string TempScenePath = "Assets/TinyBvhSamplePlayerScene.unity";

		[MenuItem( "TinyBVH/Build Sample Player (Windows 64)" )]
		public static void BuildWindows64()
		{
			// Only the Mono player is installed; Standalone defaults to Mono2x, so leave the
			// project settings alone unless something actually switched them to IL2CPP.
			if ( PlayerSettings.GetScriptingBackend( NamedBuildTarget.Standalone ) != ScriptingImplementation.Mono2x )
			{
				PlayerSettings.SetScriptingBackend( NamedBuildTarget.Standalone, ScriptingImplementation.Mono2x );
			}
			string projectRoot = Directory.GetParent( Application.dataPath ).FullName;
			string locationPathName = Path.Combine( projectRoot, "Builds", "SamplePlayer", "TinyBvhSample.exe" );
			// From the menu, the build replaces the open scenes with the temporary one: offer to
			// save them first and put them back afterwards. Batch mode has nothing to preserve.
			SceneSetup[] previousScenes = null;
			if ( !Application.isBatchMode )
			{
				if ( !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo() )
				{
					return;
				}
				previousScenes = EditorSceneManager.GetSceneManagerSetup();
				// An untitled scene has no path to restore from; leave the empty scene in that case.
				foreach ( SceneSetup setup in previousScenes )
				{
					if ( string.IsNullOrEmpty( setup.path ) )
					{
						previousScenes = null;
						break;
					}
				}
			}
			bool succeeded = false;
			try
			{
				CpuRaytracerSceneSetup.CreateScene();
				Scene scene = SceneManager.GetActiveScene();
				EditorSceneManager.SaveScene( scene, TempScenePath );
				BuildPlayerOptions options = new BuildPlayerOptions
				{
					scenes = new string[] { TempScenePath },
					locationPathName = locationPathName,
					target = BuildTarget.StandaloneWindows64,
					targetGroup = BuildTargetGroup.Standalone,
					options = BuildOptions.None
				};
				BuildReport report = BuildPipeline.BuildPlayer( options );
				succeeded = report.summary.result == BuildResult.Succeeded;
				Debug.Log( $"TinyBVH sample build: {report.summary.result}, {report.summary.totalSize} bytes, {report.summary.totalTime}, output '{report.summary.outputPath}'" );
				if ( !succeeded )
				{
					Debug.LogError( $"TinyBVH sample build failed: {report.summary.result}, {report.summary.totalErrors} error(s)" );
				}
			}
			finally
			{
				// Close the temporary scene before deleting its asset, then drop it so the repo
				// is left exactly as it was found.
				EditorSceneManager.NewScene( NewSceneSetup.EmptyScene, NewSceneMode.Single );
				AssetDatabase.DeleteAsset( TempScenePath );
				if ( previousScenes != null && previousScenes.Length > 0 )
				{
					EditorSceneManager.RestoreSceneManagerSetup( previousScenes );
				}
			}
			if ( Application.isBatchMode )
			{
				EditorApplication.Exit( succeeded ? 0 : 1 );
			}
		}
	}
}
