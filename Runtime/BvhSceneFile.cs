using System;
using System.IO;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace TinyBVH
{
	/// <summary>Loader for tinybvh's ".bin" scene files: int32 triCount followed by triCount * 3 float4 vertices.</summary>
	public static class BvhSceneFile
	{
		/// <summary>Reads a scene file into a newly allocated NativeArray of triCount * 3 vertices.</summary>
		public static NativeArray<float4> Load( string path, Allocator allocator, out uint triCount )
		{
			byte[] bytes = File.ReadAllBytes( path );
			int count = BitConverter.ToInt32( bytes, 0 );
			triCount = ( uint )count;
			int vertCount = count * 3;

			NativeArray<float4> verts = new NativeArray<float4>( vertCount, allocator, NativeArrayOptions.UninitializedMemory );
			if ( vertCount > 0 )
			{
				unsafe
				{
					fixed ( byte* src = &bytes[ 4 ] )
					{
						UnsafeUtility.MemCpy( verts.GetUnsafePtr(), src, ( long )vertCount * sizeof( float4 ) );
					}
				}
			}
			return verts;
		}

		/// <summary>Resolves a file name to "&lt;projectRoot&gt;/TestData/&lt;fileName&gt;".</summary>
		public static string TestDataPath( string fileName )
		{
			string projectRoot = Directory.GetParent( Application.dataPath ).FullName;
			return Path.Combine( projectRoot, "TestData", fileName );
		}
	}
}
