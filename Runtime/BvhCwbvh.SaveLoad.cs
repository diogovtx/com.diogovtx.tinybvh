using System;
using System.IO;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Port of BVH8_CWBVH::Save and BVH8_CWBVH::Load: caching a converted layout on disk. Unlike
	/// the C++, which dumps the BVH8_CWBVH struct itself, the header fields are written one by
	/// one, so the file format does not depend on the struct layout. A loaded layout has no
	/// source MBVH&lt;8&gt;, so only traversal and SahCost forwarding through Source are affected -
	/// SahCost still throws, exactly like the other loaded layouts, since Source.Source is gone.
	/// </summary>
	public unsafe partial struct BvhCwbvh
	{
		/// <summary>File magic; the C++ uses a version/layout header word instead.</summary>
		private const string SaveMagic = "TBVHCSCW";
		/// <summary>Bumped whenever the layout of the file changes.</summary>
		private const uint SaveVersion = 1;
		/// <summary>Magic + version + usedBlocks + triBlocks + triCount + idxCount + flags + two bounds.</summary>
		private const long HeaderSize = 8 + 4 + 4 + 4 + 4 + 4 + 4 + 24;

		private const uint FlagRefittable = 1;
		private const uint FlagMayHaveHoles = 2;
		private const uint FlagBvhOverAabbs = 4;
		private const uint FlagBvhOverIndices = 8;

		/// <summary>Writes the node blocks and triangle data to a file. The geometry it was built over is not stored.</summary>
		public void Save( string path )
		{
			if ( Data == null || UsedBlocks == 0 )
			{
				throw new InvalidOperationException( "BvhCwbvh.Save( .. ), bvhCwbvh was not built." );
			}
			uint flags = 0;
			if ( Refittable )
			{
				flags |= FlagRefittable;
			}
			if ( MayHaveHoles )
			{
				flags |= FlagMayHaveHoles;
			}
			if ( BvhOverAabbs )
			{
				flags |= FlagBvhOverAabbs;
			}
			if ( BvhOverIndices )
			{
				flags |= FlagBvhOverIndices;
			}
			using ( FileStream stream = new FileStream( path, FileMode.Create, FileAccess.Write ) )
			{
				using ( BinaryWriter writer = new BinaryWriter( stream ) )
				{
					for ( int i = 0; i < SaveMagic.Length; i++ )
					{
						writer.Write( ( byte )SaveMagic[ i ] );
					}
					writer.Write( SaveVersion );
					writer.Write( UsedBlocks );
					writer.Write( TriBlocks );
					writer.Write( TriCount );
					writer.Write( IdxCount );
					writer.Write( flags );
					writer.Write( AabbMin.x );
					writer.Write( AabbMin.y );
					writer.Write( AabbMin.z );
					writer.Write( AabbMax.x );
					writer.Write( AabbMax.y );
					writer.Write( AabbMax.z );
					WriteBlock( writer, Data, ( long )UsedBlocks * 16 );
					// the converter allocates IdxCount * 4 float4 blocks (AllocatedTriBlocks) but only
					// fills TriBlocks of them; write the full allocation so Load reproduces it exactly.
					WriteBlock( writer, Tris, ( long )IdxCount * 4 * 16 );
				}
			}
		}

		/// <summary>
		/// Loads a layout saved by <see cref="Save"/>. Returns false when the file is missing, has a
		/// different format or version, or does not match expectedTriCount; the layout is then left
		/// untouched. On success, Source is cleared to default, exactly like the C++ BVH8_CWBVH::Load
		/// discarding bvh8.
		/// </summary>
		public bool Load( string path, uint expectedTriCount )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "BvhCwbvh.Load( .. ), bvhCwbvh was not created." );
			}
			if ( !File.Exists( path ) )
			{
				return false;
			}
			using ( FileStream stream = new FileStream( path, FileMode.Open, FileAccess.Read ) )
			{
				using ( BinaryReader reader = new BinaryReader( stream ) )
				{
					if ( stream.Length < HeaderSize )
					{
						return false;
					}
					for ( int i = 0; i < SaveMagic.Length; i++ )
					{
						if ( reader.ReadByte() != ( byte )SaveMagic[ i ] )
						{
							return false;
						}
					}
					if ( reader.ReadUInt32() != SaveVersion )
					{
						return false;
					}
					uint fileUsedBlocks = reader.ReadUInt32();
					uint fileTriBlocks = reader.ReadUInt32();
					uint fileTriCount = reader.ReadUInt32();
					uint fileIdxCount = reader.ReadUInt32();
					uint flags = reader.ReadUInt32();
					float3 aabbMin = new float3( reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle() );
					float3 aabbMax = new float3( reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle() );
					if ( fileTriCount != expectedTriCount )
					{
						return false;
					}
					if ( fileUsedBlocks == 0 || fileIdxCount == 0 )
					{
						return false;
					}
					long dataBytes = ( long )fileUsedBlocks * 16;
					long triBytes = ( long )fileIdxCount * 4 * 16;
					if ( stream.Length != HeaderSize + dataBytes + triBytes )
					{
						return false;
					}
					// all checks passed; safe to overwrite this.
					Free( Data );
					Free( Tris );
					Data = ( float4* )Alloc( dataBytes );
					AllocatedBlocks = fileUsedBlocks;
					Tris = ( float4* )Alloc( triBytes );
					AllocatedTriBlocks = fileIdxCount * 4;
					ReadBlock( reader, Data, dataBytes );
					ReadBlock( reader, Tris, triBytes );
					UsedBlocks = fileUsedBlocks;
					TriBlocks = fileTriBlocks;
					TriCount = fileTriCount;
					IdxCount = fileIdxCount;
					Refittable = ( flags & FlagRefittable ) != 0;
					MayHaveHoles = ( flags & FlagMayHaveHoles ) != 0;
					BvhOverAabbs = ( flags & FlagBvhOverAabbs ) != 0;
					BvhOverIndices = ( flags & FlagBvhOverIndices ) != 0;
					AabbMin = aabbMin;
					AabbMax = aabbMax;
					// we can't load the source MBVH<8> since this layout doesn't own vertex data.
					Source = default;
					return true;
				}
			}
		}

		/// <summary>Writes an unmanaged block through a staging buffer, in chunks.</summary>
		private static void WriteBlock( BinaryWriter writer, void* source, long bytes )
		{
			byte[] buffer = new byte[ Math.Min( bytes, 1 << 16 ) ];
			long done = 0;
			while ( done < bytes )
			{
				int chunk = ( int )Math.Min( bytes - done, buffer.Length );
				fixed ( byte* dst = buffer )
				{
					UnsafeUtility.MemCpy( dst, ( byte* )source + done, chunk );
				}
				writer.Write( buffer, 0, chunk );
				done += chunk;
			}
		}

		/// <summary>Reads an unmanaged block through a staging buffer, in chunks.</summary>
		private static void ReadBlock( BinaryReader reader, void* dest, long bytes )
		{
			byte[] buffer = new byte[ Math.Min( bytes, 1 << 16 ) ];
			long done = 0;
			while ( done < bytes )
			{
				int chunk = reader.Read( buffer, 0, ( int )Math.Min( bytes - done, buffer.Length ) );
				if ( chunk == 0 )
				{
					throw new EndOfStreamException( "BvhCwbvh.Load( .. ), file ended early." );
				}
				fixed ( byte* src = buffer )
				{
					UnsafeUtility.MemCpy( ( byte* )dest + done, src, chunk );
				}
				done += chunk;
			}
		}
	}
}
