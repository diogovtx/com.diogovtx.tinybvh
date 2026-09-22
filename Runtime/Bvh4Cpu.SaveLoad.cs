using System;
using System.IO;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Port of BVH4_CPU::Save and BVH4_CPU::Load: caching a converted layout on disk. Unlike the
	/// C++, which dumps the BVH4_CPU struct itself, the header fields are written one by one, so
	/// the file format does not depend on the struct layout. A loaded layout has no base BVH or
	/// intermediate MBVH&lt;4&gt;, so it cannot be refit, optimized or asked for its SAH cost -
	/// only traversal is available, exactly like the C++, which discards bvh4 on load.
	/// </summary>
	public unsafe partial struct Bvh4Cpu
	{
		/// <summary>File magic; the C++ uses a version/layout header word instead.</summary>
		private const string SaveMagic = "TBVHCS4C";
		/// <summary>Bumped whenever the layout of the file changes.</summary>
		private const uint SaveVersion = 1;
		/// <summary>Magic + version + usedBlocks + triCount + idxCount + flags + two bounds + two costs.</summary>
		private const long HeaderSize = 8 + 4 + 4 + 4 + 4 + 4 + 24 + 4 + 4;

		private const uint FlagRefittable = 1;
		private const uint FlagMayHaveHoles = 2;
		private const uint FlagBvhOverAabbs = 4;
		private const uint FlagBvhOverIndices = 8;

		/// <summary>Writes the node/leaf blocks to a file. The geometry it was built over is not stored.</summary>
		public void Save( string path )
		{
			if ( Data == null || UsedBlocks == 0 )
			{
				throw new InvalidOperationException( "Bvh4Cpu.Save( .. ), bvh4Cpu was not built." );
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
					writer.Write( TriCount );
					writer.Write( IdxCount );
					writer.Write( flags );
					writer.Write( AabbMin.x );
					writer.Write( AabbMin.y );
					writer.Write( AabbMin.z );
					writer.Write( AabbMax.x );
					writer.Write( AabbMax.y );
					writer.Write( AabbMax.z );
					writer.Write( TraversalCost );
					writer.Write( IntersectionCost );
					WriteBlock( writer, Data, ( long )UsedBlocks * BlockSize );
				}
			}
		}

		/// <summary>
		/// Loads a layout saved by <see cref="Save"/>. Returns false when the file is missing, has a
		/// different format or version, or does not match expectedTriCount; the layout is then left
		/// untouched. On success, Base/Mbvh4 are cleared to default and OwnsSource is false, exactly
		/// like the C++ BVH4_CPU::Load discarding bvh4.
		/// </summary>
		public bool Load( string path, uint expectedTriCount )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "Bvh4Cpu.Load( .. ), bvh4Cpu was not created." );
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
					uint fileTriCount = reader.ReadUInt32();
					uint fileIdxCount = reader.ReadUInt32();
					uint flags = reader.ReadUInt32();
					float3 aabbMin = new float3( reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle() );
					float3 aabbMax = new float3( reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle() );
					float traversalCost = reader.ReadSingle();
					float intersectionCost = reader.ReadSingle();
					if ( fileTriCount != expectedTriCount )
					{
						return false;
					}
					if ( fileUsedBlocks == 0 || fileIdxCount == 0 )
					{
						return false;
					}
					long dataBytes = ( long )fileUsedBlocks * BlockSize;
					if ( stream.Length != HeaderSize + dataBytes )
					{
						return false;
					}
					// all checks passed; safe to overwrite this.
					Free( Data );
					Data = ( byte* )Alloc( dataBytes );
					AllocatedBlocks = fileUsedBlocks;
					ReadBlock( reader, Data, dataBytes );
					UsedBlocks = fileUsedBlocks;
					TriCount = fileTriCount;
					IdxCount = fileIdxCount;
					Refittable = ( flags & FlagRefittable ) != 0;
					MayHaveHoles = ( flags & FlagMayHaveHoles ) != 0;
					BvhOverAabbs = ( flags & FlagBvhOverAabbs ) != 0;
					BvhOverIndices = ( flags & FlagBvhOverIndices ) != 0;
					AabbMin = aabbMin;
					AabbMax = aabbMax;
					TraversalCost = traversalCost;
					IntersectionCost = intersectionCost;
					// we can't load the source BVH/MBVH<4> since this layout doesn't own vertex data.
					Base = default;
					Mbvh4 = default;
					OwnsSource = false;
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
					throw new EndOfStreamException( "Bvh4Cpu.Load( .. ), file ended early." );
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
