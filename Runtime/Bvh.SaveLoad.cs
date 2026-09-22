using System;
using System.IO;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Port of BVH::Save and BVH::Load: caching a built tree on disk. Unlike the C++, which dumps
	/// the BVH struct itself, the header fields are written one by one, so the file format does not
	/// depend on the struct layout. A loaded BVH owns its nodes and primitive indices but has no
	/// fragments; like the C++ it can still be refit, which only needs the nodes, the indices and
	/// the vertices, and a Build over it reallocates everything it needs.
	/// </summary>
	public unsafe partial struct Bvh
	{
		/// <summary>File magic; the C++ uses a 16-bit cache version plus the layout in the top byte.</summary>
		private const string SaveMagic = "TBVHCS";
		/// <summary>Bumped whenever the layout of the file changes.</summary>
		private const uint SaveVersion = 1;
		/// <summary>Magic + version + triCount + idxCount + usedNodes + flags + two costs + two bounds.</summary>
		private const long HeaderSize = 6 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 24;

		private const uint FlagRefittable = 1;
		private const uint FlagMayHaveHoles = 2;
		private const uint FlagBvhOverAabbs = 4;
		private const uint FlagBvhOverIndices = 8;
		private const uint FlagUseSpatialSplits = 16;

		/// <summary>Writes the tree to a file. The geometry it was built over is not stored.</summary>
		public void Save( string path )
		{
			if ( Nodes == null || UsedNodes == 0 )
			{
				throw new InvalidOperationException( "Bvh.Save( .. ), bvh was not built." );
			}
			if ( IsTlas )
			{
				throw new InvalidOperationException( "Bvh.Save( .. ), cannot save a TLAS." );
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
			if ( UseSpatialSplits )
			{
				flags |= FlagUseSpatialSplits;
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
					writer.Write( TriCount );
					writer.Write( IdxCount );
					writer.Write( UsedNodes );
					writer.Write( flags );
					writer.Write( TraversalCost );
					writer.Write( IntersectionCost );
					writer.Write( AabbMin.x );
					writer.Write( AabbMin.y );
					writer.Write( AabbMin.z );
					writer.Write( AabbMax.x );
					writer.Write( AabbMax.y );
					writer.Write( AabbMax.z );
					WriteBlock( writer, Nodes, ( long )UsedNodes * sizeof( BvhNode ) );
					WriteBlock( writer, PrimIdx, ( long )IdxCount * sizeof( uint ) );
				}
			}
		}

		/// <summary>Loads a tree saved over a triangle soup: three consecutive 16-byte vertices per primitive.</summary>
		public bool Load( string path, NativeArray<float4> vertices, uint triCount )
		{
			return Load( path, ( byte* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( vertices ), ( uint )vertices.Length, 16, null, triCount );
		}

		/// <summary>Loads a tree saved over indexed triangles.</summary>
		public bool Load( string path, NativeArray<float4> vertices, NativeArray<uint> indices, uint triCount )
		{
			return Load( path,
				( byte* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( vertices ), ( uint )vertices.Length, 16,
				( uint* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( indices ), triCount );
		}

		/// <summary>
		/// General form, matching Bvh.Build: a strided vertex buffer, optionally addressed through
		/// indices. Returns false when the file is missing, has a different format or version, or
		/// does not match the geometry passed in; the BVH is then left untouched.
		/// </summary>
		public bool Load( string path, byte* vertices, uint vertexCount, int vertexStride, uint* indices, uint primCount )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "Bvh.Load( .. ), bvh was not created." );
			}
			if ( vertices == null )
			{
				throw new ArgumentException( "Bvh.Load( .. ), vertices == null.", nameof( vertices ) );
			}
			if ( vertexStride < 16 )
			{
				throw new ArgumentException( "Bvh.Load( .. ), vertexStride < 16.", nameof( vertexStride ) );
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
					uint fileTriCount = reader.ReadUInt32();
					uint fileIdxCount = reader.ReadUInt32();
					uint fileUsedNodes = reader.ReadUInt32();
					uint flags = reader.ReadUInt32();
					float traversalCost = reader.ReadSingle();
					float intersectionCost = reader.ReadSingle();
					float3 aabbMin = new float3( reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle() );
					float3 aabbMax = new float3( reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle() );
					// the geometry must be the geometry the tree was built over.
					bool expectIndexed = indices != null;
					if ( expectIndexed != ( ( flags & FlagBvhOverIndices ) != 0 ) )
					{
						return false;
					}
					if ( fileTriCount != ( expectIndexed ? primCount : vertexCount / 3 ) )
					{
						return false;
					}
					if ( fileUsedNodes == 0 || fileIdxCount == 0 )
					{
						return false;
					}
					long nodeBytes = ( long )fileUsedNodes * sizeof( BvhNode );
					long idxBytes = ( long )fileIdxCount * sizeof( uint );
					if ( stream.Length != HeaderSize + nodeBytes + idxBytes )
					{
						return false;
					}
					// all checks passed; safe to overwrite this.
					Free( Nodes );
					Free( PrimIdx );
					Free( Fragments );
					Nodes = ( BvhNode* )Alloc( nodeBytes );
					AllocatedNodes = fileUsedNodes;
					PrimIdx = ( uint* )Alloc( idxBytes );
					AllocatedPrimIdx = fileIdxCount;
					Fragments = null; // no need for this in a BVH that can't be rebuilt.
					AllocatedFragments = 0;
					ReadBlock( reader, Nodes, nodeBytes );
					ReadBlock( reader, PrimIdx, idxBytes );
					UsedNodes = fileUsedNodes;
					TriCount = fileTriCount;
					IdxCount = fileIdxCount;
					Refittable = ( flags & FlagRefittable ) != 0;
					MayHaveHoles = ( flags & FlagMayHaveHoles ) != 0;
					BvhOverAabbs = ( flags & FlagBvhOverAabbs ) != 0;
					BvhOverIndices = ( flags & FlagBvhOverIndices ) != 0;
					UseSpatialSplits = ( flags & FlagUseSpatialSplits ) != 0;
					TraversalCost = traversalCost;
					IntersectionCost = intersectionCost;
					AabbMin = aabbMin;
					AabbMax = aabbMax;
					// we can't load vertices since the BVH doesn't own this data.
					Verts = vertices;
					VertCount = vertexCount;
					VertStride = vertexStride;
					VertIdx = indices;
					Instances = null;
					InstanceCount = 0;
					Blasses = null;
					BlasRefs = null;
					BlasCount = 0;
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
					throw new EndOfStreamException( "Bvh.Load( .. ), file ended early." );
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
