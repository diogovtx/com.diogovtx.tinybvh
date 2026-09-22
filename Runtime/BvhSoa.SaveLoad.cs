using System;
using Unity.Collections;
using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// Port of BVH_SoA::Save and BVH_SoA::Load. Unlike BVH4_CPU, which dumps its own converted
	/// blob, BVH_SoA stores the underlying BVH and rebuilds the SoA nodes on load:
	/// "void BVH_SoA::Save( const char* f ) { bvh.Save( f ); }" and a Load that calls bvh.Load
	/// followed by ConvertFrom( bvh, false ). This port does the same, so the file it writes is an
	/// ordinary Bvh file ("TBVHCS") and a loaded layout keeps a full base BVH - it can still be
	/// optimized and asked for its SAH cost, unlike a loaded Bvh4Cpu.
	/// </summary>
	public unsafe partial struct BvhSoa
	{
		/// <summary>Writes the underlying BVH to a file. The geometry it was built over is not stored.</summary>
		public void Save( string path )
		{
			if ( Source.Nodes == null || Source.UsedNodes == 0 )
			{
				throw new InvalidOperationException( "BvhSoa.Save( .. ), bvhSoa was not built." );
			}
			Source.Save( path );
		}

		/// <summary>
		/// Loads a layout saved by <see cref="Save"/> over a triangle soup, and re-converts. Returns
		/// false when the file is missing, has a different format or version, or does not match the
		/// geometry passed in; the layout is then left untouched.
		/// </summary>
		public bool Load( string path, NativeArray<float4> vertices, uint triCount )
		{
			if ( !IsCreated )
			{
				throw new InvalidOperationException( "BvhSoa.Load( .. ), bvhSoa was not created." );
			}
			if ( !Source.IsCreated )
			{
				Source = Bvh.Create( Allocator );
			}
			if ( !Source.Load( path, vertices, triCount ) )
			{
				return false;
			}
			Bvh source = Source;
			ConvertFrom( ref source, false );
			// the loaded BVH was allocated here, so this layout owns it; ConvertFrom cleared the flag.
			OwnsSource = true;
			return true;
		}
	}
}
