using Unity.Mathematics;

namespace TinyBVH
{
	/// <summary>
	/// The mixed-layout half of BLASInstance::Update. The C++ takes a BVHBase pointer and only
	/// reads the BLAS root bounds from it, so one core does for every layout; these overloads
	/// exist because the port has no common base class.
	/// </summary>
	public unsafe partial struct BlasInstance
	{
		/// <summary>BLASInstance::Update for a BLAS in the BVH4_CPU layout.</summary>
		public void Update( ref Bvh4Cpu blas )
		{
			UpdateBounds( blas.AabbMin, blas.AabbMax );
		}

		/// <summary>BLASInstance::Update for a BLAS in the BVH8_CPU layout.</summary>
		public void Update( ref Bvh8Cpu blas )
		{
			UpdateBounds( blas.AabbMin, blas.AabbMax );
		}

		/// <summary>BLASInstance::Update for a BLAS in the BVH_SoA layout.</summary>
		public void Update( ref BvhSoa blas )
		{
			UpdateBounds( blas.AabbMin, blas.AabbMax );
		}

		/// <summary>BLASInstance::Update for a voxel set, whose bounds are always the unit cube.</summary>
		public void Update( ref VoxelSet blas )
		{
			UpdateBounds( blas.AabbMin, blas.AabbMax );
		}

		/// <summary>BLASInstance::Update for a BLAS of any layout, i.e. the C++ BVHBase* form.</summary>
		public void Update( in BlasRef blas )
		{
			BlasRef.GetBounds( blas, out float3 aabbMin, out float3 aabbMax );
			UpdateBounds( aabbMin, aabbMax );
		}

		/// <summary>
		/// The body of BLASInstance::Update, over the root bounds the C++ reads from the BVHBase
		/// it is handed.
		/// </summary>
		private void UpdateBounds( float3 bmin, float3 bmax )
		{
			InvertTransform(); // TODO: done unconditionally; for a big TLAS this may be wasteful.
			// transform the eight corners of the root node aabb using the
			// instance transform and calculate the worldspace aabb over those.
			AabbMin = new float3( BvhConstants.Far );
			AabbMax = new float3( -BvhConstants.Far );
			for ( int j = 0; j < 8; j++ )
			{
				float3 p = new float3(
					( j & 1 ) != 0 ? bmax.x : bmin.x,
					( j & 2 ) != 0 ? bmax.y : bmin.y,
					( j & 4 ) != 0 ? bmax.z : bmin.z );
				float3 t = Transform.TransformPoint( p );
				AabbMin = math.min( AabbMin, t );
				AabbMax = math.max( AabbMax, t );
			}
		}
	}
}
