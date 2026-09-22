using System;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace TinyBVH.Samples
{
	public enum DisplayMode
	{
		Shaded,
		Normals,
		Depth,
		TraversalSteps,
		Barycentrics
	}

	/// <summary>Which of the builders of TinyBVH.Bvh a backend builds its triangle BVH with.</summary>
	public enum BvhBuilder
	{
		/// <summary>The binned SAH reference builder, Bvh.Build.</summary>
		Binned,
		/// <summary>The spatial-split builder, Bvh.UseSpatialSplits.</summary>
		Sbvh,
		/// <summary>The full-sweep SAH builder, Bvh.UseFullSweep.</summary>
		FullSweep,
		/// <summary>The mid-point split builder, Bvh.BuildQuick. Ignores every build setting.</summary>
		Quick
	}

	/// <summary>Maps a <see cref="BvhBuilder"/> choice onto the build settings of a TinyBVH.Bvh.</summary>
	public static class BvhBuilderOptions
	{
		/// <summary>
		/// True when presplitting actually applies: the SBVH builder ignores it, and BuildQuick
		/// ignores every build setting.
		/// </summary>
		public static bool PresplitApplies( BvhBuilder builder, bool presplit )
		{
			return presplit && builder != BvhBuilder.Sbvh && builder != BvhBuilder.Quick;
		}

		/// <summary>Writes the builder choice into the settings fields of a base BVH.</summary>
		public static void Apply( ref Bvh bvh, BvhBuilder builder, bool presplit )
		{
			bvh.UseSpatialSplits = builder == BvhBuilder.Sbvh;
			bvh.UseFullSweep = builder == BvhBuilder.FullSweep;
			bvh.UsePresplitting = PresplitApplies( builder, presplit );
		}

		/// <summary>Applies the builder choice to a base BVH and runs the build it selects.</summary>
		public static void Build( ref Bvh bvh, BvhBuilder builder, bool presplit, NativeArray<float4> vertices, uint triCount )
		{
			Apply( ref bvh, builder, presplit );
			if ( builder == BvhBuilder.Quick )
			{
				bvh.BuildQuick( vertices, triCount );
			}
			else
			{
				bvh.Build( vertices, triCount );
			}
		}

		/// <summary>
		/// The same for a Bvh4Cpu. The layout owns its base BVH and only forwards the two settings
		/// it has fields for, so the rest of the choice is written straight onto that base; the base
		/// and the 4-wide intermediate, which Bvh4Cpu.Build would create on demand, are created here
		/// so the settings have somewhere to go and the Quick path has a base to build into.
		/// </summary>
		public static void Build( ref Bvh4Cpu bvh4, BvhBuilder builder, bool presplit, bool threaded, NativeArray<float4> vertices, uint triCount )
		{
			if ( !bvh4.Mbvh4.IsCreated )
			{
				bvh4.Mbvh4 = Mbvh.Create( 4, bvh4.Allocator );
			}
			if ( !bvh4.Mbvh4.Source.IsCreated )
			{
				bvh4.Mbvh4.Source = Bvh.Create( bvh4.Allocator );
			}
			bvh4.UseSpatialSplits = builder == BvhBuilder.Sbvh;
			bvh4.UseThreadedBuild = threaded;
			Apply( ref bvh4.Mbvh4.Source, builder, presplit );
			if ( builder == BvhBuilder.Quick )
			{
				// Bvh4Cpu.Build with Bvh.BuildQuick in place of Bvh.Build. Ownership of the two
				// structures goes back to bvh4 afterwards, exactly as Bvh4Cpu.Build leaves it.
				bvh4.Mbvh4.TraversalCost = bvh4.TraversalCost;
				bvh4.Mbvh4.IntersectionCost = bvh4.IntersectionCost;
				bvh4.Mbvh4.Source.TraversalCost = bvh4.TraversalCost;
				bvh4.Mbvh4.Source.IntersectionCost = bvh4.IntersectionCost;
				bvh4.Mbvh4.Source.BuildQuick( vertices, triCount );
				Mbvh mbvh4 = bvh4.Mbvh4;
				bvh4.ConvertFrom( ref mbvh4 );
				bvh4.OwnsSource = true;
			}
			else
			{
				bvh4.Build( vertices, triCount );
			}
		}

		/// <summary>The same for a Bvh8Cpu; see the Bvh4Cpu overload. Bvh8Cpu.Build also compacts the base.</summary>
		public static void Build( ref Bvh8Cpu bvh8, BvhBuilder builder, bool presplit, bool threaded, NativeArray<float4> vertices, uint triCount )
		{
			if ( !bvh8.Mbvh8.IsCreated )
			{
				bvh8.Mbvh8 = Mbvh.Create( 8, bvh8.Allocator );
			}
			if ( !bvh8.Mbvh8.Source.IsCreated )
			{
				bvh8.Mbvh8.Source = Bvh.Create( bvh8.Allocator );
			}
			bvh8.UseSpatialSplits = builder == BvhBuilder.Sbvh;
			bvh8.UseThreadedBuild = threaded;
			Apply( ref bvh8.Mbvh8.Source, builder, presplit );
			if ( builder == BvhBuilder.Quick )
			{
				bvh8.Mbvh8.TraversalCost = bvh8.TraversalCost;
				bvh8.Mbvh8.IntersectionCost = bvh8.IntersectionCost;
				bvh8.Mbvh8.Source.TraversalCost = bvh8.TraversalCost;
				bvh8.Mbvh8.Source.IntersectionCost = bvh8.IntersectionCost;
				bvh8.Mbvh8.Source.BuildQuick( vertices, triCount );
				bvh8.Mbvh8.Source.Compact();
				Mbvh mbvh8 = bvh8.Mbvh8;
				bvh8.ConvertFrom( ref mbvh8 );
				bvh8.OwnsSource = true;
			}
			else
			{
				bvh8.Build( vertices, triCount );
			}
		}

		/// <summary>The same for a BvhSoa, which keeps its base BVH in Source rather than behind an MBVH.</summary>
		public static void Build( ref BvhSoa soa, BvhBuilder builder, bool presplit, bool threaded, NativeArray<float4> vertices, uint triCount )
		{
			if ( !soa.Source.IsCreated )
			{
				soa.Source = Bvh.Create( soa.Allocator );
			}
			soa.UseSpatialSplits = builder == BvhBuilder.Sbvh;
			soa.UseThreadedBuild = threaded;
			Apply( ref soa.Source, builder, presplit );
			if ( builder == BvhBuilder.Quick )
			{
				// BvhSoa.Build with Bvh.BuildQuick in place of Bvh.Build, down to the compact = false
				// of its conversion, and the ownership it claims back afterwards.
				soa.Source.TraversalCost = soa.TraversalCost;
				soa.Source.IntersectionCost = soa.IntersectionCost;
				soa.Source.BuildQuick( vertices, triCount );
				Bvh source = soa.Source;
				soa.ConvertFrom( ref source, false );
				soa.OwnsSource = true;
			}
			else
			{
				soa.Build( vertices, triCount );
			}
		}
	}

	/// <summary>Everything a backend needs to render one frame. Camera basis vectors are for a view plane at distance 1.</summary>
	public struct RenderSettings
	{
		public int Width;
		public int Height;
		public float3 Origin;
		public float3 TopLeftDir;
		public float3 Horizontal;
		public float3 Vertical;
		public DisplayMode Mode;
		public bool Shadows;
		public float3 SunDir;
		public float SceneDiagonal;
	}

	/// <summary>
	/// A way of tracing the scene: a BVH layout plus the code that traverses it, on the CPU or the GPU.
	/// Build is called once per scene; Render is called every frame and returns the texture to display.
	/// </summary>
	public interface IRaytraceBackend : IDisposable
	{
		string Name { get; }
		bool IsGpu { get; }
		/// <summary>Which builder the next Build / BuildInstanced builds the triangle BVH (the BLAS for instanced backends) with. TLASes are always binned.</summary>
		BvhBuilder Builder { get; set; }
		/// <summary>When true, the next Build / BuildInstanced presplits the triangle BVH (Bvh.UsePresplitting). Ignored by the SBVH and Quick builders, and by TLASes.</summary>
		bool Presplit { get; set; }
		/// <summary>When true, the next Build / BuildInstanced uses the job system for large inputs.</summary>
		bool ThreadedBuild { get; set; }
		/// <summary>When true, after the next Build / BuildInstanced the triangle BVH is optimized with 25 tree-rotation iterations. TLASes are never optimized.</summary>
		bool Optimize { get; set; }
		/// <summary>
		/// A per-triangle opacity micromap applied to the triangle BVH (the BLAS for instanced
		/// backends) at the next Build / BuildInstanced. Default/unset (not created) means no map.
		/// Indexed by the triangle index of the array passed to Build / BuildInstanced.
		/// </summary>
		NativeArray<uint> OpacityMap { get; set; }
		/// <summary>Subdivision of OpacityMap; see Bvh.SetOpacityMicroMaps. Ignored unless OpacityMap is set.</summary>
		uint OpacityMapN { get; set; }
		/// <summary>True when this backend's layout can trace an opacity micromap.</summary>
		bool SupportsOpacityMap { get; }
		/// <summary>Milliseconds spent in the last Build, including layout conversion and upload.</summary>
		float BuildMs { get; }
		/// <summary>Node count of the traversed structure, for display.</summary>
		long NodeCount { get; }
		/// <summary>Builds over a triangle soup: three float4 per triangle. The array stays alive and unchanged until Dispose or the next Build.</summary>
		void Build( NativeArray<float4> vertices, uint triCount );
		/// <summary>Renders synchronously and returns the result. GPU backends must make sure the work has completed before returning when waitForGpu is set, so timings are comparable.</summary>
		Texture Render( in RenderSettings settings, bool waitForGpu );
	}

	/// <summary>
	/// A backend that can update its acceleration structure after the vertex data it was built
	/// over changed in place, without rebuilding it. Refitting keeps the tree topology, so it is
	/// far cheaper than a rebuild, but the tree degrades as the geometry moves away from the pose
	/// it was built in. Trees that cannot be refit - a spatial-split build, for one - are the
	/// caller's problem: Refit throws for those, so the sample rebuilds instead.
	/// </summary>
	public interface IRefittableBackend : IRaytraceBackend
	{
		/// <summary>Milliseconds spent in the last Refit.</summary>
		float RefitMs { get; }
		/// <summary>The vertex array this backend was built over changed in place; update the structure without rebuilding it.</summary>
		void Refit();
	}

	/// <summary>
	/// A backend that can place one mesh in the scene several times through a TLAS instead of
	/// tracing a flattened copy of it. The sample hands these backends the source mesh plus the
	/// instance list, and moving the instances only costs a TLAS rebuild; the other backends get
	/// the mesh flattened into every instance through the regular Build, so they pay for a full
	/// rebuild whenever an instance moves.
	/// </summary>
	public interface IInstancedBackend : IRaytraceBackend
	{
		/// <summary>Milliseconds spent in the last UpdateInstances.</summary>
		float UpdateMs { get; }
		/// <summary>Builds one BLAS over the triangle soup and a TLAS over the instances. Both arrays stay alive and at the same address until Dispose or the next build.</summary>
		void BuildInstanced( NativeArray<float4> vertices, uint triCount, NativeArray<BlasInstance> instances );
		/// <summary>Rebuilds only the TLAS after the instance transforms changed. Same array as the last BuildInstanced.</summary>
		void UpdateInstances( NativeArray<BlasInstance> instances );
	}
}
