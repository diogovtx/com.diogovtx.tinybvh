using Unity.Burst;

namespace TinyBVH
{
	/// <summary>
	/// Reports whether Burst direct calls into this assembly actually run Burst-compiled code.
	/// A single Burst compile error anywhere in the assembly (for example a non-blittable bool
	/// field in a struct passed by ref) silently drops every direct call to the Mono fallback,
	/// which builds slightly different trees because Mono evaluates float math in double.
	/// </summary>
	[BurstCompile]
	public static class BvhBurst
	{
		/// <summary>True when the direct call below ran Burst-compiled code; false under the Mono fallback.</summary>
		public static bool IsActive
		{
			get
			{
				int managed = 0;
				Probe( ref managed );
				return managed == 0;
			}
		}

		[BurstCompile( CompileSynchronously = true )]
		private static void Probe( ref int managed )
		{
			managed = 0;
			MarkManaged( ref managed );
		}

		/// <summary>Stripped from the Burst build, so it only runs when the call fell back to Mono.</summary>
		[BurstDiscard]
		private static void MarkManaged( ref int managed )
		{
			managed = 1;
		}
	}
}
