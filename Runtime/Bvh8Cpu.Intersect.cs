using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Burst.Intrinsics.X86;

namespace TinyBVH
{
	/// <summary>
	/// Traversal half of tinybvh's BVH8_CPU class, i.e. the AVX2 traversal of Fuetterling et al.
	/// The C++ compiles eight template variants of each function, keyed on the sign of the ray
	/// direction; here the same predicates are evaluated once at the top and passed down, as
	/// Bvh4Cpu does.
	///
	/// Every function has a Burst SIMD path and a scalar fallback. Burst resolves
	/// Avx2.IsAvx2Supported / Fma.IsFmaSupported at compile time; under Mono they are false, so
	/// the editor - and any CPU without AVX2/FMA - runs the scalar code. The two paths pick the
	/// same children in the same order and apply the same tie-break rules; they differ in that
	/// the SIMD path uses fused multiply-add and divides by the fastrcp4 reciprocal of the
	/// Moeller-Trumbore determinant where the scalar path multiplies, adds and divides
	/// separately, so hit distances can differ in the last few bits.
	/// </summary>
	public unsafe partial struct Bvh8Cpu
	{
		/// <summary>
		/// Traversal stack depth. The C++ uses 256 entries; eight more are reserved here because
		/// the SIMD path stores a full 32-byte group at stackPtr before advancing by
		/// validNodes - 1, and the stack compression reads a full group past its last entry.
		/// </summary>
		private const int StackSize = 264;

		/// <summary>_MM_SHUFFLE( 2, 1, 0, 3 ): rotate the lanes up by one.</summary>
		private const int ShuffleRotate = ( 2 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | 3;
		/// <summary>_MM_SHUFFLE( 1, 0, 3, 2 ): swap the two lane pairs.</summary>
		private const int ShuffleSwap = ( 1 << 6 ) | ( 0 << 4 ) | ( 3 << 2 ) | 2;

		// Opacity micro maps (not owned), the opmap / opmapN BVH8_CPU inherits from BVHBase in the
		// C++. They live here rather than with the rest of the layout data because, like in the
		// C++, they are not part of the conversion: CopyBasePropertiesFrom does not copy them, so
		// SetOpacityMicroMaps has to be called on this layout as well as on the base BVH.
		[NativeDisableUnsafePtrRestriction] public uint* OpMap;
		public uint OpMapN;

		/// <summary>Port of BVHBase::hasOpacityMicroMaps.</summary>
		public bool HasOpacityMicroMaps => OpMapN > 0;

		/// <summary>
		/// Port of BVHBase::SetOpacityMicroMaps. The map is referenced, not owned, and is indexed
		/// by the primitive index the leaves carry, so the same buffer serves the base BVH and
		/// this layout. See Bvh.SetOpacityMicroMaps.
		/// </summary>
		public void SetOpacityMicroMaps( uint* mapData, uint n )
		{
			OpMap = mapData;
			OpMapN = n;
		}

		/// <summary>SetOpacityMicroMaps over a NativeArray; the array must outlive the layout.</summary>
		public void SetOpacityMicroMaps( NativeArray<uint> mapData, uint n )
		{
			SetOpacityMicroMaps( ( uint* )mapData.GetUnsafePtr(), n );
		}

		/// <summary>Removes a previously set opacity micro map; equivalent to SetOpacityMicroMaps( null, 0 ).</summary>
		public void ClearOpacityMicroMaps()
		{
			SetOpacityMicroMaps( null, 0u );
		}

		/// <summary>
		/// Scalar opacity-map lookup for one lane, matching Bvh.OpacityOpaque. The SIMD paths
		/// compute the four indices with SSE, as the C++ does, and only gather in scalar; this is
		/// the fallback paths' equivalent.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private bool OpacityOpaque( uint triIdx, float u, float v )
		{
			float fN = OpMapN;
			int row = ( int )( ( u + v ) * fN );
			int diag = ( int )( ( 1f - u ) * fN );
			int idx = ( row * row ) + ( int )( v * fN ) + ( diag - ( ( int )OpMapN - 1 - row ) );
			uint* om = OpMap + ( triIdx * ( ( ( OpMapN * OpMapN ) + 31 ) >> 5 ) );
			return ( om[ idx >> 5 ] & ( 1u << ( idx & 31 ) ) ) != 0;
		}

		/// <summary>
		/// Port of the SSE opacity-map block the C++ runs after the four-triangle test: the four
		/// micro-triangle indices are computed with SSE and the map itself is gathered in scalar.
		/// Returns the four indices; the caller loops over the lanes it cares about.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private v128 OpacityIndices4( v128 u4, v128 v4, v128 one4 )
		{
			v128 fN4 = Sse.set1_ps( OpMapN );
			v128 row4 = Sse2.cvttps_epi32( Sse.mul_ps( Sse.add_ps( u4, v4 ), fN4 ) );
			v128 dia4 = Sse2.cvttps_epi32( Sse.mul_ps( Sse.sub_ps( one4, u4 ), fN4 ) );
			v128 term0 = Sse4_1.mullo_epi32( row4, row4 );
			v128 term1 = Sse2.cvttps_epi32( Sse.mul_ps( v4, fN4 ) );
			v128 term2 = Sse2.sub_epi32( dia4, Sse2.sub_epi32( Sse2.set1_epi32( ( int )OpMapN - 1 ), row4 ) );
			return Sse2.add_epi32( Sse2.add_epi32( term0, term1 ), term2 );
		}

		/// <summary>
		/// Port of BVH8_CPU::Intersect( Ray&amp; ), including its octant dispatcher. The hit, if any,
		/// is written to ray.Hit. Returns the number of leaves visited, which is the step count the
		/// C++ returns in a _DEBUG build; the release build returns 0 there.
		/// </summary>
		public int Intersect( ref Ray ray )
		{
			if ( Data == null )
			{
				return 0;
			}
			bool posX = ray.D.x >= 0f;
			bool posY = ray.D.y >= 0f;
			bool posZ = ray.D.z >= 0f;
			int signShift = ( posX ? 3 : 0 ) + ( posY ? 6 : 0 ) + ( posZ ? 12 : 0 );
			if ( Avx2.IsAvx2Supported && Fma.IsFmaSupported )
			{
				return IntersectSimd( ref ray, posX, posY, posZ, signShift );
			}
			return IntersectScalar( ref ray, posX, posY, posZ, signShift );
		}

		/// <summary>
		/// Port of BVH8_CPU::IsOccluded( const Ray&amp; ), including its octant dispatcher. Returns
		/// true as soon as any triangle is hit within ray.Hit.T; the ray itself is left untouched.
		/// </summary>
		public bool IsOccluded( in Ray ray )
		{
			if ( Data == null )
			{
				return false;
			}
			bool posX = ray.D.x >= 0f;
			bool posY = ray.D.y >= 0f;
			bool posZ = ray.D.z >= 0f;
			if ( Avx2.IsAvx2Supported && Fma.IsFmaSupported )
			{
				return IsOccludedSimd( ray, posX, posY, posZ );
			}
			return IsOccludedScalar( ray, posX, posY, posZ );
		}

		/// <summary>
		/// True when the caller's code runs the AVX2 traversal rather than the scalar fallback.
		/// The Unity.Burst.Intrinsics support flags report false from managed code whatever the
		/// CPU is, so this asks them through a Burst direct call instead.
		/// </summary>
		public static bool IsSimdSupported
		{
			get
			{
				Bvh8CpuSimd.Query( out int supported );
				return supported != 0;
			}
		}

		/// <summary>
		/// Runs the scalar fallback of <see cref="Intersect"/> whatever the CPU supports. Regular
		/// callers want Intersect; this entry point exists so the fallback can also be validated
		/// from inside a Burst job, where float arithmetic is not widened to double.
		/// </summary>
		public int IntersectScalarPath( ref Ray ray )
		{
			if ( Data == null )
			{
				return 0;
			}
			bool posX = ray.D.x >= 0f;
			bool posY = ray.D.y >= 0f;
			bool posZ = ray.D.z >= 0f;
			int signShift = ( posX ? 3 : 0 ) + ( posY ? 6 : 0 ) + ( posZ ? 12 : 0 );
			return IntersectScalar( ref ray, posX, posY, posZ, signShift );
		}

		/// <summary>Scalar-fallback counterpart of <see cref="IsOccluded"/>; see IntersectScalarPath.</summary>
		public bool IsOccludedScalarPath( in Ray ray )
		{
			if ( Data == null )
			{
				return false;
			}
			return IsOccludedScalar( ray, ray.D.x >= 0f, ray.D.y >= 0f, ray.D.z >= 0f );
		}

		/// <summary>Port of the templated BVH8_CPU::Intersect body, AVX2 path.</summary>
		private int IntersectSimd( ref Ray ray, bool posX, bool posY, bool posZ, int signShift )
		{
			uint* nodeStack = stackalloc uint[ StackSize ];
			float* distStack = stackalloc float[ StackSize ];
			v256 zero8 = Avx.mm256_setzero_ps();
			v256 t8 = Avx.mm256_set1_ps( ray.Hit.T );
			int stackPtr = 0, steps = 0;
			uint nodeIdx = 0;
			v256 rx8 = Avx.mm256_set1_ps( ray.O.x * ray.RD.x ), rdx8 = Avx.mm256_set1_ps( ray.RD.x );
			v256 ry8 = Avx.mm256_set1_ps( ray.O.y * ray.RD.y ), rdy8 = Avx.mm256_set1_ps( ray.RD.y );
			v256 rz8 = Avx.mm256_set1_ps( ray.O.z * ray.RD.z ), rdz8 = Avx.mm256_set1_ps( ray.RD.z );
			v128 ox4 = Sse.set1_ps( ray.O.x ), oy4 = Sse.set1_ps( ray.O.y ), oz4 = Sse.set1_ps( ray.O.z );
			v128 dx4 = Sse.set1_ps( ray.D.x ), dy4 = Sse.set1_ps( ray.D.y ), dz4 = Sse.set1_ps( ray.D.z );
			v128 zero4 = Sse.setzero_ps(), one4 = Sse.set1_ps( 1f ), inf4 = Sse.set1_ps( 1e34f );
			// The C++ shift amount is a template constant; here it is a runtime value, so the
			// variable-count form of the shift is used instead of the immediate one.
			v128 shift4 = new v128( signShift, 0, 0, 0 );
			while ( true )
			{
				steps++;
				while ( ( nodeIdx & LeafBit ) == 0 )
				{
					Bvh8CpuNode* n = ( Bvh8CpuNode* )( Data + ( nodeIdx * BlockSize ) );
					v256 tx1 = Fma.mm256_fmsub_ps( Avx.mm256_load_ps( posX ? n->XMin8 : n->XMax8 ), rdx8, rx8 );
					v256 ty1 = Fma.mm256_fmsub_ps( Avx.mm256_load_ps( posY ? n->YMin8 : n->YMax8 ), rdy8, ry8 );
					v256 tz1 = Fma.mm256_fmsub_ps( Avx.mm256_load_ps( posZ ? n->ZMin8 : n->ZMax8 ), rdz8, rz8 );
					v256 tx2 = Fma.mm256_fmsub_ps( Avx.mm256_load_ps( posX ? n->XMax8 : n->XMin8 ), rdx8, rx8 );
					v256 ty2 = Fma.mm256_fmsub_ps( Avx.mm256_load_ps( posY ? n->YMax8 : n->YMin8 ), rdy8, ry8 );
					v256 tz2 = Fma.mm256_fmsub_ps( Avx.mm256_load_ps( posZ ? n->ZMax8 : n->ZMin8 ), rdz8, rz8 );
					v256 tmin = Avx.mm256_max_ps( Avx.mm256_max_ps( Avx.mm256_max_ps( zero8, tx1 ), ty1 ), tz1 );
					v256 tmax = Avx.mm256_min_ps( Avx.mm256_min_ps( Avx.mm256_min_ps( tx2, t8 ), ty2 ), tz2 );
					v256 mask8 = Avx.mm256_cmp_ps( tmin, tmax, ( int )Avx.CMP.LE_OQ );
					uint mask = ( uint )Avx.mm256_movemask_ps( mask8 );
					int validNodes = math.countbits( mask );
					if ( validNodes == 1 )
					{
						nodeIdx = n->Child8[ Bfind( mask ) ];
					}
					else if ( validNodes != 0 )
					{
						v256 index = Avx2.mm256_srl_epi32( Avx.mm256_load_si256( n->Perm8 ), shift4 );
						uint m = ( uint )Avx.mm256_movemask_ps( Avx2.mm256_permutevar8x32_ps( mask8, index ) );
						tmin = Avx2.mm256_permutevar8x32_ps( tmin, index );
						v256 cpi = IdxLut256( m );
						v256 c8 = Avx2.mm256_permutevar8x32_epi32( Avx.mm256_load_si256( n->Child8 ), index );
						v256 dist8 = Avx2.mm256_permutevar8x32_ps( tmin, cpi );
						v256 child8 = Avx2.mm256_permutevar8x32_epi32( c8, cpi );
						Avx.mm256_storeu_si256( nodeStack + stackPtr, child8 );
						Avx.mm256_storeu_ps( distStack + stackPtr, dist8 );
						stackPtr += validNodes - 1;
						nodeIdx = nodeStack[ stackPtr ];
					}
					else
					{
						if ( stackPtr == 0 )
						{
							return steps;
						}
						nodeIdx = nodeStack[ --stackPtr ];
					}
				}
				// Moeller-Trumbore ray/triangle intersection algorithm for four triangles
				Bvh4CpuTri4Leaf* leaf = ( Bvh4CpuTri4Leaf* )( Data + ( ( nodeIdx & LeafOffsetMask ) * BlockSize ) );
				v128 t4 = Avx.mm256_extractf128_ps( t8, 0 );
				v128 e1x4 = Sse.load_ps( &leaf->E1x4 ), e1y4 = Sse.load_ps( &leaf->E1y4 ), e1z4 = Sse.load_ps( &leaf->E1z4 );
				v128 e2x4 = Sse.load_ps( &leaf->E2x4 ), e2y4 = Sse.load_ps( &leaf->E2y4 ), e2z4 = Sse.load_ps( &leaf->E2z4 );
				v128 hx4 = Fma.fmsub_ps( dy4, e2z4, Sse.mul_ps( dz4, e2y4 ) );
				v128 hy4 = Fma.fmsub_ps( dz4, e2x4, Sse.mul_ps( dx4, e2z4 ) );
				v128 hz4 = Fma.fmsub_ps( dx4, e2y4, Sse.mul_ps( dy4, e2x4 ) );
				v128 sx4 = Sse.sub_ps( ox4, Sse.load_ps( &leaf->V0x4 ) );
				v128 sy4 = Sse.sub_ps( oy4, Sse.load_ps( &leaf->V0y4 ) );
				v128 sz4 = Sse.sub_ps( oz4, Sse.load_ps( &leaf->V0z4 ) );
				v128 det4 = Fma.fmadd_ps( e1z4, hz4, Fma.fmadd_ps( e1x4, hx4, Sse.mul_ps( e1y4, hy4 ) ) );
				v128 qz4 = Fma.fmsub_ps( sx4, e1y4, Sse.mul_ps( sy4, e1x4 ) );
				v128 qx4 = Fma.fmsub_ps( sy4, e1z4, Sse.mul_ps( sz4, e1y4 ) );
				v128 qy4 = Fma.fmsub_ps( sz4, e1x4, Sse.mul_ps( sx4, e1z4 ) );
				v128 invDet4 = FastRcp4( det4 );
				v128 u4 = Sse.mul_ps( Fma.fmadd_ps( sz4, hz4, Fma.fmadd_ps( sx4, hx4, Sse.mul_ps( sy4, hy4 ) ) ), invDet4 );
				v128 v4 = Sse.mul_ps( Fma.fmadd_ps( dz4, qz4, Fma.fmadd_ps( dx4, qx4, Sse.mul_ps( dy4, qy4 ) ) ), invDet4 );
				v128 ta4 = Sse.mul_ps( Fma.fmadd_ps( e2z4, qz4, Fma.fmadd_ps( e2x4, qx4, Sse.mul_ps( e2y4, qy4 ) ) ), invDet4 );
				v128 mask1 = Sse.cmpge_ps( u4, zero4 ), mask2 = Sse.cmpge_ps( v4, zero4 );
				v128 mask3 = Sse.cmple_ps( Sse.add_ps( u4, v4 ), one4 );
				v128 mask4 = Sse.cmpgt_ps( ta4, zero4 );
				v128 mask5 = Sse.cmplt_ps( ta4, t4 );
				v128 combined = Sse.and_ps( Sse.and_ps( Sse.and_ps( mask1, mask2 ), Sse.and_ps( mask3, mask4 ) ), mask5 );
				uint imask = ( uint )Sse.movemask_ps( combined );
				// evaluate opacity map, if present (SSE version).
				if ( OpMap != null && imask != 0 )
				{
					v128 idx4 = OpacityIndices4( u4, v4, one4 );
					v128 omask4 = Sse.setzero_ps();
					uint words = ( ( OpMapN * OpMapN ) + 31 ) >> 5;
					// proceed with scalar code for the gather operation, as the C++ does.
					for ( int i = 0; i < 4; i++ )
					{
						if ( ( imask & ( 1u << i ) ) != 0 )
						{
							uint idx = ( ( uint* )&idx4 )[ i ];
							uint* om = OpMap + ( ( ( uint* )&leaf->PrimIdx )[ i ] * words );
							if ( ( om[ idx >> 5 ] & ( 1u << ( int )( idx & 31 ) ) ) != 0 )
							{
								( ( uint* )&omask4 )[ i ] = 0xffffffff;
							}
						}
					}
					// combine
					combined = Sse.and_ps( combined, omask4 );
					imask = ( uint )Sse.movemask_ps( combined );
				}
				if ( imask != 0 )
				{
					// compute broadcasted horizontal minimum of dist4
					v128 dist4 = Sse4_1.blendv_ps( inf4, ta4, combined );
					v128 a = Sse.min_ps( dist4, Sse.shuffle_ps( dist4, dist4, ShuffleRotate ) );
					v128 c = Sse.min_ps( a, Sse.shuffle_ps( a, a, ShuffleSwap ) );
					int lane = Bfind( ( uint )Sse.movemask_ps( Sse.cmpeq_ps( c, dist4 ) ) );
					// update hit record
					float t = ( ( float* )&dist4 )[ lane ];
					ray.Hit.T = t;
					ray.Hit.U = ( ( float* )&u4 )[ lane ];
					ray.Hit.V = ( ( float* )&v4 )[ lane ];
					// INST_IDX_BITS == 32: the instance index lives in its own field.
					ray.Hit.Prim = ( ( uint* )&leaf->PrimIdx )[ lane ];
					ray.Hit.Inst = ray.InstIdx;
					t8 = Avx.mm256_set1_ps( t );
					// compress stack
					int outStackPtr = 0;
					for ( int i = 0; i < stackPtr; i += 8 )
					{
						v256 node8 = Avx.mm256_loadu_si256( nodeStack + i );
						v256 d8 = Avx.mm256_loadu_ps( distStack + i );
						uint keep = ( uint )Avx.mm256_movemask_ps( Avx.mm256_cmp_ps( d8, t8, ( int )Avx.CMP.LE_OQ ) );
						v256 cpi = IdxLut256( keep );
						v256 dst8 = Avx2.mm256_permutevar8x32_ps( d8, cpi );
						node8 = Avx2.mm256_permutevar8x32_epi32( node8, cpi );
						Avx.mm256_storeu_ps( distStack + outStackPtr, dst8 );
						Avx.mm256_storeu_si256( nodeStack + outStackPtr, node8 );
						int numItems = math.min( 8, stackPtr - i );
						uint validMask = ( 1u << numItems ) - 1u;
						outStackPtr += math.countbits( keep & validMask );
					}
					stackPtr = outStackPtr;
				}
				if ( stackPtr == 0 )
				{
					break;
				}
				nodeIdx = nodeStack[ --stackPtr ];
			}
			return steps;
		}

		/// <summary>Port of the templated BVH8_CPU::IsOccluded body, AVX2 path.</summary>
		private bool IsOccludedSimd( in Ray ray, bool posX, bool posY, bool posZ )
		{
			uint* nodeStack = stackalloc uint[ StackSize ];
			int stackPtr = 0;
			uint nodeIdx = 0;
			v256 t8 = Avx.mm256_set1_ps( ray.Hit.T );
			v256 rx8 = Avx.mm256_set1_ps( ray.O.x * ray.RD.x ), rdx8 = Avx.mm256_set1_ps( ray.RD.x );
			v256 ry8 = Avx.mm256_set1_ps( ray.O.y * ray.RD.y ), rdy8 = Avx.mm256_set1_ps( ray.RD.y );
			v256 rz8 = Avx.mm256_set1_ps( ray.O.z * ray.RD.z ), rdz8 = Avx.mm256_set1_ps( ray.RD.z );
			v128 ox4 = Sse.set1_ps( ray.O.x ), oy4 = Sse.set1_ps( ray.O.y ), oz4 = Sse.set1_ps( ray.O.z );
			v128 dx4 = Sse.set1_ps( ray.D.x ), dy4 = Sse.set1_ps( ray.D.y ), dz4 = Sse.set1_ps( ray.D.z );
			v128 t4 = Sse.set1_ps( ray.Hit.T );
			v128 one4 = Sse.set1_ps( 1f ), zero4 = Sse.setzero_ps();
			while ( true )
			{
				while ( ( nodeIdx & LeafBit ) == 0 )
				{
					Bvh8CpuNode* n = ( Bvh8CpuNode* )( Data + ( nodeIdx * BlockSize ) );
					v256 c8 = Avx.mm256_load_si256( n->Child8 );
					v256 tx1 = Fma.mm256_fmsub_ps( Avx.mm256_load_ps( posX ? n->XMin8 : n->XMax8 ), rdx8, rx8 );
					v256 ty1 = Fma.mm256_fmsub_ps( Avx.mm256_load_ps( posY ? n->YMin8 : n->YMax8 ), rdy8, ry8 );
					v256 tz1 = Fma.mm256_fmsub_ps( Avx.mm256_load_ps( posZ ? n->ZMin8 : n->ZMax8 ), rdz8, rz8 );
					v256 tx2 = Fma.mm256_fmsub_ps( Avx.mm256_load_ps( posX ? n->XMax8 : n->XMin8 ), rdx8, rx8 );
					v256 ty2 = Fma.mm256_fmsub_ps( Avx.mm256_load_ps( posY ? n->YMax8 : n->YMin8 ), rdy8, ry8 );
					v256 tz2 = Fma.mm256_fmsub_ps( Avx.mm256_load_ps( posZ ? n->ZMax8 : n->ZMin8 ), rdz8, rz8 );
					v256 tmin = Avx.mm256_max_ps( Avx.mm256_max_ps( Avx.mm256_max_ps( Avx.mm256_setzero_ps(), tx1 ), ty1 ), tz1 );
					v256 tmax = Avx.mm256_min_ps( Avx.mm256_min_ps( Avx.mm256_min_ps( tx2, t8 ), ty2 ), tz2 );
					v256 mask8 = Avx.mm256_cmp_ps( tmin, tmax, ( int )Avx.CMP.LE_OQ );
					uint mask = ( uint )Avx.mm256_movemask_ps( mask8 );
					int validNodes = math.countbits( mask );
					if ( validNodes == 1 )
					{
						nodeIdx = n->Child8[ Bfind( mask ) ];
					}
					else if ( validNodes != 0 )
					{
						v256 cpi = IdxLut256( mask );
						v256 child8 = Avx2.mm256_permutevar8x32_epi32( c8, cpi );
						Avx.mm256_storeu_si256( nodeStack + stackPtr, child8 );
						stackPtr += validNodes - 1;
						nodeIdx = nodeStack[ stackPtr ];
					}
					else
					{
						if ( stackPtr == 0 )
						{
							return false;
						}
						nodeIdx = nodeStack[ --stackPtr ];
					}
				}
				// Moeller-Trumbore ray/triangle intersection algorithm for four triangles
				Bvh4CpuTri4Leaf* leaf = ( Bvh4CpuTri4Leaf* )( Data + ( ( nodeIdx & LeafOffsetMask ) * BlockSize ) );
				v128 e1x4 = Sse.load_ps( &leaf->E1x4 ), e1y4 = Sse.load_ps( &leaf->E1y4 ), e1z4 = Sse.load_ps( &leaf->E1z4 );
				v128 e2x4 = Sse.load_ps( &leaf->E2x4 ), e2y4 = Sse.load_ps( &leaf->E2y4 ), e2z4 = Sse.load_ps( &leaf->E2z4 );
				v128 hx4 = Fma.fmsub_ps( dy4, e2z4, Sse.mul_ps( dz4, e2y4 ) );
				v128 hy4 = Fma.fmsub_ps( dz4, e2x4, Sse.mul_ps( dx4, e2z4 ) );
				v128 hz4 = Fma.fmsub_ps( dx4, e2y4, Sse.mul_ps( dy4, e2x4 ) );
				v128 sx4 = Sse.sub_ps( ox4, Sse.load_ps( &leaf->V0x4 ) );
				v128 sy4 = Sse.sub_ps( oy4, Sse.load_ps( &leaf->V0y4 ) );
				v128 sz4 = Sse.sub_ps( oz4, Sse.load_ps( &leaf->V0z4 ) );
				v128 det4 = Fma.fmadd_ps( e1z4, hz4, Fma.fmadd_ps( e1x4, hx4, Sse.mul_ps( e1y4, hy4 ) ) );
				v128 qz4 = Fma.fmsub_ps( sx4, e1y4, Sse.mul_ps( sy4, e1x4 ) );
				v128 qx4 = Fma.fmsub_ps( sy4, e1z4, Sse.mul_ps( sz4, e1y4 ) );
				v128 qy4 = Fma.fmsub_ps( sz4, e1x4, Sse.mul_ps( sx4, e1z4 ) );
				v128 invDet4 = FastRcp4( det4 );
				v128 u4 = Sse.mul_ps( Fma.fmadd_ps( sz4, hz4, Fma.fmadd_ps( sx4, hx4, Sse.mul_ps( sy4, hy4 ) ) ), invDet4 );
				v128 v4 = Sse.mul_ps( Fma.fmadd_ps( dz4, qz4, Fma.fmadd_ps( dx4, qx4, Sse.mul_ps( dy4, qy4 ) ) ), invDet4 );
				v128 ta4 = Sse.mul_ps( Fma.fmadd_ps( e2z4, qz4, Fma.fmadd_ps( e2x4, qx4, Sse.mul_ps( e2y4, qy4 ) ) ), invDet4 );
				v128 mask1 = Sse.cmpge_ps( u4, zero4 );
				v128 mask2 = Sse.cmpge_ps( v4, zero4 );
				v128 mask3 = Sse.cmple_ps( Sse.add_ps( u4, v4 ), one4 );
				v128 mask4 = Sse.cmplt_ps( ta4, t4 );
				v128 mask5 = Sse.cmpgt_ps( ta4, zero4 );
				v128 combined = Sse.and_ps( Sse.and_ps( Sse.and_ps( mask1, mask2 ), Sse.and_ps( mask3, mask4 ) ), mask5 );
				uint imask = ( uint )Sse.movemask_ps( combined );
				if ( imask != 0 )
				{
					if ( OpMap == null )
					{
						return true;
					}
					// evaluate opacity map, SSE version.
					v128 idx4 = OpacityIndices4( u4, v4, one4 );
					uint words = ( ( OpMapN * OpMapN ) + 31 ) >> 5;
					// proceed with scalar code for the gather operation, as the C++ does.
					for ( int i = 0; i < 4; i++ )
					{
						if ( ( imask & ( 1u << i ) ) != 0 )
						{
							uint idx = ( ( uint* )&idx4 )[ i ];
							uint* om = OpMap + ( ( ( uint* )&leaf->PrimIdx )[ i ] * words );
							if ( ( om[ idx >> 5 ] & ( 1u << ( int )( idx & 31 ) ) ) != 0 )
							{
								return true;
							}
						}
					}
				}
				// continue
				if ( stackPtr == 0 )
				{
					return false;
				}
				nodeIdx = nodeStack[ --stackPtr ];
			}
		}

		/// <summary>
		/// Scalar fallback for <see cref="IntersectSimd"/>: the same traversal as an eight-lane loop.
		/// Children are visited in the order the node's perm8 field prescribes for the ray's octant,
		/// pushed farthest first so the nearest is popped first; equal minimum triangle distances
		/// within one leaf resolve to the highest lane, matching the C++ __bfind tie-break.
		/// </summary>
		private int IntersectScalar( ref Ray ray, bool posX, bool posY, bool posZ, int signShift )
		{
			uint* nodeStack = stackalloc uint[ StackSize ];
			float* distStack = stackalloc float[ StackSize ];
			float* tmin = stackalloc float[ 8 ];
			float* u = stackalloc float[ 4 ];
			float* v = stackalloc float[ 4 ];
			float* dist = stackalloc float[ 4 ];
			int stackPtr = 0, steps = 0;
			uint nodeIdx = 0;
			float t = ray.Hit.T;
			float rx = ray.O.x * ray.RD.x, ry = ray.O.y * ray.RD.y, rz = ray.O.z * ray.RD.z;
			while ( true )
			{
				steps++;
				while ( ( nodeIdx & LeafBit ) == 0 )
				{
					Bvh8CpuNode* n = ( Bvh8CpuNode* )( Data + ( nodeIdx * BlockSize ) );
					float* xmin = posX ? n->XMin8 : n->XMax8;
					float* xmax = posX ? n->XMax8 : n->XMin8;
					float* ymin = posY ? n->YMin8 : n->YMax8;
					float* ymax = posY ? n->YMax8 : n->YMin8;
					float* zmin = posZ ? n->ZMin8 : n->ZMax8;
					float* zmax = posZ ? n->ZMax8 : n->ZMin8;
					uint* child = n->Child8;
					uint mask = 0;
					for ( int lane = 0; lane < 8; lane++ )
					{
						float tx1 = ( xmin[ lane ] * ray.RD.x ) - rx;
						float ty1 = ( ymin[ lane ] * ray.RD.y ) - ry;
						float tz1 = ( zmin[ lane ] * ray.RD.z ) - rz;
						float tx2 = ( xmax[ lane ] * ray.RD.x ) - rx;
						float ty2 = ( ymax[ lane ] * ray.RD.y ) - ry;
						float tz2 = ( zmax[ lane ] * ray.RD.z ) - rz;
						float lo = math.max( math.max( math.max( 0f, tx1 ), ty1 ), tz1 );
						float hi = math.min( math.min( math.min( tx2, t ), ty2 ), tz2 );
						tmin[ lane ] = lo;
						if ( lo <= hi )
						{
							mask |= 1u << lane;
						}
					}
					int validNodes = math.countbits( mask );
					if ( validNodes == 1 )
					{
						nodeIdx = child[ Bfind( mask ) ];
					}
					else if ( validNodes != 0 )
					{
						// The SIMD path permutes all eight lanes and then compacts the surviving
						// ones with the shuffle LUT; this is the same thing, done in place.
						uint* perm = n->Perm8;
						int outPtr = 0;
						for ( int i = 0; i < 8; i++ )
						{
							int lane = ( int )( ( perm[ i ] >> signShift ) & 7 );
							if ( ( mask & ( 1u << lane ) ) != 0 )
							{
								nodeStack[ stackPtr + outPtr ] = child[ lane ];
								distStack[ stackPtr + outPtr ] = tmin[ lane ];
								outPtr++;
							}
						}
						stackPtr += validNodes - 1;
						nodeIdx = nodeStack[ stackPtr ];
					}
					else
					{
						if ( stackPtr == 0 )
						{
							return steps;
						}
						nodeIdx = nodeStack[ --stackPtr ];
					}
				}
				// Moeller-Trumbore ray/triangle intersection algorithm for four triangles
				Bvh4CpuTri4Leaf* leaf = ( Bvh4CpuTri4Leaf* )( Data + ( ( nodeIdx & LeafOffsetMask ) * BlockSize ) );
				uint imask = 0;
				for ( int lane = 0; lane < 4; lane++ )
				{
					float e1x = ( ( float* )&leaf->E1x4 )[ lane ], e1y = ( ( float* )&leaf->E1y4 )[ lane ], e1z = ( ( float* )&leaf->E1z4 )[ lane ];
					float e2x = ( ( float* )&leaf->E2x4 )[ lane ], e2y = ( ( float* )&leaf->E2y4 )[ lane ], e2z = ( ( float* )&leaf->E2z4 )[ lane ];
					float hx = ( ray.D.y * e2z ) - ( ray.D.z * e2y );
					float hy = ( ray.D.z * e2x ) - ( ray.D.x * e2z );
					float hz = ( ray.D.x * e2y ) - ( ray.D.y * e2x );
					float sx = ray.O.x - ( ( float* )&leaf->V0x4 )[ lane ];
					float sy = ray.O.y - ( ( float* )&leaf->V0y4 )[ lane ];
					float sz = ray.O.z - ( ( float* )&leaf->V0z4 )[ lane ];
					float det = ( e1z * hz ) + ( ( e1x * hx ) + ( e1y * hy ) );
					float qz = ( sx * e1y ) - ( sy * e1x );
					float qx = ( sy * e1z ) - ( sz * e1y );
					float qy = ( sz * e1x ) - ( sx * e1z );
					// Deviation: the SIMD path fuses these multiply-adds and uses fastrcp4, a
					// reciprocal plus one Newton step, instead of a division.
					float invDet = 1f / det;
					float lu = ( ( sz * hz ) + ( ( sx * hx ) + ( sy * hy ) ) ) * invDet;
					float lv = ( ( ray.D.z * qz ) + ( ( ray.D.x * qx ) + ( ray.D.y * qy ) ) ) * invDet;
					float lt = ( ( e2z * qz ) + ( ( e2x * qx ) + ( e2y * qy ) ) ) * invDet;
					u[ lane ] = lu;
					v[ lane ] = lv;
					bool hit = lu >= 0f && lv >= 0f && ( lu + lv ) <= 1f && lt > 0f && lt < t;
					// evaluate opacity map, if present.
					if ( hit && OpMap != null )
					{
						hit = OpacityOpaque( ( ( uint* )&leaf->PrimIdx )[ lane ], lu, lv );
					}
					dist[ lane ] = hit ? lt : 1e34f;
					if ( hit )
					{
						imask |= 1u << lane;
					}
				}
				if ( imask != 0 )
				{
					float best = math.min( math.min( dist[ 0 ], dist[ 1 ] ), math.min( dist[ 2 ], dist[ 3 ] ) );
					int lane = 0;
					for ( int i = 0; i < 4; i++ )
					{
						if ( dist[ i ] == best )
						{
							lane = i; // the highest matching lane wins, like the C++ __bfind.
						}
					}
					t = dist[ lane ];
					ray.Hit.T = t;
					ray.Hit.U = u[ lane ];
					ray.Hit.V = v[ lane ];
					ray.Hit.Prim = ( ( uint* )&leaf->PrimIdx )[ lane ];
					ray.Hit.Inst = ray.InstIdx;
					// compress stack
					int outStackPtr = 0;
					for ( int i = 0; i < stackPtr; i++ )
					{
						if ( distStack[ i ] <= t )
						{
							distStack[ outStackPtr ] = distStack[ i ];
							nodeStack[ outStackPtr ] = nodeStack[ i ];
							outStackPtr++;
						}
					}
					stackPtr = outStackPtr;
				}
				if ( stackPtr == 0 )
				{
					break;
				}
				nodeIdx = nodeStack[ --stackPtr ];
			}
			return steps;
		}

		/// <summary>
		/// Scalar fallback for <see cref="IsOccludedSimd"/>. Like the C++ AVX2 version, this one does
		/// not consult perm8: the surviving children are pushed in lane order.
		/// </summary>
		private bool IsOccludedScalar( in Ray ray, bool posX, bool posY, bool posZ )
		{
			uint* nodeStack = stackalloc uint[ StackSize ];
			int stackPtr = 0;
			uint nodeIdx = 0;
			float t = ray.Hit.T;
			float rx = ray.O.x * ray.RD.x, ry = ray.O.y * ray.RD.y, rz = ray.O.z * ray.RD.z;
			while ( true )
			{
				while ( ( nodeIdx & LeafBit ) == 0 )
				{
					Bvh8CpuNode* n = ( Bvh8CpuNode* )( Data + ( nodeIdx * BlockSize ) );
					float* xmin = posX ? n->XMin8 : n->XMax8;
					float* xmax = posX ? n->XMax8 : n->XMin8;
					float* ymin = posY ? n->YMin8 : n->YMax8;
					float* ymax = posY ? n->YMax8 : n->YMin8;
					float* zmin = posZ ? n->ZMin8 : n->ZMax8;
					float* zmax = posZ ? n->ZMax8 : n->ZMin8;
					uint* child = n->Child8;
					uint mask = 0;
					for ( int lane = 0; lane < 8; lane++ )
					{
						float tx1 = ( xmin[ lane ] * ray.RD.x ) - rx;
						float ty1 = ( ymin[ lane ] * ray.RD.y ) - ry;
						float tz1 = ( zmin[ lane ] * ray.RD.z ) - rz;
						float tx2 = ( xmax[ lane ] * ray.RD.x ) - rx;
						float ty2 = ( ymax[ lane ] * ray.RD.y ) - ry;
						float tz2 = ( zmax[ lane ] * ray.RD.z ) - rz;
						float lo = math.max( math.max( math.max( 0f, tx1 ), ty1 ), tz1 );
						float hi = math.min( math.min( math.min( tx2, t ), ty2 ), tz2 );
						if ( lo <= hi )
						{
							mask |= 1u << lane;
						}
					}
					int validNodes = math.countbits( mask );
					if ( validNodes == 1 )
					{
						nodeIdx = child[ Bfind( mask ) ];
					}
					else if ( validNodes != 0 )
					{
						int outPtr = 0;
						for ( int lane = 0; lane < 8; lane++ )
						{
							if ( ( mask & ( 1u << lane ) ) != 0 )
							{
								nodeStack[ stackPtr + outPtr ] = child[ lane ];
								outPtr++;
							}
						}
						stackPtr += validNodes - 1;
						nodeIdx = nodeStack[ stackPtr ];
					}
					else
					{
						if ( stackPtr == 0 )
						{
							return false;
						}
						nodeIdx = nodeStack[ --stackPtr ];
					}
				}
				// Moeller-Trumbore ray/triangle intersection algorithm for four triangles
				Bvh4CpuTri4Leaf* leaf = ( Bvh4CpuTri4Leaf* )( Data + ( ( nodeIdx & LeafOffsetMask ) * BlockSize ) );
				for ( int lane = 0; lane < 4; lane++ )
				{
					float e1x = ( ( float* )&leaf->E1x4 )[ lane ], e1y = ( ( float* )&leaf->E1y4 )[ lane ], e1z = ( ( float* )&leaf->E1z4 )[ lane ];
					float e2x = ( ( float* )&leaf->E2x4 )[ lane ], e2y = ( ( float* )&leaf->E2y4 )[ lane ], e2z = ( ( float* )&leaf->E2z4 )[ lane ];
					float hx = ( ray.D.y * e2z ) - ( ray.D.z * e2y );
					float hy = ( ray.D.z * e2x ) - ( ray.D.x * e2z );
					float hz = ( ray.D.x * e2y ) - ( ray.D.y * e2x );
					float sx = ray.O.x - ( ( float* )&leaf->V0x4 )[ lane ];
					float sy = ray.O.y - ( ( float* )&leaf->V0y4 )[ lane ];
					float sz = ray.O.z - ( ( float* )&leaf->V0z4 )[ lane ];
					float det = ( e1z * hz ) + ( ( e1x * hx ) + ( e1y * hy ) );
					float qz = ( sx * e1y ) - ( sy * e1x );
					float qx = ( sy * e1z ) - ( sz * e1y );
					float qy = ( sz * e1x ) - ( sx * e1z );
					float invDet = 1f / det;
					float lu = ( ( sz * hz ) + ( ( sx * hx ) + ( sy * hy ) ) ) * invDet;
					float lv = ( ( ray.D.z * qz ) + ( ( ray.D.x * qx ) + ( ray.D.y * qy ) ) ) * invDet;
					float lt = ( ( e2z * qz ) + ( ( e2x * qx ) + ( e2y * qy ) ) ) * invDet;
					if ( lu >= 0f && lv >= 0f && ( lu + lv ) <= 1f && lt < t && lt > 0f )
					{
						// evaluate opacity map, if present.
						if ( OpMap == null || OpacityOpaque( ( ( uint* )&leaf->PrimIdx )[ lane ], lu, lv ) )
						{
							return true;
						}
					}
				}
				// continue
				if ( stackPtr == 0 )
				{
					return false;
				}
				nodeIdx = nodeStack[ --stackPtr ];
			}
		}

		/// <summary>Port of fastrcp4: a reciprocal estimate refined with one Newton-Raphson step.</summary>
		private static v128 FastRcp4( v128 a )
		{
			v128 res = Sse.rcp_ps( a );
			v128 muls = Sse.mul_ps( a, Sse.mul_ps( res, res ) );
			return Sse.sub_ps( Sse.add_ps( res, res ), muls );
		}

		/// <summary>Port of __bfind: the index of the most significant set bit.</summary>
		private static int Bfind( uint x )
		{
			return 31 - math.lzcnt( x );
		}

		/// <summary>
		/// Port of the idxLUT256 table: a _mm256_permutevar8x32 control vector that compacts the
		/// lanes selected by the eight-bit mask m into the low dwords, keeping their order.
		/// Unselected dwords read lane 0, exactly as in the C++ table; the traversal never uses
		/// them. The C++ indexes its table with 255 - m, so entry m here is idxLUT256[255 - m];
		/// all 256 values below were generated from that rule and checked against the literals in
		/// tiny_bvh.h. TO256 zero-extends eight packed bytes to eight int32 lanes.
		/// </summary>
		private static v256 IdxLut256( uint m )
		{
			return Avx2.mm256_cvtepu8_epi32( new v128( LutPacked( m ), 0ul ) );
		}

		/// <summary>The eight packed lane indices of <see cref="IdxLut256"/>, one byte each.</summary>
		private static ulong LutPacked( uint m )
		{
			switch ( m )
			{
				case 0: return 0x0000000000000000ul;
				case 1: return 0x0000000000000000ul;
				case 2: return 0x0000000000000001ul;
				case 3: return 0x0000000000000100ul;
				case 4: return 0x0000000000000002ul;
				case 5: return 0x0000000000000200ul;
				case 6: return 0x0000000000000201ul;
				case 7: return 0x0000000000020100ul;
				case 8: return 0x0000000000000003ul;
				case 9: return 0x0000000000000300ul;
				case 10: return 0x0000000000000301ul;
				case 11: return 0x0000000000030100ul;
				case 12: return 0x0000000000000302ul;
				case 13: return 0x0000000000030200ul;
				case 14: return 0x0000000000030201ul;
				case 15: return 0x0000000003020100ul;
				case 16: return 0x0000000000000004ul;
				case 17: return 0x0000000000000400ul;
				case 18: return 0x0000000000000401ul;
				case 19: return 0x0000000000040100ul;
				case 20: return 0x0000000000000402ul;
				case 21: return 0x0000000000040200ul;
				case 22: return 0x0000000000040201ul;
				case 23: return 0x0000000004020100ul;
				case 24: return 0x0000000000000403ul;
				case 25: return 0x0000000000040300ul;
				case 26: return 0x0000000000040301ul;
				case 27: return 0x0000000004030100ul;
				case 28: return 0x0000000000040302ul;
				case 29: return 0x0000000004030200ul;
				case 30: return 0x0000000004030201ul;
				case 31: return 0x0000000403020100ul;
				case 32: return 0x0000000000000005ul;
				case 33: return 0x0000000000000500ul;
				case 34: return 0x0000000000000501ul;
				case 35: return 0x0000000000050100ul;
				case 36: return 0x0000000000000502ul;
				case 37: return 0x0000000000050200ul;
				case 38: return 0x0000000000050201ul;
				case 39: return 0x0000000005020100ul;
				case 40: return 0x0000000000000503ul;
				case 41: return 0x0000000000050300ul;
				case 42: return 0x0000000000050301ul;
				case 43: return 0x0000000005030100ul;
				case 44: return 0x0000000000050302ul;
				case 45: return 0x0000000005030200ul;
				case 46: return 0x0000000005030201ul;
				case 47: return 0x0000000503020100ul;
				case 48: return 0x0000000000000504ul;
				case 49: return 0x0000000000050400ul;
				case 50: return 0x0000000000050401ul;
				case 51: return 0x0000000005040100ul;
				case 52: return 0x0000000000050402ul;
				case 53: return 0x0000000005040200ul;
				case 54: return 0x0000000005040201ul;
				case 55: return 0x0000000504020100ul;
				case 56: return 0x0000000000050403ul;
				case 57: return 0x0000000005040300ul;
				case 58: return 0x0000000005040301ul;
				case 59: return 0x0000000504030100ul;
				case 60: return 0x0000000005040302ul;
				case 61: return 0x0000000504030200ul;
				case 62: return 0x0000000504030201ul;
				case 63: return 0x0000050403020100ul;
				case 64: return 0x0000000000000006ul;
				case 65: return 0x0000000000000600ul;
				case 66: return 0x0000000000000601ul;
				case 67: return 0x0000000000060100ul;
				case 68: return 0x0000000000000602ul;
				case 69: return 0x0000000000060200ul;
				case 70: return 0x0000000000060201ul;
				case 71: return 0x0000000006020100ul;
				case 72: return 0x0000000000000603ul;
				case 73: return 0x0000000000060300ul;
				case 74: return 0x0000000000060301ul;
				case 75: return 0x0000000006030100ul;
				case 76: return 0x0000000000060302ul;
				case 77: return 0x0000000006030200ul;
				case 78: return 0x0000000006030201ul;
				case 79: return 0x0000000603020100ul;
				case 80: return 0x0000000000000604ul;
				case 81: return 0x0000000000060400ul;
				case 82: return 0x0000000000060401ul;
				case 83: return 0x0000000006040100ul;
				case 84: return 0x0000000000060402ul;
				case 85: return 0x0000000006040200ul;
				case 86: return 0x0000000006040201ul;
				case 87: return 0x0000000604020100ul;
				case 88: return 0x0000000000060403ul;
				case 89: return 0x0000000006040300ul;
				case 90: return 0x0000000006040301ul;
				case 91: return 0x0000000604030100ul;
				case 92: return 0x0000000006040302ul;
				case 93: return 0x0000000604030200ul;
				case 94: return 0x0000000604030201ul;
				case 95: return 0x0000060403020100ul;
				case 96: return 0x0000000000000605ul;
				case 97: return 0x0000000000060500ul;
				case 98: return 0x0000000000060501ul;
				case 99: return 0x0000000006050100ul;
				case 100: return 0x0000000000060502ul;
				case 101: return 0x0000000006050200ul;
				case 102: return 0x0000000006050201ul;
				case 103: return 0x0000000605020100ul;
				case 104: return 0x0000000000060503ul;
				case 105: return 0x0000000006050300ul;
				case 106: return 0x0000000006050301ul;
				case 107: return 0x0000000605030100ul;
				case 108: return 0x0000000006050302ul;
				case 109: return 0x0000000605030200ul;
				case 110: return 0x0000000605030201ul;
				case 111: return 0x0000060503020100ul;
				case 112: return 0x0000000000060504ul;
				case 113: return 0x0000000006050400ul;
				case 114: return 0x0000000006050401ul;
				case 115: return 0x0000000605040100ul;
				case 116: return 0x0000000006050402ul;
				case 117: return 0x0000000605040200ul;
				case 118: return 0x0000000605040201ul;
				case 119: return 0x0000060504020100ul;
				case 120: return 0x0000000006050403ul;
				case 121: return 0x0000000605040300ul;
				case 122: return 0x0000000605040301ul;
				case 123: return 0x0000060504030100ul;
				case 124: return 0x0000000605040302ul;
				case 125: return 0x0000060504030200ul;
				case 126: return 0x0000060504030201ul;
				case 127: return 0x0006050403020100ul;
				case 128: return 0x0000000000000007ul;
				case 129: return 0x0000000000000700ul;
				case 130: return 0x0000000000000701ul;
				case 131: return 0x0000000000070100ul;
				case 132: return 0x0000000000000702ul;
				case 133: return 0x0000000000070200ul;
				case 134: return 0x0000000000070201ul;
				case 135: return 0x0000000007020100ul;
				case 136: return 0x0000000000000703ul;
				case 137: return 0x0000000000070300ul;
				case 138: return 0x0000000000070301ul;
				case 139: return 0x0000000007030100ul;
				case 140: return 0x0000000000070302ul;
				case 141: return 0x0000000007030200ul;
				case 142: return 0x0000000007030201ul;
				case 143: return 0x0000000703020100ul;
				case 144: return 0x0000000000000704ul;
				case 145: return 0x0000000000070400ul;
				case 146: return 0x0000000000070401ul;
				case 147: return 0x0000000007040100ul;
				case 148: return 0x0000000000070402ul;
				case 149: return 0x0000000007040200ul;
				case 150: return 0x0000000007040201ul;
				case 151: return 0x0000000704020100ul;
				case 152: return 0x0000000000070403ul;
				case 153: return 0x0000000007040300ul;
				case 154: return 0x0000000007040301ul;
				case 155: return 0x0000000704030100ul;
				case 156: return 0x0000000007040302ul;
				case 157: return 0x0000000704030200ul;
				case 158: return 0x0000000704030201ul;
				case 159: return 0x0000070403020100ul;
				case 160: return 0x0000000000000705ul;
				case 161: return 0x0000000000070500ul;
				case 162: return 0x0000000000070501ul;
				case 163: return 0x0000000007050100ul;
				case 164: return 0x0000000000070502ul;
				case 165: return 0x0000000007050200ul;
				case 166: return 0x0000000007050201ul;
				case 167: return 0x0000000705020100ul;
				case 168: return 0x0000000000070503ul;
				case 169: return 0x0000000007050300ul;
				case 170: return 0x0000000007050301ul;
				case 171: return 0x0000000705030100ul;
				case 172: return 0x0000000007050302ul;
				case 173: return 0x0000000705030200ul;
				case 174: return 0x0000000705030201ul;
				case 175: return 0x0000070503020100ul;
				case 176: return 0x0000000000070504ul;
				case 177: return 0x0000000007050400ul;
				case 178: return 0x0000000007050401ul;
				case 179: return 0x0000000705040100ul;
				case 180: return 0x0000000007050402ul;
				case 181: return 0x0000000705040200ul;
				case 182: return 0x0000000705040201ul;
				case 183: return 0x0000070504020100ul;
				case 184: return 0x0000000007050403ul;
				case 185: return 0x0000000705040300ul;
				case 186: return 0x0000000705040301ul;
				case 187: return 0x0000070504030100ul;
				case 188: return 0x0000000705040302ul;
				case 189: return 0x0000070504030200ul;
				case 190: return 0x0000070504030201ul;
				case 191: return 0x0007050403020100ul;
				case 192: return 0x0000000000000706ul;
				case 193: return 0x0000000000070600ul;
				case 194: return 0x0000000000070601ul;
				case 195: return 0x0000000007060100ul;
				case 196: return 0x0000000000070602ul;
				case 197: return 0x0000000007060200ul;
				case 198: return 0x0000000007060201ul;
				case 199: return 0x0000000706020100ul;
				case 200: return 0x0000000000070603ul;
				case 201: return 0x0000000007060300ul;
				case 202: return 0x0000000007060301ul;
				case 203: return 0x0000000706030100ul;
				case 204: return 0x0000000007060302ul;
				case 205: return 0x0000000706030200ul;
				case 206: return 0x0000000706030201ul;
				case 207: return 0x0000070603020100ul;
				case 208: return 0x0000000000070604ul;
				case 209: return 0x0000000007060400ul;
				case 210: return 0x0000000007060401ul;
				case 211: return 0x0000000706040100ul;
				case 212: return 0x0000000007060402ul;
				case 213: return 0x0000000706040200ul;
				case 214: return 0x0000000706040201ul;
				case 215: return 0x0000070604020100ul;
				case 216: return 0x0000000007060403ul;
				case 217: return 0x0000000706040300ul;
				case 218: return 0x0000000706040301ul;
				case 219: return 0x0000070604030100ul;
				case 220: return 0x0000000706040302ul;
				case 221: return 0x0000070604030200ul;
				case 222: return 0x0000070604030201ul;
				case 223: return 0x0007060403020100ul;
				case 224: return 0x0000000000070605ul;
				case 225: return 0x0000000007060500ul;
				case 226: return 0x0000000007060501ul;
				case 227: return 0x0000000706050100ul;
				case 228: return 0x0000000007060502ul;
				case 229: return 0x0000000706050200ul;
				case 230: return 0x0000000706050201ul;
				case 231: return 0x0000070605020100ul;
				case 232: return 0x0000000007060503ul;
				case 233: return 0x0000000706050300ul;
				case 234: return 0x0000000706050301ul;
				case 235: return 0x0000070605030100ul;
				case 236: return 0x0000000706050302ul;
				case 237: return 0x0000070605030200ul;
				case 238: return 0x0000070605030201ul;
				case 239: return 0x0007060503020100ul;
				case 240: return 0x0000000007060504ul;
				case 241: return 0x0000000706050400ul;
				case 242: return 0x0000000706050401ul;
				case 243: return 0x0000070605040100ul;
				case 244: return 0x0000000706050402ul;
				case 245: return 0x0000070605040200ul;
				case 246: return 0x0000070605040201ul;
				case 247: return 0x0007060504020100ul;
				case 248: return 0x0000000706050403ul;
				case 249: return 0x0000070605040300ul;
				case 250: return 0x0000070605040301ul;
				case 251: return 0x0007060504030100ul;
				case 252: return 0x0000070605040302ul;
				case 253: return 0x0007060504030200ul;
				case 254: return 0x0007060504030201ul;
				default: return 0x0706050403020100ul;
			}
		}
	}

	/// <summary>
	/// Burst-compiled probe behind <see cref="Bvh8Cpu.IsSimdSupported"/>. Direct calls must be
	/// synchronous, otherwise the Mono fallback runs and always reports no SIMD.
	/// </summary>
	[BurstCompile]
	internal static class Bvh8CpuSimd
	{
		[BurstCompile( CompileSynchronously = true )]
		internal static void Query( out int supported )
		{
			supported = Avx2.IsAvx2Supported && Fma.IsFmaSupported ? 1 : 0;
		}
	}
}
