using System.Runtime.CompilerServices;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Burst.Intrinsics.X86;

namespace TinyBVH
{
	/// <summary>
	/// Traversal half of tinybvh's BVH4_CPU class, i.e. the SSE 'WiVe' traversal. The C++ compiles
	/// eight template variants of each function, keyed on the sign of the ray direction; here the
	/// same predicates are evaluated once at the top and passed down, as Bvh.Intersect does.
	///
	/// Every function has a Burst SIMD path and a scalar fallback. Burst resolves
	/// Sse4_1.IsSse41Supported / Ssse3.IsSsse3Supported at compile time; under Mono they are false,
	/// so the editor - and any CPU without SSE4.1/SSSE3 - runs the scalar code. The two paths pick
	/// the same children in the same order and apply the same tie-break rules; they differ only in
	/// that the SIMD path divides by the fastrcp4 reciprocal of the Moeller-Trumbore determinant
	/// where the scalar path divides exactly, so hit distances can differ in the last few bits.
	/// </summary>
	public unsafe partial struct Bvh4Cpu
	{
		/// <summary>
		/// Traversal stack depth. The C++ uses 256 entries; four more are reserved here because the
		/// SIMD path stores a full 16-byte group at stackPtr before advancing by validNodes - 1.
		/// </summary>
		private const int StackSize = 260;

		/// <summary>_MM_SHUFFLE( 2, 1, 0, 3 ): rotate the lanes up by one.</summary>
		private const int ShuffleRotate = ( 2 << 6 ) | ( 1 << 4 ) | ( 0 << 2 ) | 3;
		/// <summary>_MM_SHUFFLE( 1, 0, 3, 2 ): swap the two lane pairs.</summary>
		private const int ShuffleSwap = ( 1 << 6 ) | ( 0 << 4 ) | ( 3 << 2 ) | 2;

		// Opacity micro maps (not owned), the opmap / opmapN BVH4_CPU inherits from BVHBase in the
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
		/// Port of BVH4_CPU::Intersect( Ray&amp; ), including its octant dispatcher. The hit, if any,
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
			int signShift = ( posX ? 2 : 0 ) + ( posY ? 4 : 0 ) + ( posZ ? 8 : 0 );
			if ( Sse4_1.IsSse41Supported && Ssse3.IsSsse3Supported )
			{
				return IntersectSimd( ref ray, posX, posY, posZ, signShift );
			}
			return IntersectScalar( ref ray, posX, posY, posZ, signShift );
		}

		/// <summary>
		/// Port of BVH4_CPU::IsOccluded( const Ray&amp; ), including its octant dispatcher. Returns
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
			if ( Sse4_1.IsSse41Supported && Ssse3.IsSsse3Supported )
			{
				return IsOccludedSimd( ray, posX, posY, posZ );
			}
			return IsOccludedScalar( ray, posX, posY, posZ );
		}

		/// <summary>
		/// Runs the scalar fallback of <see cref="Intersect"/> whatever the CPU supports. Regular
		/// callers want Intersect; this entry point exists so the fallback can also be validated
		/// from inside a Burst job, where float arithmetic is not widened to double. It mirrors
		/// Bvh8Cpu.IntersectScalarPath.
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
			int signShift = ( posX ? 2 : 0 ) + ( posY ? 4 : 0 ) + ( posZ ? 8 : 0 );
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

		/// <summary>Port of the templated BVH4_CPU::Intersect body, SSE4.2 path (not the AVX one).</summary>
		private int IntersectSimd( ref Ray ray, bool posX, bool posY, bool posZ, int signShift )
		{
			uint* nodeStack = stackalloc uint[ StackSize ];
			float* distStack = stackalloc float[ StackSize ];
			v128 zero4 = Sse.setzero_ps();
			v128 t4 = Sse.set1_ps( ray.Hit.T );
			int stackPtr = 0, steps = 0;
			uint nodeIdx = 0;
			v128 rx4 = Sse.set1_ps( ray.O.x * ray.RD.x ), rdx4 = Sse.set1_ps( ray.RD.x );
			v128 ry4 = Sse.set1_ps( ray.O.y * ray.RD.y ), rdy4 = Sse.set1_ps( ray.RD.y );
			v128 rz4 = Sse.set1_ps( ray.O.z * ray.RD.z ), rdz4 = Sse.set1_ps( ray.RD.z );
			v128 ox4 = Sse.set1_ps( ray.O.x ), oy4 = Sse.set1_ps( ray.O.y ), oz4 = Sse.set1_ps( ray.O.z );
			v128 dx4 = Sse.set1_ps( ray.D.x ), dy4 = Sse.set1_ps( ray.D.y ), dz4 = Sse.set1_ps( ray.D.z );
			v128 one4 = Sse.set1_ps( 1f ), inf4 = Sse.set1_ps( 1e34f );
			v128 shftmsk4 = Sse2.set1_epi32( 3 );
			v128 mul4 = Sse2.set1_epi32( 0x04040404 );
			v128 add4 = Sse2.set1_epi32( 0x03020100 );
			// The C++ shift amount is a template constant; here it is a runtime value, so the
			// variable-count form of the shift is used instead of the immediate one.
			v128 shift4 = new v128( signShift, 0, 0, 0 );
			while ( true )
			{
				steps++;
				while ( ( nodeIdx & LeafBit ) == 0 )
				{
					Bvh4CpuNode* n = ( Bvh4CpuNode* )( Data + ( nodeIdx * BlockSize ) );
					v128 tx1 = Sse.sub_ps( Sse.mul_ps( Sse.load_ps( posX ? &n->XMin4 : &n->XMax4 ), rdx4 ), rx4 );
					v128 ty1 = Sse.sub_ps( Sse.mul_ps( Sse.load_ps( posY ? &n->YMin4 : &n->YMax4 ), rdy4 ), ry4 );
					v128 tz1 = Sse.sub_ps( Sse.mul_ps( Sse.load_ps( posZ ? &n->ZMin4 : &n->ZMax4 ), rdz4 ), rz4 );
					v128 tx2 = Sse.sub_ps( Sse.mul_ps( Sse.load_ps( posX ? &n->XMax4 : &n->XMin4 ), rdx4 ), rx4 );
					v128 ty2 = Sse.sub_ps( Sse.mul_ps( Sse.load_ps( posY ? &n->YMax4 : &n->YMin4 ), rdy4 ), ry4 );
					v128 tz2 = Sse.sub_ps( Sse.mul_ps( Sse.load_ps( posZ ? &n->ZMax4 : &n->ZMin4 ), rdz4 ), rz4 );
					v128 tmin = Sse.max_ps( Sse.max_ps( Sse.max_ps( zero4, tx1 ), ty1 ), tz1 );
					v128 tmax = Sse.min_ps( Sse.min_ps( Sse.min_ps( tx2, t4 ), ty2 ), tz2 );
					v128 mask4 = Sse.cmple_ps( tmin, tmax );
					uint mask = ( uint )Sse.movemask_ps( mask4 );
					int validNodes = math.countbits( mask );
					if ( validNodes == 1 )
					{
						nodeIdx = ( ( uint* )&n->Child4 )[ Bfind( mask ) ];
					}
					else if ( validNodes != 0 )
					{
						// sse4.2 path, 3 extra ops to emulate _mm_permutevar_ps via _mm_shuffle_epi8
						v128 raw4 = Sse2.and_si128( Sse2.srl_epi32( Sse2.load_si128( &n->Perm4 ), shift4 ), shftmsk4 );
						v128 shfl16 = Sse2.add_epi32( Sse4_1.mullo_epi32( raw4, mul4 ), add4 );
						uint m = ( uint )Sse.movemask_ps( Ssse3.shuffle_epi8( mask4, shfl16 ) );
						tmin = Ssse3.shuffle_epi8( tmin, shfl16 );
						v128 c4 = Ssse3.shuffle_epi8( Sse2.load_si128( &n->Child4 ), shfl16 );
						v128 cpi = IdxLut4( m );
						v128 dist4 = Ssse3.shuffle_epi8( tmin, cpi );
						v128 child4 = Ssse3.shuffle_epi8( c4, cpi );
						Sse2.storeu_si128( nodeStack + stackPtr, child4 );
						Sse.storeu_ps( distStack + stackPtr, dist4 );
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
				v128 e1x4 = Sse.load_ps( &leaf->E1x4 ), e1y4 = Sse.load_ps( &leaf->E1y4 ), e1z4 = Sse.load_ps( &leaf->E1z4 );
				v128 e2x4 = Sse.load_ps( &leaf->E2x4 ), e2y4 = Sse.load_ps( &leaf->E2y4 ), e2z4 = Sse.load_ps( &leaf->E2z4 );
				v128 hx4 = Sse.sub_ps( Sse.mul_ps( dy4, e2z4 ), Sse.mul_ps( dz4, e2y4 ) );
				v128 hy4 = Sse.sub_ps( Sse.mul_ps( dz4, e2x4 ), Sse.mul_ps( dx4, e2z4 ) );
				v128 hz4 = Sse.sub_ps( Sse.mul_ps( dx4, e2y4 ), Sse.mul_ps( dy4, e2x4 ) );
				v128 sx4 = Sse.sub_ps( ox4, Sse.load_ps( &leaf->V0x4 ) );
				v128 sy4 = Sse.sub_ps( oy4, Sse.load_ps( &leaf->V0y4 ) );
				v128 sz4 = Sse.sub_ps( oz4, Sse.load_ps( &leaf->V0z4 ) );
				v128 det4 = Sse.add_ps( Sse.mul_ps( e1z4, hz4 ), Sse.add_ps( Sse.mul_ps( e1x4, hx4 ), Sse.mul_ps( e1y4, hy4 ) ) );
				v128 qz4 = Sse.sub_ps( Sse.mul_ps( sx4, e1y4 ), Sse.mul_ps( sy4, e1x4 ) );
				v128 qx4 = Sse.sub_ps( Sse.mul_ps( sy4, e1z4 ), Sse.mul_ps( sz4, e1y4 ) );
				v128 qy4 = Sse.sub_ps( Sse.mul_ps( sz4, e1x4 ), Sse.mul_ps( sx4, e1z4 ) );
				v128 invDet4 = FastRcp4( det4 );
				v128 u4 = Sse.mul_ps( Sse.add_ps( Sse.mul_ps( sz4, hz4 ), Sse.add_ps( Sse.mul_ps( sx4, hx4 ), Sse.mul_ps( sy4, hy4 ) ) ), invDet4 );
				v128 v4 = Sse.mul_ps( Sse.add_ps( Sse.mul_ps( dz4, qz4 ), Sse.add_ps( Sse.mul_ps( dx4, qx4 ), Sse.mul_ps( dy4, qy4 ) ) ), invDet4 );
				v128 ta4 = Sse.mul_ps( Sse.add_ps( Sse.mul_ps( e2z4, qz4 ), Sse.add_ps( Sse.mul_ps( e2x4, qx4 ), Sse.mul_ps( e2y4, qy4 ) ) ), invDet4 );
				v128 mask1 = Sse.and_ps( Sse.cmpge_ps( u4, zero4 ), Sse.cmpge_ps( v4, zero4 ) );
				v128 mask2 = Sse.cmple_ps( Sse.add_ps( u4, v4 ), one4 );
				v128 mask3 = Sse.and_ps( Sse.cmplt_ps( ta4, t4 ), Sse.cmpgt_ps( ta4, zero4 ) );
				v128 combined = Sse.and_ps( Sse.and_ps( mask1, mask2 ), mask3 );
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
					v128 dist4 = Sse4_1.blendv_ps( inf4, ta4, combined );
					// compute broadcasted horizontal minimum of dist4
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
					t4 = Sse.set1_ps( t );
					// compress stack
					int outStackPtr = 0;
					for ( int i = 0; i < stackPtr; i += 4 )
					{
						v128 node4 = Sse2.loadu_si128( nodeStack + i );
						v128 d4 = Sse.loadu_ps( distStack + i );
						uint keep = ( uint )Sse.movemask_ps( Sse.cmple_ps( d4, t4 ) );
						v128 shfl16 = IdxLut4( keep );
						v128 dst4 = Ssse3.shuffle_epi8( d4, shfl16 );
						node4 = Ssse3.shuffle_epi8( node4, shfl16 );
						Sse2.storeu_si128( distStack + outStackPtr, dst4 );
						Sse2.storeu_si128( nodeStack + outStackPtr, node4 );
						int numItems = math.min( 4, stackPtr - i );
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

		/// <summary>Port of the templated BVH4_CPU::IsOccluded body, SSE path.</summary>
		private bool IsOccludedSimd( in Ray ray, bool posX, bool posY, bool posZ )
		{
			uint* nodeStack = stackalloc uint[ StackSize ];
			int stackPtr = 0;
			uint nodeIdx = 0;
			v128 t4 = Sse.set1_ps( ray.Hit.T );
			v128 rx4 = Sse.set1_ps( ray.O.x * ray.RD.x ), rdx4 = Sse.set1_ps( ray.RD.x );
			v128 ry4 = Sse.set1_ps( ray.O.y * ray.RD.y ), rdy4 = Sse.set1_ps( ray.RD.y );
			v128 rz4 = Sse.set1_ps( ray.O.z * ray.RD.z ), rdz4 = Sse.set1_ps( ray.RD.z );
			v128 ox4 = Sse.set1_ps( ray.O.x ), oy4 = Sse.set1_ps( ray.O.y ), oz4 = Sse.set1_ps( ray.O.z );
			v128 dx4 = Sse.set1_ps( ray.D.x ), dy4 = Sse.set1_ps( ray.D.y ), dz4 = Sse.set1_ps( ray.D.z );
			v128 one4 = Sse.set1_ps( 1f ), zero4 = Sse.setzero_ps();
			while ( true )
			{
				while ( ( nodeIdx & LeafBit ) == 0 )
				{
					Bvh4CpuNode* n = ( Bvh4CpuNode* )( Data + ( nodeIdx * BlockSize ) );
					v128 tx1 = Sse.sub_ps( Sse.mul_ps( Sse.load_ps( posX ? &n->XMin4 : &n->XMax4 ), rdx4 ), rx4 );
					v128 ty1 = Sse.sub_ps( Sse.mul_ps( Sse.load_ps( posY ? &n->YMin4 : &n->YMax4 ), rdy4 ), ry4 );
					v128 tz1 = Sse.sub_ps( Sse.mul_ps( Sse.load_ps( posZ ? &n->ZMin4 : &n->ZMax4 ), rdz4 ), rz4 );
					v128 tx2 = Sse.sub_ps( Sse.mul_ps( Sse.load_ps( posX ? &n->XMax4 : &n->XMin4 ), rdx4 ), rx4 );
					v128 ty2 = Sse.sub_ps( Sse.mul_ps( Sse.load_ps( posY ? &n->YMax4 : &n->YMin4 ), rdy4 ), ry4 );
					v128 tz2 = Sse.sub_ps( Sse.mul_ps( Sse.load_ps( posZ ? &n->ZMax4 : &n->ZMin4 ), rdz4 ), rz4 );
					v128 tmin = Sse.max_ps( Sse.max_ps( Sse.max_ps( Sse.setzero_ps(), tx1 ), ty1 ), tz1 );
					v128 tmax = Sse.min_ps( Sse.min_ps( Sse.min_ps( tx2, t4 ), ty2 ), tz2 );
					v128 mask4 = Sse.cmple_ps( tmin, tmax );
					uint mask = ( uint )Sse.movemask_ps( mask4 );
					int validNodes = math.countbits( mask );
					if ( validNodes == 1 )
					{
						nodeIdx = ( ( uint* )&n->Child4 )[ Bfind( mask ) ];
					}
					else if ( validNodes != 0 )
					{
						v128 cpi = IdxLut4( mask );
						v128 child4 = Ssse3.shuffle_epi8( Sse2.load_si128( &n->Child4 ), cpi );
						Sse2.storeu_si128( nodeStack + stackPtr, child4 );
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
				v128 hx4 = Sse.sub_ps( Sse.mul_ps( dy4, e2z4 ), Sse.mul_ps( dz4, e2y4 ) );
				v128 hy4 = Sse.sub_ps( Sse.mul_ps( dz4, e2x4 ), Sse.mul_ps( dx4, e2z4 ) );
				v128 hz4 = Sse.sub_ps( Sse.mul_ps( dx4, e2y4 ), Sse.mul_ps( dy4, e2x4 ) );
				v128 sx4 = Sse.sub_ps( ox4, Sse.load_ps( &leaf->V0x4 ) );
				v128 sy4 = Sse.sub_ps( oy4, Sse.load_ps( &leaf->V0y4 ) );
				v128 sz4 = Sse.sub_ps( oz4, Sse.load_ps( &leaf->V0z4 ) );
				v128 det4 = Sse.add_ps( Sse.mul_ps( e1z4, hz4 ), Sse.add_ps( Sse.mul_ps( e1x4, hx4 ), Sse.mul_ps( e1y4, hy4 ) ) );
				v128 qz4 = Sse.sub_ps( Sse.mul_ps( sx4, e1y4 ), Sse.mul_ps( sy4, e1x4 ) );
				v128 qx4 = Sse.sub_ps( Sse.mul_ps( sy4, e1z4 ), Sse.mul_ps( sz4, e1y4 ) );
				v128 qy4 = Sse.sub_ps( Sse.mul_ps( sz4, e1x4 ), Sse.mul_ps( sx4, e1z4 ) );
				v128 invDet4 = FastRcp4( det4 );
				v128 u4 = Sse.mul_ps( Sse.add_ps( Sse.mul_ps( sz4, hz4 ), Sse.add_ps( Sse.mul_ps( sx4, hx4 ), Sse.mul_ps( sy4, hy4 ) ) ), invDet4 );
				v128 v4 = Sse.mul_ps( Sse.add_ps( Sse.mul_ps( dz4, qz4 ), Sse.add_ps( Sse.mul_ps( dx4, qx4 ), Sse.mul_ps( dy4, qy4 ) ) ), invDet4 );
				v128 ta4 = Sse.mul_ps( Sse.add_ps( Sse.mul_ps( e2z4, qz4 ), Sse.add_ps( Sse.mul_ps( e2x4, qx4 ), Sse.mul_ps( e2y4, qy4 ) ) ), invDet4 );
				v128 mask1 = Sse.and_ps( Sse.cmpge_ps( u4, zero4 ), Sse.cmpge_ps( v4, zero4 ) );
				v128 mask2 = Sse.cmple_ps( Sse.add_ps( u4, v4 ), one4 );
				v128 mask3 = Sse.and_ps( Sse.cmplt_ps( ta4, t4 ), Sse.cmpgt_ps( ta4, zero4 ) );
				v128 combined = Sse.and_ps( Sse.and_ps( mask1, mask2 ), mask3 );
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
				// we continue.
				if ( stackPtr == 0 )
				{
					return false;
				}
				nodeIdx = nodeStack[ --stackPtr ];
			}
		}

		/// <summary>
		/// Scalar fallback for <see cref="IntersectSimd"/>: the same traversal as a four-lane loop.
		/// Children are visited in the order the node's perm4 field prescribes for the ray's octant,
		/// pushed farthest first so the nearest is popped first; equal minimum triangle distances
		/// within one leaf resolve to the highest lane, matching the C++ __bfind tie-break.
		/// </summary>
		private int IntersectScalar( ref Ray ray, bool posX, bool posY, bool posZ, int signShift )
		{
			uint* nodeStack = stackalloc uint[ StackSize ];
			float* distStack = stackalloc float[ StackSize ];
			float* tmin = stackalloc float[ 4 ];
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
					Bvh4CpuNode* n = ( Bvh4CpuNode* )( Data + ( nodeIdx * BlockSize ) );
					float* xmin = ( float* )( posX ? &n->XMin4 : &n->XMax4 );
					float* xmax = ( float* )( posX ? &n->XMax4 : &n->XMin4 );
					float* ymin = ( float* )( posY ? &n->YMin4 : &n->YMax4 );
					float* ymax = ( float* )( posY ? &n->YMax4 : &n->YMin4 );
					float* zmin = ( float* )( posZ ? &n->ZMin4 : &n->ZMax4 );
					float* zmax = ( float* )( posZ ? &n->ZMax4 : &n->ZMin4 );
					uint* child = ( uint* )&n->Child4;
					uint mask = 0;
					for ( int lane = 0; lane < 4; lane++ )
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
						// The SIMD path permutes all four lanes and then compacts the surviving
						// ones with the shuffle LUT; this is the same thing, done in place.
						uint* perm = ( uint* )&n->Perm4;
						int outPtr = 0;
						for ( int i = 0; i < 4; i++ )
						{
							int lane = ( int )( ( perm[ i ] >> signShift ) & 3 );
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
					// Deviation: the SIMD path uses fastrcp4, a reciprocal plus one Newton step.
					float invDet = 1f / det;
					float lu = ( ( sz * hz ) + ( ( sx * hx ) + ( sy * hy ) ) ) * invDet;
					float lv = ( ( ray.D.z * qz ) + ( ( ray.D.x * qx ) + ( ray.D.y * qy ) ) ) * invDet;
					float lt = ( ( e2z * qz ) + ( ( e2x * qx ) + ( e2y * qy ) ) ) * invDet;
					u[ lane ] = lu;
					v[ lane ] = lv;
					bool hit = lu >= 0f && lv >= 0f && ( lu + lv ) <= 1f && lt < t && lt > 0f;
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
		/// Scalar fallback for <see cref="IsOccludedSimd"/>. Like the C++ SSE version, this one does
		/// not consult perm4: the surviving children are pushed in lane order.
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
					Bvh4CpuNode* n = ( Bvh4CpuNode* )( Data + ( nodeIdx * BlockSize ) );
					float* xmin = ( float* )( posX ? &n->XMin4 : &n->XMax4 );
					float* xmax = ( float* )( posX ? &n->XMax4 : &n->XMin4 );
					float* ymin = ( float* )( posY ? &n->YMin4 : &n->YMax4 );
					float* ymax = ( float* )( posY ? &n->YMax4 : &n->YMin4 );
					float* zmin = ( float* )( posZ ? &n->ZMin4 : &n->ZMax4 );
					float* zmax = ( float* )( posZ ? &n->ZMax4 : &n->ZMin4 );
					uint* child = ( uint* )&n->Child4;
					uint mask = 0;
					for ( int lane = 0; lane < 4; lane++ )
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
						for ( int lane = 0; lane < 4; lane++ )
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
				// we continue.
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
		/// Port of the idxLUT4_ table: a _mm_shuffle_epi8 control mask that compacts the lanes
		/// selected by the four-bit mask into the low dwords, keeping their order. Unselected
		/// dwords are zero, exactly as in the C++ table; they address source byte 0 rather than
		/// zeroing, but the traversal never reads them.
		/// </summary>
		private static v128 IdxLut4( uint m )
		{
			const int a = 0x03020100, b = 0x07060504, c = 0x0B0A0908, d = 0x0F0E0D0C;
			switch ( m )
			{
				case 0: return new v128( 0, 0, 0, 0 );
				case 1: return new v128( a, 0, 0, 0 );
				case 2: return new v128( b, 0, 0, 0 );
				case 3: return new v128( a, b, 0, 0 );
				case 4: return new v128( c, 0, 0, 0 );
				case 5: return new v128( a, c, 0, 0 );
				case 6: return new v128( b, c, 0, 0 );
				case 7: return new v128( a, b, c, 0 );
				case 8: return new v128( d, 0, 0, 0 );
				case 9: return new v128( a, d, 0, 0 );
				case 10: return new v128( b, d, 0, 0 );
				case 11: return new v128( a, b, d, 0 );
				case 12: return new v128( c, d, 0, 0 );
				case 13: return new v128( a, c, d, 0 );
				case 14: return new v128( b, c, d, 0 );
				default: return new v128( a, b, c, d );
			}
		}
	}
}
