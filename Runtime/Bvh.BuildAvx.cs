using System;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Burst.Intrinsics.X86;

namespace TinyBVH
{
	public unsafe partial struct Bvh
	{
		/// <summary>
		/// True when the AVX binned builder can run on this machine, i.e. when Burst compiled the
		/// intrinsics of <see cref="BvhAvxBuilder"/> for a target that has AVX and SSE4.1. Always
		/// false under the Mono fallback, where the intrinsics would be emulated in double precision
		/// and the resulting tree would not match the C++.
		/// </summary>
		public static bool AvxBuilderSupported
		{
			get
			{
				BvhAvxBuilder.QueryAvxSupport( out int supported );
				return supported != 0;
			}
		}

		/// <summary>
		/// Port of BVH::BuildAVX( const bvhvec4*, const uint32_t ): the fast AVX binned-SAH builder.
		/// It produces a different tree from the scalar binned builder in Bvh.Build.cs, because the
		/// binning arithmetic differs (see BvhAvxBuilder). Triangle soup: three consecutive 16-byte
		/// vertices per primitive.
		/// </summary>
		public void BuildAvx( float4* vertices, uint primCount )
		{
			BuildAvx( ( byte* )vertices, primCount * 3, 16, null, primCount );
		}

		/// <summary>General form: a strided vertex buffer, optionally addressed through indices.</summary>
		public void BuildAvx( byte* vertices, uint vertexCount, int vertexStride, uint* indices, uint primCount )
		{
			ValidateBuildInput( vertices, vertexCount, vertexStride, indices, primCount );
			RunAvxBuild( vertices, vertexCount, vertexStride, indices, primCount );
			if ( PostOptimize )
			{
				Optimize( OptimizeIterations );
			}
		}

		/// <summary>
		/// PrepareAVXBuild + BuildAVXSubtree + BuildAVXFinalize, the body of BVH::BuildAVX. Shared
		/// with the UseSimdIfAvailable branch of Build, which applies postOptimize itself.
		/// </summary>
		internal void RunAvxBuild( byte* vertices, uint vertexCount, int vertexStride, uint* indices, uint primCount )
		{
			if ( ( vertexStride & 15 ) != 0 )
			{
				throw new ArgumentException( "Bvh.BuildAvx( .. ), stride must be multiple of 16.", nameof( vertexStride ) );
			}
			if ( !AvxBuilderSupported )
			{
				throw new InvalidOperationException( "Bvh.BuildAvx( .. ) requires AVX." );
			}
			BvhAvxBuilder.PrepareAvxBuild( ref this, vertices, vertexCount, vertexStride, indices, primCount );
			BvhAvxBuilder.BuildAvxSubtree( ref this );
			BvhAvxBuilder.BuildAvxFinalize( ref this );
		}

		/// <summary>BuildAvx over a NativeArray; the array must outlive the BVH.</summary>
		public void BuildAvx( NativeArray<float4> vertices, uint primCount )
		{
			BuildAvx( ( float4* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( vertices ), primCount );
		}

		/// <summary>BuildAvx over indexed geometry; both arrays must outlive the BVH.</summary>
		public void BuildAvx( NativeArray<float4> vertices, NativeArray<uint> indices, uint primCount )
		{
			BuildAvx(
				( byte* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( vertices ), ( uint )vertices.Length, 16,
				( uint* )NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr( indices ), primCount );
		}
	}

	/// <summary>
	/// Burst-compiled port of tinybvh's fast AVX binned-SAH builder: PrepareAVXBuild,
	/// BuildAVXSubtree and BuildAVXFinalize, all from the BVH_USEAVX block of tiny_bvh.h.
	///
	/// This is a second builder, not an optimisation of the one in Bvh.Build.cs: the AVX path bins
	/// with a different expression (a round-to-nearest convert of centroid*2 - 0.5 instead of a
	/// truncating convert of the centroid, and a bin scale of AVXBINS * 0.49999 over twice the node
	/// extent), uses 1e-7f rather than 1e-20f for the degenerate-axis epsilon, and evaluates the
	/// seven split planes in a fixed order with strict less-than, so it breaks ties differently.
	/// The trees the two produce therefore differ, which is why this has its own reference data.
	///
	/// Only the serial path is ported. The C++ runs threaded when ENABLE_THREADED_BUILDS is defined
	/// and triCount >= MT_BUILD_THRESHOLD and the context has a spawn and a barrier hook; it then
	/// bins in slices and spawns subtrees. Tools/RefDump/simddump.cpp is built with
	/// NO_THREADED_BUILDS and clears the context hooks, so the reference is the serial tree.
	///
	/// Every _mm256_* / _mm_* operation is mirrored one to one and in the same order, so the
	/// floating point results are bit-identical. The MSVC variant of halfArea is used, because the
	/// reference tool is built with cl.exe; the g++ variant computes the same value in the same
	/// order anyway.
	/// </summary>
	[BurstCompile]
	internal static unsafe class BvhAvxBuilder
	{
		/// <summary>tinybvh's AVXBINS; the split evaluation below is hardcoded for 8.</summary>
		private const int AvxBins = 8;

		/// <summary>Reports whether Burst compiled the intrinsics below for this target.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void QueryAvxSupport( out int supported )
		{
			supported = Avx.IsAvxSupported && Sse4_1.IsSse41Supported ? 1 : 0;
		}

		/// <summary>BVH::signFlip8: negates the min half of a sign-flipped 8-lane AABB.</summary>
		private static v256 SignFlip8
		{
			get { return Avx.mm256_setr_ps( -0.0f, -0.0f, -0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f ); }
		}

		/// <summary>BVH::max8: the empty sign-flipped AABB the bins start out at.</summary>
		private static v256 Max8
		{
			get { return Avx.mm256_set1_ps( -BvhConstants.Far ); }
		}

		/// <summary>
		/// Port of BVH::PrepareAVXBuildFragSlice for the whole range, i.e. the serial case.
		///
		/// Deviation: the C++ loads the three vertices with _mm_load_ps and reduces them with
		/// _mm_min_ps / _mm_max_ps; math.min / math.max over a float4 are the same instructions on
		/// the same operands, so this needs no AVX guard and stays correct under Mono. Note that
		/// both sides store all sixteen bytes of the min and the max into the fragment, so the w
		/// components of the vertices land in Fragment.PrimIdx and Fragment.Clipped. That is what
		/// the C++ does, and the binning below reads those lanes back; they only ever end up in the
		/// two integer fields of a node, which are overwritten right after.
		/// </summary>
		private static void PrepareAvxBuildFragSlice( ref Bvh bvh, uint first, uint last, uint* indices, float4* rootMin, float4* rootMax )
		{
			Fragment* frag4 = bvh.Fragments;
			uint* primIdx = bvh.PrimIdx;
			float4 rmin = new float4( BvhConstants.Far ), rmax = new float4( -BvhConstants.Far );
			if ( indices != null )
			{
				for ( uint i = first; i < last; i++ )
				{
					uint i0 = indices[ i * 3 ], i1 = indices[ ( i * 3 ) + 1 ], i2 = indices[ ( i * 3 ) + 2 ];
					float4 v0 = bvh.Vertex( i0 ), v1 = bvh.Vertex( i1 ), v2 = bvh.Vertex( i2 );
					float4 t1 = math.min( math.min( v0, v1 ), v2 ), t2 = math.max( math.max( v0, v1 ), v2 );
					( ( float4* )( frag4 + i ) )[ 0 ] = t1;
					( ( float4* )( frag4 + i ) )[ 1 ] = t2;
					rmin = math.min( rmin, t1 );
					rmax = math.max( rmax, t2 );
					primIdx[ i ] = i;
				}
			}
			else
			{
				for ( uint i = first; i < last; i++ )
				{
					float4 v0 = bvh.Vertex( i * 3 ), v1 = bvh.Vertex( ( i * 3 ) + 1 ), v2 = bvh.Vertex( ( i * 3 ) + 2 );
					float4 t1 = math.min( math.min( v0, v1 ), v2 ), t2 = math.max( math.max( v0, v1 ), v2 );
					( ( float4* )( frag4 + i ) )[ 0 ] = t1;
					( ( float4* )( frag4 + i ) )[ 1 ] = t2;
					rmin = math.min( rmin, t1 );
					rmax = math.max( rmax, t2 );
					primIdx[ i ] = i;
				}
			}
			*rootMin = rmin;
			*rootMax = rmax;
		}

		/// <summary>
		/// Port of BVH::PrepareAVXBuild: allocate memory and prepare the fragment list. Mirrors
		/// BvhBuilder.PrepareBuild except for the fragment bounds, which the C++ computes over the
		/// raw 16-byte vertices rather than over their xyz.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void PrepareAvxBuild( ref Bvh bvh, byte* vertices, uint vertexCount, int vertexStride, uint* indices, uint prims )
		{
			// reset node pool
			uint primCount = prims > 0 ? prims : vertexCount / 3;
			uint splitBudget = bvh.UsePresplitting ? ( uint )( int )( primCount * bvh.PresplitFactor ) : 0;
			uint spaceNeeded = ( primCount + splitBudget ) * 2; // upper limit
			bvh.AllocateNodes( spaceNeeded );
			bvh.AllocatePrimIdx( primCount + splitBudget );
			bvh.AllocateFragments( primCount + splitBudget );
			bvh.Nodes[ 1 ] = default; // avoid crash in refit.
			bvh.TriCount = primCount;
			bvh.Verts = vertices; // note: we're not copying this data; don't delete.
			bvh.VertCount = vertexCount;
			bvh.VertStride = vertexStride;
			bvh.VertIdx = indices;
			// prepare threading: the serial path is the only one ported; see the class comment.
			bvh.ThreadedSubtrees = 0;
			// initialize fragments
			float4 rootMin = new float4( BvhConstants.Far ), rootMax = new float4( -BvhConstants.Far );
			PrepareAvxBuildFragSlice( ref bvh, 0, bvh.TriCount, indices, &rootMin, &rootMax );
			BvhNode* root = bvh.Nodes;
			root->AabbMin = rootMin.xyz;
			root->AabbMax = rootMax.xyz;
			// presplitting
			uint fragCount = primCount;
			if ( bvh.UsePresplitting )
			{
				Fragment* fragment = bvh.Fragments;
				for ( uint i = 0; i < primCount; i++ )
				{
					fragment[ i ].PrimIdx = i;
					fragment[ i ].Clipped = 0;
				}
				fragCount = BvhPresplitter.Presplit( ref bvh );
			}
			// finalize root node
			root->LeftFirst = 0;
			root->TriCount = fragCount;
			bvh.IdxCount = fragCount;
			bvh.TriCount = fragCount;
			// reset node pool. The port keeps the C++ newNodePtr member in Bvh.UsedNodes, as
			// BvhBuilder.PrepareBuild does; BuildAvxFinalize leaves it as the subdivision left it.
			bvh.UsedNodes = 2;
			bvh.BvhOverIndices = indices != null;
			// all set; actual build happens in BuildAvxSubtree.
		}

		/// <summary>
		/// Port of BVH::BuildAVXBinTask: bin the fragments of one node into the three axes at once.
		/// The bin boxes are kept sign-flipped (min negated) so both halves accumulate with a single
		/// _mm256_max_ps. The loop is software-pipelined exactly as the C++ is, so that the bin index
		/// of fragment k+1 is computed while fragment k is merged.
		/// </summary>
		private static void BuildAvxBinTask( Fragment* fragment, uint* primIdx, uint first, uint last,
			v256* binbox, uint* count, v128 nmin4, v128 rpd4 )
		{
			v128 half4 = Sse.set1_ps( 0.5f );
			v256 signFlip8 = SignFlip8;
			uint fi = primIdx[ first ];
			// memset( count, 0, 3 * AVXBINS * 4 ) plus the copy of the all-empty bin template.
			v256 max8 = Max8;
			for ( int i = 0; i < 3 * AvxBins; i++ )
			{
				count[ i ] = 0;
				binbox[ i ] = max8;
			}
			v256 r0, r1, r2, f = Avx.mm256_xor_ps( Avx.mm256_loadu_ps( ( float* )( fragment + fi ) ), signFlip8 );
			v128 zero4i = Sse2.setzero_si128();
			v128 bc4 = Sse4_1.max_epi32( Sse2.cvtps_epi32( Sse.sub_ps( Sse.mul_ps( Sse.sub_ps( Sse.add_ps(
				Sse.loadu_ps( ( float* )( fragment + fi ) + 4 ), Sse.loadu_ps( ( float* )( fragment + fi ) ) ), nmin4 ), rpd4 ), half4 ) ), zero4i );
			uint i0 = bc4.UInt0, i1 = bc4.UInt1, i2 = bc4.UInt2;
			uint* ti = primIdx + first + 1;
			for ( uint i = first; i < last - 1; i++ )
			{
				uint fid = *ti++;
				v256 b0 = binbox[ i0 ], b1 = binbox[ AvxBins + i1 ], b2 = binbox[ ( 2 * AvxBins ) + i2 ];
				v128 frmin = Sse.loadu_ps( ( float* )( fragment + fid ) ), frmax = Sse.loadu_ps( ( float* )( fragment + fid ) + 4 );
				r0 = Avx.mm256_max_ps( b0, f );
				r1 = Avx.mm256_max_ps( b1, f );
				r2 = Avx.mm256_max_ps( b2, f );
				bc4 = Sse4_1.max_epi32( Sse2.cvtps_epi32( Sse.sub_ps( Sse.mul_ps( Sse.sub_ps( Sse.add_ps( frmax, frmin ), nmin4 ), rpd4 ), half4 ) ), zero4i );
				f = Avx.mm256_xor_ps( Avx.mm256_loadu_ps( ( float* )( fragment + fid ) ), signFlip8 );
				count[ i0 ]++;
				count[ AvxBins + i1 ]++;
				count[ ( AvxBins * 2 ) + i2 ]++;
				binbox[ i0 ] = r0;
				i0 = bc4.UInt0;
				binbox[ AvxBins + i1 ] = r1;
				i1 = bc4.UInt1;
				binbox[ ( 2 * AvxBins ) + i2 ] = r2;
				i2 = bc4.UInt2;
			}
			// final business for final fragment
			{
				v256 b0 = binbox[ i0 ], b1 = binbox[ AvxBins + i1 ], b2 = binbox[ ( 2 * AvxBins ) + i2 ];
				count[ i0 ]++;
				count[ AvxBins + i1 ]++;
				count[ ( AvxBins * 2 ) + i2 ]++;
				r0 = Avx.mm256_max_ps( b0, f );
				r1 = Avx.mm256_max_ps( b1, f );
				r2 = Avx.mm256_max_ps( b2, f );
				binbox[ i0 ] = r0;
				binbox[ AvxBins + i1 ] = r1;
				binbox[ ( 2 * AvxBins ) + i2 ] = r2;
			}
		}

		/// <summary>
		/// Port of BVH::BuildAVXSubtree( 0, 0 ), serial. The C++ only ever changes 'depth' when it
		/// spawns a subtree, so without threaded builds the whole tree is built at depth 0 and the
		/// task stack needs no depth companion.
		/// </summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void BuildAvxSubtree( ref Bvh bvh )
		{
			if ( Avx.IsAvxSupported && Sse4_1.IsSse41Supported )
			{
				// static variable declarations
				v128 binmul3 = Sse.set1_ps( AvxBins * 0.49999f );
				v128 min1 = Sse.set1_ps( -1f );
				v128 mask3 = Sse.cmpeq_ps( Sse.setr_ps( 0, 0, 0, 1 ), Sse.setzero_ps() );
				v256 signFlip8 = SignFlip8;
				// aligned data
				v256* binbox = stackalloc v256[ 3 * AvxBins ];
				uint* count = stackalloc uint[ 3 * AvxBins ];
				// the C++ leaves bestLBox / bestRBox uninitialized and suppresses the warning: a
				// node that finds no split plane always breaks out before they are read.
				v256 bestLBox = Max8, bestRBox = Max8;
				// subdivide recursively
				uint* task = stackalloc uint[ 512 ];
				uint taskCount = 0;
				BvhNode* bvhNode = bvh.Nodes;
				Fragment* fragment = bvh.Fragments;
				uint* primIdx = bvh.PrimIdx;
				uint newNodePtr = bvh.UsedNodes;
				uint nodeIdx = 0;
				BvhNode* root = bvhNode;
				float3 minDim = ( root->AabbMax - root->AabbMin ) * 1e-7f;
				while ( true )
				{
					while ( true )
					{
						BvhNode* node = bvhNode + nodeIdx;
						float* node4 = ( float* )node;
						// find optimal object split
						v128 d4 = Sse4_1.blendv_ps( min1, Sse.sub_ps( Sse.loadu_ps( node4 + 4 ), Sse.loadu_ps( node4 ) ), mask3 );
						v128 nmin4 = Sse.add_ps( Sse.loadu_ps( node4 ), Sse.loadu_ps( node4 ) );
						v128 rpd4 = Sse.and_ps( Sse.div_ps( binmul3, d4 ), Sse.cmpneq_ps( d4, Sse.setzero_ps() ) );
						// implementation of Section 4.1 of "Parallel Spatial Splits in Bounding Volume Hierarchies":
						// main loop operates on two fragments to minimize dependencies and maximize ILP.
						BuildAvxBinTask( fragment, primIdx, node->LeftFirst, node->LeftFirst + node->TriCount, binbox, count, nmin4, rpd4 );
						// calculate per-split totals
						float splitCost = BvhConstants.Far;
						float rSAV = 1f / node->SurfaceArea;
						uint bestAxis = 0, bestPos = 0;
						v256* bb = binbox;
						for ( int a = 0; a < 3; a++, bb += AvxBins )
						{
							if ( ( node->AabbMax[ a ] - node->AabbMin[ a ] ) > minDim[ a ] )
							{
								// hardcoded bin processing for AVXBINS == 8
								uint* cnt = count + ( a * AvxBins );
								uint lN0 = cnt[ 0 ], rN0 = cnt[ 7 ];
								v256 lb0 = bb[ 0 ], rb0 = bb[ 7 ];
								uint lN1 = lN0 + cnt[ 1 ], rN1 = rN0 + cnt[ 6 ], lN2 = lN1 + cnt[ 2 ];
								uint rN2 = rN1 + cnt[ 5 ], lN3 = lN2 + cnt[ 3 ], rN3 = rN2 + cnt[ 4 ];
								v256 lb1 = Avx.mm256_max_ps( lb0, bb[ 1 ] ), rb1 = Avx.mm256_max_ps( rb0, bb[ 6 ] );
								v256 lb2 = Avx.mm256_max_ps( lb1, bb[ 2 ] ), rb2 = Avx.mm256_max_ps( rb1, bb[ 5 ] );
								v256 lb3 = Avx.mm256_max_ps( lb2, bb[ 3 ] ), rb3 = Avx.mm256_max_ps( rb2, bb[ 4 ] );
								uint lN4 = lN3 + cnt[ 4 ], rN4 = rN3 + cnt[ 3 ], lN5 = lN4 + cnt[ 5 ];
								uint rN5 = rN4 + cnt[ 2 ], lN6 = lN5 + cnt[ 6 ], rN6 = rN5 + cnt[ 1 ];
								v256 lb4 = Avx.mm256_max_ps( lb3, bb[ 4 ] ), rb4 = Avx.mm256_max_ps( rb3, bb[ 3 ] );
								v256 lb5 = Avx.mm256_max_ps( lb4, bb[ 5 ] ), rb5 = Avx.mm256_max_ps( rb4, bb[ 2 ] );
								v256 lb6 = Avx.mm256_max_ps( lb5, bb[ 6 ] ), rb6 = Avx.mm256_max_ps( rb5, bb[ 1 ] );
								ProcessPlane( a, 3, lN3, rN3, lb3, rb3, ref splitCost, ref bestAxis, ref bestPos, ref bestLBox, ref bestRBox ); // most likely split
								ProcessPlane( a, 2, lN2, rN4, lb2, rb4, ref splitCost, ref bestAxis, ref bestPos, ref bestLBox, ref bestRBox );
								ProcessPlane( a, 4, lN4, rN2, lb4, rb2, ref splitCost, ref bestAxis, ref bestPos, ref bestLBox, ref bestRBox );
								ProcessPlane( a, 5, lN5, rN1, lb5, rb1, ref splitCost, ref bestAxis, ref bestPos, ref bestLBox, ref bestRBox );
								ProcessPlane( a, 1, lN1, rN5, lb1, rb5, ref splitCost, ref bestAxis, ref bestPos, ref bestLBox, ref bestRBox );
								ProcessPlane( a, 0, lN0, rN6, lb0, rb6, ref splitCost, ref bestAxis, ref bestPos, ref bestLBox, ref bestRBox );
								ProcessPlane( a, 6, lN6, rN0, lb6, rb0, ref splitCost, ref bestAxis, ref bestPos, ref bestLBox, ref bestRBox ); // least likely split
							}
						}
						splitCost = bvh.TraversalCost + ( ( bvh.IntersectionCost * rSAV ) * splitCost );
						float noSplitCost = ( float )node->TriCount * bvh.IntersectionCost;
						if ( splitCost >= noSplitCost )
						{
							break; // not splitting is better.
						}
						// in-place partition
						float rpd = ( ( float* )&rpd4 )[ bestAxis ], nmin = ( ( float* )&nmin4 )[ bestAxis ];
						uint i = node->LeftFirst, j = node->LeftFirst + node->TriCount, last = j, fr = primIdx[ i ];
						for ( uint k = 0; k < node->TriCount; k++ )
						{
							// The C++ casts the scalar bin coordinate to uint32_t; MSVC truncates it
							// towards zero into a 64-bit register and keeps the low half, which the
							// cast through long reproduces. The value is never negative here: the
							// node bounds are the exact min over the fragments it holds.
							uint bi = ( uint )( long )( ( ( fragment[ fr ].BMax[ ( int )bestAxis ] + fragment[ fr ].BMin[ ( int )bestAxis ] ) - nmin ) * rpd );
							if ( bi <= bestPos )
							{
								++i;
								// Deviation: the C++ reads primIdx[i] unconditionally, which runs one
								// past the range when every fragment went left. The value is dead in
								// that case - the loop is over and the node cannot be split - so the
								// read is skipped rather than reaching past the allocation.
								if ( i < last )
								{
									fr = primIdx[ i ];
								}
							}
							else
							{
								uint t = fr;
								--j;
								fr = primIdx[ j ];
								primIdx[ i ] = fr;
								primIdx[ j ] = t;
							}
						}
						// create child nodes and recurse
						uint n = newNodePtr;
						newNodePtr += 2;
						uint leftCount = i - node->LeftFirst, rightCount = node->TriCount - leftCount;
						if ( leftCount == 0 || rightCount == 0 || taskCount == 512 )
						{
							break; // should not happen.
						}
						Avx.mm256_storeu_ps( ( float* )( bvhNode + n ), Avx.mm256_xor_ps( bestLBox, signFlip8 ) );
						bvhNode[ n ].LeftFirst = node->LeftFirst;
						bvhNode[ n ].TriCount = leftCount;
						node->LeftFirst = n;
						node->TriCount = 0;
						Avx.mm256_storeu_ps( ( float* )( bvhNode + n + 1 ), Avx.mm256_xor_ps( bestRBox, signFlip8 ) );
						bvhNode[ n + 1 ].LeftFirst = i;
						bvhNode[ n + 1 ].TriCount = rightCount;
						task[ taskCount++ ] = n + 1;
						nodeIdx = n;
					}
					// fetch subdivision task from stack
					if ( taskCount == 0 )
					{
						break;
					}
					nodeIdx = task[ --taskCount ];
				}
				bvh.UsedNodes = newNodePtr;
			}
		}

		/// <summary>Port of BVH::BuildAVXFinalize, serial: the tree has been built.</summary>
		[BurstCompile( CompileSynchronously = true )]
		internal static void BuildAvxFinalize( ref Bvh bvh )
		{
			bvh.AabbMin = bvh.Nodes[ 0 ].AabbMin;
			bvh.AabbMax = bvh.Nodes[ 0 ].AabbMax;
			bvh.Refittable = !bvh.UsePresplitting; // only if not using spatial splits
			bvh.MayHaveHoles = false; // there are no holes in the list of nodes.
			// usedNodes = newNodePtr: BuildAvxSubtree already stored the node pointer there.
			// Deviation: BuildAVXFinalize does not touch bvh_over_aabbs, which leaves it stale when a
			// Bvh that held a TLAS is rebuilt; this build always has vertices, so it is cleared here.
			bvh.BvhOverAabbs = false;
			BvhPresplitter.FinishPresplit( ref bvh );
		}

		/// <summary>
		/// Port of the PROCESS_PLANE macro: evaluate one split plane and keep it when it is cheaper
		/// than the best so far. Strictly cheaper, so the plane order the caller uses decides ties.
		/// </summary>
		private static void ProcessPlane( int a, uint pos, uint lN, uint rN, v256 lb, v256 rb,
			ref float splitCost, ref uint bestAxis, ref uint bestPos, ref v256 bestLBox, ref v256 bestRBox )
		{
			if ( ( lN * rN ) != 0 )
			{
				float ANLR = ( HalfArea( lb ) * ( float )lN ) + ( HalfArea( rb ) * ( float )rN );
				if ( ANLR < splitCost )
				{
					splitCost = ANLR;
					bestAxis = ( uint )a;
					bestPos = pos;
					bestLBox = lb;
					bestRBox = rb;
				}
			}
		}

		/// <summary>
		/// Port of halfArea( const __m256&amp; ), the MSVC variant: 'a' holds the aabb with min.xyz
		/// negated, so folding the two 128-bit halves together yields the extent.
		/// </summary>
		private static float HalfArea( v256 a )
		{
			v128 q = Avx.mm256_castps256_ps128( Avx.mm256_add_ps( Avx.mm256_permute2f128_ps( a, a, 5 ), a ) );
			v128 v = Sse.mul_ps( q, Sse.shuffle_ps( q, q, 9 ) );
			return v.Float0 + v.Float1 + v.Float2;
		}
	}
}
