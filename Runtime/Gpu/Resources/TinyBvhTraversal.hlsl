#ifndef TINYBVH_TRAVERSAL_INCLUDED
#define TINYBVH_TRAVERSAL_INCLUDED

// HLSL ports of tinybvh's OpenCL traversal kernels (kernels/traverse_bvh2.cl,
// traverse_bvh4.cl, traverse_cwbvh.cl). Four layouts are supported:
//
//   Bvh2     - Wald 32-byte nodes, primitive indices and a shared vertex buffer.
//              This one is a port of the *CPU* code (Bvh.Intersect.cs) rather than
//              of traverse_bvh2.cl, so it reproduces the reference tie-breaking.
//   BvhGpu   - Aila & Laine 64-byte nodes; port of traverse_ailalaine.
//   Bvh4Gpu  - BVH4_GPU 16-byte block blob with inline triangles; port of traverse_gpu4way.
//   Cwbvh    - BVH8_CWBVH compressed wide nodes; port of traverse_cwbvh (SIMD_AABBTEST path).
//
// A fifth path, Tlas, traverses an Aila & Laine TLAS over BLAS instances and enters the
// BLASses through the BvhGpu or Cwbvh traversals above; port of traverse_tlas. Those two
// take base offsets into the shared buffers so several BLASses can be concatenated; the
// "At" suffix marks that variant and the offset-free ones just pass zeroes.
//
// Deviations from the .cl sources are marked with "Deviation:" comments.

#define TINYBVH_FAR 1e30f
#define TINYBVH_RCP_MAX 1e30f
// The C++ CPU code uses 1e-6 as the Moeller-Trumbore determinant epsilon while the
// OpenCL kernels use 1e-7. Deviation: 1e-6 everywhere, so all four layouts reject the
// same degenerate triangles as the CPU reference does.
#define TINYBVH_TRI_EPS 0.000001f
// Deviation: the .cl kernels use a 32-entry stack for every layout. A binary BVH gets
// deeper than a 4- or 8-wide one, so the two BVH2-shaped traversals get 64 entries.
#define TINYBVH_BVH2_STACK_SIZE 64
#define TINYBVH_BVH4_STACK_SIZE 32
#define TINYBVH_CWBVH_STACK_SIZE 32

// ---------------------------------------------------------------------------
// Geometry buffers
// ---------------------------------------------------------------------------

// Wald 32-byte nodes: two float4 per node, (aabbMin, asfloat(leftFirst)) and
// (aabbMax, asfloat(triCount)).
StructuredBuffer<float4> Bvh2Nodes;
// Aila-Laine 64-byte nodes: four float4 per node, lmin/lmax/rmin/rmax with
// left, right, triCount and firstTri in the w components.
StructuredBuffer<float4> BvhGpuNodes;
// Shared by Bvh2 and BvhGpu.
StructuredBuffer<uint> PrimIdx;
StructuredBuffer<float4> Verts;
// Vertex indices of a BVH built over indexed geometry, three per primitive; the port of
// BVH::vertIdx. Indexed is 1 when the uploaded BVH addresses Verts through them and 0 when a
// primitive's three vertices are Verts[prim * 3 .. prim * 3 + 2], which is what every
// non-indexed upload and every TLAS upload sets.
StructuredBuffer<uint> VertIdx;
uint Indexed;
// BVH4_GPU: node blocks and inline triangles in one blob of 16-byte blocks.
StructuredBuffer<float4> Bvh4Data;
// CWBVH: 80-byte nodes plus a separate triangle blob.
StructuredBuffer<float4> CwbvhNodes;
StructuredBuffer<float4> CwbvhTris;
// Opacity micro maps: OpMapN^2 bits per primitive, packed low bit first into
// ( ( OpMapN * OpMapN ) + 31 ) / 32 words per primitive, one map after another for a TLAS.
// OpMapN == 0 means no maps at all and every triangle stays fully opaque. Only the Bvh2 and
// BvhGpu paths consult it; the .cl kernels for BVH4_GPU and CWBVH have no opacity support
// either, so neither do TinyTraceBvh4Gpu and TinyTraceCwbvh.
StructuredBuffer<uint> OpMap;
uint OpMapN;

// Deviation from traverse_bvh2.cl: that kernel hardcodes a 32x32 map, so it can fold the
// word count into the index arithmetic; here OpMapN is a uniform and the word count is
// computed from it, matching the C++ CPU code instead.
#define TINYBVH_OPMAP_WORDS ( ( ( OpMapN * OpMapN ) + 31 ) >> 5 )
/// Value of a BLAS descriptor's OpMapOffset meaning "this BLAS has no map"; the .cl's
/// blasDesc.opmapOffset sentinel.
#define TINYBVH_NO_OPMAP 0x99999999u

/// Result of a traversal. TriAddr is the block index of the hit triangle inside
/// Bvh4Data / CwbvhTris; it is unused by the Bvh2 and BvhGpu paths. Inst is the instance
/// index and is only written by the TLAS traversal; Prim always stays the BLAS-local
/// primitive index rather than packing the instance into its high bits, unlike
/// traverse_tlas.cl which returns "prim | (instIdx << 24)".
struct TinyHit
{
	float T;
	float U;
	float V;
	uint Prim;
	uint Steps;
	uint TriAddr;
	uint Inst;
};

TinyHit TinyHitMiss( float tmax )
{
	TinyHit hit;
	hit.T = tmax;
	hit.U = 0.0f;
	hit.V = 0.0f;
	hit.Prim = 0;
	hit.Steps = 0;
	hit.TriAddr = 0;
	hit.Inst = 0;
	return hit;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Port of BvhMath.Rcp (tinybvh_safercp with the C# port's magnitude cap): 1/x, or
/// +-1e30 when that is not finite. Capping instead of returning +-FLT_MAX keeps the
/// products with scene coordinates finite, exactly like the C# reference.
float TinyRcp( float x )
{
	float r = 1.0f / x;
	if ( !( abs( r ) < TINYBVH_RCP_MAX ) )
	{
		r = ( x < 0.0f || ( x == 0.0f && ( asuint( x ) & 0x80000000u ) != 0u ) ) ? -TINYBVH_RCP_MAX : TINYBVH_RCP_MAX;
	}
	return r;
}

float3 TinyRcp3( float3 d )
{
	return float3( TinyRcp( d.x ), TinyRcp( d.y ), TinyRcp( d.z ) );
}

/// OpenCL's convert_float4( as_uchar4( v ) ): the four bytes of v, lowest byte in x.
float4 TinyUnpackUchar4( float v )
{
	uint u = asuint( v );
	return float4( u & 255u, ( u >> 8 ) & 255u, ( u >> 16 ) & 255u, u >> 24 );
}

/// OpenCL's __bfind: 31 - clz( v ). HLSL's firstbithigh returns the index of the
/// most significant set bit counted from the LSB, which is the same thing. Like the
/// .cl version this is only correct for v != 0, which holds at every call site.
uint TinyBfind( uint v )
{
	return firstbithigh( v );
}

/// Port of sign_extend_s8x4 (the portable, non-PTX branch).
uint TinySignExtendS8x4( uint i )
{
	uint b0 = ( i & 0x80000000u ) != 0u ? 0xff000000u : 0u;
	uint b1 = ( i & 0x00800000u ) != 0u ? 0x00ff0000u : 0u;
	uint b2 = ( i & 0x00008000u ) != 0u ? 0x0000ff00u : 0u;
	uint b3 = ( i & 0x00000080u ) != 0u ? 0x000000ffu : 0u;
	return b0 + b1 + b2 + b3;
}

// ---------------------------------------------------------------------------
// Geometric normals, per layout
// ---------------------------------------------------------------------------

/// Port of GET_PRIM_INDICES_I0_I1_I2: the three Verts offsets of a primitive, relative to the
/// BLAS's vertBase. With Indexed at 0 this is the plain prim * 3 addressing, so nothing about
/// a non-indexed trace changes.
uint3 TinyPrimIndices( uint prim )
{
	uint3 vi = uint3( prim * 3, ( prim * 3 ) + 1, ( prim * 3 ) + 2 );
	if ( Indexed != 0 )
	{
		vi = uint3( VertIdx[ prim * 3 ], VertIdx[ ( prim * 3 ) + 1 ], VertIdx[ ( prim * 3 ) + 2 ] );
	}
	return vi;
}

/// Normal from the shared vertex buffer, matching CpuBvhBackend.RenderJob.ComputeNormal.
/// vertBase is the first float4 of the BLAS inside the shared Verts buffer, zero for the
/// single-BVH kernels.
float3 TinyNormalVertsAt( uint vertBase, uint prim, float3 dir )
{
	uint3 vi = TinyPrimIndices( prim );
	float3 a = Verts[ vertBase + vi.x ].xyz;
	float3 b = Verts[ vertBase + vi.y ].xyz;
	float3 c = Verts[ vertBase + vi.z ].xyz;
	float3 n = normalize( cross( b - a, c - a ) );
	return dot( n, dir ) > 0.0f ? -n : n;
}

float3 TinyNormalVerts( uint prim, float3 dir )
{
	return TinyNormalVertsAt( 0, prim, dir );
}

/// BVH4_GPU stores v0, v1 - v0 and v2 - v0 per triangle (BVH4_GPU::ConvertFrom).
float3 TinyNormalBvh4( uint triAddr, float3 dir )
{
	float3 e1 = Bvh4Data[ triAddr + 1 ].xyz;
	float3 e2 = Bvh4Data[ triAddr + 2 ].xyz;
	float3 n = normalize( cross( e1, e2 ) );
	return dot( n, dir ) > 0.0f ? -n : n;
}

/// CWBVH stores v2 - v0, v1 - v0 and v0 per triangle (BVH8_CWBVH::ConvertFrom).
float3 TinyNormalCwbvh( uint triAddr, float3 dir )
{
	float3 v2v0 = CwbvhTris[ triAddr ].xyz;
	float3 v1v0 = CwbvhTris[ triAddr + 1 ].xyz;
	float3 n = normalize( cross( v1v0, v2v0 ) );
	return dot( n, dir ) > 0.0f ? -n : n;
}

// ---------------------------------------------------------------------------
// Bvh2: port of Bvh.Intersect.cs (Wald nodes, octant-specialised slab test)
// ---------------------------------------------------------------------------

/// Port of the SLAB_TEST_TWO_NODES macro. Outputs TINYBVH_FAR for a missed child.
void TinySlabTestTwoNodes( uint child1, uint child2, float3 rD, float3 ro,
	bool posX, bool posY, bool posZ, float tmax, out float dist1, out float dist2 )
{
	float4 a0 = Bvh2Nodes[ child1 * 2 ];
	float4 a1 = Bvh2Nodes[ ( child1 * 2 ) + 1 ];
	float4 b0 = Bvh2Nodes[ child2 * 2 ];
	float4 b1 = Bvh2Nodes[ ( child2 * 2 ) + 1 ];
	float tx1a = ( ( posX ? a0.x : a1.x ) * rD.x ) - ro.x;
	float ty1a = ( ( posY ? a0.y : a1.y ) * rD.y ) - ro.y;
	float tz1a = ( ( posZ ? a0.z : a1.z ) * rD.z ) - ro.z;
	float tx1b = ( ( posX ? b0.x : b1.x ) * rD.x ) - ro.x;
	float ty1b = ( ( posY ? b0.y : b1.y ) * rD.y ) - ro.y;
	float tz1b = ( ( posZ ? b0.z : b1.z ) * rD.z ) - ro.z;
	float tx2a = ( ( posX ? a1.x : a0.x ) * rD.x ) - ro.x;
	float ty2a = ( ( posY ? a1.y : a0.y ) * rD.y ) - ro.y;
	float tz2a = ( ( posZ ? a1.z : a0.z ) * rD.z ) - ro.z;
	float tx2b = ( ( posX ? b1.x : b0.x ) * rD.x ) - ro.x;
	float ty2b = ( ( posY ? b1.y : b0.y ) * rD.y ) - ro.y;
	float tz2b = ( ( posZ ? b1.z : b0.z ) * rD.z ) - ro.z;
	float tmina = max( max( tx1a, ty1a ), max( tz1a, 0.0f ) );
	float tminb = max( max( tx1b, ty1b ), max( tz1b, 0.0f ) );
	float tmaxa = min( min( tx2a, ty2a ), min( tz2a, tmax ) );
	float tmaxb = min( min( tx2b, ty2b ), min( tz2b, tmax ) );
	dist1 = TINYBVH_FAR;
	dist2 = TINYBVH_FAR;
	if ( tmaxa >= tmina )
	{
		dist1 = tmina;
	}
	if ( tmaxb >= tminb )
	{
		dist2 = tminb;
	}
}

/// Port of the opacity-map evaluation in BVHBase::IntersectTri / traverse_ailalaine: maps the
/// barycentrics onto one of the OpMapN^2 micro-triangles and returns its bit. mapOffset is the
/// first word of the owning BLAS's map inside OpMap, zero for a single non-TLAS structure.
/// Only called when OpMapN > 0.
bool TinyOpacityOpaque( uint mapOffset, uint triIdx, float u, float v )
{
	float fN = ( float )OpMapN;
	int row = ( int )( ( u + v ) * fN );
	int diag = ( int )( ( 1.0f - u ) * fN );
	int idx = ( row * row ) + ( int )( v * fN ) + ( diag - ( ( int )OpMapN - 1 - row ) );
	uint word = OpMap[ mapOffset + ( triIdx * TINYBVH_OPMAP_WORDS ) + ( uint )( idx >> 5 ) ];
	return ( word & ( 1u << ( idx & 31 ) ) ) != 0;
}

/// Port of BVHBase::IntersectTri. Note the "t > hit.T" rejection: a triangle at
/// exactly the current hit distance replaces the recorded hit, as on the CPU.
void TinyIntersectTriVerts( uint triIdx, float3 O, float3 D, inout TinyHit hit )
{
	uint3 vi = TinyPrimIndices( triIdx );
	float4 v0_ = Verts[ vi.x ];
	float3 v0 = v0_.xyz;
	float3 e1 = ( Verts[ vi.y ] - v0_ ).xyz;
	float3 e2 = ( Verts[ vi.z ] - v0_ ).xyz;
	float3 h = cross( D, e2 );
	float a = dot( e1, h );
	if ( abs( a ) < TINYBVH_TRI_EPS )
	{
		return;
	}
	float f = 1.0f / a;
	float3 s = O - v0;
	float u = f * dot( s, h );
	float3 q = cross( s, e1 );
	float v = f * dot( D, q );
	if ( u < 0.0f || v < 0.0f || ( u + v ) > 1.0f )
	{
		return;
	}
	float t = f * dot( e2, q );
	if ( t < 0.0f || t > hit.T )
	{
		return;
	}
	// evaluate opacity map, if present. A single Bvh2 upload owns the whole OpMap buffer.
	if ( OpMapN > 0 && !TinyOpacityOpaque( 0, triIdx, u, v ) )
	{
		return;
	}
	hit.T = t;
	hit.U = u;
	hit.V = v;
	hit.Prim = triIdx;
}

/// Port of BVHBase::TriOccludes.
bool TinyTriOccludesVerts( uint triIdx, float3 O, float3 D, float tmax )
{
	uint3 vi = TinyPrimIndices( triIdx );
	float4 v0_ = Verts[ vi.x ];
	float3 v0 = v0_.xyz;
	float3 e1 = ( Verts[ vi.y ] - v0_ ).xyz;
	float3 e2 = ( Verts[ vi.z ] - v0_ ).xyz;
	float3 h = cross( D, e2 );
	float a = dot( e1, h );
	if ( abs( a ) < TINYBVH_TRI_EPS )
	{
		return false;
	}
	float f = 1.0f / a;
	float3 s = O - v0;
	float u = f * dot( s, h );
	float3 q = cross( s, e1 );
	float v = f * dot( D, q );
	if ( u < 0.0f || v < 0.0f || ( u + v ) > 1.0f )
	{
		return false;
	}
	float t = f * dot( e2, q );
	if ( t < 0.0f || t > tmax )
	{
		return false;
	}
	// evaluate opacity map, if present. A single Bvh2 upload owns the whole OpMap buffer.
	if ( OpMapN > 0 && !TinyOpacityOpaque( 0, triIdx, u, v ) )
	{
		return false;
	}
	return true;
}

/// Port of the templated BVH::Intersect body. Steps holds the traversal cost the CPU
/// reports: one per node visit plus one per triangle test.
TinyHit TinyTraceBvh2( float3 O, float3 D, float3 rD, float tmax )
{
	TinyHit hit = TinyHitMiss( tmax );
	uint stack[ TINYBVH_BVH2_STACK_SIZE ];
	uint stackPtr = 0;
	uint node = 0;
	bool posX = D.x >= 0.0f;
	bool posY = D.y >= 0.0f;
	bool posZ = D.z >= 0.0f;
	float3 ro = O * rD;
	float cost = 0.0f;
	[loop] while ( true )
	{
		cost += 1.0f;
		float4 n0 = Bvh2Nodes[ node * 2 ];
		float4 n1 = Bvh2Nodes[ ( node * 2 ) + 1 ];
		uint leftFirst = asuint( n0.w );
		uint triCount = asuint( n1.w );
		if ( triCount > 0 )
		{
			[loop] for ( uint i = 0; i < triCount; i++ )
			{
				TinyIntersectTriVerts( PrimIdx[ leftFirst + i ], O, D, hit );
				cost += 1.0f;
			}
			if ( stackPtr == 0 )
			{
				break;
			}
			node = stack[ --stackPtr ];
			continue;
		}
		uint child1 = leftFirst;
		uint child2 = leftFirst + 1;
		float dist1;
		float dist2;
		TinySlabTestTwoNodes( child1, child2, rD, ro, posX, posY, posZ, hit.T, dist1, dist2 );
		if ( dist1 > dist2 )
		{
			float td = dist1;
			dist1 = dist2;
			dist2 = td;
			uint tn = child1;
			child1 = child2;
			child2 = tn;
		}
		if ( dist1 == TINYBVH_FAR )
		{
			if ( stackPtr == 0 )
			{
				break;
			}
			node = stack[ --stackPtr ];
		}
		else
		{
			node = child1;
			if ( dist2 != TINYBVH_FAR )
			{
				stack[ stackPtr++ ] = child2;
			}
		}
	}
	hit.Steps = ( uint )cost;
	return hit;
}

/// Port of the templated BVH::IsOccluded body.
bool TinyOccludedBvh2( float3 O, float3 D, float3 rD, float tmax )
{
	uint stack[ TINYBVH_BVH2_STACK_SIZE ];
	uint stackPtr = 0;
	uint node = 0;
	bool posX = D.x >= 0.0f;
	bool posY = D.y >= 0.0f;
	bool posZ = D.z >= 0.0f;
	float3 ro = O * rD;
	[loop] while ( true )
	{
		float4 n0 = Bvh2Nodes[ node * 2 ];
		float4 n1 = Bvh2Nodes[ ( node * 2 ) + 1 ];
		uint leftFirst = asuint( n0.w );
		uint triCount = asuint( n1.w );
		if ( triCount > 0 )
		{
			[loop] for ( uint i = 0; i < triCount; i++ )
			{
				if ( TinyTriOccludesVerts( PrimIdx[ leftFirst + i ], O, D, tmax ) )
				{
					return true;
				}
			}
			if ( stackPtr == 0 )
			{
				break;
			}
			node = stack[ --stackPtr ];
			continue;
		}
		uint child1 = leftFirst;
		uint child2 = leftFirst + 1;
		float dist1;
		float dist2;
		TinySlabTestTwoNodes( child1, child2, rD, ro, posX, posY, posZ, tmax, dist1, dist2 );
		if ( dist1 > dist2 )
		{
			float td = dist1;
			dist1 = dist2;
			dist2 = td;
			uint tn = child1;
			child1 = child2;
			child2 = tn;
		}
		if ( dist1 == TINYBVH_FAR )
		{
			if ( stackPtr == 0 )
			{
				break;
			}
			node = stack[ --stackPtr ];
		}
		else
		{
			node = child1;
			if ( dist2 != TINYBVH_FAR )
			{
				stack[ stackPtr++ ] = child2;
			}
		}
	}
	return false;
}

// ---------------------------------------------------------------------------
// BvhGpu: port of traverse_ailalaine / isoccluded_ailalaine (traverse_bvh2.cl)
// ---------------------------------------------------------------------------

/// Port of traverse_ailalaine. Deviation: the .cl rejects with "d >= hit.x" (first hit
/// wins a tie) and uses a 1e-7 determinant epsilon; the epsilon is 1e-6 here, the tie
/// rule is left as the .cl has it.
///
/// The three base offsets locate this BLAS inside the shared buffers, as blasDesc does in
/// traverse_tlas.cl: nodeBase is a float4 index into BvhGpuNodes (four per node), idxBase
/// an element index into PrimIdx and vertBase a float4 index into Verts. All three are
/// zero for the single-BVH kernels. mapOffset is the fourth, a word index into OpMap, and
/// is TINYBVH_NO_OPMAP for a BLAS without an opacity map - the .cl's blasDesc.opmapOffset.
TinyHit TinyTraceBvhGpuAt( uint nodeBase, uint idxBase, uint vertBase, uint mapOffset, float3 O, float3 D, float3 rD, float tmax )
{
	TinyHit hit = TinyHitMiss( tmax );
	float3 rO = O * -rD;
	uint stack[ TINYBVH_BVH2_STACK_SIZE ];
	uint stackPtr = 0;
	uint node = 0;
	uint steps = 0;
	[loop] while ( true )
	{
		steps++;
		float4 rmin = BvhGpuNodes[ nodeBase + ( node * 4 ) + 2 ];
		float4 rmax = BvhGpuNodes[ nodeBase + ( node * 4 ) + 3 ];
		uint triCount = asuint( rmin.w );
		if ( triCount > 0 )
		{
			uint firstTri = asuint( rmax.w );
			[loop] for ( uint i = 0; i < triCount; i++ )
			{
				uint triIdx = PrimIdx[ idxBase + firstTri + i ];
				uint3 vi = TinyPrimIndices( triIdx );
				float4 v0 = Verts[ vertBase + vi.x ];
				float3 edge1 = ( Verts[ vertBase + vi.y ] - v0 ).xyz;
				float3 edge2 = ( Verts[ vertBase + vi.z ] - v0 ).xyz;
				float3 h = cross( D, edge2 );
				float a = dot( edge1, h );
				if ( abs( a ) < TINYBVH_TRI_EPS )
				{
					continue;
				}
				float f = 1.0f / a;
				float3 s = O - v0.xyz;
				float u = f * dot( s, h );
				float3 q = cross( s, edge1 );
				float v = f * dot( D, q );
				if ( u < 0.0f || v < 0.0f || ( u + v ) > 1.0f )
				{
					continue;
				}
				float d = f * dot( edge2, q );
				if ( d <= 0.0f || d >= hit.T )
				{
					continue;
				}
				// evaluate opacity map, if this BLAS has one.
				if ( OpMapN > 0 && mapOffset != TINYBVH_NO_OPMAP && !TinyOpacityOpaque( mapOffset, triIdx, u, v ) )
				{
					continue;
				}
				hit.T = d;
				hit.U = u;
				hit.V = v;
				hit.Prim = triIdx;
			}
			if ( stackPtr == 0 )
			{
				break;
			}
			node = stack[ --stackPtr ];
			continue;
		}
		float4 lmin = BvhGpuNodes[ nodeBase + ( node * 4 ) ];
		float4 lmax = BvhGpuNodes[ nodeBase + ( node * 4 ) + 1 ];
		uint left = asuint( lmin.w );
		uint right = asuint( lmax.w );
		float3 t1a = ( lmin.xyz * rD ) + rO;
		float3 t2a = ( lmax.xyz * rD ) + rO;
		float3 t1b = ( rmin.xyz * rD ) + rO;
		float3 t2b = ( rmax.xyz * rD ) + rO;
		float3 minta = min( t1a, t2a );
		float3 maxta = max( t1a, t2a );
		float3 mintb = min( t1b, t2b );
		float3 maxtb = max( t1b, t2b );
		float tmina = max( max( max( minta.x, minta.y ), minta.z ), 0.0f );
		float tminb = max( max( max( mintb.x, mintb.y ), mintb.z ), 0.0f );
		float tmaxa = min( min( min( maxta.x, maxta.y ), maxta.z ), hit.T );
		float tmaxb = min( min( min( maxtb.x, maxtb.y ), maxtb.z ), hit.T );
		float dist1 = tmina > tmaxa ? TINYBVH_FAR : tmina;
		float dist2 = tminb > tmaxb ? TINYBVH_FAR : tminb;
		if ( dist1 > dist2 )
		{
			if ( dist2 == TINYBVH_FAR )
			{
				if ( stackPtr == 0 )
				{
					break;
				}
				node = stack[ --stackPtr ];
			}
			else
			{
				node = right;
				if ( dist1 < TINYBVH_FAR )
				{
					stack[ stackPtr++ ] = left;
				}
			}
		}
		else
		{
			if ( dist1 == TINYBVH_FAR )
			{
				if ( stackPtr == 0 )
				{
					break;
				}
				node = stack[ --stackPtr ];
			}
			else
			{
				node = left;
				if ( dist2 < TINYBVH_FAR )
				{
					stack[ stackPtr++ ] = right;
				}
			}
		}
	}
	hit.Steps = steps;
	return hit;
}

TinyHit TinyTraceBvhGpu( float3 O, float3 D, float3 rD, float tmax )
{
	// A single BvhGpu upload owns the whole OpMap buffer, so its map starts at word zero.
	return TinyTraceBvhGpuAt( 0, 0, 0, 0, O, D, rD, tmax );
}

/// Port of isoccluded_ailalaine. Deviation: the .cl omits the determinant epsilon here
/// (it divides by zero for degenerate triangles); the epsilon is kept, as in the CPU code.
/// See TinyTraceBvhGpuAt for the meaning of the four base offsets.
bool TinyOccludedBvhGpuAt( uint nodeBase, uint idxBase, uint vertBase, uint mapOffset, float3 O, float3 D, float3 rD, float tmax )
{
	float3 rO = O * -rD;
	uint stack[ TINYBVH_BVH2_STACK_SIZE ];
	uint stackPtr = 0;
	uint node = 0;
	[loop] while ( true )
	{
		float4 lmin = BvhGpuNodes[ nodeBase + ( node * 4 ) ];
		float4 lmax = BvhGpuNodes[ nodeBase + ( node * 4 ) + 1 ];
		float4 rmin = BvhGpuNodes[ nodeBase + ( node * 4 ) + 2 ];
		float4 rmax = BvhGpuNodes[ nodeBase + ( node * 4 ) + 3 ];
		uint triCount = asuint( rmin.w );
		if ( triCount > 0 )
		{
			uint firstTri = asuint( rmax.w );
			[loop] for ( uint i = 0; i < triCount; i++ )
			{
				uint triIdx = PrimIdx[ idxBase + firstTri + i ];
				uint3 vi = TinyPrimIndices( triIdx );
				float4 v0 = Verts[ vertBase + vi.x ];
				float3 edge1 = ( Verts[ vertBase + vi.y ] - v0 ).xyz;
				float3 edge2 = ( Verts[ vertBase + vi.z ] - v0 ).xyz;
				float3 h = cross( D, edge2 );
				float a = dot( edge1, h );
				if ( abs( a ) < TINYBVH_TRI_EPS )
				{
					continue;
				}
				float f = 1.0f / a;
				float3 s = O - v0.xyz;
				float u = f * dot( s, h );
				float3 q = cross( s, edge1 );
				float v = f * dot( D, q );
				if ( u < 0.0f || v < 0.0f || ( u + v ) > 1.0f )
				{
					continue;
				}
				float d = f * dot( edge2, q );
				if ( d <= 0.0f || d >= tmax )
				{
					continue;
				}
				// evaluate opacity map, if this BLAS has one.
				if ( OpMapN > 0 && mapOffset != TINYBVH_NO_OPMAP && !TinyOpacityOpaque( mapOffset, triIdx, u, v ) )
				{
					continue;
				}
				return true;
			}
			if ( stackPtr == 0 )
			{
				break;
			}
			node = stack[ --stackPtr ];
			continue;
		}
		uint left = asuint( lmin.w );
		uint right = asuint( lmax.w );
		float3 t1a = ( lmin.xyz * rD ) + rO;
		float3 t2a = ( lmax.xyz * rD ) + rO;
		float3 t1b = ( rmin.xyz * rD ) + rO;
		float3 t2b = ( rmax.xyz * rD ) + rO;
		float3 minta = min( t1a, t2a );
		float3 maxta = max( t1a, t2a );
		float3 mintb = min( t1b, t2b );
		float3 maxtb = max( t1b, t2b );
		float tmina = max( max( max( minta.x, minta.y ), minta.z ), 0.0f );
		float tminb = max( max( max( mintb.x, mintb.y ), mintb.z ), 0.0f );
		float tmaxa = min( min( min( maxta.x, maxta.y ), maxta.z ), tmax );
		float tmaxb = min( min( min( maxtb.x, maxtb.y ), maxtb.z ), tmax );
		float dist1 = tmina > tmaxa ? TINYBVH_FAR : tmina;
		float dist2 = tminb > tmaxb ? TINYBVH_FAR : tminb;
		if ( dist1 > dist2 )
		{
			if ( dist2 == TINYBVH_FAR )
			{
				if ( stackPtr == 0 )
				{
					break;
				}
				node = stack[ --stackPtr ];
			}
			else
			{
				node = right;
				if ( dist1 < TINYBVH_FAR )
				{
					stack[ stackPtr++ ] = left;
				}
			}
		}
		else
		{
			if ( dist1 == TINYBVH_FAR )
			{
				if ( stackPtr == 0 )
				{
					break;
				}
				node = stack[ --stackPtr ];
			}
			else
			{
				node = left;
				if ( dist2 < TINYBVH_FAR )
				{
					stack[ stackPtr++ ] = right;
				}
			}
		}
	}
	return false;
}

bool TinyOccludedBvhGpu( float3 O, float3 D, float3 rD, float tmax )
{
	// A single BvhGpu upload owns the whole OpMap buffer, so its map starts at word zero.
	return TinyOccludedBvhGpuAt( 0, 0, 0, 0, O, D, rD, tmax );
}

// ---------------------------------------------------------------------------
// Bvh4Gpu: port of traverse_gpu4way / isoccluded_gpu4way (traverse_bvh4.cl)
// No opacity micro map support: traverse_bvh4.cl has none either, and the leaves carry
// inline triangles rather than the primitive indices the map is keyed on.
// ---------------------------------------------------------------------------

// Uncompressed triangles: three float4 per triangle (v0 with asfloat(primIdx) in w,
// v1 - v0, v2 - v0), so STRIDE is 3.
#define TINYBVH_BVH4_TRI_STRIDE 3

void TinyIntersectTriBvh4( uint vertIdx, float3 O, float3 D, inout TinyHit hit )
{
	float4 edge2 = Bvh4Data[ vertIdx + 2 ];
	float4 edge1 = Bvh4Data[ vertIdx + 1 ];
	float4 v0 = Bvh4Data[ vertIdx ];
	float3 h = cross( D, edge2.xyz );
	float a = dot( edge1.xyz, h );
	if ( abs( a ) < TINYBVH_TRI_EPS )
	{
		return;
	}
	float f = 1.0f / a;
	float3 s = O - v0.xyz;
	float u = f * dot( s, h );
	float3 q = cross( s, edge1.xyz );
	float v = f * dot( D, q );
	if ( u < 0.0f || v < 0.0f || ( u + v ) > 1.0f )
	{
		return;
	}
	float d = f * dot( edge2.xyz, q );
	if ( d > 0.0f && d < hit.T )
	{
		hit.T = d;
		hit.U = u;
		hit.V = v;
		hit.Prim = asuint( v0.w );
		hit.TriAddr = vertIdx;
	}
}

bool TinyTriOccludedBvh4( uint vertIdx, float3 O, float3 D, float tmax )
{
	float4 edge2 = Bvh4Data[ vertIdx + 2 ];
	float4 edge1 = Bvh4Data[ vertIdx + 1 ];
	float4 v0 = Bvh4Data[ vertIdx ];
	float3 h = cross( D, edge2.xyz );
	float a = dot( edge1.xyz, h );
	if ( abs( a ) < TINYBVH_TRI_EPS )
	{
		return false;
	}
	float f = 1.0f / a;
	float3 s = O - v0.xyz;
	float u = f * dot( s, h );
	float3 q = cross( s, edge1.xyz );
	float v = f * dot( D, q );
	if ( u < 0.0f || v < 0.0f || ( u + v ) > 1.0f )
	{
		return false;
	}
	float d = f * dot( edge2.xyz, q );
	return d > 0.0f && d < tmax;
}

/// The 4-wide quantized slab test of traverse_gpu4way, shared by the two traversals.
/// Returns the four entry distances (TINYBVH_FAR for a miss) and the four child info
/// words, both sorted by descending distance.
void TinyBvh4ChildTest( uint offset, float3 O, float3 rD, float tmax, out float4 dst4, out uint4 data3 )
{
	float4 data0 = Bvh4Data[ offset ];
	float4 data1 = Bvh4Data[ offset + 1 ];
	float4 data2 = Bvh4Data[ offset + 2 ];
	float4 cminx4 = TinyUnpackUchar4( data0.w );
	float4 cmaxx4 = TinyUnpackUchar4( data1.w );
	float4 cminy4 = TinyUnpackUchar4( data2.x );
	float3 bminO = ( O - data0.xyz ) * rD;
	float3 rDe = rD * data1.xyz;
	float4 cmaxy4 = TinyUnpackUchar4( data2.y );
	float4 cminz4 = TinyUnpackUchar4( data2.z );
	float4 cmaxz4 = TinyUnpackUchar4( data2.w );
	float4 t1x4 = ( cminx4 * rDe.xxxx ) - bminO.xxxx;
	float4 t2x4 = ( cmaxx4 * rDe.xxxx ) - bminO.xxxx;
	float4 t1y4 = ( cminy4 * rDe.yyyy ) - bminO.yyyy;
	float4 t2y4 = ( cmaxy4 * rDe.yyyy ) - bminO.yyyy;
	float4 t1z4 = ( cminz4 * rDe.zzzz ) - bminO.zzzz;
	float4 t2z4 = ( cmaxz4 * rDe.zzzz ) - bminO.zzzz;
	data3 = asuint( Bvh4Data[ offset + 3 ] );
	float4 mintx4 = min( t1x4, t2x4 );
	float4 maxtx4 = max( t1x4, t2x4 );
	float4 minty4 = min( t1y4, t2y4 );
	float4 maxty4 = max( t1y4, t2y4 );
	float4 mintz4 = min( t1z4, t2z4 );
	float4 maxtz4 = max( t1z4, t2z4 );
	// The .cl's select( a, b, isless( a, b ) ) chains are plain max / min.
	dst4 = max( max( max( mintx4, minty4 ), mintz4 ), 0.0f );
	float4 tmax4 = min( min( min( maxtx4, maxty4 ), maxtz4 ), tmax );
	// Written per component: a vector-condition ?: is component-wise in HLSL, but being
	// explicit avoids relying on that across compilers.
	dst4.x = dst4.x > tmax4.x ? TINYBVH_FAR : dst4.x;
	dst4.y = dst4.y > tmax4.y ? TINYBVH_FAR : dst4.y;
	dst4.z = dst4.z > tmax4.z ? TINYBVH_FAR : dst4.z;
	dst4.w = dst4.w > tmax4.w ? TINYBVH_FAR : dst4.w;
	// Sorting network, descending; bertdobbelaere.github.io/sorting_networks.html
	if ( dst4.x < dst4.z )
	{
		dst4 = dst4.zyxw;
		data3 = data3.zyxw;
	}
	if ( dst4.y < dst4.w )
	{
		dst4 = dst4.xwzy;
		data3 = data3.xwzy;
	}
	if ( dst4.x < dst4.y )
	{
		dst4 = dst4.yxzw;
		data3 = data3.yxzw;
	}
	if ( dst4.z < dst4.w )
	{
		dst4 = dst4.xywz;
		data3 = data3.xywz;
	}
	if ( dst4.y < dst4.z )
	{
		dst4 = dst4.xzyw;
		data3 = data3.xzyw;
	}
}

TinyHit TinyTraceBvh4Gpu( float3 O, float3 D, float3 rD, float tmax )
{
	TinyHit hit = TinyHitMiss( tmax );
	uint stack[ TINYBVH_BVH4_STACK_SIZE ];
	uint stackPtr = 0;
	uint offset = 0;
	uint steps = 0;
	[loop] while ( true )
	{
		steps++;
		float4 dst4;
		uint4 data3;
		TinyBvh4ChildTest( offset, O, rD, hit.T, dst4, data3 );
		// Process results, starting with the farthest child, so the nearest ends on top.
		uint nextNode = 0;
		if ( dst4.x < TINYBVH_FAR )
		{
			if ( ( data3.x >> 31 ) == 0 )
			{
				nextNode = data3.x;
			}
			else
			{
				uint triCount = ( data3.x >> 16 ) & 0x7fffu;
				[loop] for ( uint i = 0; i < triCount; i++ )
				{
					TinyIntersectTriBvh4( ( data3.x & 0xffffu ) + offset + ( i * TINYBVH_BVH4_TRI_STRIDE ), O, D, hit );
				}
			}
		}
		if ( dst4.y < TINYBVH_FAR )
		{
			if ( ( data3.y >> 31 ) != 0 )
			{
				uint triCount = ( data3.y >> 16 ) & 0x7fffu;
				[loop] for ( uint i = 0; i < triCount; i++ )
				{
					TinyIntersectTriBvh4( ( data3.y & 0xffffu ) + offset + ( i * TINYBVH_BVH4_TRI_STRIDE ), O, D, hit );
				}
			}
			else
			{
				if ( nextNode != 0 )
				{
					stack[ stackPtr++ ] = nextNode;
				}
				nextNode = data3.y;
			}
		}
		if ( dst4.z < TINYBVH_FAR )
		{
			if ( ( data3.z >> 31 ) != 0 )
			{
				uint triCount = ( data3.z >> 16 ) & 0x7fffu;
				[loop] for ( uint i = 0; i < triCount; i++ )
				{
					TinyIntersectTriBvh4( ( data3.z & 0xffffu ) + offset + ( i * TINYBVH_BVH4_TRI_STRIDE ), O, D, hit );
				}
			}
			else
			{
				if ( nextNode != 0 )
				{
					stack[ stackPtr++ ] = nextNode;
				}
				nextNode = data3.z;
			}
		}
		if ( dst4.w < TINYBVH_FAR )
		{
			if ( ( data3.w >> 31 ) != 0 )
			{
				uint triCount = ( data3.w >> 16 ) & 0x7fffu;
				[loop] for ( uint i = 0; i < triCount; i++ )
				{
					TinyIntersectTriBvh4( ( data3.w & 0xffffu ) + offset + ( i * TINYBVH_BVH4_TRI_STRIDE ), O, D, hit );
				}
			}
			else
			{
				if ( nextNode != 0 )
				{
					stack[ stackPtr++ ] = nextNode;
				}
				nextNode = data3.w;
			}
		}
		if ( nextNode != 0 )
		{
			offset = nextNode;
		}
		else
		{
			if ( stackPtr == 0 )
			{
				break;
			}
			offset = stack[ --stackPtr ];
		}
	}
	hit.Steps = steps;
	return hit;
}

bool TinyOccludedBvh4Gpu( float3 O, float3 D, float3 rD, float tmax )
{
	uint stack[ TINYBVH_BVH4_STACK_SIZE ];
	uint stackPtr = 0;
	uint offset = 0;
	[loop] while ( true )
	{
		float4 dst4;
		uint4 data3;
		TinyBvh4ChildTest( offset, O, rD, tmax, dst4, data3 );
		uint nextNode = 0;
		if ( dst4.x < TINYBVH_FAR )
		{
			if ( ( data3.x >> 31 ) == 0 )
			{
				nextNode = data3.x;
			}
			else
			{
				uint triCount = ( data3.x >> 16 ) & 0x7fffu;
				[loop] for ( uint i = 0; i < triCount; i++ )
				{
					if ( TinyTriOccludedBvh4( ( data3.x & 0xffffu ) + offset + ( i * TINYBVH_BVH4_TRI_STRIDE ), O, D, tmax ) )
					{
						return true;
					}
				}
			}
		}
		if ( dst4.y < TINYBVH_FAR )
		{
			if ( ( data3.y >> 31 ) != 0 )
			{
				uint triCount = ( data3.y >> 16 ) & 0x7fffu;
				[loop] for ( uint i = 0; i < triCount; i++ )
				{
					if ( TinyTriOccludedBvh4( ( data3.y & 0xffffu ) + offset + ( i * TINYBVH_BVH4_TRI_STRIDE ), O, D, tmax ) )
					{
						return true;
					}
				}
			}
			else
			{
				if ( nextNode != 0 )
				{
					stack[ stackPtr++ ] = nextNode;
				}
				nextNode = data3.y;
			}
		}
		if ( dst4.z < TINYBVH_FAR )
		{
			if ( ( data3.z >> 31 ) != 0 )
			{
				uint triCount = ( data3.z >> 16 ) & 0x7fffu;
				[loop] for ( uint i = 0; i < triCount; i++ )
				{
					if ( TinyTriOccludedBvh4( ( data3.z & 0xffffu ) + offset + ( i * TINYBVH_BVH4_TRI_STRIDE ), O, D, tmax ) )
					{
						return true;
					}
				}
			}
			else
			{
				if ( nextNode != 0 )
				{
					stack[ stackPtr++ ] = nextNode;
				}
				nextNode = data3.z;
			}
		}
		if ( dst4.w < TINYBVH_FAR )
		{
			if ( ( data3.w >> 31 ) != 0 )
			{
				uint triCount = ( data3.w >> 16 ) & 0x7fffu;
				[loop] for ( uint i = 0; i < triCount; i++ )
				{
					if ( TinyTriOccludedBvh4( ( data3.w & 0xffffu ) + offset + ( i * TINYBVH_BVH4_TRI_STRIDE ), O, D, tmax ) )
					{
						return true;
					}
				}
			}
			else
			{
				if ( nextNode != 0 )
				{
					stack[ stackPtr++ ] = nextNode;
				}
				nextNode = data3.w;
			}
		}
		if ( nextNode != 0 )
		{
			offset = nextNode;
		}
		else
		{
			if ( stackPtr == 0 )
			{
				break;
			}
			offset = stack[ --stackPtr ];
		}
	}
	return false;
}

// ---------------------------------------------------------------------------
// Cwbvh: port of traverse_cwbvh / isoccluded_cwbvh (traverse_cwbvh.cl)
// No opacity micro map support: traverse_cwbvh.cl has none either, so a CWBVH BLAS ignores
// the OpMapOffset in its descriptor, exactly as traverse_tlas.cl does.
// ---------------------------------------------------------------------------

/// One half of a CWBVH node's child test (four children). This is the SIMD_AABBTEST
/// branch of the .cl, which is what the "unknown GPU" configuration selects; the scalar
/// branch computes the same values with explicit fma calls.
uint TinyCwbvhChildHits( uint meta4, uint octinv4,
	float4 lox4, float4 hix4, float4 loy4, float4 hiy4, float4 loz4, float4 hiz4,
	float3 idir, float3 orig, float tmax )
{
	uint isInner4 = ( meta4 & ( meta4 << 1 ) ) & 0x10101010u;
	uint innerMask4 = TinySignExtendS8x4( isInner4 << 3 );
	uint bitIndex4 = ( meta4 ^ ( octinv4 & innerMask4 ) ) & 0x1F1F1F1Fu;
	uint childBits4 = ( meta4 >> 5 ) & 0x07070707u;
	float4 tminx4 = ( lox4 * idir.xxxx ) + orig.xxxx;
	float4 tmaxx4 = ( hix4 * idir.xxxx ) + orig.xxxx;
	float4 tminy4 = ( loy4 * idir.yyyy ) + orig.yyyy;
	float4 tmaxy4 = ( hiy4 * idir.yyyy ) + orig.yyyy;
	float4 tminz4 = ( loz4 * idir.zzzz ) + orig.zzzz;
	float4 tmaxz4 = ( hiz4 * idir.zzzz ) + orig.zzzz;
	float4 cmin4 = max( max( max( tminx4, tminy4 ), tminz4 ), 0.0f );
	float4 cmax4 = min( min( min( tmaxx4, tmaxy4 ), tmaxz4 ), tmax );
	uint hitmask = 0;
	if ( cmin4.x <= cmax4.x )
	{
		hitmask |= ( childBits4 & 255u ) << ( bitIndex4 & 31u );
	}
	if ( cmin4.y <= cmax4.y )
	{
		hitmask |= ( ( childBits4 >> 8 ) & 255u ) << ( ( bitIndex4 >> 8 ) & 31u );
	}
	if ( cmin4.z <= cmax4.z )
	{
		hitmask |= ( ( childBits4 >> 16 ) & 255u ) << ( ( bitIndex4 >> 16 ) & 31u );
	}
	if ( cmin4.w <= cmax4.w )
	{
		hitmask |= ( childBits4 >> 24 ) << ( bitIndex4 >> 24 );
	}
	return hitmask;
}

/// Decodes one CWBVH node and returns its hit mask; also outputs the child node base
/// index and the triangle base index for the node's leaves.
uint TinyCwbvhNodeHits( uint childNodeIndex, uint octinv4, float3 O, float3 rD, float tmax,
	out uint childBaseIndex, out uint triangleBaseIndex, out uint imask )
{
	float4 n0 = CwbvhNodes[ childNodeIndex + 0 ];
	float4 n1 = CwbvhNodes[ childNodeIndex + 1 ];
	float4 n2 = CwbvhNodes[ childNodeIndex + 2 ];
	float4 n3 = CwbvhNodes[ childNodeIndex + 3 ];
	float4 n4 = CwbvhNodes[ childNodeIndex + 4 ];
	// n0.w packs the three signed exponent bytes and the imask byte.
	uint ew = asuint( n0.w );
	int ex = asint( ew << 24 ) >> 24;
	int ey = asint( ew << 16 ) >> 24;
	int ez = asint( ew << 8 ) >> 24;
	childBaseIndex = asuint( n1.x );
	triangleBaseIndex = asuint( n1.y );
	imask = ew >> 24;
	float3 idir = float3(
		asfloat( ( ex + 127 ) << 23 ) * rD.x,
		asfloat( ( ey + 127 ) << 23 ) * rD.y,
		asfloat( ( ez + 127 ) << 23 ) * rD.z );
	float3 orig = ( n0.xyz - O ) * rD;
	uint hitmask = TinyCwbvhChildHits( asuint( n1.z ), octinv4,
		TinyUnpackUchar4( rD.x < 0.0f ? n3.z : n2.x ), TinyUnpackUchar4( rD.x < 0.0f ? n2.x : n3.z ),
		TinyUnpackUchar4( rD.y < 0.0f ? n4.x : n2.z ), TinyUnpackUchar4( rD.y < 0.0f ? n2.z : n4.x ),
		TinyUnpackUchar4( rD.z < 0.0f ? n4.z : n3.x ), TinyUnpackUchar4( rD.z < 0.0f ? n3.x : n4.z ),
		idir, orig, tmax );
	hitmask |= TinyCwbvhChildHits( asuint( n1.w ), octinv4,
		TinyUnpackUchar4( rD.x < 0.0f ? n3.w : n2.y ), TinyUnpackUchar4( rD.x < 0.0f ? n2.y : n3.w ),
		TinyUnpackUchar4( rD.y < 0.0f ? n4.y : n2.w ), TinyUnpackUchar4( rD.y < 0.0f ? n2.w : n4.y ),
		TinyUnpackUchar4( rD.z < 0.0f ? n4.w : n3.y ), TinyUnpackUchar4( rD.z < 0.0f ? n3.y : n4.w ),
		idir, orig, tmax );
	return hitmask;
}

uint TinyCwbvhOctinv4( float3 D )
{
	return ( 7u - ( ( D.x < 0.0f ? 4u : 0u ) | ( D.y < 0.0f ? 2u : 0u ) | ( D.z < 0.0f ? 1u : 0u ) ) ) * 0x1010101u;
}

/// Port of traverse_cwbvh. Deviation: the .cl's hit path has no determinant epsilon,
/// which lets a degenerate triangle produce a NaN hit distance; the epsilon from
/// isoccluded_cwbvh (and from the CPU code) is used here instead.
///
/// nodeBase and triBase locate this BLAS inside the shared CwbvhNodes / CwbvhTris buffers
/// (float4 indices), as blasDesc does in traverse_tlas.cl; both are zero for the single-BVH
/// kernels. hit.TriAddr is absolute, so the normal helper needs no offset of its own.
TinyHit TinyTraceCwbvhAt( uint nodeBase, uint triBase, float3 O, float3 D, float3 rD, float t )
{
	TinyHit hit = TinyHitMiss( t );
	uint2 stack[ TINYBVH_CWBVH_STACK_SIZE ];
	uint stackPtr = 0;
	uint steps = 0;
	uint hitAddr = 0;
	uint hitTriAddr = 0;
	float2 uv = float2( 0.0f, 0.0f );
	float tmax = t;
	uint octinv4 = TinyCwbvhOctinv4( D );
	uint2 ngroup = uint2( 0, 0x80000000u );
	uint2 tgroup = uint2( 0, 0 );
	[loop] while ( true )
	{
		steps++;
		if ( ngroup.y > 0x00FFFFFFu )
		{
			uint imask = ngroup.y;
			uint childBitIndex = TinyBfind( ngroup.y );
			uint childNodeBaseIndex = ngroup.x;
			ngroup.y &= ~( 1u << childBitIndex );
			if ( ngroup.y > 0x00FFFFFFu )
			{
				stack[ stackPtr++ ] = ngroup;
			}
			uint slotIndex = ( childBitIndex - 24u ) ^ ( octinv4 & 255u );
			uint relativeIndex = countbits( imask & ~( 0xFFFFFFFFu << slotIndex ) );
			uint childNodeIndex = nodeBase + ( ( childNodeBaseIndex + relativeIndex ) * 5u );
			uint childBaseIndex;
			uint triangleBaseIndex;
			uint nodeImask;
			uint hitmask = TinyCwbvhNodeHits( childNodeIndex, octinv4, O, rD, tmax,
				childBaseIndex, triangleBaseIndex, nodeImask );
			ngroup.x = childBaseIndex;
			ngroup.y = ( hitmask & 0xFF000000u ) | nodeImask;
			tgroup = uint2( triangleBaseIndex, hitmask & 0x00FFFFFFu );
		}
		else
		{
			tgroup = ngroup;
			ngroup = uint2( 0, 0 );
		}
		[loop] while ( tgroup.y != 0 )
		{
			// Moeller-Trumbore; triangles are stored as 3x16 bytes: v2 - v0, v1 - v0 and
			// v0 with the original primitive index in its w component.
			uint triangleIndex = TinyBfind( tgroup.y );
			uint triAddr = triBase + tgroup.x + ( triangleIndex * 3u );
			float3 e1 = CwbvhTris[ triAddr ].xyz;
			float3 e2 = CwbvhTris[ triAddr + 1 ].xyz;
			float4 v0 = CwbvhTris[ triAddr + 2 ];
			tgroup.y -= 1u << triangleIndex;
			float3 r = cross( D, e1 );
			float a = dot( e2, r );
			if ( abs( a ) < TINYBVH_TRI_EPS )
			{
				continue;
			}
			float f = 1.0f / a;
			float3 s = O - v0.xyz;
			float u = f * dot( s, r );
			float3 q = cross( s, e2 );
			float v = f * dot( D, q );
			if ( u < 0.0f || v < 0.0f || ( u + v ) > 1.0f )
			{
				continue;
			}
			float d = f * dot( e1, q );
			if ( d <= 0.0f || d >= tmax )
			{
				continue;
			}
			uv = float2( u, v );
			tmax = d;
			hitAddr = asuint( v0.w );
			hitTriAddr = triAddr;
		}
		if ( ngroup.y <= 0x00FFFFFFu )
		{
			if ( stackPtr > 0 )
			{
				ngroup = stack[ --stackPtr ];
			}
			else
			{
				hit.T = tmax;
				hit.U = uv.x;
				hit.V = uv.y;
				hit.Prim = hitAddr;
				hit.TriAddr = hitTriAddr;
				break;
			}
		}
	}
	hit.Steps = steps;
	return hit;
}

TinyHit TinyTraceCwbvh( float3 O, float3 D, float3 rD, float t )
{
	return TinyTraceCwbvhAt( 0, 0, O, D, rD, t );
}

/// Port of isoccluded_cwbvh. See TinyTraceCwbvhAt for the two base offsets.
bool TinyOccludedCwbvhAt( uint nodeBase, uint triBase, float3 O, float3 D, float3 rD, float t )
{
	uint2 stack[ TINYBVH_CWBVH_STACK_SIZE ];
	uint stackPtr = 0;
	float tmax = t;
	uint octinv4 = TinyCwbvhOctinv4( D );
	uint2 ngroup = uint2( 0, 0x80000000u );
	uint2 tgroup = uint2( 0, 0 );
	[loop] while ( true )
	{
		if ( ngroup.y > 0x00FFFFFFu )
		{
			uint imask = ngroup.y;
			uint childBitIndex = TinyBfind( ngroup.y );
			uint childNodeBaseIndex = ngroup.x;
			ngroup.y &= ~( 1u << childBitIndex );
			if ( ngroup.y > 0x00FFFFFFu )
			{
				stack[ stackPtr++ ] = ngroup;
			}
			uint slotIndex = ( childBitIndex - 24u ) ^ ( octinv4 & 255u );
			uint relativeIndex = countbits( imask & ~( 0xFFFFFFFFu << slotIndex ) );
			uint childNodeIndex = nodeBase + ( ( childNodeBaseIndex + relativeIndex ) * 5u );
			uint childBaseIndex;
			uint triangleBaseIndex;
			uint nodeImask;
			uint hitmask = TinyCwbvhNodeHits( childNodeIndex, octinv4, O, rD, tmax,
				childBaseIndex, triangleBaseIndex, nodeImask );
			ngroup.x = childBaseIndex;
			ngroup.y = ( hitmask & 0xFF000000u ) | nodeImask;
			tgroup = uint2( triangleBaseIndex, hitmask & 0x00FFFFFFu );
		}
		else
		{
			tgroup = ngroup;
			ngroup = uint2( 0, 0 );
		}
		[loop] while ( tgroup.y != 0 )
		{
			uint triangleIndex = TinyBfind( tgroup.y );
			uint triAddr = triBase + tgroup.x + ( triangleIndex * 3u );
			float3 e1 = CwbvhTris[ triAddr ].xyz;
			float3 e2 = CwbvhTris[ triAddr + 1 ].xyz;
			float3 v0 = CwbvhTris[ triAddr + 2 ].xyz;
			tgroup.y -= 1u << triangleIndex;
			float3 r = cross( D, e1 );
			float a = dot( e2, r );
			if ( abs( a ) < TINYBVH_TRI_EPS )
			{
				continue;
			}
			float f = 1.0f / a;
			float3 s = O - v0;
			float u = f * dot( s, r );
			float3 q = cross( s, e2 );
			float v = f * dot( D, q );
			if ( u < 0.0f || v < 0.0f || ( u + v ) > 1.0f )
			{
				continue;
			}
			float d = f * dot( e1, q );
			if ( d > 0.0f && d < tmax )
			{
				return true;
			}
		}
		if ( ngroup.y <= 0x00FFFFFFu )
		{
			if ( stackPtr == 0 )
			{
				break;
			}
			ngroup = stack[ --stackPtr ];
		}
	}
	return false;
}

bool TinyOccludedCwbvh( float3 O, float3 D, float3 rD, float t )
{
	return TinyOccludedCwbvhAt( 0, 0, O, D, rD, t );
}

// ---------------------------------------------------------------------------
// Tlas: port of traverse_tlas / isoccluded_tlas (traverse_tlas.cl), the branch that
// handles arbitrary tlas/blas scenes rather than the DEPRECATED_TLAS_PATH one.
// ---------------------------------------------------------------------------

#define TINYBVH_BLAS_BVHGPU 0
#define TINYBVH_BLAS_CWBVH 1

/// One entry of the instance array, matching TinyBVH.BlasInstance (160 bytes) field for
/// field, which in turn is the OpenCL "struct Instance" without its dummy[8] padding.
/// The matrices are the rows of tinybvh's bvhmat4, so cell i lives at Row(i / 4)[i % 4]
/// and the translation is in the w components of the first three rows.
struct TinyInstance
{
	float4 TRow0;
	float4 TRow1;
	float4 TRow2;
	float4 TRow3;
	float4 IRow0;
	float4 IRow1;
	float4 IRow2;
	float4 IRow3;
	float4 AabbMinBlas; // xyz: world-space bounds min, w: asfloat( blasIdx )
	float4 AabbMaxMask; // xyz: world-space bounds max, w: asfloat( mask )
};

/// Where one BLAS lives inside the shared buffers; the equivalent of the .cl's BLASDesc.
struct TinyBlasDesc
{
	uint NodeOffset;  // float4 index into BvhGpuNodes, or into CwbvhNodes
	uint IndexOffset; // element index into PrimIdx; unused by CWBVH
	uint VertOffset;  // float4 index into Verts, or into CwbvhTris
	uint BlasType;    // TINYBVH_BLAS_BVHGPU or TINYBVH_BLAS_CWBVH
	uint OpMapOffset; // word index into OpMap, or TINYBVH_NO_OPMAP; unused by CWBVH
};

// TLAS in the Aila-Laine layout, four float4 per node, exactly like BvhGpuNodes.
StructuredBuffer<float4> TlasNodes;
// Instance indices the TLAS leaves point at.
StructuredBuffer<uint> TlasIdx;
StructuredBuffer<TinyInstance> Instances;
StructuredBuffer<TinyBlasDesc> BlasDesc;

/// Port of tools.cl's TransformPoint, same operation order as BvhMat4.TransformPoint.
float3 TinyTransformPoint( float3 v, float4 r0, float4 r1, float4 r2, float4 r3 )
{
	float3 res = float3(
		( r0.x * v.x ) + ( r0.y * v.y ) + ( r0.z * v.z ) + r0.w,
		( r1.x * v.x ) + ( r1.y * v.y ) + ( r1.z * v.z ) + r1.w,
		( r2.x * v.x ) + ( r2.y * v.y ) + ( r2.z * v.z ) + r2.w );
	float w = ( r3.x * v.x ) + ( r3.y * v.y ) + ( r3.z * v.z ) + r3.w;
	return w == 1.0f ? res : res * ( 1.0f / w );
}

/// Port of tools.cl's TransformVector.
float3 TinyTransformVector( float3 v, float4 r0, float4 r1, float4 r2 )
{
	return float3(
		( r0.x * v.x ) + ( r0.y * v.y ) + ( r0.z * v.z ),
		( r1.x * v.x ) + ( r1.y * v.y ) + ( r1.z * v.z ),
		( r2.x * v.x ) + ( r2.y * v.y ) + ( r2.z * v.z ) );
}

/// Closest-hit traversal of one BLAS in object space, dispatched on its layout. Written with
/// a single result local and one return, which is also what keeps fxc from warning about a
/// potentially uninitialised return value.
TinyHit TinyTraceBlas( TinyBlasDesc desc, float3 O, float3 D, float3 rD, float tmax )
{
	TinyHit hit = TinyHitMiss( tmax );
	if ( desc.BlasType == TINYBVH_BLAS_CWBVH )
	{
		hit = TinyTraceCwbvhAt( desc.NodeOffset, desc.VertOffset, O, D, rD, tmax );
	}
	else
	{
		hit = TinyTraceBvhGpuAt( desc.NodeOffset, desc.IndexOffset, desc.VertOffset, desc.OpMapOffset, O, D, rD, tmax );
	}
	return hit;
}

bool TinyOccludedBlas( TinyBlasDesc desc, float3 O, float3 D, float3 rD, float tmax )
{
	bool occluded = false;
	if ( desc.BlasType == TINYBVH_BLAS_CWBVH )
	{
		occluded = TinyOccludedCwbvhAt( desc.NodeOffset, desc.VertOffset, O, D, rD, tmax );
	}
	else
	{
		occluded = TinyOccludedBvhGpuAt( desc.NodeOffset, desc.IndexOffset, desc.VertOffset, desc.OpMapOffset, O, D, rD, tmax );
	}
	return occluded;
}

/// Geometric normal of a TLAS hit, in world space. Deviation from raytracer.cl: only the
/// geometric normal is available here, and it is pushed through the 3x3 part of the
/// instance transform rather than through its inverse transpose. That is correct for the
/// rigid and uniformly scaled transforms the sample and the reference data use; a
/// non-uniform scale would need the inverse transpose.
float3 TinyNormalTlas( TinyHit hit, float3 dir )
{
	TinyInstance inst = Instances[ hit.Inst ];
	TinyBlasDesc desc = BlasDesc[ asuint( inst.AabbMinBlas.w ) ];
	float3 nObj;
	if ( desc.BlasType == TINYBVH_BLAS_CWBVH )
	{
		// CWBVH stores v2 - v0, v1 - v0 and v0 per triangle; TriAddr is already absolute.
		nObj = cross( CwbvhTris[ hit.TriAddr + 1 ].xyz, CwbvhTris[ hit.TriAddr ].xyz );
	}
	else
	{
		float3 a = Verts[ desc.VertOffset + ( hit.Prim * 3 ) ].xyz;
		float3 b = Verts[ desc.VertOffset + ( hit.Prim * 3 ) + 1 ].xyz;
		float3 c = Verts[ desc.VertOffset + ( hit.Prim * 3 ) + 2 ].xyz;
		nObj = cross( b - a, c - a );
	}
	float3 n = normalize( TinyTransformVector( nObj, inst.TRow0, inst.TRow1, inst.TRow2 ) );
	return dot( n, dir ) > 0.0f ? -n : n;
}

/// Port of traverse_tlas. Deviation: instance masks are ignored, so every instance the
/// TLAS reaches is intersected, exactly as the .cl does and unlike the CPU IntersectTlas.
/// Steps accumulates the TLAS node visits plus the steps of every BLAS entered.
TinyHit TinyTraceTlas( float3 O, float3 D, float3 rD, float tmax )
{
	TinyHit hit = TinyHitMiss( tmax );
	uint stack[ TINYBVH_BVH2_STACK_SIZE ];
	uint stackPtr = 0;
	uint node = 0;
	uint steps = 0;
	[loop] while ( true )
	{
		steps++;
		float4 lmin = TlasNodes[ node * 4 ];
		float4 lmax = TlasNodes[ ( node * 4 ) + 1 ];
		float4 rmin = TlasNodes[ ( node * 4 ) + 2 ];
		float4 rmax = TlasNodes[ ( node * 4 ) + 3 ];
		uint triCount = asuint( rmin.w );
		if ( triCount > 0 )
		{
			uint firstTri = asuint( rmax.w );
			[loop] for ( uint i = 0; i < triCount; i++ )
			{
				uint instIdx = TlasIdx[ firstTri + i ];
				TinyInstance inst = Instances[ instIdx ];
				float3 oBlas = TinyTransformPoint( O, inst.IRow0, inst.IRow1, inst.IRow2, inst.IRow3 );
				float3 dBlas = TinyTransformVector( D, inst.IRow0, inst.IRow1, inst.IRow2 );
				TinyHit blasHit = TinyTraceBlas( BlasDesc[ asuint( inst.AabbMinBlas.w ) ],
					oBlas, dBlas, TinyRcp3( dBlas ), hit.T );
				steps += blasHit.Steps;
				if ( blasHit.T < hit.T )
				{
					hit.T = blasHit.T;
					hit.U = blasHit.U;
					hit.V = blasHit.V;
					hit.Prim = blasHit.Prim;
					hit.TriAddr = blasHit.TriAddr;
					hit.Inst = instIdx;
				}
			}
			if ( stackPtr == 0 )
			{
				break;
			}
			node = stack[ --stackPtr ];
			continue;
		}
		uint left = asuint( lmin.w );
		uint right = asuint( lmax.w );
		float3 t1a = ( lmin.xyz - O ) * rD;
		float3 t2a = ( lmax.xyz - O ) * rD;
		float3 t1b = ( rmin.xyz - O ) * rD;
		float3 t2b = ( rmax.xyz - O ) * rD;
		float3 minta = min( t1a, t2a );
		float3 maxta = max( t1a, t2a );
		float3 mintb = min( t1b, t2b );
		float3 maxtb = max( t1b, t2b );
		float tmina = max( max( max( minta.x, minta.y ), minta.z ), 0.0f );
		float tminb = max( max( max( mintb.x, mintb.y ), mintb.z ), 0.0f );
		float tmaxa = min( min( min( maxta.x, maxta.y ), maxta.z ), hit.T );
		float tmaxb = min( min( min( maxtb.x, maxtb.y ), maxtb.z ), hit.T );
		float dist1 = tmina > tmaxa ? TINYBVH_FAR : tmina;
		float dist2 = tminb > tmaxb ? TINYBVH_FAR : tminb;
		if ( dist1 > dist2 )
		{
			float h = dist1;
			dist1 = dist2;
			dist2 = h;
			uint t = left;
			left = right;
			right = t;
		}
		if ( dist1 == TINYBVH_FAR )
		{
			if ( stackPtr == 0 )
			{
				break;
			}
			node = stack[ --stackPtr ];
		}
		else
		{
			node = left;
			if ( dist2 != TINYBVH_FAR )
			{
				stack[ stackPtr++ ] = right;
			}
		}
	}
	hit.Steps = steps;
	return hit;
}

/// Port of isoccluded_tlas; instance masks are ignored here as well.
bool TinyOccludedTlas( float3 O, float3 D, float3 rD, float tmax )
{
	uint stack[ TINYBVH_BVH2_STACK_SIZE ];
	uint stackPtr = 0;
	uint node = 0;
	[loop] while ( true )
	{
		float4 lmin = TlasNodes[ node * 4 ];
		float4 lmax = TlasNodes[ ( node * 4 ) + 1 ];
		float4 rmin = TlasNodes[ ( node * 4 ) + 2 ];
		float4 rmax = TlasNodes[ ( node * 4 ) + 3 ];
		uint triCount = asuint( rmin.w );
		if ( triCount > 0 )
		{
			uint firstTri = asuint( rmax.w );
			[loop] for ( uint i = 0; i < triCount; i++ )
			{
				uint instIdx = TlasIdx[ firstTri + i ];
				TinyInstance inst = Instances[ instIdx ];
				float3 oBlas = TinyTransformPoint( O, inst.IRow0, inst.IRow1, inst.IRow2, inst.IRow3 );
				float3 dBlas = TinyTransformVector( D, inst.IRow0, inst.IRow1, inst.IRow2 );
				if ( TinyOccludedBlas( BlasDesc[ asuint( inst.AabbMinBlas.w ) ],
					oBlas, dBlas, TinyRcp3( dBlas ), tmax ) )
				{
					return true;
				}
			}
			if ( stackPtr == 0 )
			{
				break;
			}
			node = stack[ --stackPtr ];
			continue;
		}
		uint left = asuint( lmin.w );
		uint right = asuint( lmax.w );
		float3 t1a = ( lmin.xyz - O ) * rD;
		float3 t2a = ( lmax.xyz - O ) * rD;
		float3 t1b = ( rmin.xyz - O ) * rD;
		float3 t2b = ( rmax.xyz - O ) * rD;
		float3 minta = min( t1a, t2a );
		float3 maxta = max( t1a, t2a );
		float3 mintb = min( t1b, t2b );
		float3 maxtb = max( t1b, t2b );
		float tmina = max( max( max( minta.x, minta.y ), minta.z ), 0.0f );
		float tminb = max( max( max( mintb.x, mintb.y ), mintb.z ), 0.0f );
		float tmaxa = min( min( min( maxta.x, maxta.y ), maxta.z ), tmax );
		float tmaxb = min( min( min( maxtb.x, maxtb.y ), maxtb.z ), tmax );
		float dist1 = tmina > tmaxa ? TINYBVH_FAR : tmina;
		float dist2 = tminb > tmaxb ? TINYBVH_FAR : tminb;
		if ( dist1 > dist2 )
		{
			float h = dist1;
			dist1 = dist2;
			dist2 = h;
			uint t = left;
			left = right;
			right = t;
		}
		if ( dist1 == TINYBVH_FAR )
		{
			if ( stackPtr == 0 )
			{
				break;
			}
			node = stack[ --stackPtr ];
		}
		else
		{
			node = left;
			if ( dist2 != TINYBVH_FAR )
			{
				stack[ stackPtr++ ] = right;
			}
		}
	}
	return false;
}

#endif // TINYBVH_TRAVERSAL_INCLUDED
