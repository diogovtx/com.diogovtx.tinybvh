# TinyBVH

A managed C# port of [tinybvh](https://github.com/jbikker/tinybvh) (v1.8.0, MIT) for Unity 6:
BVH construction, refit, optimization and traversal for CPU and GPU, validated bit-exact
against the original C++ library.

The package version tracks the upstream tinybvh version the port matches; port-only fixes bump
the patch number.

## Requirements

Unity 6000.3 with `com.unity.burst`, `com.unity.mathematics` and `com.unity.collections`
(declared as package dependencies). The GPU path needs compute shader support.

## Install

Package Manager > **Add package from git URL**:

```
https://github.com/diogovtx/com.diogovtx.tinybvh.git
```

or, pinned to a release, `https://github.com/diogovtx/com.diogovtx.tinybvh.git#v1.8.0`. The
same URL goes into `Packages/manifest.json` as
`"com.diogovtx.tinybvh": "https://github.com/diogovtx/com.diogovtx.tinybvh.git#v1.8.0"`.
Cloning the repository into a project's `Packages/` folder works too.

## What is ported

- `Bvh`, the base BVH: binned SAH build over a triangle soup or indexed triangles, `Refit`,
  `Intersect`, `IsOccluded`, `Compact`, `SplitLeafs`, `CombineLeafs`, SAH and EPO cost, TLAS
  over `BlasInstance`s with masks, and a TLAS over BLASes of mixed layouts (`BlasRef`).
- Builders: SBVH (`UseSpatialSplits`), full sweep, `BuildQuick`, presplitting, the AVX binned
  builder, threaded construction on the job system, the tree-rotation optimizer (including
  its stochastic mode), `Save` / `Load` for every layout.
- CPU layouts and traversal: `Bvh4Cpu` (SSE4 through Burst intrinsics), `Bvh8Cpu` (AVX2),
  `BvhSoa`, `BvhCwbvh` CPU traversal, `Intersect256Rays` packets, `IntersectSphere`,
  `BvhDouble` with a double-precision TLAS, `VoxelSet` (256^3 voxel object, three-level DDA).
- GPU: `BvhGpu`, `Bvh4Gpu` and CWBVH compute kernels ported from tinybvh's OpenCL, a GPU TLAS,
  and `GpuTracer` to upload and dispatch them.
- Opacity micromaps and custom geometry (user AABBs plus intersection callbacks).

Every SIMD path has a scalar fallback that produces identical results under Mono.

## Usage

```csharp
using TinyBVH;

// three float4 vertices per triangle; only xyz is used. Keep the array alive while the BVH is in use.
NativeArray<float4> vertices = ...;
Bvh bvh = Bvh.Create( Allocator.Persistent );
bvh.Build( vertices, triangleCount );

Ray ray = new Ray( origin, direction );
int steps = bvh.Intersect( ref ray );      // hit in ray.Hit; ray.Hit.T == BvhConstants.Far on a miss
bool shadowed = bvh.IsOccluded( new Ray( origin, toLight, maxDistance ) );

bvh.Refit();                                // after moving vertices without changing topology

// faster CPU traversal: 4-wide layout (builds its own base BVH, like the C++)
Bvh4Cpu bvh4 = Bvh4Cpu.Create( Allocator.Persistent );
bvh4.Build( vertices, triangleCount );
bvh4.Intersect( ref ray );

// GPU: convert, upload, dispatch
Mbvh mbvh8 = Mbvh.Create( 8, Allocator.Persistent );
bvh.Compact();
bvh.SplitLeafs( 3 );                        // CWBVH leaves hold at most 3 triangles
mbvh8.ConvertFrom( ref bvh );
BvhCwbvh cwbvh = BvhCwbvh.Create( Allocator.Persistent );
cwbvh.ConvertFrom( ref mbvh8 );
GpuTracer tracer = new GpuTracer();
tracer.Upload( ref cwbvh );
tracer.Render( renderTexture, origin, topLeftDir, horizontal, vertical, mode, shadows, sunDir, sceneDiagonal );
```

All structs are unmanaged and can live inside Burst jobs. Derived layouts keep a shallow copy
of their source (`Source`), which they do not own: dispose the source separately, after the
derived layout. `Bvh.Compact` and `SplitLeafs` mutate the base BVH in place.

## Sample

Import **CPU and GPU raytracer** from this package's Samples tab in the Package Manager window:
a raytracer with one backend per layout, an on-screen UI to switch backends, builders and
display modes, and a standalone benchmark mode. It needs `com.unity.inputsystem`, and loads
tinybvh's `.bin` test scenes from a `TestData` folder next to the project's `Assets` folder,
where `Tools~/fetch.ps1` downloads them; see the sample's README.

## Tests

`Tests/Editor` holds the NUnit edit-mode tests. They compare trees, index arrays and tens of
thousands of rays per scene bit for bit against reference dumps produced by the original C++
library, and skip themselves when the dumps are missing. To run them:

1. List the package under `testables` in `Packages/manifest.json`.
2. Download tinybvh's test scenes into a `TestData` folder next to the project's `Assets` folder
   with `Tools~/fetch.ps1`, or by hand from tinybvh's 1.8.0 release:
   [suzanne.bin](https://raw.githubusercontent.com/jbikker/tinybvh/0e4584287823252cf83f0e9cd072848bec5f79c5/testdata/suzanne.bin),
   [bunny.bin](https://raw.githubusercontent.com/jbikker/tinybvh/0e4584287823252cf83f0e9cd072848bec5f79c5/testdata/bunny.bin) and
   [cryteksponza.bin](https://raw.githubusercontent.com/jbikker/tinybvh/0e4584287823252cf83f0e9cd072848bec5f79c5/testdata/cryteksponza.bin).
3. Build the reference tools with `Tools~/RefDump/build.bat` (MSVC on Windows x64). It downloads
   `tiny_bvh.h` from the same tinybvh commit when it is missing. `simddump` is built for AVX2
   and needs a CPU that has it.
4. Generate the dumps, about 900 MB, with `Tools~/RefDump/run_all.bat`.

`fetch.ps1` and `run_all.bat` take that folder as an optional argument. Without one they find it
from their own location, which works when the package is embedded in the project's `Packages/`
folder.

## Deviations from the C++

- `BvhMath.Rcp` caps the reciprocal direction at `BvhConstants.RcpMax` (1e30) instead of
  `FLT_MAX`. The C++ relies on the slab test overflowing to NaN for axis-aligned rays, which
  only works when every intermediate is rounded to single precision; Mono evaluates float
  arithmetic in double. The GPU kernels apply the same cap.
- The partition step of the builder evaluates the bin index with the same `float3` expression
  as the binning pass, so both agree regardless of intermediate precision.
- Fatal errors throw exceptions instead of calling `exit`.
- The per-octant template specialisation of the traversals is a single function with the ray
  sign predicates evaluated at runtime.
- `Bvh4Cpu` ports the SSE4 traversal only (not the AVX variant) and zeroes leaf padding.
- `Bvh8Cpu` zeroes the same leaf padding; its scalar fallback multiplies, adds and divides
  where the AVX2 path fuses the multiply-adds and uses the `fastrcp4` reciprocal.
- `Bvh4Cpu.Refit` and `Bvh8Cpu.Refit` refit the *base* BVH and re-convert. The C++ refits the
  intermediate `MBVH<M>` instead, but `ConvertFrom` re-derives the wide nodes from the base
  BVH right after, so those refitted bounds are dropped and the emitted tree keeps the bounds
  of the geometry it was built over.
- The GPU kernels use a 1e-6 determinant epsilon everywhere and add the missing epsilon
  guards in the any-hit CWBVH and Aila-Laine kernels.
- The threaded builders collect the subtrees at `MT_SPAWN_DEPTH` and run them as one parallel
  loop; the C++ instead spawns a task per node at every level above that depth. The set of
  subtrees, and therefore the tree, is the same - only the top levels stay on one thread.
- The GPU TLAS kernels ignore instance masks and intersect every instance the TLAS reaches,
  like `traverse_tlas.cl` and unlike the CPU `IntersectTlas`. They also return the instance
  index in its own field instead of packing it into the high bits of the primitive index.
- `HqBvhOddEven` is accepted but has no effect on a serial build, exactly as in the C++: the
  odd/even bin count is evaluated with the depth a subtree task was spawned at, and a serial
  build never spawns one. The port also rejects `HqBvhBins` outside `2 .. 256` up front, where
  the C++ would overrun its stack buffers.
- The stochastic optimizer keeps its random state per `BvhVerbose` and `Bvh.Optimize` restarts
  it from `randomSeed` on every call, where the C++ draws from the process-wide `rand()` stream.
- `Bvh.SahCost` runs under Burst so it is bit-identical to the C++; Mono would sum it in double.
- The CWBVH converter clears the whole triangle buffer where the C++ clears three quarters of it
  and relies on fresh pages being zero; `Refit`, `Optimize` and `SahCost` throw on a wide layout
  that was loaded from a file, where the C++ dereferences an empty base BVH.
- `BvhDouble.IsOccludedTlas` queries the BLAS with a zero-length ray, the defined equivalent of
  the uninitialised `tmp.hit` the C++ passes there (it never reports occlusion, matching the dump).
- `UseSimdIfAvailable` defaults to false, so `Build` stays on the scalar reference builder unless
  asked; the C++ defaults to the AVX builder. The threaded full-sweep and AVX builds are not
  ported: both run serially here.
- `BlasRef` replaces the C++ `BVHBase` pointer: the port has no common base struct, so the TLAS
  carries a layout tag and a pointer per BLAS, and a TLAS built over the `Bvh*` overload keeps a
  separate direct call. `BvhSoa` has no opacity-map API of its own and reads the base BVH's.
- `VoxelSet.Intersect` needs no layout field: the C++ constructor forgets to set one, which makes
  a voxel BLAS invisible to its TLAS unless the caller sets it; `BlasRef.From( VoxelSet* )` tags it.
- The voxel DDA nudges its entry point into each level by a fixed `0.0000025f` of the ray
  parameter, which only works for a (near) unit-length direction. The TLAS does not renormalise the
  direction it hands a BLAS, so a voxel instance scaled by `s` receives a direction of length
  `1 / s` and the nudge drops below the rounding error of `O + D * t`: the DDA steps a cell without
  advancing `t` and reports voxels the ray never enters (upstream bug, kept bit-exact; see the
  comment on `VoxelSet.Setup3DDDA`). Callers must give the world ray a direction of length `s`, as
  the sample's voxel backend does; `VoxelSampleTests` measures the effect both ways. The same nudge
  also loses outright when a ray sits exactly on a cell plane it cannot be moved off - always so for
  a direction component of exactly zero whose origin coordinate is a multiple of `1/8`, `1/32` or
  `1/256` - and the walk then steps back to `t = 0` and reports nonsense distances; that one has no
  caller-side workaround and is counted, not asserted, by `VoxelSampleTests`.


## License

MIT, see `LICENSE.md`. tinybvh is (c) Jacco Bikker, MIT.
