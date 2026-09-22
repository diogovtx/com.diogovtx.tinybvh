# Raytracer sample

A sample that hosts one or more ray tracing backends (CPU and GPU) against the
same scene, lets you switch between them and their display modes from an
on-screen UI, orbit the camera, and benchmark the backends against each other.

## Opening the sample

This sample must first be imported from the Package Manager (TinyBVH package > Samples tab >
"CPU and GPU raytracer"), which copies it into `Assets/Samples`, and it needs the Input System
package.

Use the menu item **TinyBVH > Create CPU Raytracer Scene**. This creates a new
(unsaved) scene with a directional light and a camera holding the
`RaytracerSample` component, configured to load `TestData/bunny.bin`. Press
Play. The scene is intentionally left unsaved; save it yourself if you want to
keep it.

## Controls

- Right mouse drag — orbit the camera around the scene pivot
- Middle mouse drag — pan the pivot
- Mouse wheel — zoom in/out (scaled to the scene's bounding-box diagonal)
- `Tab` — show/hide the UI panel (visible by default)
- `1`-`5` — switch display mode (Shaded, Normals, Depth, TraversalSteps, Barycentrics)
- `S` — toggle shadows

Camera input is ignored while the mouse is over the UI panel.

## UI panel

- **Backend** — selects which registered `IRaytraceBackend` renders the scene;
  GPU backends are marked as such.
- **Display mode** / **Shadows** — as above, mirrored from the keyboard shortcuts.
- **Packets (256 rays)** — shown only while `CPU BVH2 (Burst)` is the active
  backend. Traces the primary rays with `Bvh.Intersect256Rays`, one packet per
  16x16 pixel block, instead of one ray at a time; the shading and the shadow
  rays are unchanged, so only the primary hit takes a different route. The
  packet traversal keeps no step counter, so **TraversalSteps** shades a
  constant zero while this is on. Blocks that hang over the right or bottom
  edge still trace a full 256-ray packet — the traversal needs the corner rays
  to be the extremes of the bundle — with the surplus rays clamped to the last
  valid pixel and not written. The packet leaf test has no opacity micromap
  test, so with **Opacity map** on the cut-outs show in the shadows but not in
  the primary hit.
- **Builder** — which builder every backend builds its triangle BVH (the BLAS
  for the TLAS backends) with; the TLAS itself is always binned. `Binned` is
  `Bvh.Build`, `SBVH` is `UseSpatialSplits`, `FullSweep` is `UseFullSweep` and
  `Quick` is `Bvh.BuildQuick` (mid-point splits, no SAH, much faster and much
  worse). `CPU BVH2 double (Burst)` only has the binned builder and says so in
  the stats line. Changing it rebuilds every backend.
- **Presplit** — presplits the fragments before the subdivision starts
  (`Bvh.UsePresplitting`), which trades index entries for tree quality and
  makes the tree unrefittable. The SBVH builder ignores it and `BuildQuick`
  ignores every build setting; the panel says so under the toggle when one of
  those two is selected. Toggling it rebuilds every backend.
- **Threaded build** — builds every backend's triangle BVH (the BLAS for the
  TLAS backends) with the job system instead of single-threaded, for inputs of
  50,000 or more primitives. Toggling it rebuilds every backend.
- **Optimize (25 iterations)** — optimizes every backend's triangle BVH (the
  BLAS for the TLAS backends) with 25 tree-rotation iterations after the next
  build; the TLAS itself is never optimized. Toggling it rebuilds every backend.
- **Opacity map (N=8)** — applies a procedural per-triangle cut-out pattern (an
  8x8 opacity micromap, one micro-triangle in four a hole) to every backend that
  supports it; `GPU BVH4_GPU`, `GPU CWBVH` and `GPU TLAS over CWBVH` ignore it, as
  noted in the stats line when the active backend is one of them. Toggling it
  rebuilds every backend.
- **Render scale** — output resolution divisor relative to the camera's pixel size (1-8).
- **Scene source** / scene list / **Reload** — pick `BinFile` (one of
  `binScenes`, loaded via `BvhSceneFile.Load`) or `SceneMeshes` (every enabled,
  readable `MeshFilter` in the scene, transformed to world space), then reload
  to rebuild every backend against it and reframe the camera.
- **Instances** / **Animate** — place the loaded mesh once (`1`, the plain
  un-instanced scene) or on a `3x3` or `5x5` grid, spaced by the mesh bounds and
  each copy rotated and scaled a little. Changing the count rebuilds every
  backend and reframes the camera. **Animate** spins the instances every frame:
  the TLAS backends only rebuild their TLAS (`UpdateInstances`), the others get
  a full rebuild over the re-flattened geometry — that contrast is the point of
  the control. Only the active backend is refreshed while animating; the rest
  are resynchronised when it is switched off. A 5x5 grid of a large scene is a
  lot of triangles for the non-instanced backends to rebuild every frame.
- **Deform** — displaces the source mesh every frame with a travelling sine wave
  of about 2% of its diagonal, written into the same vertex array the backends
  were built over; the undeformed mesh is restored when it is switched off.
- **Deform update** (shown while **Deform** is on) — `Refit` updates the active
  backend's tree in place over the moved vertices (`IRefittableBackend.Refit`),
  `Rebuild` rebuilds it from scratch, which is the comparison this demo is for.
  A spatial-split tree and a presplit tree cannot be refit, so the sample
  rebuilds and says so.
- **Stats** — triangle count, the active backend's node count, build time and
  last refit time, an exponential moving average of the frame time, Mrays/s,
  and resolution.
- **Benchmark** — renders every registered backend in turn with the current
  view and settings (5 warm-up frames, then 30 timed frames each), then
  restores the previously active backend and shows a results table (avg ms,
  Mrays/s, build ms, node count per backend) in the panel and in the console.

## Display modes

- **Shaded** — grey-albedo Lambert shading against the `sun` light (or a fixed
  fallback direction), with optional hard shadows.
- **Normals** — geometric (per-triangle) normal, remapped from [-1,1] to [0,1].
- **Depth** — hit distance divided by the scene's bounding-box diagonal, in greyscale.
- **TraversalSteps** — heatmap of the traversal cost returned by the backend
  (blue = few steps, red = ~64 or more steps). Flat blue where the traversal
  reports no cost: `CPU CWBVH (Burst)`, whose walk always returns zero, and
  `CPU BVH2 (Burst)` while **Packets** is on.
- **Barycentrics** — the hit's (u, v, 1-u-v) triangle weights as RGB.

## Scene sources

- **BinFile** — loads a tinybvh `.bin` triangle-soup file from `TestData/`
  (one of `binScenes`, default `bunny.bin`) via `BvhSceneFile.Load`.
- **SceneMeshes** — gathers every enabled `MeshFilter` in the scene, transforms
  their triangles to world space, and builds a single BLAS from the result.
  Meshes must have **Read/Write Enabled** checked in their import settings
  (Model Import Settings > Model > Read/Write Enabled); non-readable meshes are
  skipped with a console warning.

## Backends

Backends implement `IRaytraceBackend` (`Name`, `IsGpu`, `BuildMs`, `NodeCount`,
`Build`, `Render`). `RaytracerSample` owns the scene's vertex data and calls
`Build` on every backend when the scene (re)loads, then calls `Render` on the
active one every frame.

`BackendList.Create()` is the registry: it returns the list of backends the
sample offers. Add a backend by appending one line to that method. Registered:

- `CPU BVH2 (Burst)` — `CpuBvhBackend`, traces a `TinyBVH.Bvh` from a Burst job.
- `CPU BVH4 (Burst SSE)` — `CpuBvh4Backend`, traces a `TinyBVH.Bvh4Cpu` (4-wide SSE
  traversal through Burst intrinsics; scalar fallback without Burst).
- `CPU BVH8 (AVX2)` — `CpuBvh8Backend`, traces a `TinyBVH.Bvh8Cpu` (8-wide AVX2 + FMA
  traversal through Burst intrinsics; named `CPU BVH8 (scalar)` when the CPU lacks them).
- `GPU BVH2`, `GPU BVH_GPU (Aila-Laine)`, `GPU BVH4_GPU`, `GPU CWBVH` — `GpuBackend`
  over `TinyBVH.GpuTracer`, one per compute-shader layout. Only registered when the
  device supports compute shaders.
- `CPU TLAS (Burst)` — `CpuTlasBackend`, one `TinyBVH.Bvh` BLAS placed through a
  `Bvh.BuildTlas` TLAS, traced from a Burst job.
- `CPU CWBVH (Burst)` — `CpuCwbvhBackend`, a `TinyBVH.BvhCwbvh` converted the way
  `BVH8_CWBVH::Build` prepares it (`Compact` + `SplitLeafs( 3 )`, then an 8-wide
  `Mbvh`) and traced on the CPU. The compressed layout is meant for the GPU
  kernels and its CPU walk is the scalar reference traversal, so this is here to
  put the same layout on both sides of the benchmark, not to be fast. No opacity
  micromap support, and no step counter (see **TraversalSteps** above).
- `CPU BVH2 double (Burst)` — `CpuDoubleBackend`, a `TinyBVH.BvhDouble` traced
  with `RayDouble` from a Burst job. The scene is a float triangle soup, so the
  backend owns a widened `double3` copy of it and refills that copy on every
  build; the widening is part of the reported build time. `BvhDouble` has only
  the binned builder and no optimizer or opacity micromap, so **Builder**,
  **Presplit**, **Threaded build**, **Optimize** and **Opacity map** do not reach
  it; the stats line says `(binned only)` when a different builder is selected.
- `CPU BVH SoA (Burst)` — `CpuSoaBackend`, traces a `TinyBVH.BvhSoa` (the binary
  tree with both child slabs interleaved over four SIMD lanes). The layout has no
  opacity micromap API and no `Refit`.
- `CPU TLAS mixed BLAS (Burst)` — `CpuMixedTlasBackend`, a `Bvh.BuildTlas` TLAS
  over **four** BLASses built from the same mesh in four layouts — `Bvh`,
  `Bvh4Cpu`, `Bvh8Cpu` and `BvhSoa` — with instance i entering BLAS i % 4, so a
  single traversal walks all four leaf layouts through the tagged `BlasRef`
  dispatch. Use the **Instances** control to see it: with `1` instance only the
  `Bvh` BLAS is reached, `3x3` and `5x5` spread the copies over all four. Every
  BLAS gets the current **Builder** / **Presplit** / **Threaded build** /
  **Optimize** choice; the TLAS stays binned. `BvhSoa` has no opacity micromap,
  so the mix reports no opacity support. Moving the instances only rebuilds the
  TLAS, as with `CPU TLAS (Burst)`, but a deforming mesh costs four BLAS builds.
- `CPU Voxels (Burst)` — `CpuVoxelBackend`, the scene shown as voxels instead of
  triangles. The source mesh is voxelised into one 256^3 `TinyBVH.VoxelSet` that
  occupies the unit cube, and that cube is placed through a `Bvh.BuildTlas` TLAS
  over a tagged `BlasRef`, so a ray leaves the TLAS into the three-level DDA.
  Details worth knowing:
  - **What is voxelised** — the mesh's own triangles, mapped onto the unit cube
    by a single uniform scale over its largest extent (1% margin on each side, so
    the surface never touches the cube wall). Every triangle is stamped on a
    barycentric sample grid spaced a third of a voxel apart along both of its
    edges, so no voxel the surface crosses is missed. `VoxelSet.Set` is not
    thread safe, so the whole fill is one Burst job; its time is the reported
    build time. The voxel object is recreated from scratch on every build,
    because `VoxelSet` has no clear.
  - **Resolution** — 256 voxels per cube edge, i.e. the whole scene inside a
    256^3 grid, so thin or distant geometry becomes blocky. `NodeCount` reports
    the TLAS nodes plus the number of 8^3 bricks the fill handed out.
  - **Palette** — each triangle writes the voxel value
    `1 + ( ( primIdx * 2654435761 ) >> 24 ) & 254` (never 0, which means "empty").
    **Barycentrics** turns that value into a colour, standing in for the u/v a
    voxel hit does not have; neighbouring triangles hash to unrelated entries, so
    it reads as speckle rather than as facets, which is why **Shaded** uses the
    same flat grey albedo as the triangle backends instead. **Normals** and the
    shading use `VoxelSet.GetNormal`, the face of the voxel the ray entered
    through.
  - **Ray length** — this backend hands the TLAS a direction of length `scale`,
    the world edge of the unit cube, not a unit one. `BVH::IntersectTLAS` pushes
    the direction through the instance's inverse transform without renormalising
    it, so a unit world direction would reach the voxel DDA `1 / scale` long. The
    DDA nudges its entry point into each new cell by a fixed `0.0000025f` of the
    ray parameter, which moves the point by `|D| * 0.0000025f`; shrink `|D|` by
    the scale and that nudge drops below the rounding error of `O + D * t`, the
    `ceilf` that picks the cell plane returns the next plane instead of the
    current one, and the DDA ends up one cell ahead of the ray for the rest of
    that axis — reporting voxels the ray never enters, as hits floating outside
    the mesh. That is an upstream bug (see the comment on
    `VoxelSet.Setup3DDDA`); scaling the direction makes the object-space ray unit
    length, which is what the epsilon assumes. `Hit.T` is then in units of
    `scale`, so `O + D * Hit.T` is still the hit point but **Depth** multiplies
    by `scale` to get world units. `VoxelSampleTests` measures both.
  - **Degenerate rays** — the same nudge has a second upstream failure: a ray
    sitting exactly on a cell plane that the nudge cannot move it off gets
    `tmax = ( plane - O ) * rD = 0` on that axis, below the current `t`, and the
    walk steps back to `t = 0`. A direction component of exactly zero always
    triggers it when the origin coordinate on that axis is a multiple of `1/8`,
    `1/32` or `1/256`. Camera rays and the default sun direction have no exact
    zeros, so the sample never builds such a ray, but a caller that does will get
    nonsense distances back.
  - **Shadow rays** — they start half a voxel (`scale / 512`) off the surface
    along the voxel normal, not at the `1e-3 * sceneDiagonal` the triangle
    backends use. The DDA reports the distance at which the ray entered the
    voxel, so the hit point lies exactly on a voxel face; half a voxel is the
    largest offset that certainly leaves the voxel that was hit without reaching
    past the empty neighbour the primary ray came through.
  - Instance placement, **Instances** and **Animate** work exactly as for the
    other TLAS backends (only the TLAS is rebuilt when instances move).
    **Builder**, **Presplit**, **Threaded build**, **Optimize** and **Opacity
    map** have no meaning without a triangle BVH and are ignored; the stats line
    says `(voxels)`. **Deform** re-voxelises every frame, which is the expensive
    part for a large mesh.
- `GPU TLAS over BVH_GPU`, `GPU TLAS over CWBVH` — `GpuTlasBackend`, the same TLAS
  uploaded with `GpuTracer.UploadTlas` and traced by the `Render_Tlas` kernel, one per
  supported BLAS layout. Only registered when the device supports compute shaders.

Rendering and shading are identical across backends (same five display modes and
shadow rays), so the benchmark compares traversal cost only. The one exception is
`CPU Voxels (Burst)`, which scales its ray directions and its shadow-ray offset to
the voxel grid; see its entry above.

`CPU CWBVH (Burst)`, `CPU BVH2 double (Burst)`, `CPU BVH SoA (Burst)`,
`CPU TLAS mixed BLAS (Burst)` and `CPU Voxels (Burst)` have no `Refit`, so a
deforming mesh always costs them a full rebuild whatever **Deform update** is
set to.

Backends that can instance also implement `IInstancedBackend` (`UpdateMs`,
`BuildInstanced`, `UpdateInstances`): they get the source mesh plus the instance
list and build one BLAS and a TLAS over it, and an instance move only rebuilds the
TLAS. The others receive the mesh flattened into every instance through the normal
`Build`, so an instance move costs them a full rebuild.

## Other settings

- `autoFrame` — on load (including via the Reload button), calls the public
  `FrameScene()` method to position the camera and orbit pivot from the
  scene's vertex bounds so the geometry is in view.
