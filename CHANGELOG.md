# Changelog

All notable changes to this package will be documented in this file.

## [Unreleased]

- The reference dump tools and the scene download script now ship with the package in `Tools~`,
  so the tests can run outside the development repository.

## [1.8.0] - 2026-09-22

First packaged release: a complete managed C# port of tinybvh 1.8.0.

- Base `Bvh` (binned SAH build, refit, intersect, TLAS) and its builders: SBVH (`BuildHQ`),
  full sweep, quick build, presplitting, the AVX binned builder, and threaded construction on
  the Unity job system.
- The tree-rotation optimizer, including its stochastic mode.
- Wide CPU layouts with SIMD traversal: `Bvh4Cpu` (SSE4), `Bvh8Cpu` (AVX2), `BvhSoa`, and CPU
  traversal for the compressed wide BVH (CWBVH).
- GPU layouts and compute kernels, GPU TLAS, and a TLAS over mixed BLAS layouts.
- `VoxelSet`, `BvhDouble`, Save/Load for every layout, opacity micromaps, custom geometry, ray
  packets, the sphere query and the EPO tree-quality metric.
- Tests validated bit-exact against reference dumps produced by the original C++ library.
