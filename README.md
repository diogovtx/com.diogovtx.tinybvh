# TinyBVH

A managed C# port of [tinybvh](https://github.com/jbikker/tinybvh) (v1.8.0, MIT) for Unity 6:
BVH construction, refit, optimization and traversal for CPU and GPU, validated bit-exact
against the original C++ library.

## Requirements

Unity 6000.3 with `com.unity.burst`, `com.unity.mathematics` and `com.unity.collections`
(declared as package dependencies). The GPU path needs compute shader support.

## Install

This package is embedded under `Packages/com.diogovtx.tinybvh` in its repository. To use it
in another project, copy that folder into the target project's `Packages/` directory, or
reference it from `manifest.json` as a local or git package. See the repository root README
for the exact options.

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

Import "CPU and GPU raytracer" from this package's Samples tab in the Package Manager window
for a runnable demo of every backend.

## More

See the repository root README for the test setup, the standalone benchmark, and the list of
deviations from the C++ reference.
