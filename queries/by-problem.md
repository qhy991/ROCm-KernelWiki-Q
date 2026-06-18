# Index: By Problem


Symptom → Pattern → Technique → Solution


### Compute-Bound Optimization Patterns (算力密集优化模式)

- ID: `pattern-compute-bound`
- Path: [wiki/patterns/compute-bound-optimization.md](../wiki/patterns/compute-bound-optimization.md)
- Tags: optimization, compute, mfma, pipeline, tiling, vgpr

### Cooperative Loading

- ID: `pattern-cooperative-loading`
- Path: [wiki/patterns/cooperative-loading.md](../wiki/patterns/cooperative-loading.md)
- Tags: memory, bandwidth, optimization, lds, vectorized-load

### Grid-Stride Loop

- ID: `pattern-grid-stride-loop`
- Path: [wiki/patterns/grid-stride-loop.md](../wiki/patterns/grid-stride-loop.md)
- Tags: optimization, programming-model, memory, memory-bound

### Latency Hiding (延迟隐藏)

- ID: `pattern-latency-hiding`
- Path: [wiki/patterns/latency-hiding.md](../wiki/patterns/latency-hiding.md)
- Tags: optimization, memory, occupancy, scheduling
- Related: `hw-wavefront`, `hw-compute-unit`

### Memory-Bound Kernel Optimization on AMD CDNA

- ID: `pattern-memory-bound-amd`
- Path: [wiki/patterns/memory-bound-amd.md](../wiki/patterns/memory-bound-amd.md)
- Tags: memory-bound, optimization, bandwidth, hbm
- Related: `hw-lds`, `technique-vectorized-load`, `technique-double-buffering`

### Memory-Bound Optimization Patterns

- ID: `pattern-memory-bound-optimization`
- Path: [wiki/patterns/memory-bound-optimization.md](../wiki/patterns/memory-bound-optimization.md)
- Tags: memory, memory-bound, optimization, bandwidth, vectorization

### 生产者-消费者流水线 (Producer-Consumer Pipeline)

- ID: `pattern-producer-consumer-pipeline`
- Path: [wiki/patterns/producer-consumer-pipeline.md](../wiki/patterns/producer-consumer-pipeline.md)
- Tags: pipeline, memory-bound, double-buffering, async-copy, mfma-scheduling, ck-tile-programming, optimization

### Reduction Tree

- ID: `pattern-reduction-tree`
- Path: [wiki/patterns/reduction-tree.md](../wiki/patterns/reduction-tree.md)
- Tags: reduction, cross-lane, dpp, bpermute, lds, wave-reduction, optimization

### Scatter/Gather Memory Access Patterns

- ID: `pattern-scatter-gather`
- Path: [wiki/patterns/scatter-gather.md](../wiki/patterns/scatter-gather.md)
- Tags: memory, bandwidth, optimization, vectorized-load

### Tile Quantization and Dequantization

- ID: `pattern-tile-quantize-dequant`
- Path: [wiki/patterns/tile-quantize-dequant.md](../wiki/patterns/tile-quantize-dequant.md)
- Tags: quantization, fp8, int8, fp16, vgpr, bandwidth, memory-bound

### Wavefront Specialization (Warp Specialization)

- ID: `pattern-warp-specialization`
- Path: [wiki/patterns/warp-specialization.md](../wiki/patterns/warp-specialization.md)
- Tags: memory, compute, scheduling, synchronization, wavefront, lds, async-copy, double-buffering