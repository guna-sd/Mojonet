Conservative Caching (100MB max)
Pros: Low memory overhead
Cons: More cache misses, some OS calls

Use for:
- Embedded systems
- Memory-constrained
- Large models
Aggressive Caching (10GB max)
Pros: Almost all allocations from cache
Cons: More memory held

Use for:
- GPU workloads (16GB+)
- Server inference
- Batch processing
No Caching (max = 0)
Pros: Minimal memory overhead
Cons: Every allocation → OS call

Use for:
- Embedded systems
- Memory-critical
- One-time allocations