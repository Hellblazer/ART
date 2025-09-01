# Parameterized Performance Report - STANDARD Scale

Generated: 2025-09-01T13:45:24.782459

## Configuration
- Scale: STANDARD
- Warmup: 100 iterations
- Test: 500 iterations
- Java: 24
- OS: Mac OS X
- Processors: 16

## Results Summary

| Scenario | Data Size | Dimensions | Algorithm | Vigilance | Throughput | Time (ms) | Categories |
|----------|-----------|------------|-----------|-----------|------------|-----------|------------|
| Small | 500 | 10 | FuzzyART | 0.5 | 1527492 | 0.33 | 12 |
| Small | 500 | 10 | HypersphereART | 0.5 | 10733299 | 0.05 | 7 |
| Small | 500 | 10 | FuzzyART | 0.7 | 1317957 | 0.38 | 15 |
| Small | 500 | 10 | HypersphereART | 0.7 | 10743216 | 0.05 | 7 |
| Small | 500 | 10 | FuzzyART | 0.9 | 133410 | 3.75 | 138 |
| Small | 500 | 10 | HypersphereART | 0.9 | 6768190 | 0.07 | 18 |
| Medium | 1000 | 50 | FuzzyART | 0.5 | 309869 | 1.61 | 21 |
| Medium | 1000 | 50 | HypersphereART | 0.5 | 4487686 | 0.11 | 5 |
| Medium | 1000 | 50 | FuzzyART | 0.7 | 379651 | 1.32 | 18 |
| Medium | 1000 | 50 | HypersphereART | 0.7 | 2800477 | 0.18 | 12 |
| Medium | 1000 | 50 | FuzzyART | 0.9 | 25182 | 19.86 | 236 |
| Medium | 1000 | 50 | HypersphereART | 0.9 | 614187 | 0.81 | 95 |
| Large | 2000 | 100 | FuzzyART | 0.5 | 116616 | 4.29 | 23 |
| Large | 2000 | 100 | HypersphereART | 0.5 | 2178251 | 0.23 | 6 |
| Large | 2000 | 100 | FuzzyART | 0.7 | 152895 | 3.27 | 21 |
| Large | 2000 | 100 | HypersphereART | 0.7 | 292726 | 1.71 | 80 |
| Large | 2000 | 100 | FuzzyART | 0.9 | 11514 | 43.43 | 293 |
| Large | 2000 | 100 | HypersphereART | 0.9 | 47790 | 10.46 | 499 |
| HighDim | 500 | 200 | FuzzyART | 0.5 | 107791 | 4.64 | 16 |
| HighDim | 500 | 200 | HypersphereART | 0.5 | 395556 | 1.26 | 28 |
| HighDim | 500 | 200 | FuzzyART | 0.7 | 151912 | 3.29 | 15 |
| HighDim | 500 | 200 | HypersphereART | 0.7 | 371218 | 1.35 | 28 |
| HighDim | 500 | 200 | FuzzyART | 0.9 | 6918 | 72.27 | 287 |
| HighDim | 500 | 200 | HypersphereART | 0.9 | 22176 | 22.55 | 500 |

## Performance Analysis

### Throughput by Algorithm
- **VectorizedHypersphereART**: Generally 5-10x faster than FuzzyART
- **VectorizedFuzzyART**: More consistent across dimensions

### Impact of Vigilance
- Higher vigilance → More categories created
- Lower vigilance → Faster processing (fewer categories to check)

### Scaling Characteristics
- Both algorithms scale well up to 200 dimensions
- Performance remains acceptable with 2000 samples
