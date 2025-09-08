# Parameterized Performance Report - STANDARD Scale

Generated: 2025-09-06T18:54:39.082499

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
| Small | 500 | 10 | FuzzyART | 0.5 | 1563922 | 0.32 | 12 |
| Small | 500 | 10 | HypersphereART | 0.5 | 12473805 | 0.04 | 7 |
| Small | 500 | 10 | FuzzyART | 0.7 | 1283562 | 0.39 | 15 |
| Small | 500 | 10 | HypersphereART | 0.7 | 10791445 | 0.05 | 7 |
| Small | 500 | 10 | FuzzyART | 0.9 | 131893 | 3.79 | 138 |
| Small | 500 | 10 | HypersphereART | 0.9 | 6546645 | 0.08 | 18 |
| Medium | 1000 | 50 | FuzzyART | 0.5 | 307235 | 1.63 | 21 |
| Medium | 1000 | 50 | HypersphereART | 0.5 | 4540295 | 0.11 | 5 |
| Medium | 1000 | 50 | FuzzyART | 0.7 | 381291 | 1.31 | 18 |
| Medium | 1000 | 50 | HypersphereART | 0.7 | 2802424 | 0.18 | 12 |
| Medium | 1000 | 50 | FuzzyART | 0.9 | 23979 | 20.85 | 236 |
| Medium | 1000 | 50 | HypersphereART | 0.9 | 580607 | 0.86 | 95 |
| Large | 2000 | 100 | FuzzyART | 0.5 | 121151 | 4.13 | 23 |
| Large | 2000 | 100 | HypersphereART | 0.5 | 2430630 | 0.21 | 6 |
| Large | 2000 | 100 | FuzzyART | 0.7 | 141064 | 3.54 | 21 |
| Large | 2000 | 100 | HypersphereART | 0.7 | 279174 | 1.79 | 80 |
| Large | 2000 | 100 | FuzzyART | 0.9 | 10972 | 45.57 | 293 |
| Large | 2000 | 100 | HypersphereART | 0.9 | 46256 | 10.81 | 499 |
| HighDim | 500 | 200 | FuzzyART | 0.5 | 99813 | 5.01 | 16 |
| HighDim | 500 | 200 | HypersphereART | 0.5 | 352993 | 1.42 | 28 |
| HighDim | 500 | 200 | FuzzyART | 0.7 | 135774 | 3.68 | 15 |
| HighDim | 500 | 200 | HypersphereART | 0.7 | 351720 | 1.42 | 28 |
| HighDim | 500 | 200 | FuzzyART | 0.9 | 6368 | 78.51 | 287 |
| HighDim | 500 | 200 | HypersphereART | 0.9 | 21620 | 23.13 | 500 |

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
