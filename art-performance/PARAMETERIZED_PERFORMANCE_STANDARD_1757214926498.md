# Parameterized Performance Report - STANDARD Scale

Generated: 2025-09-06T20:15:26.269261

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
| Small | 500 | 10 | FuzzyART | 0.5 | 1594474 | 0.31 | 12 |
| Small | 500 | 10 | HypersphereART | 0.5 | 11183681 | 0.04 | 7 |
| Small | 500 | 10 | FuzzyART | 0.7 | 1253133 | 0.40 | 15 |
| Small | 500 | 10 | HypersphereART | 0.7 | 10939004 | 0.05 | 7 |
| Small | 500 | 10 | FuzzyART | 0.9 | 136136 | 3.67 | 138 |
| Small | 500 | 10 | HypersphereART | 0.9 | 6420546 | 0.08 | 18 |
| Medium | 1000 | 50 | FuzzyART | 0.5 | 313725 | 1.59 | 21 |
| Medium | 1000 | 50 | HypersphereART | 0.5 | 4480969 | 0.11 | 5 |
| Medium | 1000 | 50 | FuzzyART | 0.7 | 383509 | 1.30 | 18 |
| Medium | 1000 | 50 | HypersphereART | 0.7 | 2771357 | 0.18 | 12 |
| Medium | 1000 | 50 | FuzzyART | 0.9 | 24696 | 20.25 | 236 |
| Medium | 1000 | 50 | HypersphereART | 0.9 | 583999 | 0.86 | 95 |
| Large | 2000 | 100 | FuzzyART | 0.5 | 126009 | 3.97 | 23 |
| Large | 2000 | 100 | HypersphereART | 0.5 | 2219756 | 0.23 | 6 |
| Large | 2000 | 100 | FuzzyART | 0.7 | 144335 | 3.46 | 21 |
| Large | 2000 | 100 | HypersphereART | 0.7 | 283119 | 1.77 | 80 |
| Large | 2000 | 100 | FuzzyART | 0.9 | 11158 | 44.81 | 293 |
| Large | 2000 | 100 | HypersphereART | 0.9 | 44971 | 11.12 | 499 |
| HighDim | 500 | 200 | FuzzyART | 0.5 | 104499 | 4.78 | 16 |
| HighDim | 500 | 200 | HypersphereART | 0.5 | 351813 | 1.42 | 28 |
| HighDim | 500 | 200 | FuzzyART | 0.7 | 139095 | 3.59 | 15 |
| HighDim | 500 | 200 | HypersphereART | 0.7 | 353607 | 1.41 | 28 |
| HighDim | 500 | 200 | FuzzyART | 0.9 | 6376 | 78.42 | 287 |
| HighDim | 500 | 200 | HypersphereART | 0.9 | 21689 | 23.05 | 500 |

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
