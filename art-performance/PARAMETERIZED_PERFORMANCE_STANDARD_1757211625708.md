# Parameterized Performance Report - STANDARD Scale

Generated: 2025-09-06T19:20:25.481087

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
| Small | 500 | 10 | FuzzyART | 0.5 | 1609440 | 0.31 | 12 |
| Small | 500 | 10 | HypersphereART | 0.5 | 12539185 | 0.04 | 7 |
| Small | 500 | 10 | FuzzyART | 0.7 | 1499062 | 0.33 | 15 |
| Small | 500 | 10 | HypersphereART | 0.7 | 11142061 | 0.04 | 7 |
| Small | 500 | 10 | FuzzyART | 0.9 | 139937 | 3.57 | 138 |
| Small | 500 | 10 | HypersphereART | 0.9 | 6868887 | 0.07 | 18 |
| Medium | 1000 | 50 | FuzzyART | 0.5 | 327583 | 1.53 | 21 |
| Medium | 1000 | 50 | HypersphereART | 0.5 | 4662005 | 0.11 | 5 |
| Medium | 1000 | 50 | FuzzyART | 0.7 | 396524 | 1.26 | 18 |
| Medium | 1000 | 50 | HypersphereART | 0.7 | 2872177 | 0.17 | 12 |
| Medium | 1000 | 50 | FuzzyART | 0.9 | 24911 | 20.07 | 236 |
| Medium | 1000 | 50 | HypersphereART | 0.9 | 575153 | 0.87 | 95 |
| Large | 2000 | 100 | FuzzyART | 0.5 | 127371 | 3.93 | 23 |
| Large | 2000 | 100 | HypersphereART | 0.5 | 2212801 | 0.23 | 6 |
| Large | 2000 | 100 | FuzzyART | 0.7 | 143104 | 3.49 | 21 |
| Large | 2000 | 100 | HypersphereART | 0.7 | 284731 | 1.76 | 80 |
| Large | 2000 | 100 | FuzzyART | 0.9 | 11023 | 45.36 | 293 |
| Large | 2000 | 100 | HypersphereART | 0.9 | 45066 | 11.09 | 499 |
| HighDim | 500 | 200 | FuzzyART | 0.5 | 103819 | 4.82 | 16 |
| HighDim | 500 | 200 | HypersphereART | 0.5 | 356771 | 1.40 | 28 |
| HighDim | 500 | 200 | FuzzyART | 0.7 | 137920 | 3.63 | 15 |
| HighDim | 500 | 200 | HypersphereART | 0.7 | 357005 | 1.40 | 28 |
| HighDim | 500 | 200 | FuzzyART | 0.9 | 6488 | 77.07 | 287 |
| HighDim | 500 | 200 | HypersphereART | 0.9 | 21737 | 23.00 | 500 |

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
