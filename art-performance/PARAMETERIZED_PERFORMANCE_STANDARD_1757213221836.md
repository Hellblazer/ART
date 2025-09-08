# Parameterized Performance Report - STANDARD Scale

Generated: 2025-09-06T19:47:01.606468

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
| Small | 500 | 10 | FuzzyART | 0.5 | 1590457 | 0.31 | 12 |
| Small | 500 | 10 | HypersphereART | 0.5 | 11152499 | 0.04 | 7 |
| Small | 500 | 10 | FuzzyART | 0.7 | 1362862 | 0.37 | 15 |
| Small | 500 | 10 | HypersphereART | 0.7 | 11059989 | 0.05 | 7 |
| Small | 500 | 10 | FuzzyART | 0.9 | 136047 | 3.68 | 138 |
| Small | 500 | 10 | HypersphereART | 0.9 | 6586140 | 0.08 | 18 |
| Medium | 1000 | 50 | FuzzyART | 0.5 | 307684 | 1.63 | 21 |
| Medium | 1000 | 50 | HypersphereART | 0.5 | 4470952 | 0.11 | 5 |
| Medium | 1000 | 50 | FuzzyART | 0.7 | 386785 | 1.29 | 18 |
| Medium | 1000 | 50 | HypersphereART | 0.7 | 2747253 | 0.18 | 12 |
| Medium | 1000 | 50 | FuzzyART | 0.9 | 25087 | 19.93 | 236 |
| Medium | 1000 | 50 | HypersphereART | 0.9 | 583175 | 0.86 | 95 |
| Large | 2000 | 100 | FuzzyART | 0.5 | 122748 | 4.07 | 23 |
| Large | 2000 | 100 | HypersphereART | 0.5 | 2092409 | 0.24 | 6 |
| Large | 2000 | 100 | FuzzyART | 0.7 | 144161 | 3.47 | 21 |
| Large | 2000 | 100 | HypersphereART | 0.7 | 281657 | 1.78 | 80 |
| Large | 2000 | 100 | FuzzyART | 0.9 | 11057 | 45.22 | 293 |
| Large | 2000 | 100 | HypersphereART | 0.9 | 45264 | 11.05 | 499 |
| HighDim | 500 | 200 | FuzzyART | 0.5 | 103848 | 4.81 | 16 |
| HighDim | 500 | 200 | HypersphereART | 0.5 | 350007 | 1.43 | 28 |
| HighDim | 500 | 200 | FuzzyART | 0.7 | 143282 | 3.49 | 15 |
| HighDim | 500 | 200 | HypersphereART | 0.7 | 352578 | 1.42 | 28 |
| HighDim | 500 | 200 | FuzzyART | 0.9 | 6421 | 77.87 | 287 |
| HighDim | 500 | 200 | HypersphereART | 0.9 | 21457 | 23.30 | 500 |

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
