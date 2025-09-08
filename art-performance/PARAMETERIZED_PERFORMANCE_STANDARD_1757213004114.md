# Parameterized Performance Report - STANDARD Scale

Generated: 2025-09-06T19:43:23.880943

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
| Small | 500 | 10 | FuzzyART | 0.5 | 1596378 | 0.31 | 12 |
| Small | 500 | 10 | HypersphereART | 0.5 | 10498688 | 0.05 | 7 |
| Small | 500 | 10 | FuzzyART | 0.7 | 1269438 | 0.39 | 15 |
| Small | 500 | 10 | HypersphereART | 0.7 | 10676233 | 0.05 | 7 |
| Small | 500 | 10 | FuzzyART | 0.9 | 135622 | 3.69 | 138 |
| Small | 500 | 10 | HypersphereART | 0.9 | 6292633 | 0.08 | 18 |
| Medium | 1000 | 50 | FuzzyART | 0.5 | 304352 | 1.64 | 21 |
| Medium | 1000 | 50 | HypersphereART | 0.5 | 4547191 | 0.11 | 5 |
| Medium | 1000 | 50 | FuzzyART | 0.7 | 374520 | 1.34 | 18 |
| Medium | 1000 | 50 | HypersphereART | 0.7 | 2503643 | 0.20 | 12 |
| Medium | 1000 | 50 | FuzzyART | 0.9 | 24510 | 20.40 | 236 |
| Medium | 1000 | 50 | HypersphereART | 0.9 | 559779 | 0.89 | 95 |
| Large | 2000 | 100 | FuzzyART | 0.5 | 103513 | 4.83 | 23 |
| Large | 2000 | 100 | HypersphereART | 0.5 | 2135994 | 0.23 | 6 |
| Large | 2000 | 100 | FuzzyART | 0.7 | 140883 | 3.55 | 21 |
| Large | 2000 | 100 | HypersphereART | 0.7 | 277034 | 1.80 | 80 |
| Large | 2000 | 100 | FuzzyART | 0.9 | 10963 | 45.61 | 293 |
| Large | 2000 | 100 | HypersphereART | 0.9 | 45166 | 11.07 | 499 |
| HighDim | 500 | 200 | FuzzyART | 0.5 | 102159 | 4.89 | 16 |
| HighDim | 500 | 200 | HypersphereART | 0.5 | 350416 | 1.43 | 28 |
| HighDim | 500 | 200 | FuzzyART | 0.7 | 136277 | 3.67 | 15 |
| HighDim | 500 | 200 | HypersphereART | 0.7 | 362900 | 1.38 | 28 |
| HighDim | 500 | 200 | FuzzyART | 0.9 | 6388 | 78.27 | 287 |
| HighDim | 500 | 200 | HypersphereART | 0.9 | 21269 | 23.51 | 500 |

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
