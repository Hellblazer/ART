# ART Algorithm Performance Report - ACTUAL MEASUREMENTS

Generated: 2025-09-06T19:43:28.017036

## Test Environment
- Java Version: 24
- OS: Mac OS X aarch64
- Available Processors: 16
- Max Memory: 512 MB

## Summary of Results

| Scenario | Algorithm | Throughput (patterns/sec) | Time (ms) | Categories |
|----------|-----------|---------------------------|-----------|------------|
| Small Dataset | VectorizedFuzzyART | 1232035 | 0.16 | 15 |
| Small Dataset | VectorizedHypersphereART | 6153846 | 0.03 | 7 |
| Medium Dataset | VectorizedFuzzyART | 473140 | 0.42 | 11 |
| Medium Dataset | VectorizedHypersphereART | 2771388 | 0.07 | 10 |
| Large Dataset | VectorizedFuzzyART | 251401 | 0.80 | 10 |
| Large Dataset | VectorizedHypersphereART | 615779 | 0.32 | 29 |
| High Dimensional | VectorizedFuzzyART | 130230 | 1.54 | 10 |
| High Dimensional | VectorizedHypersphereART | 275767 | 0.73 | 36 |

# ART Algorithm Performance Report - ACTUAL MEASUREMENTS

Generated: 2025-09-06T19:43:27.570152

## Test Environment
- Java Version: 24
- OS: Mac OS X aarch64
- Available Processors: 16
- Max Memory: 512 MB

## Summary of Results

| Scenario | Algorithm | Throughput (patterns/sec) | Time (ms) | Categories |
|----------|-----------|---------------------------|-----------|------------|

## Small Dataset

- Data Size: 500
- Dimensions: 10

### VectorizedFuzzyART Results
- Time: 0.16 ms
- Throughput: 1232035 patterns/sec
- Categories Created: 0
- Final Category Count: 15

### VectorizedHypersphereART Results
- Time: 0.03 ms
- Throughput: 6153846 patterns/sec
- Categories Created: 0
- Final Category Count: 7


## Medium Dataset

- Data Size: 1000
- Dimensions: 50

### VectorizedFuzzyART Results
- Time: 0.42 ms
- Throughput: 473140 patterns/sec
- Categories Created: 0
- Final Category Count: 11

### VectorizedHypersphereART Results
- Time: 0.07 ms
- Throughput: 2771388 patterns/sec
- Categories Created: 0
- Final Category Count: 10


## Large Dataset

- Data Size: 2000
- Dimensions: 100

### VectorizedFuzzyART Results
- Time: 0.80 ms
- Throughput: 251401 patterns/sec
- Categories Created: 0
- Final Category Count: 10

### VectorizedHypersphereART Results
- Time: 0.32 ms
- Throughput: 615779 patterns/sec
- Categories Created: 0
- Final Category Count: 29


## High Dimensional

- Data Size: 500
- Dimensions: 200

### VectorizedFuzzyART Results
- Time: 1.54 ms
- Throughput: 130230 patterns/sec
- Categories Created: 0
- Final Category Count: 10

### VectorizedHypersphereART Results
- Time: 0.73 ms
- Throughput: 275767 patterns/sec
- Categories Created: 0
- Final Category Count: 36

