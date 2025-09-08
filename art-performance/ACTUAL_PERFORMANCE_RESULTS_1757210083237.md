# ART Algorithm Performance Report - ACTUAL MEASUREMENTS

Generated: 2025-09-06T18:54:43.237829

## Test Environment
- Java Version: 24
- OS: Mac OS X aarch64
- Available Processors: 16
- Max Memory: 512 MB

## Summary of Results

| Scenario | Algorithm | Throughput (patterns/sec) | Time (ms) | Categories |
|----------|-----------|---------------------------|-----------|------------|
| Small Dataset | VectorizedFuzzyART | 1367914 | 0.15 | 15 |
| Small Dataset | VectorizedHypersphereART | 6114525 | 0.03 | 7 |
| Medium Dataset | VectorizedFuzzyART | 483774 | 0.41 | 11 |
| Medium Dataset | VectorizedHypersphereART | 2593193 | 0.08 | 10 |
| Large Dataset | VectorizedFuzzyART | 236349 | 0.85 | 10 |
| Large Dataset | VectorizedHypersphereART | 619034 | 0.32 | 29 |
| High Dimensional | VectorizedFuzzyART | 128150 | 1.56 | 10 |
| High Dimensional | VectorizedHypersphereART | 281096 | 0.71 | 36 |

# ART Algorithm Performance Report - ACTUAL MEASUREMENTS

Generated: 2025-09-06T18:54:42.787175

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
- Time: 0.15 ms
- Throughput: 1367914 patterns/sec
- Categories Created: 0
- Final Category Count: 15

### VectorizedHypersphereART Results
- Time: 0.03 ms
- Throughput: 6114525 patterns/sec
- Categories Created: 0
- Final Category Count: 7


## Medium Dataset

- Data Size: 1000
- Dimensions: 50

### VectorizedFuzzyART Results
- Time: 0.41 ms
- Throughput: 483774 patterns/sec
- Categories Created: 0
- Final Category Count: 11

### VectorizedHypersphereART Results
- Time: 0.08 ms
- Throughput: 2593193 patterns/sec
- Categories Created: 0
- Final Category Count: 10


## Large Dataset

- Data Size: 2000
- Dimensions: 100

### VectorizedFuzzyART Results
- Time: 0.85 ms
- Throughput: 236349 patterns/sec
- Categories Created: 0
- Final Category Count: 10

### VectorizedHypersphereART Results
- Time: 0.32 ms
- Throughput: 619034 patterns/sec
- Categories Created: 0
- Final Category Count: 29


## High Dimensional

- Data Size: 500
- Dimensions: 200

### VectorizedFuzzyART Results
- Time: 1.56 ms
- Throughput: 128150 patterns/sec
- Categories Created: 0
- Final Category Count: 10

### VectorizedHypersphereART Results
- Time: 0.71 ms
- Throughput: 281096 patterns/sec
- Categories Created: 0
- Final Category Count: 36

