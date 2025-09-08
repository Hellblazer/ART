# ART Algorithm Performance Report - ACTUAL MEASUREMENTS

Generated: 2025-09-06T20:15:30.430939

## Test Environment
- Java Version: 24
- OS: Mac OS X aarch64
- Available Processors: 16
- Max Memory: 512 MB

## Summary of Results

| Scenario | Algorithm | Throughput (patterns/sec) | Time (ms) | Categories |
|----------|-----------|---------------------------|-----------|------------|
| Small Dataset | VectorizedFuzzyART | 1257205 | 0.16 | 15 |
| Small Dataset | VectorizedHypersphereART | 5853601 | 0.03 | 7 |
| Medium Dataset | VectorizedFuzzyART | 479663 | 0.42 | 11 |
| Medium Dataset | VectorizedHypersphereART | 2872531 | 0.07 | 10 |
| Large Dataset | VectorizedFuzzyART | 247627 | 0.81 | 10 |
| Large Dataset | VectorizedHypersphereART | 579989 | 0.34 | 29 |
| High Dimensional | VectorizedFuzzyART | 127382 | 1.57 | 10 |
| High Dimensional | VectorizedHypersphereART | 269784 | 0.74 | 36 |

# ART Algorithm Performance Report - ACTUAL MEASUREMENTS

Generated: 2025-09-06T20:15:29.980428

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
- Throughput: 1257205 patterns/sec
- Categories Created: 0
- Final Category Count: 15

### VectorizedHypersphereART Results
- Time: 0.03 ms
- Throughput: 5853601 patterns/sec
- Categories Created: 0
- Final Category Count: 7


## Medium Dataset

- Data Size: 1000
- Dimensions: 50

### VectorizedFuzzyART Results
- Time: 0.42 ms
- Throughput: 479663 patterns/sec
- Categories Created: 0
- Final Category Count: 11

### VectorizedHypersphereART Results
- Time: 0.07 ms
- Throughput: 2872531 patterns/sec
- Categories Created: 0
- Final Category Count: 10


## Large Dataset

- Data Size: 2000
- Dimensions: 100

### VectorizedFuzzyART Results
- Time: 0.81 ms
- Throughput: 247627 patterns/sec
- Categories Created: 0
- Final Category Count: 10

### VectorizedHypersphereART Results
- Time: 0.34 ms
- Throughput: 579989 patterns/sec
- Categories Created: 0
- Final Category Count: 29


## High Dimensional

- Data Size: 500
- Dimensions: 200

### VectorizedFuzzyART Results
- Time: 1.57 ms
- Throughput: 127382 patterns/sec
- Categories Created: 0
- Final Category Count: 10

### VectorizedHypersphereART Results
- Time: 0.74 ms
- Throughput: 269784 patterns/sec
- Categories Created: 0
- Final Category Count: 36

