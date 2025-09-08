# ART Algorithm Performance Report - ACTUAL MEASUREMENTS

Generated: 2025-09-06T19:47:05.801850

## Test Environment
- Java Version: 24
- OS: Mac OS X aarch64
- Available Processors: 16
- Max Memory: 512 MB

## Summary of Results

| Scenario | Algorithm | Throughput (patterns/sec) | Time (ms) | Categories |
|----------|-----------|---------------------------|-----------|------------|
| Small Dataset | VectorizedFuzzyART | 1223242 | 0.16 | 15 |
| Small Dataset | VectorizedHypersphereART | 6068329 | 0.03 | 7 |
| Medium Dataset | VectorizedFuzzyART | 468293 | 0.43 | 11 |
| Medium Dataset | VectorizedHypersphereART | 2532960 | 0.08 | 10 |
| Large Dataset | VectorizedFuzzyART | 252207 | 0.79 | 10 |
| Large Dataset | VectorizedHypersphereART | 618079 | 0.32 | 29 |
| High Dimensional | VectorizedFuzzyART | 129747 | 1.54 | 10 |
| High Dimensional | VectorizedHypersphereART | 275040 | 0.73 | 36 |

# ART Algorithm Performance Report - ACTUAL MEASUREMENTS

Generated: 2025-09-06T19:47:05.348919

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
- Throughput: 1223242 patterns/sec
- Categories Created: 0
- Final Category Count: 15

### VectorizedHypersphereART Results
- Time: 0.03 ms
- Throughput: 6068329 patterns/sec
- Categories Created: 0
- Final Category Count: 7


## Medium Dataset

- Data Size: 1000
- Dimensions: 50

### VectorizedFuzzyART Results
- Time: 0.43 ms
- Throughput: 468293 patterns/sec
- Categories Created: 0
- Final Category Count: 11

### VectorizedHypersphereART Results
- Time: 0.08 ms
- Throughput: 2532960 patterns/sec
- Categories Created: 0
- Final Category Count: 10


## Large Dataset

- Data Size: 2000
- Dimensions: 100

### VectorizedFuzzyART Results
- Time: 0.79 ms
- Throughput: 252207 patterns/sec
- Categories Created: 0
- Final Category Count: 10

### VectorizedHypersphereART Results
- Time: 0.32 ms
- Throughput: 618079 patterns/sec
- Categories Created: 0
- Final Category Count: 29


## High Dimensional

- Data Size: 500
- Dimensions: 200

### VectorizedFuzzyART Results
- Time: 1.54 ms
- Throughput: 129747 patterns/sec
- Categories Created: 0
- Final Category Count: 10

### VectorizedHypersphereART Results
- Time: 0.73 ms
- Throughput: 275040 patterns/sec
- Categories Created: 0
- Final Category Count: 36

