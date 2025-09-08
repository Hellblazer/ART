# ART Algorithm Performance Report - ACTUAL MEASUREMENTS

Generated: 2025-09-06T19:20:29.609479

## Test Environment
- Java Version: 24
- OS: Mac OS X aarch64
- Available Processors: 16
- Max Memory: 512 MB

## Summary of Results

| Scenario | Algorithm | Throughput (patterns/sec) | Time (ms) | Categories |
|----------|-----------|---------------------------|-----------|------------|
| Small Dataset | VectorizedFuzzyART | 1220137 | 0.16 | 15 |
| Small Dataset | VectorizedHypersphereART | 6075888 | 0.03 | 7 |
| Medium Dataset | VectorizedFuzzyART | 493117 | 0.41 | 11 |
| Medium Dataset | VectorizedHypersphereART | 2855430 | 0.07 | 10 |
| Large Dataset | VectorizedFuzzyART | 252299 | 0.79 | 10 |
| Large Dataset | VectorizedHypersphereART | 607517 | 0.33 | 29 |
| High Dimensional | VectorizedFuzzyART | 130740 | 1.53 | 10 |
| High Dimensional | VectorizedHypersphereART | 277601 | 0.72 | 36 |

# ART Algorithm Performance Report - ACTUAL MEASUREMENTS

Generated: 2025-09-06T19:20:29.157802

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
- Throughput: 1220137 patterns/sec
- Categories Created: 0
- Final Category Count: 15

### VectorizedHypersphereART Results
- Time: 0.03 ms
- Throughput: 6075888 patterns/sec
- Categories Created: 0
- Final Category Count: 7


## Medium Dataset

- Data Size: 1000
- Dimensions: 50

### VectorizedFuzzyART Results
- Time: 0.41 ms
- Throughput: 493117 patterns/sec
- Categories Created: 0
- Final Category Count: 11

### VectorizedHypersphereART Results
- Time: 0.07 ms
- Throughput: 2855430 patterns/sec
- Categories Created: 0
- Final Category Count: 10


## Large Dataset

- Data Size: 2000
- Dimensions: 100

### VectorizedFuzzyART Results
- Time: 0.79 ms
- Throughput: 252299 patterns/sec
- Categories Created: 0
- Final Category Count: 10

### VectorizedHypersphereART Results
- Time: 0.33 ms
- Throughput: 607517 patterns/sec
- Categories Created: 0
- Final Category Count: 29


## High Dimensional

- Data Size: 500
- Dimensions: 200

### VectorizedFuzzyART Results
- Time: 1.53 ms
- Throughput: 130740 patterns/sec
- Categories Created: 0
- Final Category Count: 10

### VectorizedHypersphereART Results
- Time: 0.72 ms
- Throughput: 277601 patterns/sec
- Categories Created: 0
- Final Category Count: 36

