# ART Algorithm Real-World Performance Report

Generated: 2025-09-06T18:54:39.906395

## Test Environment
- Java Version: 24
- OS: Mac OS X
- Available Processors: 16
- Max Memory: 512 MB

## Small Dataset - Low Dimensions

- Data Size: 1000
- Dimensions: 10

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 5.69 ms
- Throughput: 87802 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.10 ms
- Throughput: 5020887 samples/sec
- Categories Created: 0
- Final Category Count: 23

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 0.58 ms
- Throughput: 858369 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 0.07 ms
- Throughput: 7121696 samples/sec
- Categories Created: 0
- Final Category Count: 9

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 0.38 ms
- Throughput: 1320132 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.07 ms
- Throughput: 6745363 samples/sec
- Categories Created: 0
- Final Category Count: 9

## Medium Dataset - Medium Dimensions

- Data Size: 10000
- Dimensions: 50

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 11.54 ms
- Throughput: 43332 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.88 ms
- Throughput: 570179 samples/sec
- Categories Created: 0
- Final Category Count: 95

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 1.26 ms
- Throughput: 395400 samples/sec
- Categories Created: 0
- Final Category Count: 18

**VectorizedHypersphereART Results:**
- Time: 0.19 ms
- Throughput: 2679773 samples/sec
- Categories Created: 0
- Final Category Count: 12

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 1.02 ms
- Throughput: 492227 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.12 ms
- Throughput: 4062266 samples/sec
- Categories Created: 0
- Final Category Count: 5

## Large Dataset - High Dimensions

- Data Size: 50000
- Dimensions: 100

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 21.78 ms
- Throughput: 22961 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 11.13 ms
- Throughput: 44932 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 2.29 ms
- Throughput: 218245 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 0.98 ms
- Throughput: 512120 samples/sec
- Categories Created: 0
- Final Category Count: 41

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 1.75 ms
- Throughput: 285694 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.25 ms
- Throughput: 1964312 samples/sec
- Categories Created: 0
- Final Category Count: 7

## Image Recognition Simulation

- Data Size: 5000
- Dimensions: 784

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 109.31 ms
- Throughput: 4574 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 120.73 ms
- Throughput: 4141 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 17.91 ms
- Throughput: 27914 samples/sec
- Categories Created: 0
- Final Category Count: 19

**VectorizedHypersphereART Results:**
- Time: 120.96 ms
- Throughput: 4134 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 13.52 ms
- Throughput: 36991 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 11.89 ms
- Throughput: 42051 samples/sec
- Categories Created: 0
- Final Category Count: 50

## Sensor Data Processing

- Data Size: 100000
- Dimensions: 32

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 8.73 ms
- Throughput: 57267 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.33 ms
- Throughput: 1525553 samples/sec
- Categories Created: 0
- Final Category Count: 50

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 0.93 ms
- Throughput: 535141 samples/sec
- Categories Created: 0
- Final Category Count: 19

**VectorizedHypersphereART Results:**
- Time: 0.10 ms
- Throughput: 4946430 samples/sec
- Categories Created: 0
- Final Category Count: 8

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 0.66 ms
- Throughput: 752350 samples/sec
- Categories Created: 0
- Final Category Count: 10

**VectorizedHypersphereART Results:**
- Time: 0.10 ms
- Throughput: 5006258 samples/sec
- Categories Created: 0
- Final Category Count: 8

## Text Embedding Clustering

- Data Size: 20000
- Dimensions: 300

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 45.55 ms
- Throughput: 10976 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 39.57 ms
- Throughput: 12637 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 6.21 ms
- Throughput: 80577 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 3.54 ms
- Throughput: 141329 samples/sec
- Categories Created: 0
- Final Category Count: 42

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 5.03 ms
- Throughput: 99314 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 3.52 ms
- Throughput: 141908 samples/sec
- Categories Created: 0
- Final Category Count: 42

