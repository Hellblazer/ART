# ART Algorithm Real-World Performance Report

Generated: 2025-09-06T19:43:24.678915

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
- Time: 5.80 ms
- Throughput: 86144 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.10 ms
- Throughput: 4866180 samples/sec
- Categories Created: 0
- Final Category Count: 23

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 0.51 ms
- Throughput: 971268 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 0.07 ms
- Throughput: 7412019 samples/sec
- Categories Created: 0
- Final Category Count: 9

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 0.32 ms
- Throughput: 1538660 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.07 ms
- Throughput: 7570940 samples/sec
- Categories Created: 0
- Final Category Count: 9

## Medium Dataset - Medium Dimensions

- Data Size: 10000
- Dimensions: 50

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 12.75 ms
- Throughput: 39203 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.87 ms
- Throughput: 571674 samples/sec
- Categories Created: 0
- Final Category Count: 95

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 1.39 ms
- Throughput: 359723 samples/sec
- Categories Created: 0
- Final Category Count: 18

**VectorizedHypersphereART Results:**
- Time: 0.19 ms
- Throughput: 2677376 samples/sec
- Categories Created: 0
- Final Category Count: 12

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 0.89 ms
- Throughput: 560591 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.11 ms
- Throughput: 4680202 samples/sec
- Categories Created: 0
- Final Category Count: 5

## Large Dataset - High Dimensions

- Data Size: 50000
- Dimensions: 100

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 24.02 ms
- Throughput: 20815 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 11.38 ms
- Throughput: 43931 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 2.29 ms
- Throughput: 218146 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 0.98 ms
- Throughput: 511335 samples/sec
- Categories Created: 0
- Final Category Count: 41

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 1.74 ms
- Throughput: 287095 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.26 ms
- Throughput: 1930196 samples/sec
- Categories Created: 0
- Final Category Count: 7

## Image Recognition Simulation

- Data Size: 5000
- Dimensions: 784

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 110.55 ms
- Throughput: 4523 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 121.79 ms
- Throughput: 4105 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 17.72 ms
- Throughput: 28210 samples/sec
- Categories Created: 0
- Final Category Count: 19

**VectorizedHypersphereART Results:**
- Time: 121.10 ms
- Throughput: 4129 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 13.66 ms
- Throughput: 36612 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 11.85 ms
- Throughput: 42197 samples/sec
- Categories Created: 0
- Final Category Count: 50

## Sensor Data Processing

- Data Size: 100000
- Dimensions: 32

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 8.51 ms
- Throughput: 58742 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.33 ms
- Throughput: 1522844 samples/sec
- Categories Created: 0
- Final Category Count: 50

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 0.94 ms
- Throughput: 531185 samples/sec
- Categories Created: 0
- Final Category Count: 19

**VectorizedHypersphereART Results:**
- Time: 0.11 ms
- Throughput: 4660223 samples/sec
- Categories Created: 0
- Final Category Count: 8

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 0.67 ms
- Throughput: 749158 samples/sec
- Categories Created: 0
- Final Category Count: 10

**VectorizedHypersphereART Results:**
- Time: 0.10 ms
- Throughput: 4769445 samples/sec
- Categories Created: 0
- Final Category Count: 8

## Text Embedding Clustering

- Data Size: 20000
- Dimensions: 300

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 45.46 ms
- Throughput: 10998 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 39.27 ms
- Throughput: 12734 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 6.26 ms
- Throughput: 79870 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 3.53 ms
- Throughput: 141802 samples/sec
- Categories Created: 0
- Final Category Count: 42

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 5.04 ms
- Throughput: 99270 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 3.55 ms
- Throughput: 140865 samples/sec
- Categories Created: 0
- Final Category Count: 42

