# ART Algorithm Real-World Performance Report

Generated: 2025-09-06T20:15:27.080949

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
- Time: 5.65 ms
- Throughput: 88542 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.10 ms
- Throughput: 5104228 samples/sec
- Categories Created: 0
- Final Category Count: 23

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 0.57 ms
- Throughput: 877770 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 0.07 ms
- Throughput: 7575758 samples/sec
- Categories Created: 0
- Final Category Count: 9

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 0.38 ms
- Throughput: 1310616 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.07 ms
- Throughput: 7308125 samples/sec
- Categories Created: 0
- Final Category Count: 9

## Medium Dataset - Medium Dimensions

- Data Size: 10000
- Dimensions: 50

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 12.82 ms
- Throughput: 39010 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.91 ms
- Throughput: 549300 samples/sec
- Categories Created: 0
- Final Category Count: 95

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 1.86 ms
- Throughput: 268871 samples/sec
- Categories Created: 0
- Final Category Count: 18

**VectorizedHypersphereART Results:**
- Time: 0.24 ms
- Throughput: 2120135 samples/sec
- Categories Created: 0
- Final Category Count: 12

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 1.13 ms
- Throughput: 441047 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.12 ms
- Throughput: 4189956 samples/sec
- Categories Created: 0
- Final Category Count: 5

## Large Dataset - High Dimensions

- Data Size: 50000
- Dimensions: 100

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 26.23 ms
- Throughput: 19064 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 12.88 ms
- Throughput: 38822 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 2.94 ms
- Throughput: 170049 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 1.38 ms
- Throughput: 361500 samples/sec
- Categories Created: 0
- Final Category Count: 41

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 1.86 ms
- Throughput: 268114 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.26 ms
- Throughput: 1934550 samples/sec
- Categories Created: 0
- Final Category Count: 7

## Image Recognition Simulation

- Data Size: 5000
- Dimensions: 784

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 108.87 ms
- Throughput: 4593 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 122.12 ms
- Throughput: 4094 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 18.16 ms
- Throughput: 27526 samples/sec
- Categories Created: 0
- Final Category Count: 19

**VectorizedHypersphereART Results:**
- Time: 121.86 ms
- Throughput: 4103 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 13.54 ms
- Throughput: 36916 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 11.83 ms
- Throughput: 42264 samples/sec
- Categories Created: 0
- Final Category Count: 50

## Sensor Data Processing

- Data Size: 100000
- Dimensions: 32

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 8.52 ms
- Throughput: 58668 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.33 ms
- Throughput: 1531197 samples/sec
- Categories Created: 0
- Final Category Count: 50

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 1.09 ms
- Throughput: 459647 samples/sec
- Categories Created: 0
- Final Category Count: 19

**VectorizedHypersphereART Results:**
- Time: 0.10 ms
- Throughput: 4987531 samples/sec
- Categories Created: 0
- Final Category Count: 8

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 0.76 ms
- Throughput: 661048 samples/sec
- Categories Created: 0
- Final Category Count: 10

**VectorizedHypersphereART Results:**
- Time: 0.10 ms
- Throughput: 5086884 samples/sec
- Categories Created: 0
- Final Category Count: 8

## Text Embedding Clustering

- Data Size: 20000
- Dimensions: 300

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 44.73 ms
- Throughput: 11179 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 39.02 ms
- Throughput: 12813 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 6.35 ms
- Throughput: 78700 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 3.49 ms
- Throughput: 143328 samples/sec
- Categories Created: 0
- Final Category Count: 42

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 5.14 ms
- Throughput: 97274 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 3.54 ms
- Throughput: 141196 samples/sec
- Categories Created: 0
- Final Category Count: 42

