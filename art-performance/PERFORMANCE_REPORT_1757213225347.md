# ART Algorithm Real-World Performance Report

Generated: 2025-09-06T19:47:02.465843

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
- Time: 5.26 ms
- Throughput: 95065 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.10 ms
- Throughput: 5143557 samples/sec
- Categories Created: 0
- Final Category Count: 23

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 0.49 ms
- Throughput: 1028013 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 0.07 ms
- Throughput: 7667771 samples/sec
- Categories Created: 0
- Final Category Count: 9

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 0.38 ms
- Throughput: 1300251 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.07 ms
- Throughput: 7604563 samples/sec
- Categories Created: 0
- Final Category Count: 9

## Medium Dataset - Medium Dimensions

- Data Size: 10000
- Dimensions: 50

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 11.93 ms
- Throughput: 41913 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.85 ms
- Throughput: 585081 samples/sec
- Categories Created: 0
- Final Category Count: 95

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 1.37 ms
- Throughput: 364564 samples/sec
- Categories Created: 0
- Final Category Count: 18

**VectorizedHypersphereART Results:**
- Time: 0.18 ms
- Throughput: 2734108 samples/sec
- Categories Created: 0
- Final Category Count: 12

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 1.02 ms
- Throughput: 489397 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.12 ms
- Throughput: 4198752 samples/sec
- Categories Created: 0
- Final Category Count: 5

## Large Dataset - High Dimensions

- Data Size: 50000
- Dimensions: 100

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 21.84 ms
- Throughput: 22892 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 11.23 ms
- Throughput: 44527 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 2.28 ms
- Throughput: 219166 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 0.98 ms
- Throughput: 508776 samples/sec
- Categories Created: 0
- Final Category Count: 41

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 1.76 ms
- Throughput: 283896 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.26 ms
- Throughput: 1919386 samples/sec
- Categories Created: 0
- Final Category Count: 7

## Image Recognition Simulation

- Data Size: 5000
- Dimensions: 784

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 111.12 ms
- Throughput: 4500 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 121.19 ms
- Throughput: 4126 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 18.04 ms
- Throughput: 27717 samples/sec
- Categories Created: 0
- Final Category Count: 19

**VectorizedHypersphereART Results:**
- Time: 121.77 ms
- Throughput: 4106 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 13.06 ms
- Throughput: 38288 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 11.81 ms
- Throughput: 42332 samples/sec
- Categories Created: 0
- Final Category Count: 50

## Sensor Data Processing

- Data Size: 100000
- Dimensions: 32

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 8.66 ms
- Throughput: 57742 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.33 ms
- Throughput: 1527496 samples/sec
- Categories Created: 0
- Final Category Count: 50

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 0.93 ms
- Throughput: 536672 samples/sec
- Categories Created: 0
- Final Category Count: 19

**VectorizedHypersphereART Results:**
- Time: 0.10 ms
- Throughput: 5046274 samples/sec
- Categories Created: 0
- Final Category Count: 8

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 0.66 ms
- Throughput: 752776 samples/sec
- Categories Created: 0
- Final Category Count: 10

**VectorizedHypersphereART Results:**
- Time: 0.10 ms
- Throughput: 5089059 samples/sec
- Categories Created: 0
- Final Category Count: 8

## Text Embedding Clustering

- Data Size: 20000
- Dimensions: 300

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 45.65 ms
- Throughput: 10952 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 39.93 ms
- Throughput: 12522 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 6.14 ms
- Throughput: 81399 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 3.36 ms
- Throughput: 148789 samples/sec
- Categories Created: 0
- Final Category Count: 42

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 5.01 ms
- Throughput: 99787 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 3.54 ms
- Throughput: 141113 samples/sec
- Categories Created: 0
- Final Category Count: 42

