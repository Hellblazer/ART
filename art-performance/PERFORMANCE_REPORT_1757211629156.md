# ART Algorithm Real-World Performance Report

Generated: 2025-09-06T19:20:26.267469

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
- Time: 5.75 ms
- Throughput: 86890 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.10 ms
- Throughput: 5174644 samples/sec
- Categories Created: 0
- Final Category Count: 23

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 0.60 ms
- Throughput: 837521 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 0.07 ms
- Throughput: 7619048 samples/sec
- Categories Created: 0
- Final Category Count: 9

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 0.40 ms
- Throughput: 1252740 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.06 ms
- Throughput: 7741975 samples/sec
- Categories Created: 0
- Final Category Count: 9

## Medium Dataset - Medium Dimensions

- Data Size: 10000
- Dimensions: 50

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 11.07 ms
- Throughput: 45159 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.88 ms
- Throughput: 567242 samples/sec
- Categories Created: 0
- Final Category Count: 95

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 1.23 ms
- Throughput: 407332 samples/sec
- Categories Created: 0
- Final Category Count: 18

**VectorizedHypersphereART Results:**
- Time: 0.17 ms
- Throughput: 2888087 samples/sec
- Categories Created: 0
- Final Category Count: 12

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 1.00 ms
- Throughput: 500542 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.12 ms
- Throughput: 4182630 samples/sec
- Categories Created: 0
- Final Category Count: 5

## Large Dataset - High Dimensions

- Data Size: 50000
- Dimensions: 100

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 20.86 ms
- Throughput: 23971 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 11.23 ms
- Throughput: 44535 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 2.29 ms
- Throughput: 218583 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 0.98 ms
- Throughput: 511596 samples/sec
- Categories Created: 0
- Final Category Count: 41

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 1.74 ms
- Throughput: 286752 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 0.26 ms
- Throughput: 1952805 samples/sec
- Categories Created: 0
- Final Category Count: 7

## Image Recognition Simulation

- Data Size: 5000
- Dimensions: 784

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 108.21 ms
- Throughput: 4621 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 120.70 ms
- Throughput: 4142 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 17.86 ms
- Throughput: 27998 samples/sec
- Categories Created: 0
- Final Category Count: 19

**VectorizedHypersphereART Results:**
- Time: 121.33 ms
- Throughput: 4121 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 12.94 ms
- Throughput: 38630 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 11.75 ms
- Throughput: 42551 samples/sec
- Categories Created: 0
- Final Category Count: 50

## Sensor Data Processing

- Data Size: 100000
- Dimensions: 32

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 8.55 ms
- Throughput: 58460 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 0.32 ms
- Throughput: 1582694 samples/sec
- Categories Created: 0
- Final Category Count: 50

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 0.90 ms
- Throughput: 554888 samples/sec
- Categories Created: 0
- Final Category Count: 19

**VectorizedHypersphereART Results:**
- Time: 0.11 ms
- Throughput: 4687485 samples/sec
- Categories Created: 0
- Final Category Count: 8

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 0.67 ms
- Throughput: 744278 samples/sec
- Categories Created: 0
- Final Category Count: 10

**VectorizedHypersphereART Results:**
- Time: 0.10 ms
- Throughput: 5054794 samples/sec
- Categories Created: 0
- Final Category Count: 8

## Text Embedding Clustering

- Data Size: 20000
- Dimensions: 300

### Configuration: High Vigilance - Fast Learning

**VectorizedFuzzyART Results:**
- Time: 44.98 ms
- Throughput: 11115 samples/sec
- Categories Created: 0
- Final Category Count: 51

**VectorizedHypersphereART Results:**
- Time: 39.37 ms
- Throughput: 12700 samples/sec
- Categories Created: 0
- Final Category Count: 500

### Configuration: Medium Vigilance - Moderate Learning

**VectorizedFuzzyART Results:**
- Time: 6.24 ms
- Throughput: 80190 samples/sec
- Categories Created: 0
- Final Category Count: 20

**VectorizedHypersphereART Results:**
- Time: 3.53 ms
- Throughput: 141628 samples/sec
- Categories Created: 0
- Final Category Count: 42

### Configuration: Low Vigilance - Slow Learning

**VectorizedFuzzyART Results:**
- Time: 5.04 ms
- Throughput: 99153 samples/sec
- Categories Created: 0
- Final Category Count: 9

**VectorizedHypersphereART Results:**
- Time: 3.51 ms
- Throughput: 142403 samples/sec
- Categories Created: 0
- Final Category Count: 42

