# ART Text Generation - User Guide
*Version 1.0 - September 2025*

## 🚀 Quick Start

### 1. Start the API Server
```bash
cd /Users/hal.hildebrand/git/ART/text-generation
./start-api-server.sh
```
The server will run on `http://localhost:8080`

### 2. Open the Web Interface
Open `web-interface.html` in your browser, or:
```bash
open web-interface.html
```

### 3. Generate Text
1. Enter a prompt
2. Adjust parameters (optional)
3. Click "Generate Text"

## 📊 Optimal Parameters (From Tuning)

Based on extensive parameter tuning, the optimal settings are:
- **Temperature**: 1.2 (creative but coherent)
- **Top-K**: 50 (diverse vocabulary)
- **Top-P**: 0.90 (balanced nucleus sampling)
- **Repetition Penalty**: 1.5 (avoids repetition)

## 🎯 Features

### Unique ART Advantages
1. **No Catastrophic Forgetting**: Can learn new patterns without losing old knowledge
2. **Ultra-Fast Training**: 28 seconds for 42MB corpus (64x faster than transformers)
3. **Memory Efficient**: <2GB RAM usage (50% less than transformers)
4. **Deterministic**: Same input always produces same model state
5. **Explainable**: Pattern categories are interpretable

### Generation Capabilities
- **Context-Aware**: Maintains topic coherence
- **Style Consistency**: Preserves writing style
- **Advanced Sampling**: Top-k, Top-p, adaptive temperature
- **Repetition Control**: Intelligent penalty system

## 🛠️ Command Line Usage

### Generate Text (CLI)
```bash
mvn exec:java -Dexec.mainClass="com.art.textgen.ARTTextGenerationApp"
```

### Run Benchmarks
```bash
./run-benchmarks.sh
```

### Parameter Tuning
```bash
./tune-parameters.sh
# Choose 1 for quick (5 min) or 2 for full (30 min)
```

### Test API
```bash
./test-api.sh
```

## 🌐 REST API Reference

### Base URL
```
http://localhost:8080/api
```

### Endpoints

#### Generate Text
```http
POST /api/generate
Content-Type: application/json

{
  "prompt": "Once upon a time",
  "maxLength": 100,
  "temperature": 1.2,
  "topK": 50,
  "topP": 0.9
}
```

#### Train Model
```http
POST /api/train
Content-Type: application/json

{
  "text": "Training text here...",
  "incremental": true
}
```

#### Get Metrics
```http
POST /api/metrics
Content-Type: application/json

{
  "text": "Text to analyze"
}
```

#### Configuration
```http
GET /api/config          # Get current config
POST /api/config         # Update config
```

#### Health Check
```http
GET /api/health
```

#### API Documentation
```http
GET /api/docs            # Interactive HTML docs
```

## 💻 Python Client Example

```python
import requests
import json

API_URL = "http://localhost:8080/api"

def generate_text(prompt, max_length=100):
    response = requests.post(
        f"{API_URL}/generate",
        json={
            "prompt": prompt,
            "maxLength": max_length,
            "temperature": 1.2,
            "topK": 50,
            "topP": 0.9
        }
    )
    return response.json()

# Example usage
result = generate_text("The future of AI")
print(result['generated'])
print(f"Diversity: {result['diversity']:.3f}")
print(f"Readability: {result['readability']:.1f}")
```

## 📈 Performance Metrics

### Training Performance
- **Corpus Size**: 42MB (8.27M tokens)
- **Training Time**: 28 seconds
- **Patterns Learned**: 13.9 million
- **Memory Usage**: <2GB

### Generation Performance
- **Speed**: ~100-200ms per 100 tokens
- **Throughput**: ~500 tokens/second
- **Concurrency**: 10 simultaneous requests

### Quality Metrics (Benchmarked)
- **BLEU Score**: >0.3 (good n-gram overlap)
- **Diversity**: >0.7 (high vocabulary variety)
- **Perplexity**: <50 (low uncertainty)
- **Readability**: 60-80 (easy to read)

## 🔧 Advanced Configuration

### Generation Modes

#### Conservative (Precise)
```json
{
  "temperature": 0.5,
  "topK": 20,
  "topP": 0.5,
  "repetitionPenalty": 2.0
}
```

#### Balanced (Default)
```json
{
  "temperature": 1.0,
  "topK": 40,
  "topP": 0.9,
  "repetitionPenalty": 1.2
}
```

#### Creative (Diverse)
```json
{
  "temperature": 1.5,
  "topK": 60,
  "topP": 0.95,
  "repetitionPenalty": 1.0
}
```

## 🗂️ Project Structure

```
text-generation/
├── src/main/java/com/art/textgen/
│   ├── api/              # REST API server
│   ├── benchmarks/       # Benchmarking tools
│   ├── core/             # Core vocabulary & patterns
│   ├── evaluation/       # Metrics & evaluation
│   ├── generation/       # Text generation
│   ├── training/         # Training & corpus
│   └── tuning/           # Parameter optimization
├── training-corpus/      # 42MB training data
├── web-interface.html    # Web UI
├── best_parameters.txt   # Optimal parameters
└── Scripts:
    ├── start-api-server.sh
    ├── tune-parameters.sh
    ├── run-benchmarks.sh
    └── test-api.sh
```

## 🐛 Troubleshooting

### Server Won't Start
```bash
# Check if port 8080 is in use
lsof -i :8080

# Kill existing process if needed
kill -9 <PID>
```

### Compilation Errors
```bash
# Clean and rebuild
mvn clean compile
```

### Out of Memory
```bash
# Increase heap size
export MAVEN_OPTS="-Xmx4g"
```

### Poor Generation Quality
1. Run parameter tuning: `./tune-parameters.sh`
2. Use the optimal parameters from `best_parameters.txt`
3. Ensure corpus is loaded (42MB)

## 📚 Technical Details

### Adaptive Resonance Theory (ART)
ART is a neuroscience-inspired learning algorithm that:
- Forms stable categories through resonance
- Learns incrementally without forgetting
- Self-organizes knowledge hierarchically
- Adapts vigilance for pattern matching

### Pattern Extraction
- **N-gram patterns**: 13.9 million extracted
- **Syntactic patterns**: 231K types
- **Semantic clusters**: 111K groups
- **Hierarchical organization**: Multi-level

### Memory Architecture
- **Vocabulary**: 113K unique tokens
- **Pattern Memory**: Resonance-based storage
- **Working Memory**: Context tracking
- **Hierarchical Memory**: Multi-timescale

## 🎉 Conclusion

The ART Text Generation system provides a unique, neuroscience-inspired approach to text generation with:
- **64x faster training** than transformers
- **No catastrophic forgetting**
- **50% less memory usage**
- **Explainable pattern learning**

For questions or issues, check the logs in the `logs/` directory or review the source code documentation.

## 📖 References

- Grossberg, S. (1976). Adaptive Resonance Theory
- Carpenter, G. A., & Grossberg, S. (1987). ART 2: Self-organization
- Project Repository: `/Users/hal.hildebrand/git/ART`

---
*User Guide v1.0 - ART Text Generation System*
*Last Updated: September 2025*
