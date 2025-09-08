#!/bin/bash

# Simple test script for ART Text Generation

cd /Users/hal.hildebrand/git/ART/text-generation

echo "Building the project..."
mvn clean compile

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Testing with sample corpus training..."
    
    # Create a simple test Java file that will use the classes
    cat > TestRun.java << 'EOF'
import com.art.textgen.core.Vocabulary;
import com.art.textgen.generation.PatternGenerator;
import com.art.textgen.training.TrainingPipeline;
import java.util.*;

public class TestRun {
    public static void main(String[] args) {
        System.out.println("=== ART Text Generation Quick Test ===\n");
        
        // Initialize
        Vocabulary vocabulary = new Vocabulary(64);
        PatternGenerator generator = new PatternGenerator(vocabulary, 0.7);
        TrainingPipeline pipeline = new TrainingPipeline(vocabulary, generator);
        
        // Train from samples
        System.out.println("Training from sample corpus...");
        pipeline.trainFromSamples();
        
        // Test generation
        String[] prompts = {"The future", "Artificial intelligence", "Once upon"};
        
        for (String prompt : prompts) {
            System.out.println("\nPrompt: \"" + prompt + "\"");
            List<String> context = vocabulary.tokenize(prompt);
            System.out.print("Generated: " + prompt);
            
            for (int i = 0; i < 20; i++) {
                String next = generator.generateNext(context);
                if (next.equals("<END>")) break;
                System.out.print(" " + next);
                context.add(next);
                if (context.size() > 10) context.remove(0);
            }
            System.out.println();
        }
        
        // Show statistics
        System.out.println("\n=== Statistics ===");
        Map<String, Object> stats = generator.getStatistics();
        System.out.println("Total patterns: " + stats.get("total_patterns"));
        System.out.println("Temperature: " + stats.get("temperature"));
        System.out.println("Vocabulary size: " + vocabulary.size());
        System.out.println("Embedding dimension: " + vocabulary.getEmbeddingDim());
    }
}
EOF
    
    # Compile and run the test
    javac -cp target/classes TestRun.java
    if [ $? -eq 0 ]; then
        echo ""
        java -cp .:target/classes TestRun
    else
        echo "Test compilation failed"
    fi
    
    # Clean up
    rm -f TestRun.java TestRun.class
else
    echo "Build failed!"
fi
