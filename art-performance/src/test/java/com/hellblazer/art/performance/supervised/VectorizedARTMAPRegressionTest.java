package com.hellblazer.art.performance.supervised;

import com.hellblazer.art.performance.algorithms.*;
import com.hellblazer.art.core.*;
import org.junit.jupiter.api.*;

import java.util.ArrayList;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Simple test to verify the vigilance parameter regression fix.
 * Tests that 10-class classification achieves >80% accuracy with proper category creation.
 */
class VectorizedARTMAPRegressionTest {
    
    @Test
    @DisplayName("Regression Test: 10-class classification >80% accuracy")
    void testRegressionFix() {
        // Use the fixed vigilance parameters (0.85/0.92)
        var artAParams = VectorizedParameters.createDefault().withVigilance(0.85);
        var artBParams = VectorizedParameters.createDefault().withVigilance(0.92);
        
        var testParams = VectorizedARTMAPParameters.builder()
            .mapVigilance(0.9)
            .baselineVigilance(0.0)
            .vigilanceIncrement(0.05)
            .maxVigilance(0.95)
            .enableMatchTracking(true)
            .enableParallelSearch(false)
            .maxSearchAttempts(10)
            .artAParams(artAParams)
            .artBParams(artBParams)
            .build();
        
        var artmap = new VectorizedARTMAP(testParams);
        
        // Generate 10 classes with 10 samples each (smaller dataset)
        var trainingData = generateMultiClassData(10, 10);
        
        // Train
        for (var sample : trainingData) {
            var result = artmap.train(sample.input(), sample.target());
            assertTrue(result.isSuccess(), "Training should succeed");
        }
        
        System.out.printf("Created: ArtA=%d categories, ArtB=%d categories\n",
            artmap.getArtA().getCategoryCount(), artmap.getArtB().getCategoryCount());
        
        // Test accuracy
        var correct = 0;
        var total = 0;
        
        for (var sample : trainingData) {
            var prediction = artmap.predict(sample.input());
            if (prediction.isPresent()) {
                total++;
                var predicted = prediction.get().predictedBIndex();
                var expected = (int) sample.target().get(0) - 1; // Account for +1.0 offset
                if (predicted == expected) {
                    correct++;
                }
            }
        }
        
        var accuracy = (double) correct / total;
        System.out.printf("Accuracy: %.1f%% (%d correct out of %d predictions)\n", 
            accuracy * 100, correct, total);
        
        // Verify regression is fixed
        assertTrue(accuracy > 0.8, String.format(
            "Accuracy should be >80%% (was 49%% before fix), got %.1f%%", accuracy * 100));
        
        // Verify proper category creation  
        assertTrue(artmap.getArtB().getCategoryCount() >= 8, 
            "Should create at least 8 B categories for 10 classes, got: " + artmap.getArtB().getCategoryCount());
        
        System.out.println("✅ REGRESSION FIXED: Accuracy >80% with proper categories");
    }
    
    private java.util.List<ClassificationSample> generateMultiClassData(int numClasses, int samplesPerClass) {
        var data = new ArrayList<ClassificationSample>();
        var random = new Random(42); // Fixed seed
        
        for (int classId = 0; classId < numClasses; classId++) {
            for (int sample = 0; sample < samplesPerClass; sample++) {
                // Generate well-separated class patterns
                var baseValue = 0.1 + classId * 0.08; 
                var input = Pattern.of(
                    baseValue + Math.abs(random.nextGaussian()) * 0.02,
                    0.1 + Math.abs(Math.sin(classId * Math.PI / numClasses)) * 0.6 + Math.abs(random.nextGaussian()) * 0.02,
                    0.1 + Math.abs(Math.cos(classId * Math.PI / numClasses)) * 0.6 + Math.abs(random.nextGaussian()) * 0.02
                );
                
                var target = Pattern.of(classId + 1.0); // Avoid zero
                data.add(new ClassificationSample(input, target));
            }
        }
        
        return data;
    }
    
    private record ClassificationSample(Pattern input, Pattern target) {}
}