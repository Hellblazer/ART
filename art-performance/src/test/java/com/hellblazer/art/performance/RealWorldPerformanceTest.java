package com.hellblazer.art.performance;

import com.hellblazer.art.performance.algorithms.*;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.Pattern;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Real-world performance testing for ART algorithms.
 * Tests various scenarios with realistic data patterns.
 */
public class RealWorldPerformanceTest {
    
    private static final int WARMUP_ITERATIONS = 100;
    private static final int TEST_ITERATIONS = 500;
    
    private List<TestScenario> scenarios;
    private StringBuilder performanceReport;
    
    @BeforeEach
    public void setup() {
        scenarios = new ArrayList<>();
        performanceReport = new StringBuilder();
        
        // Define real-world test scenarios
        scenarios.add(new TestScenario("Small Dataset - Low Dimensions", 1000, 10));
        scenarios.add(new TestScenario("Medium Dataset - Medium Dimensions", 10000, 50));
        scenarios.add(new TestScenario("Large Dataset - High Dimensions", 50000, 100));
        scenarios.add(new TestScenario("Image Recognition Simulation", 5000, 784)); // 28x28 images
        scenarios.add(new TestScenario("Sensor Data Processing", 100000, 32));
        scenarios.add(new TestScenario("Text Embedding Clustering", 20000, 300));
    }
    
    @Test
    public void runComprehensivePerformanceTests() throws IOException {
        performanceReport.append("# ART Algorithm Real-World Performance Report\n\n");
        performanceReport.append("Generated: ").append(LocalDateTime.now()).append("\n\n");
        performanceReport.append("## Test Environment\n");
        performanceReport.append("- Java Version: ").append(System.getProperty("java.version")).append("\n");
        performanceReport.append("- OS: ").append(System.getProperty("os.name")).append("\n");
        performanceReport.append("- Available Processors: ").append(Runtime.getRuntime().availableProcessors()).append("\n");
        performanceReport.append("- Max Memory: ").append(Runtime.getRuntime().maxMemory() / (1024 * 1024)).append(" MB\n\n");
        
        for (TestScenario scenario : scenarios) {
            System.out.println("\n=== Testing: " + scenario.name + " ===");
            performanceReport.append("## ").append(scenario.name).append("\n\n");
            performanceReport.append("- Data Size: ").append(scenario.dataSize).append("\n");
            performanceReport.append("- Dimensions: ").append(scenario.dimensions).append("\n\n");
            
            var testData = generateTestData(scenario);
            
            // Test different parameter configurations
            var paramConfigs = generateParameterConfigurations();
            
            for (var config : paramConfigs) {
                System.out.println("  Testing with config: " + config.name);
                performanceReport.append("### Configuration: ").append(config.name).append("\n\n");
                
                // Test VectorizedFuzzyART
                testVectorizedFuzzyART(scenario, testData, config);
                
                // Test VectorizedHypersphereART
                testVectorizedHypersphereART(scenario, testData, config);
                
                // Force garbage collection between tests
                System.gc();
                try { Thread.sleep(100); } catch (InterruptedException e) {}
            }
        }
        
        // Write report to target directory
        var reportFile = "target/PERFORMANCE_REPORT_" + System.currentTimeMillis() + ".md";
        try (var writer = new FileWriter(reportFile)) {
            writer.write(performanceReport.toString());
        }
        
        System.out.println("\n\nPerformance report written to: " + reportFile);
    }
    
    private void testVectorizedFuzzyART(TestScenario scenario, List<Pattern> testData, ParameterConfig config) {
        var params = new VectorizedParameters(
            config.vigilance,
            config.learningRate,
            0.001,  // alpha
            4,      // parallelismLevel
            50,     // parallelThreshold
            1000,   // maxCacheSize
            true,   // enableSIMD
            false,  // enableJOML
            0.8     // memoryOptimizationThreshold
        );
        
        var art = new VectorizedFuzzyART(params);
        
        // Warmup
        for (int i = 0; i < Math.min(WARMUP_ITERATIONS, testData.size()); i++) {
            art.learn(testData.get(i), params);
        }
        
        // Actual test
        long startTime = System.nanoTime();
        int categoriesCreated = 0;
        
        for (int i = 0; i < Math.min(TEST_ITERATIONS, testData.size()); i++) {
            var result = art.learn(testData.get(i), params);
            if (result instanceof ActivationResult.NoMatch) {
                categoriesCreated++;
            }
        }
        
        long endTime = System.nanoTime();
        double timeMs = (endTime - startTime) / 1_000_000.0;
        double throughput = TEST_ITERATIONS / (timeMs / 1000.0);
        
        performanceReport.append("**VectorizedFuzzyART Results:**\n");
        performanceReport.append("- Time: ").append(String.format("%.2f", timeMs)).append(" ms\n");
        performanceReport.append("- Throughput: ").append(String.format("%.0f", throughput)).append(" samples/sec\n");
        performanceReport.append("- Categories Created: ").append(categoriesCreated).append("\n");
        performanceReport.append("- Final Category Count: ").append(art.getCategoryCount()).append("\n\n");
        
        art.close();
    }
    
    private void testVectorizedHypersphereART(TestScenario scenario, List<Pattern> testData, ParameterConfig config) {
        var params = new VectorizedHypersphereParameters(
            config.vigilance,        // vigilance
            config.learningRate,     // learningRate
            scenario.dimensions,     // inputDimensions
            1000,                   // maxCategories
            true,                   // enableSIMD
            8,                      // simdThreshold
            4,                      // parallelismLevel
            true,                   // enableCaching
            1000,                   // cacheSize
            1.2,                    // expansionFactor
            1                       // useSIMD (1 = true)
        );
        
        var art = new VectorizedHypersphereART(params);
        
        // Warmup
        for (int i = 0; i < Math.min(WARMUP_ITERATIONS, testData.size()); i++) {
            art.learn(testData.get(i), params);
        }
        
        // Actual test
        long startTime = System.nanoTime();
        int categoriesCreated = 0;
        
        for (int i = 0; i < Math.min(TEST_ITERATIONS, testData.size()); i++) {
            var result = art.learn(testData.get(i), params);
            if (result instanceof ActivationResult.NoMatch) {
                categoriesCreated++;
            }
        }
        
        long endTime = System.nanoTime();
        double timeMs = (endTime - startTime) / 1_000_000.0;
        double throughput = TEST_ITERATIONS / (timeMs / 1000.0);
        
        performanceReport.append("**VectorizedHypersphereART Results:**\n");
        performanceReport.append("- Time: ").append(String.format("%.2f", timeMs)).append(" ms\n");
        performanceReport.append("- Throughput: ").append(String.format("%.0f", throughput)).append(" samples/sec\n");
        performanceReport.append("- Categories Created: ").append(categoriesCreated).append("\n");
        performanceReport.append("- Final Category Count: ").append(art.getCategoryCount()).append("\n\n");
        
        art.close();
    }
    
    private List<Pattern> generateTestData(TestScenario scenario) {
        var random = new Random(42);
        var data = new ArrayList<Pattern>();
        
        // Generate clustered data to simulate real-world patterns
        int numClusters = Math.min(10, scenario.dataSize / 100);
        var clusterCenters = new double[numClusters][scenario.dimensions];
        
        // Generate cluster centers
        for (int c = 0; c < numClusters; c++) {
            for (int d = 0; d < scenario.dimensions; d++) {
                clusterCenters[c][d] = random.nextDouble();
            }
        }
        
        // Generate data points around clusters
        for (int i = 0; i < scenario.dataSize; i++) {
            var values = new double[scenario.dimensions];
            int cluster = i % numClusters;
            
            for (int d = 0; d < scenario.dimensions; d++) {
                // Add noise around cluster center
                values[d] = clusterCenters[cluster][d] + (random.nextGaussian() * 0.1);
                // Clamp to [0, 1]
                values[d] = Math.max(0, Math.min(1, values[d]));
            }
            
            data.add(Pattern.of(values));
        }
        
        return data;
    }
    
    private List<ParameterConfig> generateParameterConfigurations() {
        var configs = new ArrayList<ParameterConfig>();
        configs.add(new ParameterConfig("High Vigilance - Fast Learning", 0.9, 0.5));
        configs.add(new ParameterConfig("Medium Vigilance - Moderate Learning", 0.7, 0.3));
        configs.add(new ParameterConfig("Low Vigilance - Slow Learning", 0.5, 0.1));
        return configs;
    }
    
    private static class TestScenario {
        final String name;
        final int dataSize;
        final int dimensions;
        
        TestScenario(String name, int dataSize, int dimensions) {
            this.name = name;
            this.dataSize = dataSize;
            this.dimensions = dimensions;
        }
    }
    
    private static class ParameterConfig {
        final String name;
        final double vigilance;
        final double learningRate;
        
        ParameterConfig(String name, double vigilance, double learningRate) {
            this.name = name;
            this.vigilance = vigilance;
            this.learningRate = learningRate;
        }
    }
}