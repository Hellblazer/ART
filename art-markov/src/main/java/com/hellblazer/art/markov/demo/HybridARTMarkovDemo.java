package com.hellblazer.art.markov.demo;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.parameters.FuzzyParameters;
import com.hellblazer.art.markov.core.MinimalHybridPredictor;
import com.hellblazer.art.markov.core.ValidationLayer;
import com.hellblazer.art.markov.parameters.HybridMarkovParameters;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Standalone demonstration of the hybrid ART-Markov system.
 *
 * This demo shows:
 * 1. State discovery from continuous observations using FuzzyART
 * 2. Transition learning with mathematical validation
 * 3. Hybrid prediction combining ART and Markov approaches
 * 4. Performance baseline measurements
 * 5. Comparison with pure approaches
 */
public final class HybridARTMarkovDemo {

    // Weather states (ground truth for validation)
    private static final int SUNNY = 0;
    private static final int CLOUDY = 1;
    private static final int RAINY = 2;
    private static final int STORMY = 3;

    private static final String[] STATE_NAMES = {"Sunny", "Cloudy", "Rainy", "Stormy"};

    // Ground truth transition matrix (realistic weather patterns)
    private static final double[][] GROUND_TRUTH_TRANSITIONS = {
        // From:  SUNNY    CLOUDY   RAINY    STORMY
        /*SUNNY*/  {0.60,   0.30,    0.08,    0.02},  // Sunny tends to stay sunny
        /*CLOUDY*/ {0.25,   0.40,    0.30,    0.05},  // Cloudy can go either way
        /*RAINY*/  {0.10,   0.35,    0.45,    0.10},  // Rainy persists or gets worse
        /*STORMY*/ {0.05,   0.20,    0.35,    0.40}   // Stormy tends to persist
    };

    public static void main(String[] args) {
        System.out.println("=== Hybrid ART-Markov Proof-of-Concept Demo ===\n");

        try {
            runFullDemonstration();
        } catch (Exception e) {
            System.err.println("Demo failed: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void runFullDemonstration() {
        // Step 1: Validate mathematical foundations
        validateMathematicalFoundations();

        // Step 2: Create and train hybrid system
        var hybridPredictor = createAndTrainHybridSystem();

        // Step 3: Demonstrate predictions
        demonstratePredictions(hybridPredictor);

        // Step 4: Performance baseline
        measurePerformanceBaseline(hybridPredictor);

        // Step 5: Compare approaches
        compareApproaches();

        // Cleanup
        hybridPredictor.close();

        System.out.println("\n=== Demo completed successfully! ===");
    }

    private static void validateMathematicalFoundations() {
        System.out.println("1. Validating Mathematical Foundations");
        System.out.println("=====================================");

        // Validate ground truth transition matrix
        System.out.print("Validating ground truth stochastic matrix... ");
        try {
            ValidationLayer.validateStochasticMatrix(GROUND_TRUTH_TRANSITIONS);
            System.out.println("✓ PASSED");
        } catch (Exception e) {
            System.out.println("✗ FAILED: " + e.getMessage());
            return;
        }

        // Check row sums
        System.out.print("Verifying row sums equal 1.0... ");
        boolean rowSumsValid = true;
        for (int i = 0; i < GROUND_TRUTH_TRANSITIONS.length; i++) {
            double rowSum = Arrays.stream(GROUND_TRUTH_TRANSITIONS[i]).sum();
            if (Math.abs(rowSum - 1.0) > 1e-10) {
                rowSumsValid = false;
                break;
            }
        }
        System.out.println(rowSumsValid ? "✓ PASSED" : "✗ FAILED");

        // Test convergence detection
        System.out.print("Testing convergence detection... ");
        var identityMatrix = new double[][]{{1.0, 0.0}, {0.0, 1.0}};
        boolean converged = ValidationLayer.hasConverged(identityMatrix, 0.001);
        System.out.println(converged ? "✓ PASSED" : "✗ FAILED");

        // Test steady state computation
        System.out.print("Testing steady state computation... ");
        var steadyState = ValidationLayer.computeSteadyState(GROUND_TRUTH_TRANSITIONS, 1000, 1e-8);
        boolean steadyStateValid = Math.abs(Arrays.stream(steadyState).sum() - 1.0) < 1e-6;
        System.out.println(steadyStateValid ? "✓ PASSED" : "✗ FAILED");

        if (steadyStateValid) {
            System.out.print("Ground truth steady state: ");
            for (int i = 0; i < steadyState.length; i++) {
                System.out.printf("%s: %.3f  ", STATE_NAMES[i], steadyState[i]);
            }
            System.out.println();
        }

        System.out.println();
    }

    private static MinimalHybridPredictor createAndTrainHybridSystem() {
        System.out.println("2. Creating and Training Hybrid System");
        System.out.println("======================================");

        // Create parameters with tuned vigilance for optimal state discovery
        var parameters = HybridMarkovParameters.withVigilance(0.6);
        System.out.println("Created parameters:");
        System.out.println("  Vigilance: " + parameters.fuzzyParameters().vigilance());
        System.out.println("  Hybrid weight: " + parameters.hybridWeight());
        System.out.println("  Max states: " + parameters.maxStates());

        // Create hybrid predictor
        var hybridPredictor = new MinimalHybridPredictor(parameters);
        System.out.println("Created hybrid predictor");

        // Generate more training data for better learning
        System.out.print("Generating training data... ");
        var trainingData = generateWeatherSequence(5000, 42L);
        System.out.println("Generated " + trainingData.size() + " observations");

        // Train the system and track state-to-weather mappings
        System.out.print("Training hybrid system... ");
        long startTime = System.currentTimeMillis();

        // Track which weather types map to which discovered states
        Map<Integer, Map<Integer, Integer>> stateToWeatherCounts = new HashMap<>();

        for (var observation : trainingData) {
            int state = hybridPredictor.learn(observation.observation());
            if (state >= 0) {
                // Track the mapping between discovered states and actual weather
                stateToWeatherCounts.computeIfAbsent(state, k -> new HashMap<>())
                    .merge(observation.groundTruthState(), 1, Integer::sum);
            }
        }

        // Assign labels based on majority voting
        for (var entry : stateToWeatherCounts.entrySet()) {
            int state = entry.getKey();
            var weatherCounts = entry.getValue();

            // Find the most common weather type for this state
            int mostCommonWeather = weatherCounts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(0);

            if (mostCommonWeather >= 0 && mostCommonWeather < STATE_NAMES.length) {
                hybridPredictor.setStateLabel(state, STATE_NAMES[mostCommonWeather]);
                System.out.println("  State " + state + " mapped to " + STATE_NAMES[mostCommonWeather] +
                                 " (" + weatherCounts.get(mostCommonWeather) + " occurrences)");
            }
        }

        long trainingTime = System.currentTimeMillis() - startTime;
        System.out.println("Completed in " + trainingTime + " ms");

        // Display results
        var stats = hybridPredictor.getStatistics();
        System.out.println("Training results:");
        System.out.println("  States discovered: " + hybridPredictor.getStateCount());
        System.out.println("  Total transitions: " + ((Map<String, Object>) stats.get("markovStats")).get("totalTransitions"));

        // Validate learned transition matrix
        var learnedMatrix = hybridPredictor.getTransitionMatrix();
        System.out.print("Validating learned transition matrix... ");
        try {
            ValidationLayer.validateStochasticMatrix(learnedMatrix);
            System.out.println("✓ PASSED");
        } catch (Exception e) {
            System.out.println("✗ FAILED: " + e.getMessage());
        }

        // Display learned transition matrix
        System.out.println("Learned transition matrix:");
        printTransitionMatrix(learnedMatrix, hybridPredictor);

        System.out.println();
        return hybridPredictor;
    }

    private static void demonstratePredictions(MinimalHybridPredictor hybridPredictor) {
        System.out.println("3. Demonstrating Predictions");
        System.out.println("============================");

        // Generate test data
        var testData = generateWeatherSequence(50, 999L);

        int correctPredictions = 0;
        var predictionTimes = new ArrayList<Long>();

        System.out.println("Making predictions on test data:");
        System.out.println("Obs# | Current Weather | Predicted | Actual | Correct | Confidence | Time(μs)");
        System.out.println("-".repeat(80));

        for (int i = 0; i < Math.min(testData.size() - 1, 15); i++) {
            var currentObs = testData.get(i);
            var nextObs = testData.get(i + 1);

            var prediction = hybridPredictor.predict(currentObs.observation());
            boolean correct = hybridPredictor.validatePrediction(prediction, nextObs.observation());

            if (correct) correctPredictions++;
            predictionTimes.add(prediction.predictionTimeNanos());

            // Get state labels
            String currentLabel = "Unknown";
            String predictedLabel = "Unknown";
            String actualLabel = STATE_NAMES[nextObs.groundTruthState()];

            if (prediction.observedState() >= 0) {
                currentLabel = hybridPredictor.getStateLabel(prediction.observedState());
            }
            if (prediction.mostLikelyNextState() >= 0) {
                predictedLabel = hybridPredictor.getStateLabel(prediction.mostLikelyNextState());
            }

            System.out.printf("%4d | %-15s | %-9s | %-6s | %7s | %8.3f | %7.1f\n",
                i + 1,
                currentLabel,
                predictedLabel,
                actualLabel,
                correct ? "✓" : "✗",
                prediction.getConfidence(),
                prediction.predictionTimeNanos() / 1000.0
            );
        }

        // Summary statistics
        double accuracy = (double) correctPredictions / Math.min(testData.size() - 1, 15);
        double avgPredictionTime = predictionTimes.stream().mapToLong(Long::longValue).average().orElse(0.0);

        System.out.println("-".repeat(80));
        System.out.printf("Accuracy: %.1f%% (%d/%d correct)\n", accuracy * 100, correctPredictions, Math.min(testData.size() - 1, 15));
        System.out.printf("Average prediction time: %.1f μs\n", avgPredictionTime / 1000.0);

        System.out.println();
    }

    private static void measurePerformanceBaseline(MinimalHybridPredictor hybridPredictor) {
        System.out.println("4. Performance Baseline Measurements");
        System.out.println("====================================");

        var stats = hybridPredictor.getStatistics();

        // Memory usage
        var memoryUsage = ((Double) stats.get("memoryUsageBytes")).longValue();
        System.out.printf("Memory usage: %,d bytes (%.1f KB)\n", memoryUsage, memoryUsage / 1024.0);

        // Throughput test
        System.out.print("Measuring prediction throughput... ");
        var testObs = generateWeatherSequence(1000, 12345L);

        long startTime = System.nanoTime();
        for (var obs : testObs) {
            hybridPredictor.predict(obs.observation());
        }
        long elapsedNanos = System.nanoTime() - startTime;

        double throughput = (double) testObs.size() / (elapsedNanos / 1e9);
        System.out.printf("%.0f predictions/second\n", throughput);

        // System properties
        var markovStats = (Map<String, Object>) stats.get("markovStats");
        System.out.println("System properties:");
        System.out.println("  Satisfies Markov property: " + markovStats.get("satisfiesMarkovProperty"));
        System.out.println("  Has converged: " + markovStats.get("hasConverged"));
        System.out.println("  Matrix entropy: " + String.format("%.3f", markovStats.get("matrixEntropy")));

        System.out.println();
    }

    private static void compareApproaches() {
        System.out.println("5. Comparing Hybrid vs Pure Approaches");
        System.out.println("======================================");

        var hybridWeights = new double[]{0.0, 0.25, 0.5, 0.75, 1.0};
        var approaches = new String[]{"Pure Markov", "Markov-heavy", "Balanced", "ART-heavy", "Pure ART"};

        System.out.println("Approach      | Accuracy | Avg Pred Time");
        System.out.println("-".repeat(40));

        for (int i = 0; i < hybridWeights.length; i++) {
            var accuracy = testApproach(hybridWeights[i]);
            System.out.printf("%-12s | %6.1f%% | %10.1f μs\n",
                approaches[i],
                accuracy.accuracy * 100,
                accuracy.avgPredictionTimeNanos / 1000.0
            );
        }

        System.out.println();
    }

    private static AccuracyResult testApproach(double hybridWeight) {
        // Use tuned vigilance for comparison tests too
        var parameters = HybridMarkovParameters.withVigilance(0.6);
        parameters = new HybridMarkovParameters(
            parameters.fuzzyParameters(),
            parameters.transitionSmoothingFactor(),
            hybridWeight,
            parameters.convergenceThreshold(),
            parameters.maxStates(),
            parameters.memoryWindow()
        );

        try (var predictor = new MinimalHybridPredictor(parameters)) {
            // Generate more data for comparison tests
            var trainingData = generateWeatherSequence(1000, 42L);
            var testData = generateWeatherSequence(100, 999L);

            // Train
            for (var obs : trainingData) {
                predictor.learn(obs.observation());
            }

            // Test
            int correct = 0;
            var predictionTimes = new ArrayList<Long>();

            for (int i = 0; i < testData.size() - 1; i++) {
                var currentObs = testData.get(i);
                var nextObs = testData.get(i + 1);

                var prediction = predictor.predict(currentObs.observation());
                predictionTimes.add(prediction.predictionTimeNanos());

                if (predictor.validatePrediction(prediction, nextObs.observation())) {
                    correct++;
                }
            }

            double accuracy = (double) correct / (testData.size() - 1);
            double avgTime = predictionTimes.stream().mapToLong(Long::longValue).average().orElse(0.0);

            return new AccuracyResult(accuracy, avgTime);
        }
    }

    // Helper methods and classes

    private record WeatherObservation(Pattern observation, int groundTruthState) {}

    private record AccuracyResult(double accuracy, double avgPredictionTimeNanos) {}

    private static List<WeatherObservation> generateWeatherSequence(int length, long seed) {
        var random = new Random(seed);
        var observations = new ArrayList<WeatherObservation>();

        int currentState = SUNNY; // Start sunny

        for (int i = 0; i < length; i++) {
            // Generate continuous observation from discrete state
            var observation = generateObservationFromState(currentState, random);
            observations.add(new WeatherObservation(observation, currentState));

            // Transition to next state based on ground truth probabilities
            currentState = sampleNextState(currentState, random);
        }

        return observations;
    }

    private static Pattern generateObservationFromState(int state, Random random) {
        // Generate realistic weather observations as continuous vectors
        // [temperature, humidity, pressure] normalized to [0, 1]
        // Very distinct values with minimal overlap

        double temperature, humidity, pressure;

        switch (state) {
            case SUNNY -> {
                temperature = 0.9 + random.nextGaussian() * 0.03;   // Very hot
                humidity = 0.1 + random.nextGaussian() * 0.03;      // Very dry
                pressure = 0.9 + random.nextGaussian() * 0.03;      // Very high pressure
            }
            case CLOUDY -> {
                temperature = 0.6 + random.nextGaussian() * 0.03;   // Warm
                humidity = 0.4 + random.nextGaussian() * 0.03;      // Medium-low humidity
                pressure = 0.6 + random.nextGaussian() * 0.03;      // Medium pressure
            }
            case RAINY -> {
                temperature = 0.3 + random.nextGaussian() * 0.03;   // Cool
                humidity = 0.8 + random.nextGaussian() * 0.03;      // High humidity
                pressure = 0.3 + random.nextGaussian() * 0.03;      // Low pressure
            }
            case STORMY -> {
                temperature = 0.1 + random.nextGaussian() * 0.03;   // Cold
                humidity = 0.95 + random.nextGaussian() * 0.02;     // Very high humidity
                pressure = 0.1 + random.nextGaussian() * 0.03;      // Very low pressure
            }
            default -> throw new IllegalArgumentException("Invalid state: " + state);
        }

        // Clamp values to [0, 1] range
        temperature = Math.max(0.0, Math.min(1.0, temperature));
        humidity = Math.max(0.0, Math.min(1.0, humidity));
        pressure = Math.max(0.0, Math.min(1.0, pressure));

        return Pattern.of(temperature, humidity, pressure);
    }

    private static int sampleNextState(int currentState, Random random) {
        double[] probabilities = GROUND_TRUTH_TRANSITIONS[currentState];
        double rand = random.nextDouble();

        double cumulative = 0.0;
        for (int i = 0; i < probabilities.length; i++) {
            cumulative += probabilities[i];
            if (rand <= cumulative) {
                return i;
            }
        }

        // Fallback (should not happen with valid probabilities)
        return currentState;
    }

    private static void printTransitionMatrix(double[][] matrix, MinimalHybridPredictor predictor) {
        int stateCount = predictor.getStateCount();

        // Header
        System.out.print("From\\To   ");
        for (int j = 0; j < stateCount; j++) {
            System.out.printf("%8s ", predictor.getStateLabel(j));
        }
        System.out.println();

        // Rows
        for (int i = 0; i < stateCount; i++) {
            System.out.printf("%-8s  ", predictor.getStateLabel(i));
            for (int j = 0; j < stateCount; j++) {
                System.out.printf("%8.3f ", matrix[i][j]);
            }
            System.out.println();
        }
    }
}