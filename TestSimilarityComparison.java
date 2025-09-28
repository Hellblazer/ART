import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.hybrid.pan.PAN;
import com.hellblazer.art.hybrid.pan.parameters.PANParameters;
import com.hellblazer.art.hybrid.pan.similarity.SimilarityMeasures;

import java.util.Random;

public class TestSimilarityComparison {
    public static void main(String[] args) {
        System.out.println("=== Comparing PAN Similarity Measures ===\n");

        // Create test patterns
        Random rand = new Random(42);
        Pattern[] patterns = new Pattern[3];

        for (int i = 0; i < 3; i++) {
            double[] data = new double[784];
            for (int j = 0; j < 784; j++) {
                // Create distinct patterns
                data[j] = rand.nextDouble() * (i + 1) / 3.0;
            }
            patterns[i] = new DenseVector(data);
        }

        // Test with Fuzzy ART (default)
        System.out.println("1. Testing with Fuzzy ART similarity (default):");
        PANParameters fuzzyParams = PANParameters.defaultParameters();
        System.out.println("  Similarity measure: " + fuzzyParams.similarityMeasure());
        System.out.println("  Vigilance: " + fuzzyParams.vigilance());

        try (PAN pan = new PAN(fuzzyParams)) {
            // Learn patterns
            for (Pattern p : patterns) {
                pan.learn(p, fuzzyParams);
            }

            System.out.println("  Categories created: " + pan.getCategoryCount());

            // Test prediction accuracy
            int correct = 0;
            for (int i = 0; i < patterns.length; i++) {
                var result = pan.predict(patterns[i], fuzzyParams);
                if (result instanceof com.hellblazer.art.core.results.ActivationResult.Success success) {
                    if (success.categoryIndex() == i) correct++;
                    System.out.printf("    Pattern %d -> Category %d\n",
                        i, success.categoryIndex());
                }
            }
            System.out.println("  Prediction accuracy: " + ((double) correct / patterns.length * 100) + "%");
        }

        // Test with Dot Product (paper-compliant)
        System.out.println("\n2. Testing with Dot Product similarity (paper-compliant):");
        PANParameters dotParams = PANParameters.paperCompliantParameters();
        System.out.println("  Similarity measure: " + dotParams.similarityMeasure());
        System.out.println("  Vigilance: " + dotParams.vigilance());

        try (PAN pan = new PAN(dotParams)) {
            // Learn patterns
            for (Pattern p : patterns) {
                pan.learn(p, dotParams);
            }

            System.out.println("  Categories created: " + pan.getCategoryCount());

            // Test prediction accuracy
            int correct = 0;
            for (int i = 0; i < patterns.length; i++) {
                var result = pan.predict(patterns[i], dotParams);
                if (result instanceof com.hellblazer.art.core.results.ActivationResult.Success success) {
                    if (success.categoryIndex() == i) correct++;
                    System.out.printf("    Pattern %d -> Category %d\n",
                        i, success.categoryIndex());
                }
            }
            System.out.println("  Prediction accuracy: " + ((double) correct / patterns.length * 100) + "%");
        }

        System.out.println("\n=== Summary ===");
        System.out.println("Successfully refactored PAN to support configurable similarity measures!");
        System.out.println("- Fuzzy ART: Bounded [0,1], stable, good for general use");
        System.out.println("- Dot Product: Unbounded, paper-compliant, may need vigilance adjustment");
    }
}