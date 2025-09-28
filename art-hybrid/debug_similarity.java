import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.hybrid.pan.similarity.SimilarityMeasures;

public class debug_similarity {
    public static void main(String[] args) {
        // Create test pattern
        double[] data = {0.1, 0.2, 0.3, 0.4, 0.5};
        Pattern pattern = new DenseVector(data);

        // Create test weights
        double[] weights = {0.15, 0.25, 0.35, 0.45, 0.55};

        // Test FUZZY_ART similarity
        double similarity = SimilarityMeasures.FUZZY_ART.compute(pattern, weights);
        System.out.println("FUZZY_ART similarity: " + similarity);

        // Expected: min(0.1,0.15) + min(0.2,0.25) + ... = 0.1+0.2+0.3+0.4+0.5 = 1.5
        // Input sum: 0.1+0.2+0.3+0.4+0.5 = 1.5
        // Result: 1.5/1.5 = 1.0

        System.out.println("Expected: 1.0");
        System.out.println("Actual: " + similarity);
        System.out.println("Match: " + (Math.abs(similarity - 1.0) < 0.001));
    }
}