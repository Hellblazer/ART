import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.algorithms.ART1;
import com.hellblazer.art.core.parameters.ART1Parameters;

/**
 * Simple demo of ART1 implementation
 */
public class ART1Demo {
    public static void main(String[] args) {
        // Create ART1 with parameters from Python test
        var params = ART1Parameters.builder()
            .vigilance(0.7)
            .L(2.0)
            .build();
        
        var art1 = new ART1();
        
        // Test binary patterns
        var pattern1 = Pattern.of(1, 0);
        var pattern2 = Pattern.of(0, 1);
        var pattern3 = Pattern.of(1, 1);
        
        System.out.println("ART1 Demo - Binary Pattern Recognition");
        System.out.println("Parameters: vigilance=0.7, L=2.0");
        System.out.println();
        
        // Learn patterns
        System.out.println("Learning patterns:");
        var result1 = art1.learn(pattern1, params);
        System.out.println("Pattern [1,0] -> Category " + result1.winnerIndex());
        
        var result2 = art1.learn(pattern2, params);
        System.out.println("Pattern [0,1] -> Category " + result2.winnerIndex());
        
        var result3 = art1.learn(pattern3, params);
        System.out.println("Pattern [1,1] -> Category " + result3.winnerIndex());
        
        System.out.println("\nTotal categories created: " + art1.getCategoryCount());
        
        // Test predictions
        System.out.println("\nTesting predictions:");
        var pred1 = art1.predict(pattern1, params);
        System.out.println("Pattern [1,0] predicted as category " + pred1.winnerIndex());
        
        var pred2 = art1.predict(pattern2, params);
        System.out.println("Pattern [0,1] predicted as category " + pred2.winnerIndex());
        
        // Test cluster centers
        System.out.println("\nCluster centers:");
        var centers = art1.getClusterCenters();
        for (int i = 0; i < centers.size(); i++) {
            var center = centers.get(i);
            System.out.println("Category " + i + ": " + java.util.Arrays.toString(center.features()));
        }
        
        System.out.println("\nART1 Demo completed successfully!");
    }
}