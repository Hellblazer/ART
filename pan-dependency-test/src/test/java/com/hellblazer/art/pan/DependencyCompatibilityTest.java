package com.hellblazer.art.pan;

import ai.onnxruntime.*;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.junit.jupiter.api.Test;
import smile.classification.OneClassSVM;
import smile.math.kernel.GaussianKernel;

import java.nio.FloatBuffer;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests to verify Java 24 compatibility with proposed PAN dependencies
 */
public class DependencyCompatibilityTest {

    @Test
    void testONNXRuntimeCompatibility() {
        // Test ONNX Runtime initialization and basic operation
        try {
            var env = OrtEnvironment.getEnvironment();
            assertNotNull(env, "ONNX Runtime environment should initialize");

            // Test session options
            var sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);

            System.out.println("✓ ONNX Runtime " + OrtEnvironment.getEnvironment().toString() +
                             " compatible with Java " + Runtime.version().feature());
        } catch (Exception e) {
            fail("ONNX Runtime initialization failed: " + e.getMessage());
        }
    }

    @Test
    void testJCudaCompatibility() {
        // Test JCuda initialization
        try {
            // Initialize the driver
            JCudaDriver.setExceptionsEnabled(true);
            int result = JCudaDriver.cuInit(0);

            if (result == JCudaDriver.cudaSuccess) {
                System.out.println("✓ JCuda driver initialized successfully");

                // Check CUDA version
                int[] version = new int[1];
                JCudaDriver.cuDriverGetVersion(version);
                System.out.println("  CUDA Driver Version: " + version[0]);
            } else if (result == JCudaDriver.cudaErrorNoDevice) {
                System.out.println("⚠ No CUDA devices found (expected in CI/non-GPU environment)");
            } else {
                System.out.println("⚠ JCuda initialization returned: " + result);
            }

            // Basic JCuda runtime should always work
            assertNotNull(JCuda.class);
            System.out.println("✓ JCuda runtime classes compatible with Java " +
                             Runtime.version().feature());

        } catch (UnsatisfiedLinkError e) {
            System.out.println("⚠ JCuda native libraries not available: " + e.getMessage());
            System.out.println("  This is expected if CUDA is not installed");
        } catch (Exception e) {
            fail("JCuda compatibility test failed: " + e.getMessage());
        }
    }

    @Test
    void testSmileMLCompatibility() {
        // Test Smile ML OneClassSVM
        try {
            // Create sample data
            double[][] data = {
                {1.0, 2.0},
                {2.0, 3.0},
                {3.0, 4.0},
                {4.0, 5.0},
                {5.0, 6.0}
            };

            // Create and train OneClassSVM
            var kernel = new GaussianKernel(1.0);
            var svm = OneClassSVM.fit(data, kernel, 0.1);
            assertNotNull(svm, "OneClassSVM should be created");

            // Test prediction
            double[] testPoint = {2.5, 3.5};
            int prediction = svm.predict(testPoint);
            assertTrue(prediction == 1 || prediction == -1,
                      "Prediction should be 1 (inlier) or -1 (outlier)");

            System.out.println("✓ Smile ML OneClassSVM compatible with Java " +
                             Runtime.version().feature());
        } catch (Exception e) {
            fail("Smile ML compatibility test failed: " + e.getMessage());
        }
    }

    @Test
    void testCircularFifoQueueCompatibility() {
        // Test Apache Commons Collections CircularFifoQueue
        try {
            var queue = new CircularFifoQueue<String>(3);

            // Test basic operations
            queue.add("item1");
            queue.add("item2");
            queue.add("item3");
            queue.add("item4"); // Should remove item1

            assertEquals(3, queue.size(), "Queue should maintain max size");
            assertFalse(queue.contains("item1"), "Oldest item should be removed");
            assertTrue(queue.contains("item4"), "Newest item should be present");

            System.out.println("✓ Apache Commons Collections CircularFifoQueue compatible with Java " +
                             Runtime.version().feature());
        } catch (Exception e) {
            fail("CircularFifoQueue compatibility test failed: " + e.getMessage());
        }
    }

    @Test
    void testJava24Features() {
        // Test that Java 24 features work
        try {
            // Test pattern matching
            Object obj = "test";
            var result = switch (obj) {
                case String s when s.length() > 0 -> "non-empty string";
                case String s -> "empty string";
                default -> "not a string";
            };
            assertEquals("non-empty string", result);

            // Test record patterns
            record Point(int x, int y) {}
            var point = new Point(3, 4);
            var description = switch (point) {
                case Point(var x, var y) when x == y -> "diagonal";
                case Point(var x, var y) when x == 0 || y == 0 -> "on axis";
                case Point(var x, var y) -> "general point at (" + x + ", " + y + ")";
            };
            assertEquals("general point at (3, 4)", description);

            System.out.println("✓ Java 24 features working correctly");
        } catch (Exception e) {
            fail("Java 24 feature test failed: " + e.getMessage());
        }
    }

    @Test
    void testIntegration() {
        // Test that all dependencies can work together
        try {
            // Simulate a simple PAN workflow
            var env = OrtEnvironment.getEnvironment();
            var experiencePool = new CircularFifoQueue<float[]>(100);

            // Add some dummy experiences
            for (int i = 0; i < 10; i++) {
                experiencePool.add(new float[]{i, i * 2, i * 3});
            }

            // Create dummy data for SVM
            double[][] svmData = experiencePool.stream()
                .map(arr -> new double[]{arr[0], arr[1]})
                .toArray(double[][]::new);

            if (svmData.length > 0) {
                var kernel = new GaussianKernel(1.0);
                var svm = OneClassSVM.fit(svmData, kernel, 0.1);
                assertNotNull(svm);
            }

            System.out.println("✓ All dependencies can work together in Java " +
                             Runtime.version().feature());
        } catch (Exception e) {
            fail("Integration test failed: " + e.getMessage());
        }
    }
}