package com.hellblazer.art.cortical.gpu.validation;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * Framework for validating FP32 (GPU) vs FP64 (CPU) precision.
 * Tests learning convergence, classification accuracy, and numerical stability.
 */
public class PrecisionValidator {

    private static final Logger log = LoggerFactory.getLogger(PrecisionValidator.class);

    /**
     * Compare FP32 and FP64 implementations.
     *
     * @param testName Test name for logging
     * @param fp64Task Task to run with FP64
     * @param fp32Task Task to run with FP32
     * @param fp64Output Supplier for FP64 output
     * @param fp32Output Supplier for FP32 output
     * @param tolerance Acceptable tolerance
     * @return Validation result
     */
    public static ValidationResult compare(
        String testName,
        Runnable fp64Task,
        Runnable fp32Task,
        Supplier<double[]> fp64Output,
        Supplier<float[]> fp32Output,
        double tolerance
    ) {
        log.info("Running precision validation: {}", testName);

        // Run both implementations
        long fp64StartTime = System.nanoTime();
        fp64Task.run();
        long fp64Time = System.nanoTime() - fp64StartTime;

        long fp32StartTime = System.nanoTime();
        fp32Task.run();
        long fp32Time = System.nanoTime() - fp32StartTime;

        // Get outputs
        var fp64 = fp64Output.get();
        var fp32 = fp32Output.get();

        // Analyze results
        var result = analyzeResults(testName, fp64, fp32, tolerance, fp64Time, fp32Time);

        log.info("Validation result for {}: {}", testName,
            result.passed ? "PASS" : "FAIL");
        log.info("  Max error: %.6e, Avg error: %.6e, Violations: {}/{}",
            result.maxError, result.avgError, result.violations, fp64.length);
        log.info("  FP64 time: %.3f ms, FP32 time: %.3f ms, Speedup: %.2fx",
            fp64Time / 1e6, fp32Time / 1e6, (double) fp64Time / fp32Time);

        return result;
    }

    /**
     * Analyze precision differences between FP64 and FP32.
     */
    private static ValidationResult analyzeResults(
        String testName,
        double[] fp64,
        float[] fp32,
        double tolerance,
        long fp64TimeNanos,
        long fp32TimeNanos
    ) {
        if (fp64.length != fp32.length) {
            throw new IllegalArgumentException(
                String.format("Size mismatch: FP64=%d, FP32=%d", fp64.length, fp32.length)
            );
        }

        double maxError = 0.0;
        double sumError = 0.0;
        int violations = 0;
        List<ErrorSample> worstErrors = new ArrayList<>();

        for (int i = 0; i < fp64.length; i++) {
            double error = Math.abs(fp64[i] - fp32[i]);
            maxError = Math.max(maxError, error);
            sumError += error;

            if (error > tolerance) {
                violations++;
            }

            // Track worst errors
            if (worstErrors.size() < 10 || error > worstErrors.get(9).error) {
                worstErrors.add(new ErrorSample(i, fp64[i], fp32[i], error));
                worstErrors.sort((a, b) -> Double.compare(b.error, a.error));
                if (worstErrors.size() > 10) {
                    worstErrors.remove(10);
                }
            }
        }

        double avgError = sumError / fp64.length;
        boolean passed = maxError < tolerance;

        return new ValidationResult(
            testName,
            fp64.length,
            maxError,
            avgError,
            violations,
            passed,
            fp64TimeNanos,
            fp32TimeNanos,
            worstErrors
        );
    }

    /**
     * Validation result.
     */
    public static class ValidationResult {
        public final String testName;
        public final int elementCount;
        public final double maxError;
        public final double avgError;
        public final int violations;
        public final boolean passed;
        public final long fp64TimeNanos;
        public final long fp32TimeNanos;
        public final List<ErrorSample> worstErrors;

        public ValidationResult(
            String testName,
            int elementCount,
            double maxError,
            double avgError,
            int violations,
            boolean passed,
            long fp64TimeNanos,
            long fp32TimeNanos,
            List<ErrorSample> worstErrors
        ) {
            this.testName = testName;
            this.elementCount = elementCount;
            this.maxError = maxError;
            this.avgError = avgError;
            this.violations = violations;
            this.passed = passed;
            this.fp64TimeNanos = fp64TimeNanos;
            this.fp32TimeNanos = fp32TimeNanos;
            this.worstErrors = worstErrors;
        }

        public double getSpeedup() {
            return (double) fp64TimeNanos / fp32TimeNanos;
        }

        public double getViolationRate() {
            return (double) violations / elementCount;
        }

        @Override
        public String toString() {
            return String.format(
                "ValidationResult{test='%s', passed=%s, maxError=%.6e, avgError=%.6e, " +
                "violations=%d/%d (%.1f%%), speedup=%.2fx}",
                testName, passed, maxError, avgError,
                violations, elementCount, getViolationRate() * 100, getSpeedup()
            );
        }
    }

    /**
     * Error sample for detailed analysis.
     */
    public static class ErrorSample {
        public final int index;
        public final double fp64Value;
        public final float fp32Value;
        public final double error;

        public ErrorSample(int index, double fp64Value, float fp32Value, double error) {
            this.index = index;
            this.fp64Value = fp64Value;
            this.fp32Value = fp32Value;
            this.error = error;
        }

        @Override
        public String toString() {
            return String.format(
                "index=%d, fp64=%.10f, fp32=%.10f, error=%.6e",
                index, fp64Value, fp32Value, error
            );
        }
    }

    /**
     * Aggregate multiple validation results.
     */
    public static class ValidationSummary {
        private final List<ValidationResult> results = new ArrayList<>();

        public void add(ValidationResult result) {
            results.add(result);
        }

        public boolean allPassed() {
            return results.stream().allMatch(r -> r.passed);
        }

        public double getMaxError() {
            return results.stream().mapToDouble(r -> r.maxError).max().orElse(0.0);
        }

        public double getAvgSpeedup() {
            return results.stream().mapToDouble(ValidationResult::getSpeedup).average().orElse(1.0);
        }

        public int getTotalViolations() {
            return results.stream().mapToInt(r -> r.violations).sum();
        }

        public int getTotalElements() {
            return results.stream().mapToInt(r -> r.elementCount).sum();
        }

        public List<ValidationResult> getResults() {
            return List.copyOf(results);
        }

        @Override
        public String toString() {
            return String.format(
                "ValidationSummary{tests=%d, passed=%d/%d, maxError=%.6e, avgSpeedup=%.2fx, violations=%d/%d}",
                results.size(),
                results.stream().filter(r -> r.passed).count(),
                results.size(),
                getMaxError(),
                getAvgSpeedup(),
                getTotalViolations(),
                getTotalElements()
            );
        }
    }
}