package com.hellblazer.art.cortical.gpu.compute;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Optional;

/**
 * Automatic GPU backend selection for ART cortical system.
 * Selects optimal backend based on platform, availability, and performance characteristics.
 *
 * Priority order:
 * 1. Metal (macOS only, highest performance)
 * 2. OpenCL (cross-platform)
 * 3. CPU fallback (always available)
 */
public class BackendSelector {

    private static final Logger log = LoggerFactory.getLogger(BackendSelector.class);

    private static GPUBackend selectedBackend = null;
    private static boolean initialized = false;

    /**
     * Get the optimal GPU backend for the current platform.
     * Caches the result after first call.
     *
     * @return Selected backend
     */
    public static GPUBackend getOptimalBackend() {
        if (!initialized) {
            selectedBackend = selectBackend();
            initialized = true;

            log.info("GPU Backend Selection: {}", selectedBackend.getDisplayName());
            log.info("Platform: {}", getPlatformDescription());
            log.info("CI Environment: {}", isCIEnvironment());
        }
        return selectedBackend;
    }

    /**
     * Select the best available backend.
     *
     * @return Selected backend
     */
    private static GPUBackend selectBackend() {
        // Check environment variables for forced selection
        var forced = getForcedBackend();
        if (forced.isPresent()) {
            var backend = forced.get();
            log.info("Backend forced via environment: {}", backend);
            return backend;
        }

        // In CI, use CPU fallback
        if (isCIEnvironment()) {
            log.info("CI environment detected, using CPU fallback");
            return GPUBackend.CPU_FALLBACK;
        }

        // Check if GPU is disabled
        if (isGPUDisabled()) {
            log.info("GPU disabled via environment, using CPU fallback");
            return GPUBackend.CPU_FALLBACK;
        }

        // Select highest priority available backend
        return Arrays.stream(GPUBackend.values())
                     .filter(GPUBackend::isGPU)  // GPU backends only
                     .filter(GPUBackend::isAvailable)
                     .max(Comparator.comparingInt(GPUBackend::getPriority))
                     .orElse(GPUBackend.CPU_FALLBACK);
    }

    /**
     * Check if a specific backend is forced via environment variable.
     *
     * @return Forced backend, if any
     */
    private static Optional<GPUBackend> getForcedBackend() {
        var backend = System.getenv("ART_GPU_BACKEND");
        if (backend != null) {
            return switch (backend.toLowerCase()) {
                case "metal" -> Optional.of(GPUBackend.METAL);
                case "opencl" -> Optional.of(GPUBackend.OPENCL);
                case "cpu" -> Optional.of(GPUBackend.CPU_FALLBACK);
                default -> {
                    log.warn("Unknown backend specified: {}. Ignoring.", backend);
                    yield Optional.empty();
                }
            };
        }
        return Optional.empty();
    }

    /**
     * Check if GPU is disabled via environment variable.
     *
     * @return true if GPU is disabled
     */
    private static boolean isGPUDisabled() {
        var disabled = System.getenv("ART_GPU_DISABLE");
        return "true".equalsIgnoreCase(disabled) || "1".equals(disabled);
    }

    /**
     * Check if running in a CI environment.
     *
     * @return true if CI environment detected
     */
    public static boolean isCIEnvironment() {
        return System.getenv("CI") != null ||
               System.getenv("GITHUB_ACTIONS") != null ||
               System.getenv("JENKINS_URL") != null ||
               System.getenv("GITLAB_CI") != null ||
               System.getenv("TRAVIS") != null ||
               System.getenv("CIRCLECI") != null;
    }

    /**
     * Get platform description.
     *
     * @return Platform description
     */
    public static String getPlatformDescription() {
        var os = System.getProperty("os.name");
        var arch = System.getProperty("os.arch");
        return String.format("%s %s", os, arch);
    }

    /**
     * Check if Metal is available on this platform.
     *
     * @return true if Metal is available
     */
    public static boolean isMetalAvailable() {
        return GPUBackend.METAL.isAvailable();
    }

    /**
     * Check if OpenCL is available on this platform.
     *
     * @return true if OpenCL is available
     */
    public static boolean isOpenCLAvailable() {
        return GPUBackend.OPENCL.isAvailable();
    }

    /**
     * Get environment information for debugging.
     *
     * @return Environment description
     */
    public static String getEnvironmentInfo() {
        var sb = new StringBuilder();
        sb.append("Platform: ").append(getPlatformDescription()).append("\n");
        sb.append("CI: ").append(isCIEnvironment()).append("\n");
        sb.append("Metal Available: ").append(isMetalAvailable()).append("\n");
        sb.append("OpenCL Available: ").append(isOpenCLAvailable()).append("\n");
        sb.append("Selected Backend: ").append(getOptimalBackend().getDisplayName()).append("\n");
        return sb.toString();
    }

    /**
     * Reset backend selection (for testing).
     */
    public static void reset() {
        selectedBackend = null;
        initialized = false;
    }
}
