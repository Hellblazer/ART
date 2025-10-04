package com.hellblazer.art.cortical.gpu.compute;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for GPU backend selection logic.
 */
class BackendSelectorTest {

    private static final Logger log = LoggerFactory.getLogger(BackendSelectorTest.class);

    @BeforeEach
    void setUp() {
        // Reset backend selection before each test
        BackendSelector.reset();
    }

    @AfterEach
    void tearDown() {
        // Clean up any environment variable overrides
        BackendSelector.reset();
    }

    @Test
    void testBackendSelection() {
        log.info("Testing backend selection");

        var backend = BackendSelector.getOptimalBackend();
        assertNotNull(backend, "Backend should not be null");

        log.info("Selected backend: {}", backend.getDisplayName());
        log.info("Backend priority: {}", backend.getPriority());
        log.info("Is GPU: {}", backend.isGPU());

        // Backend should be available
        assertTrue(backend.isAvailable(),
            "Selected backend should be available: " + backend);
    }

    @Test
    void testPlatformDetection() {
        log.info("Testing platform detection");

        var platform = BackendSelector.getPlatformDescription();
        assertNotNull(platform, "Platform should not be null");
        assertFalse(platform.isBlank(), "Platform should not be blank");

        log.info("Platform: {}", platform);

        // On macOS, Metal should be available
        var os = System.getProperty("os.name").toLowerCase();
        if (os.contains("mac")) {
            log.info("macOS detected, checking Metal availability");
            var metalAvailable = BackendSelector.isMetalAvailable();
            log.info("Metal available: {}", metalAvailable);

            // Metal should be selected on macOS (unless in CI)
            if (!BackendSelector.isCIEnvironment()) {
                var backend = BackendSelector.getOptimalBackend();
                if (metalAvailable) {
                    assertEquals(GPUBackend.METAL, backend,
                        "Metal should be selected on macOS");
                }
            }
        }
    }

    @Test
    void testCIDetection() {
        log.info("Testing CI environment detection");

        var isCI = BackendSelector.isCIEnvironment();
        log.info("CI environment: {}", isCI);

        if (isCI) {
            // In CI, should use CPU fallback
            var backend = BackendSelector.getOptimalBackend();
            assertEquals(GPUBackend.CPU_FALLBACK, backend,
                "CI environment should use CPU fallback");
        }
    }

    @Test
    void testBackendAvailability() {
        log.info("Testing backend availability");

        // CPU fallback always available
        assertTrue(GPUBackend.CPU_FALLBACK.isAvailable(),
            "CPU fallback should always be available");

        var os = System.getProperty("os.name").toLowerCase();

        // Metal only on macOS
        if (os.contains("mac")) {
            log.info("macOS detected, Metal may be available");
            var metalAvailable = GPUBackend.METAL.isAvailable();
            log.info("Metal available: {}", metalAvailable);
        } else {
            log.info("Non-macOS platform, Metal should not be available");
            assertFalse(GPUBackend.METAL.isAvailable(),
                "Metal should not be available on non-macOS");
        }

        // OpenCL availability depends on drivers
        var openclAvailable = GPUBackend.OPENCL.isAvailable();
        log.info("OpenCL available: {}", openclAvailable);
    }

    @Test
    void testEnvironmentInfo() {
        log.info("Testing environment info");

        var info = BackendSelector.getEnvironmentInfo();
        assertNotNull(info, "Environment info should not be null");
        assertFalse(info.isBlank(), "Environment info should not be blank");

        log.info("Environment Info:\n{}", info);

        // Should contain key information
        assertTrue(info.contains("Platform:"), "Should contain platform info");
        assertTrue(info.contains("Selected Backend:"), "Should contain backend info");
    }

    @Test
    void testBackendCaching() {
        log.info("Testing backend selection caching");

        // First call
        var backend1 = BackendSelector.getOptimalBackend();

        // Second call should return same instance
        var backend2 = BackendSelector.getOptimalBackend();

        assertSame(backend1, backend2,
            "Backend selection should be cached");
    }
}
