package com.hellblazer.art.cortical.gpu.kernels;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.stream.Collectors;

/**
 * Utility for loading GPU kernel source code from resources.
 */
public class KernelLoader {

    private static final Logger log = LoggerFactory.getLogger(KernelLoader.class);

    /**
     * Load a Metal kernel from resources.
     *
     * @param kernelName Kernel file name (e.g., "vector_add")
     * @return Kernel source code
     * @throws IOException if kernel file not found or cannot be read
     */
    public static String loadMetalKernel(String kernelName) throws IOException {
        return loadKernel("kernels/metal/" + kernelName + ".metal");
    }

    /**
     * Load an OpenCL kernel from resources.
     *
     * @param kernelName Kernel file name (e.g., "vector_add")
     * @return Kernel source code
     * @throws IOException if kernel file not found or cannot be read
     */
    public static String loadOpenCLKernel(String kernelName) throws IOException {
        return loadKernel("kernels/opencl/" + kernelName + ".cl");
    }

    /**
     * Load a kernel from resources.
     *
     * @param resourcePath Resource path (e.g., "kernels/metal/vector_add.metal")
     * @return Kernel source code
     * @throws IOException if kernel file not found or cannot be read
     */
    public static String loadKernel(String resourcePath) throws IOException {
        log.debug("Loading kernel: {}", resourcePath);

        var classLoader = KernelLoader.class.getClassLoader();
        var inputStream = classLoader.getResourceAsStream(resourcePath);

        if (inputStream == null) {
            throw new IOException("Kernel not found: " + resourcePath);
        }

        try (var reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
            var source = reader.lines().collect(Collectors.joining("\n"));
            log.debug("Loaded kernel {} ({} bytes)", resourcePath, source.length());
            return source;
        }
    }
}
