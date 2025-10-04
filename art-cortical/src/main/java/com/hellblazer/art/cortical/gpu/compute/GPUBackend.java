package com.hellblazer.art.cortical.gpu.compute;

/**
 * Supported GPU compute backends.
 */
public enum GPUBackend {
    /**
     * Metal 3 (macOS only, highest performance).
     */
    METAL("Metal", 100, true),

    /**
     * OpenCL 1.2+ (cross-platform).
     */
    OPENCL("OpenCL", 90, true),

    /**
     * CPU fallback (no GPU required).
     */
    CPU_FALLBACK("CPU Fallback", 10, false);

    private final String displayName;
    private final int priority;
    private final boolean isGPU;

    GPUBackend(String displayName, int priority, boolean isGPU) {
        this.displayName = displayName;
        this.priority = priority;
        this.isGPU = isGPU;
    }

    public String getDisplayName() {
        return displayName;
    }

    public int getPriority() {
        return priority;
    }

    public boolean isGPU() {
        return isGPU;
    }

    /**
     * Check if this backend is available on the current platform.
     *
     * @return true if backend is available
     */
    public boolean isAvailable() {
        return switch (this) {
            case METAL -> isMetalAvailable();
            case OPENCL -> isOpenCLAvailable();
            case CPU_FALLBACK -> true;  // Always available
        };
    }

    private static boolean isMetalAvailable() {
        var os = System.getProperty("os.name").toLowerCase();
        if (!os.contains("mac")) {
            return false;  // Metal only on macOS
        }

        // Check if bgfx is available
        try {
            Class.forName("org.lwjgl.bgfx.BGFX");
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }

    private static boolean isOpenCLAvailable() {
        // Check if OpenCL is available
        try {
            Class.forName("org.lwjgl.opencl.CL");
            // TODO: Actually check for OpenCL platform/device availability
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }
}
