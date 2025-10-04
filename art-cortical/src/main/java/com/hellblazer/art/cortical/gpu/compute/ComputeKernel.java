package com.hellblazer.art.cortical.gpu.compute;

import com.hellblazer.art.cortical.gpu.memory.GPUBuffer;

import java.nio.FloatBuffer;

/**
 * Unified interface for GPU compute kernels across Metal and OpenCL backends.
 * Provides a common API for executing compute operations on the GPU.
 */
public interface ComputeKernel extends AutoCloseable {

    /**
     * Compile the kernel from source code.
     *
     * @param source Kernel source code (Metal or OpenCL)
     * @param entryPoint Kernel entry point function name
     * @throws KernelCompilationException if compilation fails
     */
    void compile(String source, String entryPoint) throws KernelCompilationException;

    /**
     * Set a buffer argument for the kernel.
     *
     * @param index Argument index (0-based)
     * @param buffer Buffer to bind
     * @param access Access mode (READ, WRITE, READ_WRITE)
     */
    void setBufferArg(int index, GPUBuffer buffer, BufferAccess access);

    /**
     * Set a scalar float argument for the kernel.
     *
     * @param index Argument index (0-based)
     * @param value Float value
     */
    void setFloatArg(int index, float value);

    /**
     * Set a scalar int argument for the kernel.
     *
     * @param index Argument index (0-based)
     * @param value Int value
     */
    void setIntArg(int index, int value);

    /**
     * Execute the kernel with specified global work size.
     *
     * @param globalWorkSize Number of work items (1D)
     * @throws KernelExecutionException if execution fails
     */
    void execute(int globalWorkSize) throws KernelExecutionException;

    /**
     * Execute the kernel with specified global work size (2D).
     *
     * @param globalWorkSizeX Number of work items in X dimension
     * @param globalWorkSizeY Number of work items in Y dimension
     * @throws KernelExecutionException if execution fails
     */
    void execute(int globalWorkSizeX, int globalWorkSizeY) throws KernelExecutionException;

    /**
     * Execute the kernel with specified global work size (3D).
     *
     * @param globalWorkSizeX Number of work items in X dimension
     * @param globalWorkSizeY Number of work items in Y dimension
     * @param globalWorkSizeZ Number of work items in Z dimension
     * @throws KernelExecutionException if execution fails
     */
    void execute(int globalWorkSizeX, int globalWorkSizeY, int globalWorkSizeZ)
        throws KernelExecutionException;

    /**
     * Wait for kernel execution to complete.
     * Blocks until all queued operations finish.
     */
    void finish();

    /**
     * Get the backend type for this kernel.
     *
     * @return Backend type (METAL or OPENCL)
     */
    GPUBackend getBackend();

    /**
     * Check if the kernel is compiled and ready to execute.
     *
     * @return true if kernel is compiled
     */
    boolean isCompiled();

    /**
     * Release GPU resources.
     */
    @Override
    void close();

    /**
     * Buffer access modes.
     */
    enum BufferAccess {
        READ,
        WRITE,
        READ_WRITE
    }

    /**
     * Kernel compilation exception.
     */
    class KernelCompilationException extends Exception {
        public KernelCompilationException(String message) {
            super(message);
        }

        public KernelCompilationException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    /**
     * Kernel execution exception.
     */
    class KernelExecutionException extends Exception {
        public KernelExecutionException(String message) {
            super(message);
        }

        public KernelExecutionException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
