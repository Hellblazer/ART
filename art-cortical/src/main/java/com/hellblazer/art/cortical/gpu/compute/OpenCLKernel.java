package com.hellblazer.art.cortical.gpu.compute;

import com.hellblazer.art.cortical.gpu.memory.GPUBuffer;
import com.hellblazer.art.cortical.gpu.memory.OpenCLBuffer;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.IntBuffer;
import java.util.HashMap;
import java.util.Map;

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.*;

/**
 * OpenCL compute kernel implementation.
 * Provides OpenCL kernel compilation and execution for cross-platform GPU compute.
 */
public class OpenCLKernel implements ComputeKernel {

    private static final Logger log = LoggerFactory.getLogger(OpenCLKernel.class);

    private final String name;
    private final long context;
    private final long commandQueue;
    private long program = NULL;
    private long kernel = NULL;
    private boolean compiled = false;
    private final Map<Integer, BufferBinding> bufferBindings = new HashMap<>();

    public OpenCLKernel(String name, long context, long commandQueue) {
        this.name = name;
        this.context = context;
        this.commandQueue = commandQueue;
    }

    @Override
    public void compile(String source, String entryPoint) throws KernelCompilationException {
        if (compiled) {
            throw new KernelCompilationException("Kernel already compiled");
        }

        try (var stack = stackPush()) {
            // Create program from source
            var errcode = stack.mallocInt(1);
            program = clCreateProgramWithSource(context, source, errcode);
            checkCLError(errcode.get(0), "Failed to create OpenCL program");

            // Build program
            var buildStatus = clBuildProgram(program, (long) 0, "", null, NULL);
            if (buildStatus != CL_SUCCESS) {
                // Get build log
                var logSize = stack.mallocPointer(1);
                clGetProgramBuildInfo(program, 0, CL_PROGRAM_BUILD_LOG, (PointerBuffer) null, logSize);

                var log = stack.malloc((int) logSize.get(0));
                clGetProgramBuildInfo(program, 0, CL_PROGRAM_BUILD_LOG, log, null);

                var buildLog = memUTF8(log);
                throw new KernelCompilationException(
                    "OpenCL kernel compilation failed:\n" + buildLog
                );
            }

            // Create kernel
            kernel = clCreateKernel(program, entryPoint, errcode);
            checkCLError(errcode.get(0), "Failed to create OpenCL kernel: " + entryPoint);

            compiled = true;
            log.debug("Compiled OpenCL kernel: {} (entry point: {})", name, entryPoint);

        } catch (Exception e) {
            cleanup();
            if (e instanceof KernelCompilationException kce) {
                throw kce;
            }
            throw new KernelCompilationException("OpenCL kernel compilation failed: " + name, e);
        }
    }

    @Override
    public void setBufferArg(int index, GPUBuffer buffer, BufferAccess access) {
        if (!compiled) {
            throw new IllegalStateException("Kernel not compiled");
        }

        if (!(buffer instanceof OpenCLBuffer openCLBuffer)) {
            throw new IllegalArgumentException("Buffer must be OpenCLBuffer");
        }

        var binding = new BufferBinding(openCLBuffer, access);
        bufferBindings.put(index, binding);

        // Set kernel argument
        try {
            checkCLError(
                clSetKernelArg1p(kernel, index, openCLBuffer.getHandle()),
                "Failed to set buffer argument " + index
            );
        } catch (KernelCompilationException e) {
            throw new RuntimeException(e);  // Convert to unchecked for setter
        }
    }

    @Override
    public void setFloatArg(int index, float value) {
        if (!compiled) {
            throw new IllegalStateException("Kernel not compiled");
        }

        try (var stack = stackPush()) {
            var buffer = stack.mallocFloat(1).put(0, value);
            try {
                checkCLError(
                    clSetKernelArg(kernel, index, buffer),
                    "Failed to set float argument " + index
                );
            } catch (KernelCompilationException e) {
                throw new RuntimeException(e);  // Convert to unchecked for setter
            }
        }
    }

    @Override
    public void setIntArg(int index, int value) {
        if (!compiled) {
            throw new IllegalStateException("Kernel not compiled");
        }

        try (var stack = stackPush()) {
            var buffer = stack.mallocInt(1).put(0, value);
            try {
                checkCLError(
                    clSetKernelArg(kernel, index, buffer),
                    "Failed to set int argument " + index
                );
            } catch (KernelCompilationException e) {
                throw new RuntimeException(e);  // Convert to unchecked for setter
            }
        }
    }

    @Override
    public void execute(int globalWorkSize) throws KernelExecutionException {
        execute(globalWorkSize, 1, 1);
    }

    @Override
    public void execute(int globalWorkSizeX, int globalWorkSizeY) throws KernelExecutionException {
        execute(globalWorkSizeX, globalWorkSizeY, 1);
    }

    @Override
    public void execute(int globalWorkSizeX, int globalWorkSizeY, int globalWorkSizeZ)
        throws KernelExecutionException {

        if (!compiled) {
            throw new KernelExecutionException("Kernel not compiled");
        }

        try (var stack = stackPush()) {
            // Set global work size
            var globalWorkSize = stack.mallocPointer(3);
            globalWorkSize.put(0, globalWorkSizeX);
            globalWorkSize.put(1, globalWorkSizeY);
            globalWorkSize.put(2, globalWorkSizeZ);

            // Enqueue kernel
            var errcode = clEnqueueNDRangeKernel(
                commandQueue,
                kernel,
                3,  // work_dim
                null,  // global_work_offset
                globalWorkSize,
                null,  // local_work_size (auto-select)
                null,  // event_wait_list
                null   // event
            );

            checkCLError(errcode, "Failed to enqueue OpenCL kernel");

            log.trace("Executed OpenCL kernel: {} with work size [{}, {}, {}]",
                name, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ);

        } catch (Exception e) {
            throw new KernelExecutionException("OpenCL kernel execution failed: " + name, e);
        }
    }

    @Override
    public void finish() {
        // Wait for command queue to finish
        clFinish(commandQueue);
    }

    @Override
    public GPUBackend getBackend() {
        return GPUBackend.OPENCL;
    }

    @Override
    public boolean isCompiled() {
        return compiled;
    }

    @Override
    public void close() {
        cleanup();
    }

    private void cleanup() {
        if (kernel != NULL) {
            clReleaseKernel(kernel);
            kernel = NULL;
        }
        if (program != NULL) {
            clReleaseProgram(program);
            program = NULL;
        }
        compiled = false;
        bufferBindings.clear();
    }

    private void checkCLError(int errcode, String message) throws KernelCompilationException {
        if (errcode != CL_SUCCESS) {
            throw new KernelCompilationException(
                String.format("%s (error code: %d)", message, errcode)
            );
        }
    }

    private record BufferBinding(OpenCLBuffer buffer, BufferAccess access) {
    }
}
