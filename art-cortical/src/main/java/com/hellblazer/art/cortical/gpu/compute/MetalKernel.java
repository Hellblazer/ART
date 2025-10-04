package com.hellblazer.art.cortical.gpu.compute;

import com.hellblazer.art.cortical.gpu.memory.GPUBuffer;
import com.hellblazer.art.cortical.gpu.memory.MetalBuffer;
import org.lwjgl.bgfx.*;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

import static org.lwjgl.bgfx.BGFX.*;
import static org.lwjgl.system.MemoryStack.stackPush;

/**
 * Metal compute kernel implementation using bgfx.
 * Provides Metal shader compilation and execution on macOS.
 */
public class MetalKernel implements ComputeKernel {

    private static final Logger log = LoggerFactory.getLogger(MetalKernel.class);

    private final String name;
    private short programHandle = BGFX_INVALID_HANDLE;
    private short shaderHandle = BGFX_INVALID_HANDLE;
    private boolean compiled = false;
    private final Map<Integer, BufferBinding> bufferBindings = new HashMap<>();
    private final Map<Integer, Float> floatArgs = new HashMap<>();
    private final Map<Integer, Integer> intArgs = new HashMap<>();

    public MetalKernel(String name) {
        this.name = name;
    }

    @Override
    public void compile(String source, String entryPoint) throws KernelCompilationException {
        if (compiled) {
            throw new KernelCompilationException("Kernel already compiled");
        }

        try {
            // Convert source to ByteBuffer
            var sourceBuffer = MemoryUtil.memUTF8(source);
            var memory = bgfx_make_ref(sourceBuffer);

            // Create shader from Metal source
            shaderHandle = bgfx_create_shader(memory);
            if (shaderHandle == BGFX_INVALID_HANDLE) {
                throw new KernelCompilationException("Failed to create Metal shader: " + name);
            }

            // Create compute program from shader
            programHandle = bgfx_create_compute_program(shaderHandle, true);
            if (programHandle == BGFX_INVALID_HANDLE) {
                throw new KernelCompilationException("Failed to create compute program: " + name);
            }

            compiled = true;
            log.debug("Compiled Metal kernel: {}", name);

        } catch (Exception e) {
            cleanup();
            throw new KernelCompilationException("Metal kernel compilation failed: " + name, e);
        }
    }

    @Override
    public void setBufferArg(int index, GPUBuffer buffer, BufferAccess access) {
        if (!compiled) {
            throw new IllegalStateException("Kernel not compiled");
        }

        if (!(buffer instanceof MetalBuffer metalBuffer)) {
            throw new IllegalArgumentException("Buffer must be MetalBuffer");
        }

        var binding = new BufferBinding(metalBuffer, access);
        bufferBindings.put(index, binding);
    }

    @Override
    public void setFloatArg(int index, float value) {
        floatArgs.put(index, value);
    }

    @Override
    public void setIntArg(int index, int value) {
        intArgs.put(index, value);
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

        try {
            // Bind buffers
            for (var entry : bufferBindings.entrySet()) {
                var index = entry.getKey();
                var binding = entry.getValue();
                var accessFlags = switch (binding.access) {
                    case READ -> BGFX_ACCESS_READ;
                    case WRITE -> BGFX_ACCESS_WRITE;
                    case READ_WRITE -> BGFX_ACCESS_READ | BGFX_ACCESS_WRITE;
                };

                // Use bgfx_set_compute_dynamic_vertex_buffer for compute shaders
                bgfx_set_compute_dynamic_vertex_buffer(
                    (byte) index.intValue(),
                    binding.buffer.getHandle(),
                    accessFlags
                );
            }

            // TODO: Set uniform parameters (floatArgs, intArgs)
            // bgfx doesn't have a direct uniform API for compute shaders
            // May need to use uniform buffers instead

            // Dispatch compute
            var numGroups = (int) Math.ceil(globalWorkSizeX / 64.0);  // Assuming 64 threads per group
            bgfx_dispatch(
                0,  // View ID
                programHandle,
                numGroups,
                (int) Math.ceil(globalWorkSizeY / 8.0),
                (int) Math.ceil(globalWorkSizeZ / 8.0),
                (byte) 0  // Flags
            );

            log.trace("Dispatched Metal kernel: {} with work size [{}, {}, {}]",
                name, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ);

        } catch (Exception e) {
            throw new KernelExecutionException("Metal kernel execution failed: " + name, e);
        }
    }

    @Override
    public void finish() {
        // Submit frame and wait for completion
        bgfx_frame(false);
    }

    @Override
    public GPUBackend getBackend() {
        return GPUBackend.METAL;
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
        if (programHandle != BGFX_INVALID_HANDLE) {
            bgfx_destroy_program(programHandle);
            programHandle = BGFX_INVALID_HANDLE;
        }
        // Note: shader is destroyed when program is destroyed with destroyShaders=true
        shaderHandle = BGFX_INVALID_HANDLE;
        compiled = false;
        bufferBindings.clear();
        floatArgs.clear();
        intArgs.clear();
    }

    private record BufferBinding(MetalBuffer buffer, BufferAccess access) {
    }
}
