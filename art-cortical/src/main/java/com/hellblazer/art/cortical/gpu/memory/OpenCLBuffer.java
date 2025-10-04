package com.hellblazer.art.cortical.gpu.memory;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.FloatBuffer;

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.*;

/**
 * OpenCL GPU buffer implementation.
 */
public class OpenCLBuffer implements GPUBuffer {

    private static final Logger log = LoggerFactory.getLogger(OpenCLBuffer.class);

    private final int size;  // Size in floats
    private final long context;
    private final long commandQueue;
    private long handle = NULL;
    private boolean valid = false;

    /**
     * Create an OpenCL buffer.
     *
     * @param size Size in floats
     * @param context OpenCL context
     * @param commandQueue OpenCL command queue
     * @param flags Buffer flags (CL_MEM_READ_WRITE, CL_MEM_READ_ONLY, etc.)
     */
    public OpenCLBuffer(int size, long context, long commandQueue, long flags) {
        this.size = size;
        this.context = context;
        this.commandQueue = commandQueue;
        allocate(flags);
    }

    private void allocate(long flags) {
        try (var stack = stackPush()) {
            var errcode = stack.mallocInt(1);
            var byteSize = (long) size * Float.BYTES;

            handle = clCreateBuffer(context, flags, byteSize, errcode);

            if (errcode.get(0) != CL_SUCCESS) {
                throw new RuntimeException(
                    String.format("Failed to create OpenCL buffer (error: %d)", errcode.get(0))
                );
            }

            valid = true;
            log.debug("Allocated OpenCL buffer: {} floats ({} bytes)", size, byteSize);
        }
    }

    @Override
    public void upload(FloatBuffer data) {
        if (!valid) {
            throw new IllegalStateException("Buffer not valid");
        }

        if (data.remaining() != size) {
            throw new IllegalArgumentException(
                String.format("Data size mismatch: expected %d, got %d", size, data.remaining())
            );
        }

        var byteSize = (long) size * Float.BYTES;
        var errcode = clEnqueueWriteBuffer(
            commandQueue,
            handle,
            true,  // blocking_write
            0,     // offset
            data,
            null,  // event_wait_list
            null   // event
        );

        if (errcode != CL_SUCCESS) {
            throw new RuntimeException(
                String.format("Failed to upload to OpenCL buffer (error: %d)", errcode)
            );
        }

        log.trace("Uploaded {} floats to OpenCL buffer", size);
    }

    @Override
    public void upload(float[] data) {
        if (data.length != size) {
            throw new IllegalArgumentException(
                String.format("Data size mismatch: expected %d, got %d", size, data.length)
            );
        }

        var buffer = MemoryUtil.memAllocFloat(size);
        try {
            buffer.put(data).flip();
            upload(buffer);
        } finally {
            MemoryUtil.memFree(buffer);
        }
    }

    @Override
    public void download(FloatBuffer data) {
        if (!valid) {
            throw new IllegalStateException("Buffer not valid");
        }

        if (data.remaining() != size) {
            throw new IllegalArgumentException(
                String.format("Data size mismatch: expected %d, got %d", size, data.remaining())
            );
        }

        var errcode = clEnqueueReadBuffer(
            commandQueue,
            handle,
            true,  // blocking_read
            0,     // offset
            data,
            null,  // event_wait_list
            null   // event
        );

        if (errcode != CL_SUCCESS) {
            throw new RuntimeException(
                String.format("Failed to download from OpenCL buffer (error: %d)", errcode)
            );
        }

        log.trace("Downloaded {} floats from OpenCL buffer", size);
    }

    @Override
    public void download(float[] data) {
        if (data.length != size) {
            throw new IllegalArgumentException(
                String.format("Data size mismatch: expected %d, got %d", size, data.length)
            );
        }

        var buffer = MemoryUtil.memAllocFloat(size);
        try {
            download(buffer);
            buffer.get(data);
        } finally {
            MemoryUtil.memFree(buffer);
        }
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int sizeInBytes() {
        return size * Float.BYTES;
    }

    @Override
    public boolean isValid() {
        return valid;
    }

    /**
     * Get the OpenCL buffer handle.
     * Public for use by OpenCLKernel.
     *
     * @return Buffer handle
     */
    public long getHandle() {
        return handle;
    }

    @Override
    public void close() {
        if (valid && handle != NULL) {
            clReleaseMemObject(handle);
            handle = NULL;
            valid = false;
            log.debug("Destroyed OpenCL buffer: {} floats", size);
        }
    }
}
