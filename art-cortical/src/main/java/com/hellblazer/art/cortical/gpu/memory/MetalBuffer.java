package com.hellblazer.art.cortical.gpu.memory;

import org.lwjgl.bgfx.BGFXMemory;
import org.lwjgl.system.MemoryUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

import static org.lwjgl.bgfx.BGFX.*;

/**
 * Metal GPU buffer implementation using bgfx.
 */
public class MetalBuffer implements GPUBuffer {

    private static final Logger log = LoggerFactory.getLogger(MetalBuffer.class);

    private final int size;  // Size in floats
    private short handle = BGFX_INVALID_HANDLE;
    private boolean valid = false;

    /**
     * Create a Metal buffer.
     *
     * @param size Size in floats
     * @param flags Buffer flags (BGFX_BUFFER_COMPUTE_READ, BGFX_BUFFER_COMPUTE_WRITE, etc.)
     */
    public MetalBuffer(int size, int flags) {
        this.size = size;
        allocate(flags);
    }

    private void allocate(int flags) {
        // Allocate host memory
        var byteSize = size * Float.BYTES;
        var buffer = MemoryUtil.memAlloc(byteSize);

        try {
            // Create bgfx memory reference
            var memory = bgfx_make_ref(buffer);

            // Create dynamic vertex buffer (used for compute buffers)
            handle = bgfx_create_dynamic_vertex_buffer_mem(memory, null, flags);

            if (handle == BGFX_INVALID_HANDLE) {
                throw new RuntimeException("Failed to create Metal buffer");
            }

            valid = true;
            log.debug("Allocated Metal buffer: {} floats ({} bytes)", size, byteSize);

        } finally {
            MemoryUtil.memFree(buffer);
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

        // Convert FloatBuffer to ByteBuffer
        var byteBuffer = MemoryUtil.memAlloc(size * Float.BYTES);
        try {
            byteBuffer.asFloatBuffer().put(data);
            byteBuffer.flip();

            // Update dynamic buffer
            var memory = bgfx_make_ref(byteBuffer);
            bgfx_update_dynamic_vertex_buffer(handle, 0, memory);

            log.trace("Uploaded {} floats to Metal buffer", size);
        } finally {
            MemoryUtil.memFree(byteBuffer);
        }
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

        // Note: bgfx doesn't provide direct buffer read functionality
        // This is a limitation of the API - typically you'd use a staging buffer
        // or read back via a frame capture mechanism

        log.warn("Metal buffer download not fully supported by bgfx - results may not be current");

        // This is a placeholder - actual implementation would need:
        // 1. Create staging buffer
        // 2. Copy compute buffer to staging
        // 3. Map staging buffer
        // 4. Read data
        // 5. Unmap staging buffer

        throw new UnsupportedOperationException(
            "Metal buffer download requires staging buffer implementation"
        );
    }

    @Override
    public void download(float[] data) {
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
     * Get the bgfx buffer handle.
     * Public for use by MetalKernel.
     *
     * @return Buffer handle
     */
    public short getHandle() {
        return handle;
    }

    @Override
    public void close() {
        if (valid && handle != BGFX_INVALID_HANDLE) {
            bgfx_destroy_dynamic_vertex_buffer(handle);
            handle = BGFX_INVALID_HANDLE;
            valid = false;
            log.debug("Destroyed Metal buffer: {} floats", size);
        }
    }
}
