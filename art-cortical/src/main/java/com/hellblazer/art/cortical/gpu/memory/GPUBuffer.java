package com.hellblazer.art.cortical.gpu.memory;

import java.nio.FloatBuffer;

/**
 * GPU buffer abstraction for Metal and OpenCL backends.
 * Manages host-device memory transfers and buffer lifecycle.
 */
public interface GPUBuffer extends AutoCloseable {

    /**
     * Upload data from host to device.
     *
     * @param data Host data to upload
     */
    void upload(FloatBuffer data);

    /**
     * Upload data from host to device.
     *
     * @param data Host data to upload
     */
    void upload(float[] data);

    /**
     * Download data from device to host.
     *
     * @param data Host buffer to receive data
     */
    void download(FloatBuffer data);

    /**
     * Download data from device to host.
     *
     * @param data Host array to receive data
     */
    void download(float[] data);

    /**
     * Get the size of the buffer in elements (floats).
     *
     * @return Buffer size
     */
    int size();

    /**
     * Get the size of the buffer in bytes.
     *
     * @return Buffer size in bytes
     */
    int sizeInBytes();

    /**
     * Check if the buffer is valid and allocated.
     *
     * @return true if buffer is valid
     */
    boolean isValid();

    /**
     * Release GPU resources.
     */
    @Override
    void close();
}
