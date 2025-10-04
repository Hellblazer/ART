/**
 * Vector scalar multiplication kernel for testing OpenCL backend.
 * b[i] = a[i] * scale
 */
__kernel void vector_scale(
    __global const float* a,
    __global float* b,
    const float scale
) {
    int id = get_global_id(0);
    b[id] = a[id] * scale;
}
