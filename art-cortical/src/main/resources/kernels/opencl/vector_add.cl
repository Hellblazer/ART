/**
 * Simple vector addition kernel for testing OpenCL backend.
 * c[i] = a[i] + b[i]
 */
__kernel void vector_add(
    __global const float* a,
    __global const float* b,
    __global float* c
) {
    int id = get_global_id(0);
    c[id] = a[id] + b[id];
}
