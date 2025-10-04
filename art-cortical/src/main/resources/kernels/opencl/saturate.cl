/**
 * Saturation kernel for testing OpenCL backend.
 * Clamps values to [min, max] range.
 * b[i] = clamp(a[i], min_val, max_val)
 */
__kernel void saturate(
    __global const float* a,
    __global float* b,
    const float min_val,
    const float max_val
) {
    int id = get_global_id(0);
    b[id] = clamp(a[id], min_val, max_val);
}
