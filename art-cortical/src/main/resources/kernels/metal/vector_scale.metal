#include <metal_stdlib>
using namespace metal;

/**
 * Vector scalar multiplication kernel for testing Metal backend.
 * b[i] = a[i] * scale
 */
kernel void vector_scale(
    device const float* a [[buffer(0)]],
    device float* b [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    b[id] = a[id] * scale;
}
