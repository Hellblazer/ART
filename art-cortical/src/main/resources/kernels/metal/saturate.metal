#include <metal_stdlib>
using namespace metal;

/**
 * Saturation kernel for testing Metal backend.
 * Clamps values to [min, max] range.
 * b[i] = clamp(a[i], min_val, max_val)
 */
kernel void saturate(
    device const float* a [[buffer(0)]],
    device float* b [[buffer(1)]],
    constant float& min_val [[buffer(2)]],
    constant float& max_val [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    b[id] = clamp(a[id], min_val, max_val);
}
