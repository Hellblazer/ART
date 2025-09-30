/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 *
 * This file is part of Java ART Neural Networks.
 *
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.performance;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * High-performance vectorized operations for layer processing.
 * Uses Java Vector API (SIMD) to accelerate array operations that are
 * the bottleneck in layer computations (80% of runtime).
 *
 * Provides 2-4x speedup for patterns >= 64D.
 * For smaller patterns, falls back to scalar operations to avoid SIMD overhead.
 *
 * All operations are semantically equivalent to their scalar counterparts
 * within floating-point tolerance (1e-10).
 *
 * @author Hal Hildebrand
 */
public final class VectorizedArrayOperations {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int SIMD_THRESHOLD = 64; // Minimum size for SIMD benefit

    private VectorizedArrayOperations() {
        // Utility class - no instantiation
    }

    /**
     * Scale array elements by a scalar value: result[i] = array[i] * scalar
     *
     * @param array input array
     * @param scalar multiplier
     * @return new array with scaled values
     */
    public static double[] scale(double[] array, double scalar) {
        var result = new double[array.length];

        if (array.length >= SIMD_THRESHOLD) {
            // Vectorized path
            int i = 0;
            int upperBound = SPECIES.loopBound(array.length);

            for (; i < upperBound; i += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, array, i);
                va.mul(scalar).intoArray(result, i);
            }

            // Scalar cleanup for remaining elements
            for (; i < array.length; i++) {
                result[i] = array[i] * scalar;
            }
        } else {
            // Scalar path for small arrays
            for (int i = 0; i < array.length; i++) {
                result[i] = array[i] * scalar;
            }
        }

        return result;
    }

    /**
     * Scale array elements in-place: array[i] *= scalar
     *
     * @param array input/output array (modified in place)
     * @param scalar multiplier
     */
    public static void scaleInPlace(double[] array, double scalar) {
        if (array.length >= SIMD_THRESHOLD) {
            // Vectorized path
            int i = 0;
            int upperBound = SPECIES.loopBound(array.length);

            for (; i < upperBound; i += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, array, i);
                va.mul(scalar).intoArray(array, i);
            }

            // Scalar cleanup
            for (; i < array.length; i++) {
                array[i] *= scalar;
            }
        } else {
            // Scalar path
            for (int i = 0; i < array.length; i++) {
                array[i] *= scalar;
            }
        }
    }

    /**
     * Add two arrays element-wise: result[i] = a[i] + b[i]
     *
     * @param a first array
     * @param b second array (must match length)
     * @return new array with element-wise sum
     */
    public static double[] add(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Array lengths must match: " + a.length + " vs " + b.length);
        }

        var result = new double[a.length];

        if (a.length >= SIMD_THRESHOLD) {
            // Vectorized path
            int i = 0;
            int upperBound = SPECIES.loopBound(a.length);

            for (; i < upperBound; i += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, a, i);
                var vb = DoubleVector.fromArray(SPECIES, b, i);
                va.add(vb).intoArray(result, i);
            }

            // Scalar cleanup
            for (; i < a.length; i++) {
                result[i] = a[i] + b[i];
            }
        } else {
            // Scalar path
            for (int i = 0; i < a.length; i++) {
                result[i] = a[i] + b[i];
            }
        }

        return result;
    }

    /**
     * Multiply two arrays element-wise: result[i] = a[i] * b[i]
     *
     * @param a first array
     * @param b second array (must match length)
     * @return new array with element-wise product
     */
    public static double[] multiply(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Array lengths must match: " + a.length + " vs " + b.length);
        }

        var result = new double[a.length];

        if (a.length >= SIMD_THRESHOLD) {
            // Vectorized path
            int i = 0;
            int upperBound = SPECIES.loopBound(a.length);

            for (; i < upperBound; i += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, a, i);
                var vb = DoubleVector.fromArray(SPECIES, b, i);
                va.mul(vb).intoArray(result, i);
            }

            // Scalar cleanup
            for (; i < a.length; i++) {
                result[i] = a[i] * b[i];
            }
        } else {
            // Scalar path
            for (int i = 0; i < a.length; i++) {
                result[i] = a[i] * b[i];
            }
        }

        return result;
    }

    /**
     * Element-wise minimum of two arrays: result[i] = min(a[i], b[i])
     *
     * @param a first array
     * @param b second array (must match length)
     * @return new array with element-wise minimum
     */
    public static double[] min(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Array lengths must match: " + a.length + " vs " + b.length);
        }

        var result = new double[a.length];

        if (a.length >= SIMD_THRESHOLD) {
            // Vectorized path
            int i = 0;
            int upperBound = SPECIES.loopBound(a.length);

            for (; i < upperBound; i += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, a, i);
                var vb = DoubleVector.fromArray(SPECIES, b, i);
                va.min(vb).intoArray(result, i);
            }

            // Scalar cleanup
            for (; i < a.length; i++) {
                result[i] = Math.min(a[i], b[i]);
            }
        } else {
            // Scalar path
            for (int i = 0; i < a.length; i++) {
                result[i] = Math.min(a[i], b[i]);
            }
        }

        return result;
    }

    /**
     * Element-wise maximum of two arrays: result[i] = max(a[i], b[i])
     *
     * @param a first array
     * @param b second array (must match length)
     * @return new array with element-wise maximum
     */
    public static double[] max(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Array lengths must match: " + a.length + " vs " + b.length);
        }

        var result = new double[a.length];

        if (a.length >= SIMD_THRESHOLD) {
            // Vectorized path
            int i = 0;
            int upperBound = SPECIES.loopBound(a.length);

            for (; i < upperBound; i += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, a, i);
                var vb = DoubleVector.fromArray(SPECIES, b, i);
                va.max(vb).intoArray(result, i);
            }

            // Scalar cleanup
            for (; i < a.length; i++) {
                result[i] = Math.max(a[i], b[i]);
            }
        } else {
            // Scalar path
            for (int i = 0; i < a.length; i++) {
                result[i] = Math.max(a[i], b[i]);
            }
        }

        return result;
    }

    /**
     * Clamp array elements to [min, max] range: result[i] = clamp(array[i], min, max)
     *
     * @param array input array
     * @param min minimum value
     * @param max maximum value
     * @return new array with clamped values
     */
    public static double[] clamp(double[] array, double min, double max) {
        var result = new double[array.length];

        if (array.length >= SIMD_THRESHOLD) {
            // Vectorized path
            int i = 0;
            int upperBound = SPECIES.loopBound(array.length);
            var vmin = DoubleVector.broadcast(SPECIES, min);
            var vmax = DoubleVector.broadcast(SPECIES, max);

            for (; i < upperBound; i += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, array, i);
                va.max(vmin).min(vmax).intoArray(result, i);
            }

            // Scalar cleanup
            for (; i < array.length; i++) {
                result[i] = Math.max(min, Math.min(max, array[i]));
            }
        } else {
            // Scalar path
            for (int i = 0; i < array.length; i++) {
                result[i] = Math.max(min, Math.min(max, array[i]));
            }
        }

        return result;
    }

    /**
     * Clamp array elements in-place: array[i] = clamp(array[i], min, max)
     *
     * @param array input/output array (modified in place)
     * @param min minimum value
     * @param max maximum value
     */
    public static void clampInPlace(double[] array, double min, double max) {
        if (array.length >= SIMD_THRESHOLD) {
            // Vectorized path
            int i = 0;
            int upperBound = SPECIES.loopBound(array.length);
            var vmin = DoubleVector.broadcast(SPECIES, min);
            var vmax = DoubleVector.broadcast(SPECIES, max);

            for (; i < upperBound; i += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, array, i);
                va.max(vmin).min(vmax).intoArray(array, i);
            }

            // Scalar cleanup
            for (; i < array.length; i++) {
                array[i] = Math.max(min, Math.min(max, array[i]));
            }
        } else {
            // Scalar path
            for (int i = 0; i < array.length; i++) {
                array[i] = Math.max(min, Math.min(max, array[i]));
            }
        }
    }

    /**
     * Sum all elements in array: result = sum(array[i])
     *
     * @param array input array
     * @return sum of all elements
     */
    public static double sum(double[] array) {
        double result = 0.0;

        if (array.length >= SIMD_THRESHOLD) {
            // Vectorized path
            int i = 0;
            int upperBound = SPECIES.loopBound(array.length);

            for (; i < upperBound; i += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, array, i);
                result += va.reduceLanes(VectorOperators.ADD);
            }

            // Scalar cleanup
            for (; i < array.length; i++) {
                result += array[i];
            }
        } else {
            // Scalar path
            for (int i = 0; i < array.length; i++) {
                result += array[i];
            }
        }

        return result;
    }

    /**
     * Fused multiply-add: result[i] = a[i] * b[i] + c[i]
     *
     * @param a first array
     * @param b second array (must match length)
     * @param c third array (must match length)
     * @return new array with fused multiply-add result
     */
    public static double[] fma(double[] a, double[] b, double[] c) {
        if (a.length != b.length || a.length != c.length) {
            throw new IllegalArgumentException("Array lengths must match");
        }

        var result = new double[a.length];

        if (a.length >= SIMD_THRESHOLD) {
            // Vectorized path
            int i = 0;
            int upperBound = SPECIES.loopBound(a.length);

            for (; i < upperBound; i += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, a, i);
                var vb = DoubleVector.fromArray(SPECIES, b, i);
                var vc = DoubleVector.fromArray(SPECIES, c, i);
                va.fma(vb, vc).intoArray(result, i);
            }

            // Scalar cleanup
            for (; i < a.length; i++) {
                result[i] = Math.fma(a[i], b[i], c[i]);
            }
        } else {
            // Scalar path
            for (int i = 0; i < a.length; i++) {
                result[i] = Math.fma(a[i], b[i], c[i]);
            }
        }

        return result;
    }

    /**
     * Blend two arrays with interpolation: result[i] = a[i] * (1 - alpha) + b[i] * alpha
     *
     * @param a first array
     * @param b second array (must match length)
     * @param alpha blend factor [0, 1]
     * @return new array with blended values
     */
    public static double[] blend(double[] a, double[] b, double alpha) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Array lengths must match: " + a.length + " vs " + b.length);
        }

        var result = new double[a.length];
        var oneMinusAlpha = 1.0 - alpha;

        if (a.length >= SIMD_THRESHOLD) {
            // Vectorized path
            int i = 0;
            int upperBound = SPECIES.loopBound(a.length);
            var vAlpha = DoubleVector.broadcast(SPECIES, alpha);
            var vOneMinusAlpha = DoubleVector.broadcast(SPECIES, oneMinusAlpha);

            for (; i < upperBound; i += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, a, i);
                var vb = DoubleVector.fromArray(SPECIES, b, i);
                // result = a * (1-alpha) + b * alpha
                va.mul(vOneMinusAlpha).add(vb.mul(vAlpha)).intoArray(result, i);
            }

            // Scalar cleanup
            for (; i < a.length; i++) {
                result[i] = a[i] * oneMinusAlpha + b[i] * alpha;
            }
        } else {
            // Scalar path
            for (int i = 0; i < a.length; i++) {
                result[i] = a[i] * oneMinusAlpha + b[i] * alpha;
            }
        }

        return result;
    }

    /**
     * Dot product of two arrays: result = sum(a[i] * b[i])
     *
     * @param a first array
     * @param b second array (must match length)
     * @return dot product
     */
    public static double dot(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Array lengths must match: " + a.length + " vs " + b.length);
        }

        double result = 0.0;

        if (a.length >= SIMD_THRESHOLD) {
            // Vectorized path
            var sum = DoubleVector.zero(SPECIES);
            int i = 0;
            int upperBound = SPECIES.loopBound(a.length);

            for (; i < upperBound; i += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, a, i);
                var vb = DoubleVector.fromArray(SPECIES, b, i);
                sum = va.fma(vb, sum);  // Fused multiply-add for better accuracy
            }

            result = sum.reduceLanes(VectorOperators.ADD);

            // Scalar cleanup
            for (; i < a.length; i++) {
                result += a[i] * b[i];
            }
        } else {
            // Scalar path
            for (int i = 0; i < a.length; i++) {
                result += a[i] * b[i];
            }
        }

        return result;
    }
}