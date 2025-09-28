package com.hellblazer.art.laminar.impl;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.laminar.core.LayerType;
import com.hellblazer.art.laminar.parameters.ILayerParameters;

/**
 * Input layer implementation for preprocessing and initial signal processing.
 * Handles complement coding and input normalization.
 *
 * @author Hal Hildebrand
 */
public class InputLayer extends AbstractLayer {

    private final boolean useComplementCoding;
    private final int originalSize;

    public InputLayer(int size) {
        this(size, false);
    }

    public InputLayer(int size, boolean useComplementCoding) {
        super(LayerType.INPUT, useComplementCoding ? size * 2 : size);
        this.useComplementCoding = useComplementCoding;
        this.originalSize = size;
    }

    public InputLayer(String id, int size, boolean useComplementCoding) {
        super(id, LayerType.INPUT, useComplementCoding ? size * 2 : size);
        this.useComplementCoding = useComplementCoding;
        this.originalSize = size;
    }

    @Override
    public Pattern processBottomUp(Pattern input, ILayerParameters parameters) {
        // Input layer processes external input
        if (useComplementCoding) {
            return applyComplementCoding(input);
        } else {
            return preprocessInput(input, parameters);
        }
    }

    @Override
    public Pattern processTopDown(Pattern feedback, ILayerParameters parameters) {
        // Input layer typically doesn't process top-down feedback
        // But can use it for attention modulation
        return applyAttentionModulation(activation, feedback);
    }

    @Override
    public Pattern processLateral(Pattern lateral, ILayerParameters parameters) {
        // Minimal lateral processing in input layer
        return lateral;
    }

    /**
     * Apply complement coding: [x, 1-x] for better ART performance
     */
    private Pattern applyComplementCoding(Pattern input) {
        if (input.dimension() != originalSize) {
            throw new IllegalArgumentException("Input size must match original layer size");
        }

        var complementCoded = new double[size];

        // Copy original input
        for (int i = 0; i < originalSize; i++) {
            complementCoded[i] = input.get(i);
        }

        // Add complement
        for (int i = 0; i < originalSize; i++) {
            complementCoded[originalSize + i] = 1.0 - input.get(i);
        }

        return new DenseVector(complementCoded);
    }

    /**
     * Preprocess input with normalization and clamping
     */
    private Pattern preprocessInput(Pattern input, ILayerParameters parameters) {
        var processed = input;

        // Apply normalization if requested
        if (parameters.useNormalization()) {
            processed = normalizeActivation(processed, parameters.getNormalizationType());
        }

        // Clamp values to valid range
        var processedArray = new double[processed.dimension()];
        for (int i = 0; i < processed.dimension(); i++) {
            processedArray[i] = Math.max(parameters.getLowerBound(),
                                       Math.min(parameters.getUpperBound(), processed.get(i)));
        }

        return new DenseVector(processedArray);
    }

    /**
     * Apply attention modulation using top-down feedback
     */
    private Pattern applyAttentionModulation(Pattern input, Pattern attention) {
        var modulated = new double[input.dimension()];
        var attentionSize = attention.dimension();

        for (int i = 0; i < input.dimension(); i++) {
            var attentionWeight = i < attentionSize ? attention.get(i) : 1.0;
            modulated[i] = input.get(i) * attentionWeight;
        }

        return new DenseVector(modulated);
    }

    public boolean usesComplementCoding() {
        return useComplementCoding;
    }

    public int getOriginalSize() {
        return originalSize;
    }
}