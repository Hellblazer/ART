package com.hellblazer.art.laminar.parameters;

/**
 * Parameters for Layer 2/3 configuration.
 * Controls horizontal grouping, complex cells, and attention integration.
 *
 * @author Hal Hildebrand
 */
public record Layer23Parameters(
    int size,
    double timeConstant,           // 30-150ms for Layer 2/3
    double topDownWeight,           // Weight of top-down priming from Layer 1
    double bottomUpWeight,          // Weight of bottom-up input from Layer 4
    double horizontalWeight,        // Weight of horizontal grouping
    double complexCellThreshold,    // Threshold for complex cell pooling
    boolean enableHorizontalGrouping,
    boolean enableComplexCells
) implements LayerParameters {

    // LayerParameters implementation
    @Override
    public double getDecayRate() {
        return 0.1;  // Medium decay for Layer 2/3
    }

    @Override
    public double getCeiling() {
        return 1.0;
    }

    @Override
    public double getFloor() {
        return 0.0;
    }

    @Override
    public double getSelfExcitation() {
        return 0.4;  // Moderate self-excitation
    }

    @Override
    public double getLateralInhibition() {
        return 0.2;  // Some lateral inhibition
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private int size = 100;
        private double timeConstant = 0.05;  // 50ms default
        private double topDownWeight = 0.3;
        private double bottomUpWeight = 1.0;
        private double horizontalWeight = 0.5;
        private double complexCellThreshold = 0.4;
        private boolean enableHorizontalGrouping = true;
        private boolean enableComplexCells = true;

        public Builder size(int size) {
            this.size = size;
            return this;
        }

        public Builder timeConstant(double timeConstant) {
            if (timeConstant < 0.03 || timeConstant > 0.15) {
                throw new IllegalArgumentException("Layer 2/3 time constant must be between 30-150ms");
            }
            this.timeConstant = timeConstant;
            return this;
        }

        public Builder topDownWeight(double topDownWeight) {
            this.topDownWeight = topDownWeight;
            return this;
        }

        public Builder bottomUpWeight(double bottomUpWeight) {
            this.bottomUpWeight = bottomUpWeight;
            return this;
        }

        public Builder horizontalWeight(double horizontalWeight) {
            this.horizontalWeight = horizontalWeight;
            return this;
        }

        public Builder complexCellThreshold(double complexCellThreshold) {
            this.complexCellThreshold = complexCellThreshold;
            return this;
        }

        public Builder enableHorizontalGrouping(boolean enable) {
            this.enableHorizontalGrouping = enable;
            return this;
        }

        public Builder enableComplexCells(boolean enable) {
            this.enableComplexCells = enable;
            return this;
        }

        public Layer23Parameters build() {
            return new Layer23Parameters(
                size,
                timeConstant,
                topDownWeight,
                bottomUpWeight,
                horizontalWeight,
                complexCellThreshold,
                enableHorizontalGrouping,
                enableComplexCells
            );
        }
    }
}