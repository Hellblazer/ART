package com.hellblazer.art.cortical.layers;

/**
 * Marker interface for layer-specific parameters.
 * Each layer type (L1, L2/3, L4, L5, L6) has its own parameter record
 * implementing this interface with biologically-constrained parameters.
 *
 * <p>Design Pattern: Type-safe parameter objects using Java records.
 * <ul>
 *   <li>Immutable configuration objects</li>
 *   <li>Layer-specific parameter validation</li>
 *   <li>Compiler-enforced type safety</li>
 *   <li>Builder pattern for ergonomic construction</li>
 * </ul>
 *
 * @author Migrated from art-laminar to art-cortical (Phase 3, Milestone 1)
 */
public sealed interface LayerParameters permits
    Layer1Parameters,
    Layer23Parameters,
    Layer4Parameters,
    Layer5Parameters,
    Layer6Parameters {

    /**
     * Validate that all parameters are within acceptable ranges.
     * Each implementing record provides specific validation.
     *
     * @throws IllegalArgumentException if any parameter is invalid
     */
    default void validate() {
        // Default: no validation
    }

    /**
     * Get decay rate for this layer (inverse of time constant).
     *
     * @return decay rate
     */
    double decayRate();

    /**
     * Get activation ceiling (maximum activation value).
     *
     * @return ceiling value
     */
    double ceiling();

    /**
     * Get activation floor (minimum activation value).
     *
     * @return floor value
     */
    double floor();

    /**
     * Get self-excitation strength.
     *
     * @return self-excitation strength
     */
    double selfExcitation();

    /**
     * Get lateral inhibition strength.
     *
     * @return lateral inhibition strength
     */
    double lateralInhibition();
}

/**
 * Layer 1 parameters (Apical Dendrites & Top-Down Attentional Priming).
 *
 * <p>Layer 1 contains apical dendrites that receive top-down attentional signals
 * from higher cortical areas. Implements surface contour processing via BipoleCell networks.
 *
 * <p>Biological constraints:
 * <ul>
 *   <li>Very slow time constants (200-1000ms) for sustained attention</li>
 *   <li>Receives top-down signals from higher cortical areas</li>
 *   <li>Sustained attention effects that persist after input ends</li>
 *   <li>Priming without driving cells directly (modulatory)</li>
 *   <li>Integrates with Layer 2/3 apical dendrites</li>
 *   <li>Lowest firing rates of all layers (max 50Hz)</li>
 * </ul>
 *
 * @param timeConstant Time constant in milliseconds (200-1000ms)
 * @param primingStrength Strength of priming effect (0-1)
 * @param sustainedDecayRate Very slow decay for persistence (0-0.01)
 * @param apicalIntegration Integration strength with Layer 2/3 (0-1)
 * @param attentionShiftRate Rate of attention shifting (0-1)
 * @param decayRate Inverse of time constant
 * @param ceiling Maximum activation (typically 1.0)
 * @param floor Minimum activation (typically 0.0)
 * @param selfExcitation Self-excitation strength (very weak, ~0.05)
 * @param lateralInhibition Lateral inhibition strength (very weak, ~0.05)
 * @param maxFiringRate Maximum biological firing rate in Hz (0-50Hz)
 */
record Layer1Parameters(
    double timeConstant,
    double primingStrength,
    double sustainedDecayRate,
    double apicalIntegration,
    double attentionShiftRate,
    double decayRate,
    double ceiling,
    double floor,
    double selfExcitation,
    double lateralInhibition,
    double maxFiringRate
) implements LayerParameters {

    /**
     * Canonical constructor with validation.
     */
    public Layer1Parameters {
        if (timeConstant < 200.0 || timeConstant > 1000.0) {
            throw new IllegalArgumentException(
                "Layer 1 time constant must be 200-1000ms, got: " + timeConstant);
        }
        if (primingStrength < 0.0 || primingStrength > 1.0) {
            throw new IllegalArgumentException(
                "Priming strength must be 0-1, got: " + primingStrength);
        }
        if (sustainedDecayRate < 0.0 || sustainedDecayRate > 0.01) {
            throw new IllegalArgumentException(
                "Sustained decay rate must be 0-0.01 for slow decay, got: " + sustainedDecayRate);
        }
        if (apicalIntegration < 0.0 || apicalIntegration > 1.0) {
            throw new IllegalArgumentException(
                "Apical integration must be 0-1, got: " + apicalIntegration);
        }
        if (attentionShiftRate < 0.0 || attentionShiftRate > 1.0) {
            throw new IllegalArgumentException(
                "Attention shift rate must be 0-1, got: " + attentionShiftRate);
        }
        if (floor > ceiling) {
            throw new IllegalArgumentException("Floor must be <= ceiling");
        }
        if (maxFiringRate <= 0.0 || maxFiringRate > 50.0) {
            throw new IllegalArgumentException(
                "Layer 1 max firing rate must be 0-50Hz, got: " + maxFiringRate);
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double timeConstant = 500.0;           // Mid-range default
        private double primingStrength = 0.3;          // 30% priming effect
        private double sustainedDecayRate = 0.001;     // Very slow decay
        private double apicalIntegration = 0.5;        // 50% integration
        private double attentionShiftRate = 0.3;       // Moderate shift rate
        private double ceiling = 1.0;
        private double floor = 0.0;
        private double selfExcitation = 0.05;          // Very weak self-excitation
        private double lateralInhibition = 0.05;       // Very weak lateral inhibition
        private double maxFiringRate = 30.0;           // Low firing rate for Layer 1

        public Builder timeConstant(double timeConstant) {
            this.timeConstant = timeConstant;
            return this;
        }

        public Builder primingStrength(double primingStrength) {
            this.primingStrength = primingStrength;
            return this;
        }

        public Builder sustainedDecayRate(double sustainedDecayRate) {
            this.sustainedDecayRate = sustainedDecayRate;
            return this;
        }

        public Builder apicalIntegration(double apicalIntegration) {
            this.apicalIntegration = apicalIntegration;
            return this;
        }

        public Builder attentionShiftRate(double attentionShiftRate) {
            this.attentionShiftRate = attentionShiftRate;
            return this;
        }

        public Builder ceiling(double ceiling) {
            this.ceiling = ceiling;
            return this;
        }

        public Builder floor(double floor) {
            this.floor = floor;
            return this;
        }

        public Builder selfExcitation(double selfExcitation) {
            this.selfExcitation = selfExcitation;
            return this;
        }

        public Builder lateralInhibition(double lateralInhibition) {
            this.lateralInhibition = lateralInhibition;
            return this;
        }

        public Builder maxFiringRate(double maxFiringRate) {
            this.maxFiringRate = maxFiringRate;
            return this;
        }

        public Layer1Parameters build() {
            var decayRate = 1.0 / timeConstant;
            return new Layer1Parameters(
                timeConstant, primingStrength, sustainedDecayRate,
                apicalIntegration, attentionShiftRate, decayRate,
                ceiling, floor, selfExcitation, lateralInhibition, maxFiringRate
            );
        }
    }
}

/**
 * Layer 2/3 parameters (Inter-Areal Communication & Prediction).
 *
 * <p>Layer 2/3 implements category formation, working memory integration,
 * and predictive coding with ART dynamics.
 *
 * <p>Biological constraints:
 * <ul>
 *   <li>Medium time constants (30-150ms)</li>
 *   <li>Horizontal grouping via lateral connections</li>
 *   <li>Complex cell pooling for invariance</li>
 *   <li>Working memory integration</li>
 *   <li>Category learning via ART match/reset</li>
 * </ul>
 *
 * @param size Layer size (number of units/columns)
 * @param timeConstant Time constant in milliseconds (30-150ms)
 * @param topDownWeight Weight of top-down priming from Layer 1
 * @param bottomUpWeight Weight of bottom-up input from Layer 4
 * @param horizontalWeight Weight of horizontal grouping
 * @param complexCellThreshold Threshold for complex cell pooling
 * @param enableHorizontalGrouping Enable horizontal grouping mechanism
 * @param enableComplexCells Enable complex cell pooling
 * @param decayRate Inverse of time constant
 * @param ceiling Maximum activation (typically 1.0)
 * @param floor Minimum activation (typically 0.0)
 * @param selfExcitation Self-excitation strength (moderate, ~0.4)
 * @param lateralInhibition Lateral inhibition strength (~0.2)
 */
record Layer23Parameters(
    int size,
    double timeConstant,
    double topDownWeight,
    double bottomUpWeight,
    double horizontalWeight,
    double complexCellThreshold,
    boolean enableHorizontalGrouping,
    boolean enableComplexCells,
    double decayRate,
    double ceiling,
    double floor,
    double selfExcitation,
    double lateralInhibition
) implements LayerParameters {

    /**
     * Canonical constructor with validation.
     */
    public Layer23Parameters {
        if (timeConstant < 30.0 || timeConstant > 150.0) {
            throw new IllegalArgumentException(
                "Layer 2/3 time constant must be 30-150ms, got: " + timeConstant);
        }
        if (size <= 0) {
            throw new IllegalArgumentException("Layer size must be positive, got: " + size);
        }
        if (floor > ceiling) {
            throw new IllegalArgumentException("Floor must be <= ceiling");
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private int size = 100;
        private double timeConstant = 75.0;           // Mid-range default (75ms)
        private double topDownWeight = 0.3;
        private double bottomUpWeight = 1.0;
        private double horizontalWeight = 0.5;
        private double complexCellThreshold = 0.4;
        private boolean enableHorizontalGrouping = true;
        private boolean enableComplexCells = true;
        private double ceiling = 1.0;
        private double floor = 0.0;
        private double selfExcitation = 0.4;          // Moderate self-excitation
        private double lateralInhibition = 0.2;       // Some lateral inhibition

        public Builder size(int size) {
            this.size = size;
            return this;
        }

        public Builder timeConstant(double timeConstant) {
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

        public Builder ceiling(double ceiling) {
            this.ceiling = ceiling;
            return this;
        }

        public Builder floor(double floor) {
            this.floor = floor;
            return this;
        }

        public Builder selfExcitation(double selfExcitation) {
            this.selfExcitation = selfExcitation;
            return this;
        }

        public Builder lateralInhibition(double lateralInhibition) {
            this.lateralInhibition = lateralInhibition;
            return this;
        }

        public Layer23Parameters build() {
            var decayRate = 1.0 / timeConstant;
            return new Layer23Parameters(
                size, timeConstant, topDownWeight, bottomUpWeight,
                horizontalWeight, complexCellThreshold,
                enableHorizontalGrouping, enableComplexCells,
                decayRate, ceiling, floor, selfExcitation, lateralInhibition
            );
        }
    }
}

/**
 * Layer 4 parameters (Thalamic Driving Input).
 *
 * <p>Layer 4 is the primary recipient of driving input from the thalamus (LGN).
 * It initiates cortical processing with strong, fast dynamics.
 *
 * <p>Biological constraints:
 * <ul>
 *   <li>Fast time constants (10-50ms) for rapid response</li>
 *   <li>Strong driving signals that can fire cells independently</li>
 *   <li>Simple feedforward processing</li>
 *   <li>Minimal lateral inhibition in basic circuits</li>
 *   <li>Direct transformation of thalamic input to cortical representation</li>
 * </ul>
 *
 * @param timeConstant Time constant in milliseconds (10-50ms)
 * @param drivingStrength Strength of thalamic drive (0-1)
 * @param decayRate Inverse of time constant
 * @param ceiling Maximum activation (typically 1.0)
 * @param floor Minimum activation (typically 0.0)
 * @param selfExcitation Self-excitation strength (~0.3)
 * @param lateralInhibition Lateral inhibition strength (initially 0)
 */
record Layer4Parameters(
    double timeConstant,
    double drivingStrength,
    double decayRate,
    double ceiling,
    double floor,
    double selfExcitation,
    double lateralInhibition
) implements LayerParameters {

    /**
     * Canonical constructor with validation.
     */
    public Layer4Parameters {
        if (timeConstant < 10.0 || timeConstant > 50.0) {
            throw new IllegalArgumentException(
                "Layer 4 time constant must be 10-50ms, got: " + timeConstant);
        }
        if (drivingStrength < 0.0 || drivingStrength > 1.0) {
            throw new IllegalArgumentException(
                "Driving strength must be 0-1, got: " + drivingStrength);
        }
        if (floor > ceiling) {
            throw new IllegalArgumentException("Floor must be <= ceiling");
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double timeConstant = 25.0;          // Mid-range default
        private double drivingStrength = 0.8;        // Strong driving
        private double ceiling = 1.0;
        private double floor = 0.0;
        private double selfExcitation = 0.3;
        private double lateralInhibition = 0.0;      // No lateral inhibition initially

        public Builder timeConstant(double timeConstant) {
            this.timeConstant = timeConstant;
            return this;
        }

        public Builder drivingStrength(double drivingStrength) {
            this.drivingStrength = drivingStrength;
            return this;
        }

        public Builder ceiling(double ceiling) {
            this.ceiling = ceiling;
            return this;
        }

        public Builder floor(double floor) {
            this.floor = floor;
            return this;
        }

        public Builder selfExcitation(double selfExcitation) {
            this.selfExcitation = selfExcitation;
            return this;
        }

        public Builder lateralInhibition(double lateralInhibition) {
            this.lateralInhibition = lateralInhibition;
            return this;
        }

        public Layer4Parameters build() {
            var decayRate = 1.0 / timeConstant;
            return new Layer4Parameters(
                timeConstant, drivingStrength, decayRate,
                ceiling, floor, selfExcitation, lateralInhibition
            );
        }
    }
}

/**
 * Layer 5 parameters (Motor Output & Action Selection).
 *
 * <p>Layer 5 projects processed signals from Layer 2/3 to higher cortical areas
 * and subcortical structures. Implements decision formation and action selection.
 *
 * <p>Biological constraints:
 * <ul>
 *   <li>Medium time constants (50-200ms)</li>
 *   <li>Receives input from Layer 2/3 pyramidal cells</li>
 *   <li>Amplification/gating for salient features</li>
 *   <li>Output normalization for stable signaling</li>
 *   <li>Category signal generation</li>
 *   <li>Burst firing capability for important signals</li>
 *   <li>Maximum firing rate up to 200Hz</li>
 * </ul>
 *
 * @param timeConstant Time constant in milliseconds (50-200ms)
 * @param amplificationGain Signal amplification factor
 * @param outputGain Output scaling factor
 * @param outputNormalization Normalization strength
 * @param categoryThreshold Threshold for category detection (0-1)
 * @param burstThreshold Threshold for burst firing (0-1)
 * @param burstAmplification Amplification during burst (>=1.0)
 * @param decayRate Inverse of time constant
 * @param ceiling Maximum activation (typically 1.0)
 * @param floor Minimum activation (typically 0.0)
 * @param selfExcitation Self-excitation strength (~0.2)
 * @param lateralInhibition Lateral inhibition strength (~0.1)
 * @param maxFiringRate Maximum biological firing rate in Hz (0-200Hz)
 */
record Layer5Parameters(
    double timeConstant,
    double amplificationGain,
    double outputGain,
    double outputNormalization,
    double categoryThreshold,
    double burstThreshold,
    double burstAmplification,
    double decayRate,
    double ceiling,
    double floor,
    double selfExcitation,
    double lateralInhibition,
    double maxFiringRate
) implements LayerParameters {

    /**
     * Canonical constructor with validation.
     */
    public Layer5Parameters {
        if (timeConstant < 50.0 || timeConstant > 200.0) {
            throw new IllegalArgumentException(
                "Layer 5 time constant must be 50-200ms, got: " + timeConstant);
        }
        if (amplificationGain < 0.0) {
            throw new IllegalArgumentException(
                "Amplification gain must be non-negative, got: " + amplificationGain);
        }
        if (outputGain < 0.0) {
            throw new IllegalArgumentException(
                "Output gain must be non-negative, got: " + outputGain);
        }
        if (categoryThreshold < 0.0 || categoryThreshold > 1.0) {
            throw new IllegalArgumentException(
                "Category threshold must be 0-1, got: " + categoryThreshold);
        }
        if (burstThreshold < 0.0 || burstThreshold > 1.0) {
            throw new IllegalArgumentException(
                "Burst threshold must be 0-1, got: " + burstThreshold);
        }
        if (burstAmplification < 1.0) {
            throw new IllegalArgumentException(
                "Burst amplification must be >= 1.0, got: " + burstAmplification);
        }
        if (floor > ceiling) {
            throw new IllegalArgumentException("Floor must be <= ceiling");
        }
        if (maxFiringRate <= 0.0 || maxFiringRate > 200.0) {
            throw new IllegalArgumentException(
                "Max firing rate must be 0-200Hz, got: " + maxFiringRate);
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double timeConstant = 100.0;          // Mid-range default
        private double amplificationGain = 1.5;       // Moderate amplification
        private double outputGain = 1.0;              // Unity gain default
        private double outputNormalization = 0.01;    // Weak normalization
        private double categoryThreshold = 0.5;       // Mid-threshold
        private double burstThreshold = 0.8;          // High threshold for bursting
        private double burstAmplification = 2.0;      // Double during burst
        private double ceiling = 1.0;
        private double floor = 0.0;
        private double selfExcitation = 0.2;
        private double lateralInhibition = 0.1;       // Weak lateral inhibition
        private double maxFiringRate = 100.0;         // 100Hz max

        public Builder timeConstant(double timeConstant) {
            this.timeConstant = timeConstant;
            return this;
        }

        public Builder amplificationGain(double amplificationGain) {
            this.amplificationGain = amplificationGain;
            return this;
        }

        public Builder outputGain(double outputGain) {
            this.outputGain = outputGain;
            return this;
        }

        public Builder outputNormalization(double outputNormalization) {
            this.outputNormalization = outputNormalization;
            return this;
        }

        public Builder categoryThreshold(double categoryThreshold) {
            this.categoryThreshold = categoryThreshold;
            return this;
        }

        public Builder burstThreshold(double burstThreshold) {
            this.burstThreshold = burstThreshold;
            return this;
        }

        public Builder burstAmplification(double burstAmplification) {
            this.burstAmplification = burstAmplification;
            return this;
        }

        public Builder ceiling(double ceiling) {
            this.ceiling = ceiling;
            return this;
        }

        public Builder floor(double floor) {
            this.floor = floor;
            return this;
        }

        public Builder selfExcitation(double selfExcitation) {
            this.selfExcitation = selfExcitation;
            return this;
        }

        public Builder lateralInhibition(double lateralInhibition) {
            this.lateralInhibition = lateralInhibition;
            return this;
        }

        public Builder maxFiringRate(double maxFiringRate) {
            this.maxFiringRate = maxFiringRate;
            return this;
        }

        public Layer5Parameters build() {
            var decayRate = 1.0 / timeConstant;
            return new Layer5Parameters(
                timeConstant, amplificationGain, outputGain, outputNormalization,
                categoryThreshold, burstThreshold, burstAmplification,
                decayRate, ceiling, floor, selfExcitation, lateralInhibition,
                maxFiringRate
            );
        }
    }
}

/**
 * Layer 6 parameters (Corticothalamic Feedback & Attention).
 *
 * <p>Layer 6 provides modulatory feedback to Layer 4 and thalamus.
 * Implements ART matching rule - modulatory only (cannot fire cells alone).
 *
 * <p>CRITICAL: On-center, off-surround dynamics for expectation matching.
 *
 * <p>Biological constraints:
 * <ul>
 *   <li>Slow time constants (100-500ms) for sustained modulation</li>
 *   <li>Modulatory signals that cannot drive responses alone</li>
 *   <li>Implements center-surround organization</li>
 *   <li>Lower firing rates than other layers (max 100Hz)</li>
 *   <li>Top-down expectation generation</li>
 *   <li>Attentional gain control</li>
 * </ul>
 *
 * @param timeConstant Time constant in milliseconds (100-500ms)
 * @param onCenterWeight On-center excitation strength
 * @param offSurroundStrength Off-surround inhibition strength
 * @param modulationThreshold Threshold for modulatory effect (0-1)
 * @param attentionalGain Gain for attentional modulation
 * @param decayRate Inverse of time constant
 * @param ceiling Maximum activation (typically 1.0)
 * @param floor Minimum activation (typically 0.0)
 * @param selfExcitation Self-excitation strength (weak, ~0.1)
 * @param lateralInhibition Lateral inhibition strength (moderate, ~0.3)
 * @param maxFiringRate Maximum biological firing rate in Hz (0-100Hz)
 */
record Layer6Parameters(
    double timeConstant,
    double onCenterWeight,
    double offSurroundStrength,
    double modulationThreshold,
    double attentionalGain,
    double decayRate,
    double ceiling,
    double floor,
    double selfExcitation,
    double lateralInhibition,
    double maxFiringRate
) implements LayerParameters {

    /**
     * Canonical constructor with validation.
     */
    public Layer6Parameters {
        if (timeConstant < 100.0 || timeConstant > 500.0) {
            throw new IllegalArgumentException(
                "Layer 6 time constant must be 100-500ms, got: " + timeConstant);
        }
        if (onCenterWeight < 0.0) {
            throw new IllegalArgumentException(
                "On-center weight must be non-negative, got: " + onCenterWeight);
        }
        if (offSurroundStrength < 0.0) {
            throw new IllegalArgumentException(
                "Off-surround strength must be non-negative, got: " + offSurroundStrength);
        }
        if (modulationThreshold < 0.0 || modulationThreshold > 1.0) {
            throw new IllegalArgumentException(
                "Modulation threshold must be 0-1, got: " + modulationThreshold);
        }
        if (attentionalGain < 0.0) {
            throw new IllegalArgumentException(
                "Attentional gain must be non-negative, got: " + attentionalGain);
        }
        if (floor > ceiling) {
            throw new IllegalArgumentException("Floor must be <= ceiling");
        }
        if (maxFiringRate <= 0.0 || maxFiringRate > 100.0) {
            throw new IllegalArgumentException(
                "Layer 6 max firing rate must be 0-100Hz, got: " + maxFiringRate);
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private double timeConstant = 200.0;          // Mid-range default
        private double onCenterWeight = 1.0;          // Unity on-center
        private double offSurroundStrength = 0.2;     // Weak off-surround
        private double modulationThreshold = 0.1;     // Low threshold for modulation
        private double attentionalGain = 1.0;         // Unity gain default
        private double ceiling = 1.0;
        private double floor = 0.0;
        private double selfExcitation = 0.1;          // Weak self-excitation
        private double lateralInhibition = 0.3;       // Moderate lateral inhibition
        private double maxFiringRate = 50.0;          // Lower firing rate for Layer 6

        public Builder timeConstant(double timeConstant) {
            this.timeConstant = timeConstant;
            return this;
        }

        public Builder onCenterWeight(double onCenterWeight) {
            this.onCenterWeight = onCenterWeight;
            return this;
        }

        public Builder offSurroundStrength(double offSurroundStrength) {
            this.offSurroundStrength = offSurroundStrength;
            return this;
        }

        public Builder modulationThreshold(double modulationThreshold) {
            this.modulationThreshold = modulationThreshold;
            return this;
        }

        public Builder attentionalGain(double attentionalGain) {
            this.attentionalGain = attentionalGain;
            return this;
        }

        public Builder ceiling(double ceiling) {
            this.ceiling = ceiling;
            return this;
        }

        public Builder floor(double floor) {
            this.floor = floor;
            return this;
        }

        public Builder selfExcitation(double selfExcitation) {
            this.selfExcitation = selfExcitation;
            return this;
        }

        public Builder lateralInhibition(double lateralInhibition) {
            this.lateralInhibition = lateralInhibition;
            return this;
        }

        public Builder maxFiringRate(double maxFiringRate) {
            this.maxFiringRate = maxFiringRate;
            return this;
        }

        public Layer6Parameters build() {
            var decayRate = 1.0 / timeConstant;
            return new Layer6Parameters(
                timeConstant, onCenterWeight, offSurroundStrength,
                modulationThreshold, attentionalGain, decayRate,
                ceiling, floor, selfExcitation, lateralInhibition, maxFiringRate
            );
        }
    }
}
