package com.hellblazer.art.hybrid.pan.optimization;

/**
 * Learning rate scheduling strategies for PAN.
 */
public class LearningRateScheduler {

    public enum ScheduleType {
        CONSTANT,
        STEP_DECAY,
        EXPONENTIAL_DECAY,
        COSINE_ANNEALING,
        WARMUP_COSINE,
        ADAPTIVE
    }

    private final ScheduleType type;
    private final double initialLearningRate;
    private final int totalSteps;
    private final double decayRate;
    private final int warmupSteps;
    private final double minLearningRate;

    private int currentStep = 0;
    private double currentLearningRate;
    private double recentLoss = Double.MAX_VALUE;
    private int plateauCounter = 0;

    public LearningRateScheduler(ScheduleType type, double initialLearningRate,
                                int totalSteps, double decayRate,
                                int warmupSteps, double minLearningRate) {
        this.type = type;
        this.initialLearningRate = initialLearningRate;
        this.totalSteps = totalSteps;
        this.decayRate = decayRate;
        this.warmupSteps = warmupSteps;
        this.minLearningRate = minLearningRate;
        this.currentLearningRate = initialLearningRate;
    }

    /**
     * Create a constant learning rate scheduler.
     */
    public static LearningRateScheduler constant(double learningRate) {
        return new LearningRateScheduler(
            ScheduleType.CONSTANT, learningRate, 0, 0, 0, learningRate
        );
    }

    /**
     * Create a step decay scheduler.
     */
    public static LearningRateScheduler stepDecay(double initialLr, int stepSize, double decayRate) {
        return new LearningRateScheduler(
            ScheduleType.STEP_DECAY, initialLr, stepSize, decayRate, 0, initialLr * 0.01
        );
    }

    /**
     * Create an exponential decay scheduler.
     */
    public static LearningRateScheduler exponentialDecay(double initialLr, double decayRate) {
        return new LearningRateScheduler(
            ScheduleType.EXPONENTIAL_DECAY, initialLr, 0, decayRate, 0, initialLr * 0.01
        );
    }

    /**
     * Create a cosine annealing scheduler.
     */
    public static LearningRateScheduler cosineAnnealing(double initialLr, int totalSteps) {
        return new LearningRateScheduler(
            ScheduleType.COSINE_ANNEALING, initialLr, totalSteps, 0, 0, initialLr * 0.01
        );
    }

    /**
     * Create a warmup + cosine scheduler.
     */
    public static LearningRateScheduler warmupCosine(double initialLr, int warmupSteps, int totalSteps) {
        return new LearningRateScheduler(
            ScheduleType.WARMUP_COSINE, initialLr, totalSteps, 0, warmupSteps, initialLr * 0.01
        );
    }

    /**
     * Create an adaptive scheduler that reduces on plateau.
     */
    public static LearningRateScheduler adaptive(double initialLr, double decayRate) {
        return new LearningRateScheduler(
            ScheduleType.ADAPTIVE, initialLr, 0, decayRate, 0, initialLr * 0.01
        );
    }

    /**
     * Get current learning rate and advance step.
     */
    public double getLearningRate() {
        currentLearningRate = computeLearningRate();
        currentStep++;
        return Math.max(currentLearningRate, minLearningRate);
    }

    /**
     * Update loss for adaptive scheduling.
     */
    public void updateLoss(double loss) {
        if (type == ScheduleType.ADAPTIVE) {
            if (loss >= recentLoss * 0.99) {
                plateauCounter++;
                if (plateauCounter >= 10) {
                    currentLearningRate *= decayRate;
                    plateauCounter = 0;
                }
            } else {
                plateauCounter = 0;
            }
            recentLoss = loss;
        }
    }

    /**
     * Reset the scheduler.
     */
    public void reset() {
        currentStep = 0;
        currentLearningRate = initialLearningRate;
        recentLoss = Double.MAX_VALUE;
        plateauCounter = 0;
    }

    /**
     * Get current step.
     */
    public int getCurrentStep() {
        return currentStep;
    }

    private double computeLearningRate() {
        switch (type) {
            case CONSTANT:
                return initialLearningRate;

            case STEP_DECAY:
                int epochs = currentStep / totalSteps;
                return initialLearningRate * Math.pow(decayRate, epochs);

            case EXPONENTIAL_DECAY:
                return initialLearningRate * Math.exp(-decayRate * currentStep);

            case COSINE_ANNEALING:
                if (totalSteps <= 0) return initialLearningRate;
                return minLearningRate + 0.5 * (initialLearningRate - minLearningRate) *
                       (1 + Math.cos(Math.PI * currentStep / totalSteps));

            case WARMUP_COSINE:
                if (currentStep < warmupSteps) {
                    // Linear warmup
                    return initialLearningRate * currentStep / warmupSteps;
                } else {
                    // Cosine annealing after warmup
                    int adjustedStep = currentStep - warmupSteps;
                    int adjustedTotal = totalSteps - warmupSteps;
                    if (adjustedTotal <= 0) return initialLearningRate;
                    return minLearningRate + 0.5 * (initialLearningRate - minLearningRate) *
                           (1 + Math.cos(Math.PI * adjustedStep / adjustedTotal));
                }

            case ADAPTIVE:
                return currentLearningRate;

            default:
                return initialLearningRate;
        }
    }
}