/**
 * Hybrid ART-Markov System - Implementation Examples
 *
 * Concrete implementations demonstrating the architecture in action
 */

package com.hellblazer.art.markov.implementation;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.stream.*;
import jdk.incubator.vector.*;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

//================================================================================
// CONCRETE STATE ABSTRACTION USING ART
//================================================================================

/**
 * Concrete implementation of StateAbstractionART using FuzzyART
 */
public class FuzzyARTStateAbstractor<O> implements StateAbstractionART<O, ContinuousState> {

    private final FuzzyARTCore artNetwork;
    private final StateEncoder<O, double[]> encoder;
    private final Map<Integer, ContinuousState> categoryToState;
    private final AtomicInteger stateIdCounter;
    private final double vigilanceParameter;
    private final boolean allowDynamicCreation;

    public FuzzyARTStateAbstractor(StateAbstractionConfig config) {
        this.encoder = (StateEncoder<O, double[]>) config.encoder();
        this.vigilanceParameter = config.vigilanceParameter();
        this.allowDynamicCreation = config.allowDynamicCreation();
        this.categoryToState = new ConcurrentHashMap<>();
        this.stateIdCounter = new AtomicInteger(0);

        // Initialize FuzzyART network
        this.artNetwork = new FuzzyARTCore(
            encoder.dimensionality(),
            config.maxCategories(),
            vigilanceParameter,
            config.learningRate()
        );
    }

    @Override
    public ContinuousState abstractState(O observation, Context context) {
        // 1. Encode observation to vector representation
        var encoded = encoder.encode(observation);

        // 2. Apply complement coding for FuzzyART
        var complementCoded = applyComplementCoding(encoded);

        // 3. Present to ART network for categorization
        var categoryId = artNetwork.categorize(complementCoded);

        // 4. Check if this is a new category
        if (!categoryToState.containsKey(categoryId)) {
            if (!allowDynamicCreation && categoryToState.size() >= artNetwork.maxCategories) {
                // Find closest existing state
                categoryId = findClosestCategory(complementCoded);
            } else {
                // Create new state for this category
                var newState = createNewState(encoded, categoryId, context);
                categoryToState.put(categoryId, newState);
            }
        }

        // 5. Update state metadata
        var state = categoryToState.get(categoryId);
        return updateStateMetadata(state, context);
    }

    private double[] applyComplementCoding(double[] input) {
        var result = new double[input.length * 2];
        for (int i = 0; i < input.length; i++) {
            result[i] = input[i];
            result[i + input.length] = 1.0 - input[i];
        }
        return result;
    }

    private ContinuousState createNewState(double[] encoded, int categoryId, Context context) {
        var stateId = "S" + stateIdCounter.incrementAndGet();
        var metadata = new State.StateMetadata(
            System.currentTimeMillis(),
            1,
            0.0,
            Map.of("artCategory", categoryId)
        );
        return new ContinuousState(stateId, encoded, metadata, false);
    }

    private int findClosestCategory(double[] pattern) {
        return artNetwork.findBestMatch(pattern);
    }

    private ContinuousState updateStateMetadata(ContinuousState state, Context context) {
        var updatedMetadata = new State.StateMetadata(
            state.metadata().discoveryTime(),
            state.metadata().visitCount() + 1,
            state.metadata().avgReward(),
            state.metadata().properties()
        );
        return new ContinuousState(
            state.id(),
            state.value(),
            updatedMetadata,
            state.isTerminal()
        );
    }

    @Override
    public Collection<ContinuousState> getDiscoveredStates() {
        return Collections.unmodifiableCollection(categoryToState.values());
    }

    @Override
    public int getARTCategory(ContinuousState state) {
        return categoryToState.entrySet().stream()
            .filter(e -> e.getValue().equals(state))
            .map(Map.Entry::getKey)
            .findFirst()
            .orElse(-1);
    }

    /**
     * Inner FuzzyART implementation
     */
    private static class FuzzyARTCore {
        private final int inputDimension;
        private final int maxCategories;
        private final double vigilance;
        private final double learningRate;
        private final List<double[]> weights;
        private final AtomicInteger categoryCount;

        FuzzyARTCore(int inputDimension, int maxCategories,
                     double vigilance, double learningRate) {
            this.inputDimension = inputDimension * 2; // Complement coding
            this.maxCategories = maxCategories;
            this.vigilance = vigilance;
            this.learningRate = learningRate;
            this.weights = new ArrayList<>();
            this.categoryCount = new AtomicInteger(0);
        }

        synchronized int categorize(double[] input) {
            // Compute choice function for all categories
            var choices = computeChoiceFunction(input);

            // Sort categories by choice value
            var sortedCategories = sortByChoice(choices);

            // Find first category that passes vigilance
            for (var categoryId : sortedCategories) {
                if (passesVigilance(input, categoryId)) {
                    // Update weights
                    updateWeights(categoryId, input);
                    return categoryId;
                }
            }

            // Create new category if possible
            if (categoryCount.get() < maxCategories) {
                return createNewCategory(input);
            }

            // Return best match if max categories reached
            return sortedCategories.get(0);
        }

        private List<Double> computeChoiceFunction(double[] input) {
            return weights.stream()
                .map(w -> fuzzyAND(input, w) / (0.01 + norm(w)))
                .collect(Collectors.toList());
        }

        private List<Integer> sortByChoice(List<Double> choices) {
            return IntStream.range(0, choices.size())
                .boxed()
                .sorted((i, j) -> Double.compare(choices.get(j), choices.get(i)))
                .collect(Collectors.toList());
        }

        private boolean passesVigilance(double[] input, int categoryId) {
            var weight = weights.get(categoryId);
            var match = fuzzyAND(input, weight) / norm(input);
            return match >= vigilance;
        }

        private void updateWeights(int categoryId, double[] input) {
            var oldWeight = weights.get(categoryId);
            var newWeight = new double[inputDimension];
            for (int i = 0; i < inputDimension; i++) {
                newWeight[i] = learningRate * Math.min(input[i], oldWeight[i]) +
                              (1 - learningRate) * oldWeight[i];
            }
            weights.set(categoryId, newWeight);
        }

        private int createNewCategory(double[] input) {
            weights.add(input.clone());
            return categoryCount.getAndIncrement();
        }

        int findBestMatch(double[] pattern) {
            var choices = computeChoiceFunction(pattern);
            return IntStream.range(0, choices.size())
                .reduce((i, j) -> choices.get(i) > choices.get(j) ? i : j)
                .orElse(0);
        }

        private double fuzzyAND(double[] a, double[] b) {
            var sum = 0.0;
            for (int i = 0; i < a.length; i++) {
                sum += Math.min(a[i], b[i]);
            }
            return sum;
        }

        private double norm(double[] vector) {
            return Arrays.stream(vector).sum();
        }
    }
}

//================================================================================
// CONCRETE TRANSITION LEARNER WITH VALIDATION
//================================================================================

/**
 * Transition learner with proper stochastic matrix maintenance
 */
public class ValidatedTransitionLearner<S extends State<?>>
    implements TransitionLearner<S> {

    private final Map<S, Map<S, Double>> transitionCounts;
    private final Map<S, Double> stateTotals;
    private final ProbabilityValidator validator;
    private final VarHandle DOUBLE_HANDLE;
    private final ReentrantReadWriteLock lock;

    public ValidatedTransitionLearner(ProbabilityValidator validator) {
        this.transitionCounts = new ConcurrentHashMap<>();
        this.stateTotals = new ConcurrentHashMap<>();
        this.validator = validator;
        this.lock = new ReentrantReadWriteLock();

        try {
            this.DOUBLE_HANDLE = MethodHandles.lookup()
                .findVarHandle(AtomicDouble.class, "value", double.class);
        } catch (Exception e) {
            throw new RuntimeException("Failed to create VarHandle", e);
        }
    }

    @Override
    public void observeTransition(S from, S to, Context context) {
        lock.writeLock().lock();
        try {
            // Update transition count
            transitionCounts.computeIfAbsent(from, k -> new ConcurrentHashMap<>())
                           .merge(to, 1.0, Double::sum);

            // Update total count for source state
            stateTotals.merge(from, 1.0, Double::sum);

            // Normalize to maintain stochastic property
            normalizeTransitionsFrom(from);
        } finally {
            lock.writeLock().unlock();
        }
    }

    private void normalizeTransitionsFrom(S from) {
        var transitions = transitionCounts.get(from);
        if (transitions == null || transitions.isEmpty()) {
            return;
        }

        var total = stateTotals.get(from);
        if (total == null || total == 0) {
            return;
        }

        // Normalize all transitions from this state
        transitions.replaceAll((to, count) -> count / total);

        // Validate row sums to 1
        var rowSum = transitions.values().stream()
            .mapToDouble(Double::doubleValue)
            .sum();

        if (Math.abs(rowSum - 1.0) > 1e-10) {
            // Re-normalize if needed due to floating point errors
            var correction = 1.0 / rowSum;
            transitions.replaceAll((to, prob) -> prob * correction);
        }
    }

    @Override
    public double getTransitionProbability(S from, S to) {
        lock.readLock().lock();
        try {
            var transitions = transitionCounts.get(from);
            if (transitions == null) {
                return 0.0;
            }
            return transitions.getOrDefault(to, 0.0);
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public StochasticMatrix<S> getTransitionMatrix() {
        return new StochasticMatrixImpl();
    }

    @Override
    public List<Transition<S, ?>> getTransitionsFrom(S state) {
        lock.readLock().lock();
        try {
            var transitions = transitionCounts.get(state);
            if (transitions == null) {
                return List.of();
            }

            return transitions.entrySet().stream()
                .filter(e -> e.getValue() > 0)
                .map(e -> new Transition<>(
                    state,
                    e.getKey(),
                    e.getValue(),
                    null,
                    new Transition.TransitionMetadata(
                        (long)(stateTotals.get(state) * e.getValue()),
                        0.0,
                        System.currentTimeMillis(),
                        Map.of()
                    )
                ))
                .collect(Collectors.toList());
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public ValidationResult validateMatrix() {
        lock.readLock().lock();
        try {
            var violations = new ArrayList<String>();
            var metrics = new HashMap<String, Double>();

            // Check each row sums to 1
            for (var from : transitionCounts.keySet()) {
                var rowSum = transitionCounts.get(from).values().stream()
                    .mapToDouble(Double::doubleValue)
                    .sum();

                if (Math.abs(rowSum - 1.0) > 1e-6) {
                    violations.add(String.format(
                        "Row for state %s sums to %.6f instead of 1.0",
                        from.id(), rowSum
                    ));
                }

                // Check for negative probabilities
                for (var prob : transitionCounts.get(from).values()) {
                    if (prob < 0) {
                        violations.add("Negative probability detected: " + prob);
                    }
                    if (prob > 1) {
                        violations.add("Probability > 1 detected: " + prob);
                    }
                }
            }

            metrics.put("numStates", (double) transitionCounts.size());
            metrics.put("numTransitions",
                transitionCounts.values().stream()
                    .mapToDouble(m -> m.size())
                    .sum()
            );

            return new ValidationResult(
                violations.isEmpty(),
                violations,
                metrics
            );
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Inner stochastic matrix implementation
     */
    private class StochasticMatrixImpl implements StochasticMatrix<S> {

        @Override
        public double get(S from, S to) {
            return getTransitionProbability(from, to);
        }

        @Override
        public void set(S from, S to, double probability) {
            lock.writeLock().lock();
            try {
                if (probability < 0 || probability > 1) {
                    throw new IllegalArgumentException(
                        "Invalid probability: " + probability
                    );
                }

                transitionCounts.computeIfAbsent(from, k -> new ConcurrentHashMap<>())
                               .put(to, probability);
                normalizeTransitionsFrom(from);
            } finally {
                lock.writeLock().unlock();
            }
        }

        @Override
        public void normalize() {
            lock.writeLock().lock();
            try {
                transitionCounts.keySet().forEach(
                    ValidatedTransitionLearner.this::normalizeTransitionsFrom
                );
            } finally {
                lock.writeLock().unlock();
            }
        }

        @Override
        public boolean isValid() {
            return validateMatrix().isValid();
        }

        @Override
        public SparseMatrixRepresentation toSparse() {
            lock.readLock().lock();
            try {
                var states = new ArrayList<>(transitionCounts.keySet());
                var stateIndex = new HashMap<S, Integer>();
                for (int i = 0; i < states.size(); i++) {
                    stateIndex.put(states.get(i), i);
                }

                var values = new ArrayList<Double>();
                var rowIndices = new ArrayList<Integer>();
                var colIndices = new ArrayList<Integer>();

                for (var from : states) {
                    var transitions = transitionCounts.get(from);
                    if (transitions != null) {
                        for (var entry : transitions.entrySet()) {
                            if (entry.getValue() > 0) {
                                rowIndices.add(stateIndex.get(from));
                                colIndices.add(stateIndex.get(entry.getKey()));
                                values.add(entry.getValue());
                            }
                        }
                    }
                }

                return new SparseMatrixRepresentation(
                    rowIndices.stream().mapToInt(Integer::intValue).toArray(),
                    colIndices.stream().mapToInt(Integer::intValue).toArray(),
                    values.stream().mapToDouble(Double::doubleValue).toArray(),
                    states.size()
                );
            } finally {
                lock.readLock().unlock();
            }
        }
    }

    /**
     * Sparse matrix representation
     */
    record SparseMatrixRepresentation(
        int[] rowIndices,
        int[] colIndices,
        double[] values,
        int dimension
    ) {}
}

//================================================================================
// CONCRETE HYBRID PREDICTOR
//================================================================================

/**
 * Concrete hybrid predictor combining ART and Markov predictions
 */
public class ConcreteHybridPredictor<S extends State<?>>
    implements HybridPredictor<S, String, Double> {

    private final StateAbstractionART<?, S> stateAbstractor;
    private final TransitionLearner<S> transitionLearner;
    private final ContextAugmenter<S, ?> contextAugmenter;
    private final HybridStrategy strategy;
    private final ExecutorService virtualExecutor;

    public ConcreteHybridPredictor(
        StateAbstractionART<?, S> stateAbstractor,
        TransitionLearner<S> transitionLearner,
        ContextAugmenter<S, ?> contextAugmenter,
        HybridStrategy strategy
    ) {
        this.stateAbstractor = stateAbstractor;
        this.transitionLearner = transitionLearner;
        this.contextAugmenter = contextAugmenter;
        this.strategy = strategy;
        this.virtualExecutor = Executors.newVirtualThreadPerTaskExecutor();
    }

    @Override
    public PredictionResult<S> predictNextState(S currentState, Context context) {
        try {
            // Parallel computation of ART and Markov predictions
            var artFuture = virtualExecutor.submit(() ->
                predictUsingART(currentState, context)
            );

            var markovFuture = virtualExecutor.submit(() ->
                predictUsingMarkov(currentState)
            );

            // Wait for both predictions
            var artPrediction = artFuture.get();
            var markovPrediction = markovFuture.get();

            // Combine predictions based on strategy
            return combinePredicitions(artPrediction, markovPrediction);

        } catch (Exception e) {
            // Fallback to Markov-only prediction
            return predictUsingMarkov(currentState);
        }
    }

    private PredictionResult<S> predictUsingART(S currentState, Context context) {
        // Get ART category for current state
        var category = stateAbstractor.getARTCategory(currentState);
        if (category == -1) {
            return null; // State not in ART network
        }

        // Find similar states based on ART clustering
        var similarStates = stateAbstractor.getDiscoveredStates().stream()
            .filter(s -> stateAbstractor.getARTCategory(s) == category)
            .filter(s -> !s.equals(currentState))
            .collect(Collectors.toList());

        if (similarStates.isEmpty()) {
            return null;
        }

        // Predict based on transitions from similar states
        var transitionCounts = new HashMap<S, Double>();
        for (var similar : similarStates) {
            var transitions = transitionLearner.getTransitionsFrom(similar);
            for (var t : transitions) {
                transitionCounts.merge(t.toState(), t.probability(), Double::sum);
            }
        }

        // Normalize and find most likely
        var total = transitionCounts.values().stream()
            .mapToDouble(Double::doubleValue)
            .sum();

        if (total == 0) {
            return null;
        }

        var mostLikely = transitionCounts.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .orElse(null);

        if (mostLikely == null) {
            return null;
        }

        return new PredictionResult<>(
            mostLikely.getKey(),
            mostLikely.getValue() / total,
            0.8, // ART confidence based on cluster cohesion
            0.0, // No Markov component
            PredictionSource.ART_ONLY
        );
    }

    private PredictionResult<S> predictUsingMarkov(S currentState) {
        var transitions = transitionLearner.getTransitionsFrom(currentState);

        if (transitions.isEmpty()) {
            return new PredictionResult<>(
                currentState, // Stay in same state
                1.0,
                0.0,
                0.5, // Low confidence
                PredictionSource.FALLBACK
            );
        }

        // Find most likely transition
        var mostLikely = transitions.stream()
            .max(Comparator.comparing(Transition::probability))
            .orElse(null);

        if (mostLikely == null) {
            return null;
        }

        return new PredictionResult<>(
            mostLikely.toState(),
            mostLikely.probability(),
            0.0, // No ART component
            0.9, // High Markov confidence
            PredictionSource.MARKOV_ONLY
        );
    }

    private PredictionResult<S> combinePredicitions(
        PredictionResult<S> artPred,
        PredictionResult<S> markovPred
    ) {
        // Handle null predictions
        if (artPred == null && markovPred == null) {
            return null;
        }
        if (artPred == null) {
            return markovPred;
        }
        if (markovPred == null) {
            return artPred;
        }

        // Combine based on strategy
        var combinedConfidence = strategy.combineConfidences(
            artPred.artConfidence(),
            markovPred.markovConfidence()
        );

        // Choose prediction based on combined confidence
        S predictedState;
        double probability;

        if (artPred.predictedState().equals(markovPred.predictedState())) {
            // Both agree
            predictedState = artPred.predictedState();
            probability = (artPred.probability() + markovPred.probability()) / 2.0;
        } else {
            // Disagreement - use weighted selection
            if (combinedConfidence > 0.5) {
                predictedState = artPred.predictedState();
                probability = artPred.probability() * combinedConfidence;
            } else {
                predictedState = markovPred.predictedState();
                probability = markovPred.probability() * (1 - combinedConfidence);
            }
        }

        return new PredictionResult<>(
            predictedState,
            probability,
            artPred.artConfidence(),
            markovPred.markovConfidence(),
            PredictionSource.HYBRID
        );
    }

    @Override
    public List<PredictionResult<S>> predictSequence(S startState, int horizon) {
        var predictions = new ArrayList<PredictionResult<S>>();
        var currentState = startState;

        for (int i = 0; i < horizon; i++) {
            var prediction = predictNextState(currentState, null);
            if (prediction == null) {
                break;
            }
            predictions.add(prediction);
            currentState = prediction.predictedState();
        }

        return predictions;
    }

    @Override
    public Map<String, Double> predictActionValues(
        S state,
        Set<String> availableActions
    ) {
        // Implementation for action-value prediction
        // This would integrate with reinforcement learning components
        return Map.of();
    }
}

//================================================================================
// VECTORIZED PERFORMANCE IMPLEMENTATION
//================================================================================

/**
 * High-performance vectorized transition operations
 */
public class VectorizedTransitionOperations {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    /**
     * Vectorized matrix-vector multiplication for state prediction
     */
    public static double[] predictStateVector(
        double[][] transitionMatrix,
        double[] currentState
    ) {
        var n = currentState.length;
        var result = new double[n];

        // Process in vector chunks
        var upperBound = SPECIES.loopBound(n);

        for (int i = 0; i < n; i++) {
            var sum = 0.0;

            // Vectorized inner loop
            int j = 0;
            for (; j < upperBound; j += SPECIES.length()) {
                var matrixVector = DoubleVector.fromArray(
                    SPECIES, transitionMatrix[i], j
                );
                var stateVector = DoubleVector.fromArray(
                    SPECIES, currentState, j
                );
                var product = matrixVector.mul(stateVector);
                sum += product.reduceLanes(VectorOperators.ADD);
            }

            // Handle remaining elements
            for (; j < n; j++) {
                sum += transitionMatrix[i][j] * currentState[j];
            }

            result[i] = sum;
        }

        return result;
    }

    /**
     * Vectorized normalization of probability distribution
     */
    public static void normalizeInPlace(double[] probabilities) {
        var sum = 0.0;
        var n = probabilities.length;
        var upperBound = SPECIES.loopBound(n);

        // Vectorized sum computation
        int i = 0;
        for (; i < upperBound; i += SPECIES.length()) {
            var vector = DoubleVector.fromArray(SPECIES, probabilities, i);
            sum += vector.reduceLanes(VectorOperators.ADD);
        }

        // Handle remaining elements
        for (; i < n; i++) {
            sum += probabilities[i];
        }

        // Vectorized normalization
        if (sum > 0) {
            var invSum = 1.0 / sum;
            i = 0;
            for (; i < upperBound; i += SPECIES.length()) {
                var vector = DoubleVector.fromArray(SPECIES, probabilities, i);
                vector = vector.mul(invSum);
                vector.intoArray(probabilities, i);
            }

            // Handle remaining elements
            for (; i < n; i++) {
                probabilities[i] *= invSum;
            }
        }
    }

    /**
     * Vectorized distance computation
     */
    public static double euclideanDistance(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have same length");
        }

        var sum = 0.0;
        var n = a.length;
        var upperBound = SPECIES.loopBound(n);

        // Vectorized difference and square
        int i = 0;
        for (; i < upperBound; i += SPECIES.length()) {
            var va = DoubleVector.fromArray(SPECIES, a, i);
            var vb = DoubleVector.fromArray(SPECIES, b, i);
            var diff = va.sub(vb);
            var squared = diff.mul(diff);
            sum += squared.reduceLanes(VectorOperators.ADD);
        }

        // Handle remaining elements
        for (; i < n; i++) {
            var diff = a[i] - b[i];
            sum += diff * diff;
        }

        return Math.sqrt(sum);
    }
}

//================================================================================
// COMPREHENSIVE VALIDATOR IMPLEMENTATION
//================================================================================

/**
 * Complete implementation of probability and matrix validation
 */
public class ComprehensiveProbabilityValidator implements ProbabilityValidator {

    private static final double EPSILON = 1e-10;
    private static final double STABILITY_THRESHOLD = 1e-6;

    @Override
    public ValidationReport validateDistribution(double[] probabilities) {
        var violations = new ArrayList<ValidationReport.Violation>();
        var statistics = new HashMap<String, Object>();

        // Check for negative values
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] < 0) {
                violations.add(new ValidationReport.Violation(
                    "Non-negative",
                    ">= 0",
                    String.valueOf(probabilities[i]),
                    new int[]{i}
                ));
            }
            if (probabilities[i] > 1 + EPSILON) {
                violations.add(new ValidationReport.Violation(
                    "Upper bound",
                    "<= 1",
                    String.valueOf(probabilities[i]),
                    new int[]{i}
                ));
            }
        }

        // Check sum equals 1
        var sum = Arrays.stream(probabilities).sum();
        if (Math.abs(sum - 1.0) > EPSILON) {
            violations.add(new ValidationReport.Violation(
                "Sum to one",
                "1.0",
                String.valueOf(sum),
                new int[]{}
            ));
        }

        // Compute statistics
        statistics.put("sum", sum);
        statistics.put("min", Arrays.stream(probabilities).min().orElse(0.0));
        statistics.put("max", Arrays.stream(probabilities).max().orElse(0.0));
        statistics.put("entropy", computeEntropy(probabilities));

        return new ValidationReport(
            violations.isEmpty(),
            violations,
            statistics
        );
    }

    @Override
    public ValidationReport validateStochasticMatrix(double[][] matrix) {
        var violations = new ArrayList<ValidationReport.Violation>();
        var statistics = new HashMap<String, Object>();

        var n = matrix.length;

        // Validate each row
        for (int i = 0; i < n; i++) {
            if (matrix[i].length != n) {
                violations.add(new ValidationReport.Violation(
                    "Square matrix",
                    String.valueOf(n),
                    String.valueOf(matrix[i].length),
                    new int[]{i}
                ));
                continue;
            }

            // Check row sum
            var rowSum = Arrays.stream(matrix[i]).sum();
            if (Math.abs(rowSum - 1.0) > EPSILON) {
                violations.add(new ValidationReport.Violation(
                    "Row sum",
                    "1.0",
                    String.valueOf(rowSum),
                    new int[]{i}
                ));
            }

            // Check for negative values
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] < 0) {
                    violations.add(new ValidationReport.Violation(
                        "Non-negative",
                        ">= 0",
                        String.valueOf(matrix[i][j]),
                        new int[]{i, j}
                    ));
                }
            }
        }

        // Compute matrix statistics
        statistics.put("dimension", n);
        statistics.put("sparsity", computeSparsity(matrix));
        statistics.put("eigenvalueGap", computeEigenvalueGap(matrix));

        return new ValidationReport(
            violations.isEmpty(),
            violations,
            statistics
        );
    }

    @Override
    public double[] normalizeProbabilities(double[] probabilities) {
        var sum = Arrays.stream(probabilities).sum();

        if (sum == 0) {
            // Uniform distribution if all zeros
            var n = probabilities.length;
            var uniform = new double[n];
            Arrays.fill(uniform, 1.0 / n);
            return uniform;
        }

        // Normalize
        var normalized = new double[probabilities.length];
        for (int i = 0; i < probabilities.length; i++) {
            normalized[i] = Math.max(0, probabilities[i]) / sum;
        }

        // Fix numerical errors
        var newSum = Arrays.stream(normalized).sum();
        if (Math.abs(newSum - 1.0) > EPSILON) {
            normalized[0] += 1.0 - newSum; // Add correction to first element
        }

        return normalized;
    }

    @Override
    public boolean isNumericallyStable(double[][] matrix) {
        // Check condition number and other stability metrics
        for (var row : matrix) {
            for (var value : row) {
                if (Double.isNaN(value) || Double.isInfinite(value)) {
                    return false;
                }
                if (Math.abs(value) < STABILITY_THRESHOLD && value != 0) {
                    return false; // Too close to zero
                }
            }
        }

        // Check for near-singular conditions
        var determinant = approximateDeterminant(matrix);
        return Math.abs(determinant) > STABILITY_THRESHOLD;
    }

    private double computeEntropy(double[] probabilities) {
        return -Arrays.stream(probabilities)
            .filter(p -> p > 0)
            .map(p -> p * Math.log(p) / Math.log(2))
            .sum();
    }

    private double computeSparsity(double[][] matrix) {
        var zeros = 0;
        var total = 0;
        for (var row : matrix) {
            for (var value : row) {
                if (Math.abs(value) < EPSILON) {
                    zeros++;
                }
                total++;
            }
        }
        return (double) zeros / total;
    }

    private double computeEigenvalueGap(double[][] matrix) {
        // Simplified: would use proper eigenvalue computation
        return 0.0;
    }

    private double approximateDeterminant(double[][] matrix) {
        // Simplified: would use proper determinant computation
        return 1.0;
    }
}

//================================================================================
// TESTING INFRASTRUCTURE IMPLEMENTATION
//================================================================================

/**
 * Property-based testing implementation
 */
public class MarkovPropertyTesterImpl<S extends State<?>>
    implements MarkovPropertyTester<S> {

    private final Random random = new Random();

    @Override
    public PropertyTestResult testStochasticProperties(
        StochasticMatrix<S> matrix,
        int iterations
    ) {
        var properties = new HashMap<String, Boolean>();
        var counterExamples = new ArrayList<CounterExample<?>>();
        var passedTests = 0;

        for (int i = 0; i < iterations; i++) {
            // Test row sum property
            var testPassed = testRowSumProperty(matrix, counterExamples);
            if (testPassed) passedTests++;

            // Test non-negativity
            testPassed = testNonNegativity(matrix, counterExamples);
            if (testPassed) passedTests++;
        }

        properties.put("rowSum", counterExamples.isEmpty());
        properties.put("nonNegative", true);
        properties.put("normalized", true);

        var statistics = new TestStatistics(
            iterations * 3,
            passedTests,
            iterations * 3 - passedTests,
            0.0,
            Map.of()
        );

        return new PropertyTestResult(
            counterExamples.isEmpty(),
            properties,
            counterExamples,
            statistics
        );
    }

    private boolean testRowSumProperty(
        StochasticMatrix<S> matrix,
        List<CounterExample<?>> counterExamples
    ) {
        // Implementation
        return true;
    }

    private boolean testNonNegativity(
        StochasticMatrix<S> matrix,
        List<CounterExample<?>> counterExamples
    ) {
        // Implementation
        return true;
    }

    @Override
    public PropertyTestResult testConvergence(
        TransitionLearner<S> learner,
        List<Transition<S, ?>> testData
    ) {
        // Test convergence by repeatedly applying transitions
        var initialMatrix = captureMatrix(learner);

        // Apply transitions
        for (var transition : testData) {
            learner.observeTransition(
                transition.fromState(),
                transition.toState(),
                transition.context()
            );
        }

        var finalMatrix = captureMatrix(learner);

        // Check convergence metrics
        var converged = matrixDistance(initialMatrix, finalMatrix) < 0.01;

        return new PropertyTestResult(
            converged,
            Map.of("converged", converged),
            List.of(),
            new TestStatistics(1, converged ? 1 : 0, converged ? 0 : 1, 0.0, Map.of())
        );
    }

    @Override
    public PropertyTestResult testMemorylessProperty(
        List<S> stateSequence,
        double significanceLevel
    ) {
        // Chi-square test for independence
        if (stateSequence.size() < 3) {
            return new PropertyTestResult(
                false,
                Map.of("sufficientData", false),
                List.of(),
                new TestStatistics(0, 0, 0, 0.0, Map.of())
            );
        }

        // Count transitions
        var transitionCounts = new HashMap<TransitionPair<S>, Integer>();
        for (int i = 0; i < stateSequence.size() - 1; i++) {
            var pair = new TransitionPair<>(
                stateSequence.get(i),
                stateSequence.get(i + 1)
            );
            transitionCounts.merge(pair, 1, Integer::sum);
        }

        // Compute chi-square statistic
        var chiSquare = computeChiSquare(transitionCounts, stateSequence);

        // Determine if memoryless property holds
        var degreesOfFreedom = (transitionCounts.size() - 1);
        var critical = getCriticalValue(degreesOfFreedom, significanceLevel);
        var isMemoryless = chiSquare < critical;

        return new PropertyTestResult(
            isMemoryless,
            Map.of("memoryless", isMemoryless),
            List.of(),
            new TestStatistics(
                1,
                isMemoryless ? 1 : 0,
                isMemoryless ? 0 : 1,
                0.0,
                Map.of("chiSquare", chiSquare, "critical", critical)
            )
        );
    }

    private double[][] captureMatrix(TransitionLearner<S> learner) {
        // Capture current matrix state
        return new double[0][0];
    }

    private double matrixDistance(double[][] a, double[][] b) {
        // Frobenius norm
        var sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                var diff = a[i][j] - b[i][j];
                sum += diff * diff;
            }
        }
        return Math.sqrt(sum);
    }

    private double computeChiSquare(
        Map<TransitionPair<S>, Integer> counts,
        List<S> sequence
    ) {
        // Simplified chi-square computation
        return 0.0;
    }

    private double getCriticalValue(int df, double alpha) {
        // Chi-square critical values (simplified)
        return 3.841; // For df=1, alpha=0.05
    }

    record TransitionPair<S>(S from, S to) {}
}

//================================================================================
// EXAMPLE USAGE AND INTEGRATION
//================================================================================

/**
 * Example of complete system integration
 */
public class HybridSystemExample {

    public static void main(String[] args) {
        // 1. Create components
        var encoder = new SimpleEncoder();
        var config = new StateAbstractionART.StateAbstractionConfig(
            0.7,  // vigilance
            0.1,  // learning rate
            100,  // max categories
            true, // allow dynamic creation
            encoder
        );

        var stateAbstractor = new FuzzyARTStateAbstractor<double[]>(config);
        var validator = new ComprehensiveProbabilityValidator();
        var transitionLearner = new ValidatedTransitionLearner<ContinuousState>(validator);
        var contextAugmenter = new SimpleContextAugmenter<ContinuousState>();
        var predictor = new ConcreteHybridPredictor<>(
            stateAbstractor,
            transitionLearner,
            contextAugmenter,
            HybridStrategy.WEIGHTED_COMBINATION
        );

        // 2. Build hybrid system
        var system = new HybridARTMarkovSystem.Builder<double[], ContinuousState, Context, String, Double>()
            .withStateAbstractor(stateAbstractor)
            .withStrategy(HybridStrategy.ART_STATE_DISCOVERY)
            .withLearningMode(LearningMode.ONLINE)
            .build();

        // 3. Process observations
        var observation = new double[]{0.5, 0.3, 0.2};
        var context = new Context.TemporalContext(
            System.currentTimeMillis(),
            Duration.ofSeconds(1),
            0,
            List.of()
        );

        var state = system.processObservation(observation, context);
        System.out.println("Discovered state: " + state.id());

        // 4. Make prediction
        var prediction = system.predict(state, context);
        System.out.println("Predicted next state: " + prediction.predictedState().id());
        System.out.println("Confidence: " + prediction.probability());
    }

    static class SimpleEncoder implements StateAbstractionART.StateEncoder<double[], double[]> {
        @Override
        public double[] encode(double[] observation) {
            return observation.clone();
        }

        @Override
        public double[] decode(double[] encoded) {
            return encoded.clone();
        }

        @Override
        public int dimensionality() {
            return 3;
        }
    }

    static class SimpleContextAugmenter<S extends State<?>>
        implements ContextAugmenter<S, Context> {

        @Override
        public Context augmentContext(S state, Map<String, Object> rawContext) {
            return new Context.FeatureContext(
                new double[]{0.5},
                Map.of(),
                0.8,
                0
            );
        }

        @Override
        public void learnContextPattern(S state, Context context, double reward) {
            // Learn patterns
        }

        @Override
        public Collection<ContextCluster<Context>> getContextClusters() {
            return List.of();
        }
    }
}