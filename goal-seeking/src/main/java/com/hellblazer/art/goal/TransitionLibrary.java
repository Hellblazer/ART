package com.hellblazer.art.goal;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Stores and manages learned state transitions.
 * This is the actual learning component that remembers successful transitions
 * and can recall them for similar states.
 */
public class TransitionLibrary {
    private static final Logger log = LoggerFactory.getLogger(TransitionLibrary.class);

    // Storage for learned transitions
    private final Map<TransitionKey, List<LearnedTransition>> transitions;

    // Index for fast similarity search
    private final Map<Integer, Set<TransitionKey>> stateIndex;

    // Configuration
    private final float similarityThreshold;
    private final int maxTransitionsPerPair;
    private final int maxTotalTransitions;

    // Statistics
    private int totalTransitions = 0;
    private int successfulRecalls = 0;
    private int failedRecalls = 0;

    public TransitionLibrary() {
        this(0.85f, 10, 10000);
    }

    public TransitionLibrary(float similarityThreshold, int maxTransitionsPerPair, int maxTotalTransitions) {
        this.similarityThreshold = similarityThreshold;
        this.maxTransitionsPerPair = maxTransitionsPerPair;
        this.maxTotalTransitions = maxTotalTransitions;
        this.transitions = new ConcurrentHashMap<>();
        this.stateIndex = new ConcurrentHashMap<>();
    }

    /**
     * Learn a new transition from experience
     */
    public void learnTransition(State from, State to, Action action, float success) {
        if (success < 0.5f) {
            return; // Don't learn from failures
        }

        var key = new TransitionKey(from, to);
        var transition = new LearnedTransition(action, success, System.currentTimeMillis());

        transitions.compute(key, (k, list) -> {
            if (list == null) {
                list = new ArrayList<>();
            }

            // Add new transition
            list.add(transition);

            // Keep only best transitions if over limit
            if (list.size() > maxTransitionsPerPair) {
                list.sort((a, b) -> Float.compare(b.successRate, a.successRate));
                list = new ArrayList<>(list.subList(0, maxTransitionsPerPair));
            }

            return list;
        });

        // Update index
        indexState(from, key);
        indexState(to, key);

        totalTransitions++;

        // Prune if needed
        if (totalTransitions > maxTotalTransitions) {
            pruneOldTransitions();
        }

        log.debug("Learned transition: {} -> {} with success {}",
                 from.hashCode(), to.hashCode(), success);
    }

    /**
     * Find applicable transitions for moving from current state toward goal
     */
    public List<Action> findTransitions(State from, State goal) {
        var results = new ArrayList<Action>();

        // First try exact match
        var exactKey = new TransitionKey(from, goal);
        var exactTransitions = transitions.get(exactKey);
        if (exactTransitions != null && !exactTransitions.isEmpty()) {
            successfulRecalls++;
            return extractActions(exactTransitions);
        }

        // Find similar starting states
        var similarKeys = findSimilarTransitions(from);

        for (var key : similarKeys) {
            // Check if this transition moves us closer to goal
            if (isProgressiveTransition(key, from, goal)) {
                var transitionList = transitions.get(key);
                if (transitionList != null) {
                    results.addAll(extractActions(transitionList));
                    if (!results.isEmpty()) {
                        successfulRecalls++;
                        return results; // Return first good match
                    }
                }
            }
        }

        failedRecalls++;
        return results; // Empty if no matches found
    }

    /**
     * Get all transitions that can be applied from a given state
     */
    public List<TransitionOption> getApplicableTransitions(State from) {
        var options = new ArrayList<TransitionOption>();

        // Find all transitions starting from similar states
        var similarKeys = findSimilarTransitions(from);

        for (var key : similarKeys) {
            var similarity = computeSimilarity(from, key.fromState);
            if (similarity >= similarityThreshold) {
                var transitionList = transitions.get(key);
                if (transitionList != null) {
                    for (var transition : transitionList) {
                        options.add(new TransitionOption(
                            key.toState,
                            transition.action,
                            transition.successRate,
                            similarity
                        ));
                    }
                }
            }
        }

        // Sort by expected success
        options.sort((a, b) -> Float.compare(
            b.successRate * b.similarity,
            a.successRate * a.similarity
        ));

        return options;
    }

    /**
     * Update success rate of a previously used transition
     */
    public void updateTransitionSuccess(State from, State to, Action action, float newSuccess) {
        var key = new TransitionKey(from, to);
        var transitionList = transitions.get(key);

        if (transitionList != null) {
            for (var transition : transitionList) {
                if (transition.action.equals(action)) {
                    // Update success rate with exponential moving average
                    transition.successRate = 0.7f * transition.successRate + 0.3f * newSuccess;
                    transition.useCount++;
                    transition.lastUsed = System.currentTimeMillis();
                    break;
                }
            }
        }
    }

    // ===== Helper Methods =====

    private void indexState(State state, TransitionKey key) {
        int hash = computeStateHash(state);
        stateIndex.computeIfAbsent(hash, k -> ConcurrentHashMap.newKeySet()).add(key);
    }

    private Set<TransitionKey> findSimilarTransitions(State state) {
        Set<TransitionKey> similar = new HashSet<>();

        // Get exact hash matches
        int hash = computeStateHash(state);
        var exact = stateIndex.get(hash);
        if (exact != null) {
            similar.addAll(exact);
        }

        // Also check nearby hashes for similar states
        for (int offset = -1; offset <= 1; offset++) {
            if (offset != 0) {
                var nearby = stateIndex.get(hash + offset);
                if (nearby != null) {
                    similar.addAll(nearby);
                }
            }
        }

        return similar;
    }

    private boolean isProgressiveTransition(TransitionKey key, State from, State goal) {
        // Check if this transition moves us closer to the goal
        float currentDistance = from.distanceTo(goal);
        float afterDistance = key.toState.distanceTo(goal);
        return afterDistance < currentDistance * 0.95f; // Must improve by at least 5%
    }

    private float computeSimilarity(State s1, State s2) {
        if (s1 == null || s2 == null) return 0;

        float distance = s1.distanceTo(s2);
        // Convert distance to similarity (assuming normalized states)
        return Math.max(0, 1.0f - distance);
    }

    private int computeStateHash(State state) {
        // Simple hash based on state encoding
        if (state instanceof StateTransitionGenerator.State stgState) {
            float[] vector = stgState.vectorTo(stgState); // Self-vector as signature
            int hash = 0;
            for (int i = 0; i < Math.min(vector.length, 4); i++) {
                hash = hash * 31 + Float.floatToIntBits(vector[i]);
            }
            return hash;
        }
        return state.hashCode();
    }

    private List<Action> extractActions(List<LearnedTransition> transitions) {
        // Get the best actions from learned transitions
        return transitions.stream()
            .sorted((a, b) -> Float.compare(b.successRate, a.successRate))
            .limit(3) // Return top 3 options
            .map(t -> t.action)
            .toList();
    }

    private void pruneOldTransitions() {
        // Remove least recently used transitions
        long cutoffTime = System.currentTimeMillis() - (7L * 24 * 60 * 60 * 1000); // 7 days

        transitions.entrySet().removeIf(entry -> {
            entry.getValue().removeIf(t -> t.lastUsed < cutoffTime && t.useCount < 5);
            return entry.getValue().isEmpty();
        });

        // Rebuild index
        stateIndex.clear();
        transitions.forEach((key, list) -> {
            indexState(key.fromState, key);
            indexState(key.toState, key);
        });

        totalTransitions = transitions.values().stream()
            .mapToInt(List::size)
            .sum();
    }

    // ===== Inner Classes =====

    static class TransitionKey {
        final State fromState;
        final State toState;
        private final int hashCode;

        TransitionKey(State from, State to) {
            this.fromState = from;
            this.toState = to;
            this.hashCode = Objects.hash(
                System.identityHashCode(from),
                System.identityHashCode(to)
            );
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (!(obj instanceof TransitionKey other)) return false;
            return fromState == other.fromState && toState == other.toState;
        }

        @Override
        public int hashCode() {
            return hashCode;
        }
    }

    static class LearnedTransition {
        final Action action;
        float successRate;
        int useCount;
        long lastUsed;

        LearnedTransition(Action action, float successRate, long timestamp) {
            this.action = action;
            this.successRate = successRate;
            this.useCount = 1;
            this.lastUsed = timestamp;
        }
    }

    public static class TransitionOption {
        public final State targetState;
        public final Action action;
        public final float successRate;
        public final float similarity;

        TransitionOption(State target, Action action, float success, float similarity) {
            this.targetState = target;
            this.action = action;
            this.successRate = success;
            this.similarity = similarity;
        }

        public float getExpectedValue() {
            return successRate * similarity;
        }
    }

    public static class Action {
        public final String type;
        public final float[] parameters;
        public final float cost;

        public Action(String type, float[] parameters, float cost) {
            this.type = type;
            this.parameters = parameters.clone();
            this.cost = cost;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (!(obj instanceof Action other)) return false;
            return type.equals(other.type) &&
                   Arrays.equals(parameters, other.parameters);
        }

        @Override
        public int hashCode() {
            return Objects.hash(type, Arrays.hashCode(parameters));
        }
    }

    // ===== Statistics =====

    public LibraryStats getStatistics() {
        int totalStored = transitions.values().stream()
            .mapToInt(List::size)
            .sum();

        float recallRate = (successfulRecalls + failedRecalls) > 0 ?
            (float) successfulRecalls / (successfulRecalls + failedRecalls) : 0;

        return new LibraryStats(
            transitions.size(),
            totalStored,
            successfulRecalls,
            failedRecalls,
            recallRate
        );
    }

    public static class LibraryStats {
        public final int uniquePairs;
        public final int totalTransitions;
        public final int successfulRecalls;
        public final int failedRecalls;
        public final float recallRate;

        LibraryStats(int pairs, int total, int success, int failed, float rate) {
            this.uniquePairs = pairs;
            this.totalTransitions = total;
            this.successfulRecalls = success;
            this.failedRecalls = failed;
            this.recallRate = rate;
        }
    }

    // Simple State interface for independence from StateTransitionGenerator
    public interface State {
        float distanceTo(State other);
    }
}