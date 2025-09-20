package com.hellblazer.art.core;

import com.hellblazer.art.core.cvi.CalinskiHarabaszIndex;
import com.hellblazer.art.core.cvi.ClusterValidityIndex;
import com.hellblazer.art.core.results.ActivationResult;
import com.hellblazer.art.core.results.MatchResult;

import java.util.*;

/**
 * CVIART - ART with integrated Cluster Validity Indices.
 * Automatically adjusts learning parameters based on clustering quality metrics.
 */
public class CVIART extends CVIEnabledART {
    
    // Track pattern history locally
    private final List<Pattern> patternHistory = new ArrayList<>();
    private final Set<Integer> seenPatternHashes = new HashSet<>();
    private int epochCounter = 0;
    private int lastUpdateEpoch = -1;
    
    // CVI management
    private final List<ClusterValidityIndex> cvis = new ArrayList<>();
    private final Map<String, Double> currentScores = new HashMap<>();
    private final Map<String, List<Double>> scoreHistory = new HashMap<>();
    
    // Vigilance adaptation
    private double currentVigilance = -1; // -1 indicates not yet initialized
    private double vigilanceAdaptationRate = 0.01;
    private double targetClusters = 0; // 0 means no target
    
    // Optimization strategy
    private OptimizationStrategy currentStrategy = OptimizationStrategy.SINGLE_CVI;
    private String primaryCVI = "Calinski-Harabasz Index";
    private Map<String, Double> cviWeights = new HashMap<>();
    private Map<String, Double> cviThresholds = new HashMap<>();
    
    // Learning state
    private boolean adaptiveVigilance = false;
    private int epochsSinceImprovement = 0;
    private double bestCompositeScore = Double.NEGATIVE_INFINITY;
    
    public CVIART() {
        super();
        // Add default CVI
        cvis.add(new CalinskiHarabaszIndex());
        // currentVigilance will be initialized on first learn() call
    }
    
    @Override
    protected WeightVector createInitialWeight(Pattern input, Object parameters) {
        // Create a simple weight vector initialized with the input pattern values
        double[] values = new double[input.dimension()];
        for (int i = 0; i < input.dimension(); i++) {
            values[i] = input.get(i);
        }
        return new SimpleWeight(values);
    }
    
    @Override
    protected WeightVector updateWeights(Pattern input, WeightVector currentWeight, Object parameters) {
        // Simple weight update: average between current weight and input
        double[] updated = new double[input.dimension()];
        double learningRate = 0.5; // Default learning rate
        
        if (parameters instanceof CVIARTParameters params) {
            // Could extract learning rate from params if added
        }
        
        for (int i = 0; i < input.dimension(); i++) {
            double currentValue = currentWeight.get(i);
            double inputValue = input.get(i);
            updated[i] = currentValue + learningRate * (inputValue - currentValue);
        }
        
        return new SimpleWeight(updated);
    }
    
    @Override
    protected double calculateActivation(Pattern input, WeightVector weight, Object parameters) {
        // Calculate activation using FuzzyART-style min operation
        double sumMin = 0.0;
        double sumWeight = 0.0;
        
        for (int i = 0; i < input.dimension(); i++) {
            double inputVal = input.get(i);
            double weightVal = weight.get(i);
            sumMin += Math.min(inputVal, weightVal);
            sumWeight += weightVal;
        }
        
        // Calculate activation with alpha (choice parameter)
        double activation = 0.0;
        double alpha = 0.0; // Default choice parameter
        if (parameters instanceof CVIARTParameters params) {
            // Could add choice parameter to params if needed
        }
        
        if (sumWeight + alpha > 0) {
            activation = sumMin / (alpha + sumWeight);
        }
        
        return activation;
    }
    
    @Override
    protected MatchResult checkVigilance(Pattern input, WeightVector weight, Object parameters) {
        // Use current vigilance which may have been adapted
        double vigilance = currentVigilance;
        
        // Calculate match score using min operation (FuzzyART-style)
        double sumMin = 0.0;
        double sumInput = 0.0;
        
        for (int i = 0; i < input.dimension(); i++) {
            double inputVal = input.get(i);
            double weightVal = weight.get(i);
            sumMin += Math.min(inputVal, weightVal);
            sumInput += inputVal;
        }
        
        // Calculate match score
        double similarity = 0.0;
        if (sumInput > 0) {
            similarity = sumMin / sumInput;
        }
        
        // Check vigilance criterion
        if (similarity >= vigilance) {
            return new MatchResult.Accepted(similarity, vigilance);
        } else {
            return new MatchResult.Rejected(similarity, vigilance);
        }
    }
    
    // CVI Management Methods
    
    public boolean hasCVI() {
        return !cvis.isEmpty();
    }
    
    public int getCVICount() {
        return cvis.size();
    }
    
    public List<String> getCVINames() {
        return cvis.stream().map(ClusterValidityIndex::getName).toList();
    }
    
    public void addCVI(ClusterValidityIndex cvi) {
        cvis.add(cvi);
    }
    
    public void removeCVI(String name) {
        cvis.removeIf(cvi -> cvi.getName().equals(name));
    }
    
    public void clearCVIs() {
        cvis.clear();
        currentScores.clear();
        scoreHistory.clear();
    }
    
    public Map<String, Double> getCurrentCVIScores() {
        return new HashMap<>(currentScores);
    }
    
    public Map<String, List<Double>> getCVIHistory() {
        return new HashMap<>(scoreHistory);
    }
    
    /**
     * Protected method for subclasses to access the CVIs list.
     */
    protected List<ClusterValidityIndex> getCVIs() {
        return cvis;
    }
    
    // Vigilance and Optimization Methods
    
    public double getCurrentVigilance() {
        return currentVigilance;
    }
    
    // Can't override final method, use parent's count instead
    // We'll need to sync with parent's category management
    
    protected List<Pattern> getPatternHistory() {
        return new ArrayList<>(patternHistory);
    }
    
    public OptimizationStrategy getCurrentOptimizationStrategy() {
        return currentStrategy;
    }
    
    public void setOptimizationStrategy(OptimizationStrategy strategy) {
        this.currentStrategy = strategy;
    }
    
    public void setPrimaryCVI(String cviName) {
        this.primaryCVI = cviName;
    }
    
    public void setCVIWeights(Map<String, Double> weights) {
        this.cviWeights = new HashMap<>(weights);
    }
    
    public void setCVIThresholds(Map<String, Double> thresholds) {
        this.cviThresholds = new HashMap<>(thresholds);
    }
    
    // Learning Methods
    
    public LearningResult learn(Pattern pattern, CVIARTParameters params) {
        // Initialize vigilance only on first call
        if (currentVigilance < 0) {
            currentVigilance = params.getInitialVigilance();
        }
        adaptiveVigilance = params.isAdaptiveVigilance();
        vigilanceAdaptationRate = params.getVigilanceAdaptationRate();
        targetClusters = params.getTargetClusters();
        
        // Set optimization strategy and related parameters
        // Don't override if we've already selected a concrete strategy from ADAPTIVE
        if (params.getCVIOptimizationStrategy() != null) {
            if (params.getCVIOptimizationStrategy() != OptimizationStrategy.ADAPTIVE ||
                currentStrategy == OptimizationStrategy.ADAPTIVE) {
                currentStrategy = params.getCVIOptimizationStrategy();
            }
        }
        if (params.getPrimaryCVI() != null) {
            primaryCVI = params.getPrimaryCVI();
        }
        if (params.getCVIWeights() != null && !params.getCVIWeights().isEmpty()) {
            cviWeights = new HashMap<>(params.getCVIWeights());
        }
        if (params.getCVIThresholds() != null && !params.getCVIThresholds().isEmpty()) {
            cviThresholds = new HashMap<>(params.getCVIThresholds());
        }
        
        // Track pattern for CVI calculation
        patternHistory.add(pattern);
        
        // Track unique patterns to detect epochs
        double[] patternArray = new double[pattern.dimension()];
        for (int i = 0; i < pattern.dimension(); i++) {
            patternArray[i] = pattern.get(i);
        }
        int patternHash = Arrays.hashCode(patternArray);
        boolean isNewPattern = seenPatternHashes.add(patternHash);
        
        // Remember the category count before learning
        int previousCategoryCount = getCategoryCount();
        
        // Use parent class's stepFit method
        var result = stepFit(pattern, params);
        
        // Check if clustering structure changed
        boolean structureChanged = getCategoryCount() != previousCategoryCount;
        
        // Update category prototypes for CVI calculations
        if (result instanceof ActivationResult.Success success) {
            int categoryIndex = success.categoryIndex();
            
            // Update or add prototype for this category
            while (categoryPrototypes.size() <= categoryIndex) {
                categoryPrototypes.add(null);
                structureChanged = true; // New category added
            }
            
            // Use the weight as the prototype
            var weight = getCategory(categoryIndex);
            if (weight != null) {
                // Convert weight to pattern for prototype
                double[] values = new double[pattern.dimension()];
                for (int i = 0; i < pattern.dimension(); i++) {
                    values[i] = weight.get(i);
                }
                var newPrototype = new DenseVector(values);
                
                // Check if prototype changed significantly
                if (categoryPrototypes.get(categoryIndex) == null ||
                    !prototypesEqual(categoryPrototypes.get(categoryIndex), newPrototype)) {
                    categoryPrototypes.set(categoryIndex, newPrototype);
                    structureChanged = true;
                }
            }
        }
        
        // Determine if we should update CVIs
        boolean shouldUpdateCVIs = false;
        
        if (structureChanged) {
            // Always update if structure changed
            shouldUpdateCVIs = true;
        } else if (isNewPattern) {
            // Update for new patterns (building initial model)
            shouldUpdateCVIs = true;
        } else if (epochCounter - lastUpdateEpoch >= 5) {
            // Update periodically even for repeated patterns (for trend tracking)
            shouldUpdateCVIs = true;
        }
        
        // Track epochs
        if (!isNewPattern) {
            epochCounter++;
        }
        
        // Update CVIs if needed
        if (shouldUpdateCVIs && patternHistory.size() > 1 && hasCVI()) {
            updateCVIScoresInternal();
            lastUpdateEpoch = epochCounter;
            
            // Adapt vigilance if enabled
            if (adaptiveVigilance) {
                adaptVigilance();
            }
        }
        
        boolean success = result instanceof ActivationResult.Success;
        int categoryIndex = success ? ((ActivationResult.Success) result).categoryIndex() : -1;
        
        return new LearningResult(success, categoryIndex);
    }
    
    private boolean processPattern(Pattern pattern) {
        // Track the pattern
        patternHistory.add(pattern);
        
        // Determine category assignment based on similarity
        int bestCategory = -1;
        double bestActivation = 0.0;
        
        // Search existing categories
        for (int i = 0; i < getCategoryCount(); i++) {
            // Simulate activation calculation
            double activation = calculateSimulatedActivation(pattern, i);
            if (activation > bestActivation) {
                bestActivation = activation;
                bestCategory = i;
            }
        }
        
        // Check if best match meets vigilance
        boolean createNewCategory = (getCategoryCount() == 0) || (bestActivation < currentVigilance);
        
        if (createNewCategory) {
            // Create new category
            bestCategory = getCategoryCount();
            // Add prototype for new category
            double[] prototypeValues = new double[pattern.dimension()];
            for (int i = 0; i < pattern.dimension(); i++) {
                prototypeValues[i] = pattern.get(i);
            }
            categoryPrototypes.add(new DenseVector(prototypeValues));
        } else {
            // Update existing prototype (simple averaging)
            if (bestCategory < categoryPrototypes.size()) {
                var oldPrototype = categoryPrototypes.get(bestCategory);
                double[] newValues = new double[pattern.dimension()];
                for (int i = 0; i < pattern.dimension(); i++) {
                    newValues[i] = (oldPrototype.get(i) + pattern.get(i)) / 2.0;
                }
                categoryPrototypes.set(bestCategory, new DenseVector(newValues));
            }
        }
        
        // Create result for tracking
        var weight = new SimpleWeight(new double[pattern.dimension()]);
        var result = new ActivationResult.Success(bestCategory, bestActivation, weight);
        trackPattern(pattern, result);
        
        return true;
    }
    
    private boolean prototypesEqual(Pattern p1, Pattern p2) {
        if (p1 == null || p2 == null) return false;
        double epsilon = 1e-6; // Small threshold for floating point comparison
        for (int i = 0; i < p1.dimension(); i++) {
            if (Math.abs(p1.get(i) - p2.get(i)) > epsilon) {
                return false;
            }
        }
        return true;
    }
    
    private double calculateSimulatedActivation(Pattern pattern, int categoryIndex) {
        // Keep track of category prototypes for better clustering
        if (categoryPrototypes.isEmpty() || categoryIndex >= categoryPrototypes.size()) {
            return 0.0;
        }
        
        var prototype = categoryPrototypes.get(categoryIndex);
        
        // Calculate similarity as inverse of Euclidean distance
        double distance = 0.0;
        for (int i = 0; i < pattern.dimension(); i++) {
            double diff = pattern.get(i) - prototype.get(i);
            distance += diff * diff;
        }
        distance = Math.sqrt(distance);
        
        // Convert distance to similarity in [0, 1]
        return Math.exp(-distance * 2.0); // Exponential decay based on distance
    }
    
    // Store category prototypes
    private final List<Pattern> categoryPrototypes = new ArrayList<>();
    
    private void updateCVIScoresInternal() {
        var patterns = patternHistory;
        if (patterns.size() < 2) return;
        
        // Create labels (simplified - assign patterns to categories)
        int[] labels = generateLabels(patterns.size());
        
        // Calculate centroids
        var centroids = calculateCentroids(patterns, labels);
        
        // Update each CVI
        for (var cvi : cvis) {
            try {
                double score = cvi.calculate(patterns, labels, centroids);
                currentScores.put(cvi.getName(), score);
                
                // Update history
                scoreHistory.computeIfAbsent(cvi.getName(), k -> new ArrayList<>()).add(score);
            } catch (Exception e) {
                // Handle CVI calculation failure gracefully
                // Log CVI calculation failure - handled gracefully
            }
        }
    }
    
    private int[] generateLabels(int size) {
        // Assign each pattern to its nearest prototype
        int[] labels = new int[size];
        
        if (categoryPrototypes.isEmpty()) {
            return labels; // All zeros if no categories
        }
        
        for (int i = 0; i < size && i < patternHistory.size(); i++) {
            var pattern = patternHistory.get(i);
            int bestCluster = 0;
            double minDistance = Double.MAX_VALUE;
            
            // Find nearest prototype
            for (int j = 0; j < categoryPrototypes.size(); j++) {
                var prototype = categoryPrototypes.get(j);
                double distance = 0.0;
                for (int k = 0; k < pattern.dimension(); k++) {
                    double diff = pattern.get(k) - prototype.get(k);
                    distance += diff * diff;
                }
                distance = Math.sqrt(distance);
                
                if (distance < minDistance) {
                    minDistance = distance;
                    bestCluster = j;
                }
            }
            
            labels[i] = bestCluster;
        }
        
        return labels;
    }
    
    private List<Pattern> calculateCentroids(List<Pattern> patterns, int[] labels) {
        int maxLabel = Arrays.stream(labels).max().orElse(0);
        List<Pattern> centroids = new ArrayList<>();
        
        for (int k = 0; k <= maxLabel; k++) {
            var clusterPatterns = new ArrayList<Pattern>();
            for (int i = 0; i < patterns.size(); i++) {
                if (labels[i] == k) {
                    clusterPatterns.add(patterns.get(i));
                }
            }
            
            if (!clusterPatterns.isEmpty()) {
                centroids.add(calculateCentroid(clusterPatterns));
            }
        }
        
        return centroids;
    }
    
    private Pattern calculateCentroid(List<Pattern> patterns) {
        if (patterns.isEmpty()) return null;
        
        // Simple centroid calculation
        int dimensions = patterns.get(0).dimension();
        double[] centroid = new double[dimensions];
        
        for (var pattern : patterns) {
            for (int i = 0; i < dimensions; i++) {
                centroid[i] += pattern.get(i);
            }
        }
        
        for (int i = 0; i < dimensions; i++) {
            centroid[i] /= patterns.size();
        }
        
        return new DenseVector(centroid);
    }
    
    private void adaptVigilance() {
        double adjustment = 0.0;
        
        switch (currentStrategy) {
            case SINGLE_CVI -> {
                if (currentScores.containsKey(primaryCVI)) {
                    adjustment = calculateSingleCVIAdjustment(primaryCVI);
                }
            }
            case WEIGHTED_AVERAGE -> {
                adjustment = calculateWeightedAverageAdjustment();
            }
            case PARETO_OPTIMAL -> {
                adjustment = calculateParetoOptimalAdjustment();
            }
            case THRESHOLD_BASED -> {
                adjustment = calculateThresholdBasedAdjustment();
            }
            case ADAPTIVE -> {
                // Select strategy based on current performance
                selectAdaptiveStrategy();
                adaptVigilance(); // Recursive call with new strategy
                return;
            }
        }
        
        // Apply adjustment
        currentVigilance = Math.max(0.0, Math.min(1.0, 
            currentVigilance + adjustment * vigilanceAdaptationRate));
    }
    
    private double calculateSingleCVIAdjustment(String cviName) {
        var history = scoreHistory.get(cviName);
        if (history == null || history.size() < 2) return 0.0;
        
        // Compare recent trend
        double recent = history.get(history.size() - 1);
        double previous = history.get(history.size() - 2);
        
        // Find the CVI to check if higher is better
        var cvi = cvis.stream()
            .filter(c -> c.getName().equals(cviName))
            .findFirst()
            .orElse(null);
        
        if (cvi == null) return 0.0;
        
        boolean improving = cvi.isHigherBetter() ? 
            recent > previous : recent < previous;
        
        // Adjust based on target clusters if set
        if (targetClusters > 0) {
            int currentClusters = getCategoryCount();
            if (currentClusters < targetClusters) {
                return 0.1; // Increase vigilance to create more clusters
            } else if (currentClusters > targetClusters) {
                return -0.1; // Decrease vigilance to merge clusters
            }
        }
        
        return improving ? 0.0 : 0.05; // Small increase if not improving
    }
    
    private double calculateWeightedAverageAdjustment() {
        double totalWeight = 0.0;
        double weightedScore = 0.0;
        
        for (var entry : currentScores.entrySet()) {
            String cviName = entry.getKey();
            double score = entry.getValue();
            double weight = cviWeights.getOrDefault(cviName, 1.0);
            
            // Normalize score based on CVI direction
            var cvi = cvis.stream()
                .filter(c -> c.getName().equals(cviName))
                .findFirst()
                .orElse(null);
            
            if (cvi != null) {
                if (!cvi.isHigherBetter()) {
                    score = 1.0 / (1.0 + score); // Invert for lower-is-better
                }
                weightedScore += score * weight;
                totalWeight += weight;
            }
        }
        
        if (totalWeight == 0) return 0.0;
        
        double compositeScore = weightedScore / totalWeight;
        
        // Compare with best score
        if (compositeScore > bestCompositeScore) {
            bestCompositeScore = compositeScore;
            epochsSinceImprovement = 0;
            return 0.0; // No adjustment needed
        } else {
            epochsSinceImprovement++;
            return epochsSinceImprovement > 3 ? 0.1 : 0.05;
        }
    }
    
    private double calculateParetoOptimalAdjustment() {
        // Simplified Pareto optimization
        // Check if current solution dominates previous
        boolean dominates = true;
        boolean isDominated = true;
        
        for (var cvi : cvis) {
            var history = scoreHistory.get(cvi.getName());
            if (history == null || history.size() < 2) continue;
            
            double current = history.get(history.size() - 1);
            double previous = history.get(history.size() - 2);
            
            if (cvi.isHigherBetter()) {
                if (current <= previous) dominates = false;
                if (current >= previous) isDominated = false;
            } else {
                if (current >= previous) dominates = false;
                if (current <= previous) isDominated = false;
            }
        }
        
        if (dominates) return 0.0; // Current solution is better
        if (isDominated) return 0.1; // Need significant change
        return 0.05; // Mixed results, small adjustment
    }
    
    private double calculateThresholdBasedAdjustment() {
        boolean allThresholdsMet = true;
        
        for (var entry : cviThresholds.entrySet()) {
            String cviName = entry.getKey();
            double threshold = entry.getValue();
            
            // Check if we have a score for this CVI
            if (!currentScores.containsKey(cviName)) {
                // CVI hasn't been calculated yet, can't meet threshold
                allThresholdsMet = false;
                break;
            }
            
            double currentScore = currentScores.get(cviName);
            
            var cvi = cvis.stream()
                .filter(c -> c.getName().equals(cviName))
                .findFirst()
                .orElse(null);
            
            if (cvi != null) {
                boolean thresholdMet = cvi.isHigherBetter() ? 
                    currentScore >= threshold : currentScore <= threshold;
                
                if (!thresholdMet) {
                    allThresholdsMet = false;
                    break;
                }
            }
        }
        
        return allThresholdsMet ? 0.0 : 0.05;
    }
    
    private void selectAdaptiveStrategy() {
        // Simple adaptive strategy selection based on CVI count and performance
        if (cvis.size() == 1) {
            currentStrategy = OptimizationStrategy.SINGLE_CVI;
        } else if (!cviThresholds.isEmpty()) {
            currentStrategy = OptimizationStrategy.THRESHOLD_BASED;
        } else if (!cviWeights.isEmpty()) {
            currentStrategy = OptimizationStrategy.WEIGHTED_AVERAGE;
        } else {
            currentStrategy = OptimizationStrategy.PARETO_OPTIMAL;
        }
    }
    
    /**
     * Learning result for CVIART
     */
    public record LearningResult(boolean wasSuccessful, int categoryIndex) {}
    
    /**
     * Optimization strategies for CVI-based learning
     */
    public enum OptimizationStrategy {
        SINGLE_CVI,        // Optimize single primary CVI
        WEIGHTED_AVERAGE,  // Weighted average of multiple CVIs
        PARETO_OPTIMAL,    // Pareto-optimal multi-objective
        THRESHOLD_BASED,   // Meet specific thresholds
        ADAPTIVE           // Automatically select strategy
    }
    
    /**
     * Parameters for CVIART
     */
    public static class CVIARTParameters {
        private double initialVigilance = 0.5;
        private int targetClusters = 0;
        private boolean adaptiveVigilance = false;
        private double vigilanceAdaptationRate = 0.01;
        private OptimizationStrategy strategy = OptimizationStrategy.SINGLE_CVI;
        private String primaryCVI = "Calinski-Harabasz Index";
        private Map<String, Double> cviWeights = new HashMap<>();
        private Map<String, Double> cviThresholds = new HashMap<>();
        
        // Getters and setters
        public double getInitialVigilance() { return initialVigilance; }
        public void setInitialVigilance(double v) { this.initialVigilance = v; }
        
        public int getTargetClusters() { return targetClusters; }
        public void setTargetClusters(int t) { this.targetClusters = t; }
        
        public boolean isAdaptiveVigilance() { return adaptiveVigilance; }
        public void setAdaptiveVigilance(boolean a) { this.adaptiveVigilance = a; }
        
        public double getVigilanceAdaptationRate() { return vigilanceAdaptationRate; }
        public void setVigilanceAdaptationRate(double r) { this.vigilanceAdaptationRate = r; }
        
        public OptimizationStrategy getCVIOptimizationStrategy() { return strategy; }
        public void setCVIOptimizationStrategy(OptimizationStrategy s) { this.strategy = s; }
        
        public String getPrimaryCVI() { return primaryCVI; }
        public void setPrimaryCVI(String cvi) { this.primaryCVI = cvi; }
        
        public Map<String, Double> getCVIWeights() { return cviWeights; }
        public void setCVIWeights(Map<String, Double> w) { this.cviWeights = new HashMap<>(w); }
        
        public Map<String, Double> getCVIThresholds() { return cviThresholds; }
        public void setCVIThresholds(Map<String, Double> t) { this.cviThresholds = new HashMap<>(t); }
    }

    @Override
    public void close() throws Exception {
        // No-op for vanilla implementation
    }
}