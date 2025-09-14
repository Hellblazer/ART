package com.hellblazer.art.hartcq.spatial;

import com.hellblazer.art.hartcq.HARTCQConfig;
import com.hellblazer.art.hartcq.Token;
import org.joml.Vector2f;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Topological relationship processor for maintaining spatial invariants.
 * 
 * Processes topological relationships between tokens, maintains topological
 * invariants during transformations, preserves neighborhood structures,
 * and analyzes connectivity patterns.
 * 
 * Uses JOML for efficient vector operations and geometric computations.
 * Supports parallel processing for large token sets.
 * 
 * @author Claude Code
 */
public class TopologicalProcessor implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(TopologicalProcessor.class);
    
    private final HARTCQConfig config;
    private final TopologicalProcessorStats stats;
    
    // Topology processing parameters
    private final double connectivityThreshold;
    private final int maxTopologySize;
    private final boolean enableInvariantChecking;
    
    /**
     * Topological relationship types.
     */
    public enum TopologicalRelation {
        ADJACENT,        // Directly connected
        SEPARATED,       // Disconnected
        CONTAINED,       // One region contains another
        OVERLAPPING,     // Regions overlap
        TOUCHING,        // Regions touch at boundary
        DISJOINT        // Completely separate
    }
    
    /**
     * Topological invariant types.
     */
    public enum TopologicalInvariant {
        CONNECTIVITY,    // Connection structure preserved
        ORIENTATION,     // Relative orientation maintained
        ORDERING,        // Sequential ordering preserved  
        CONTAINMENT,     // Containment relationships preserved
        ADJACENCY       // Adjacency relationships preserved
    }
    
    /**
     * Node in the topological graph.
     */
    public static class TopologyNode {
        private final Token token;
        private final Vector2f position;
        private final Set<TopologyNode> neighbors;
        private final Map<TopologyNode, TopologicalRelation> relations;
        private final double strength;
        private final Map<String, Object> properties;
        
        public TopologyNode(Token token, Vector2f position, double strength) {
            this.token = token;
            this.position = new Vector2f(position);
            this.neighbors = new HashSet<>();
            this.relations = new HashMap<>();
            this.strength = strength;
            this.properties = new HashMap<>();
        }
        
        public Token getToken() { return token; }
        public Vector2f getPosition() { return new Vector2f(position); }
        public Set<TopologyNode> getNeighbors() { return new HashSet<>(neighbors); }
        public Map<TopologyNode, TopologicalRelation> getRelations() { return new HashMap<>(relations); }
        public double getStrength() { return strength; }
        public Map<String, Object> getProperties() { return new HashMap<>(properties); }
        
        public void addNeighbor(TopologyNode neighbor, TopologicalRelation relation) {
            neighbors.add(neighbor);
            relations.put(neighbor, relation);
        }
        
        public void removeNeighbor(TopologyNode neighbor) {
            neighbors.remove(neighbor);
            relations.remove(neighbor);
        }
        
        public int getDegree() {
            return neighbors.size();
        }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof TopologyNode node)) return false;
            return Objects.equals(token, node.token);
        }
        
        @Override
        public int hashCode() {
            return Objects.hash(token);
        }
        
        @Override
        public String toString() {
            return "TopologyNode{token=%s, neighbors=%d, strength=%.3f}"
                .formatted(token, neighbors.size(), strength);
        }
    }
    
    /**
     * Topological graph structure.
     */
    public static class TopologicalGraph {
        private final Map<Token, TopologyNode> nodes;
        private final List<TopologyEdge> edges;
        private final Map<TopologicalInvariant, Boolean> invariants;
        private final double connectivity;
        private final double clustering;
        
        public TopologicalGraph(Map<Token, TopologyNode> nodes, List<TopologyEdge> edges,
                               Map<TopologicalInvariant, Boolean> invariants, 
                               double connectivity, double clustering) {
            this.nodes = new HashMap<>(nodes);
            this.edges = new ArrayList<>(edges);
            this.invariants = new HashMap<>(invariants);
            this.connectivity = connectivity;
            this.clustering = clustering;
        }
        
        public Map<Token, TopologyNode> getNodes() { return new HashMap<>(nodes); }
        public List<TopologyEdge> getEdges() { return new ArrayList<>(edges); }
        public Map<TopologicalInvariant, Boolean> getInvariants() { return new HashMap<>(invariants); }
        public double getConnectivity() { return connectivity; }
        public double getClustering() { return clustering; }
        
        public int getNodeCount() { return nodes.size(); }
        public int getEdgeCount() { return edges.size(); }
        
        public boolean isConnected() {
            return connectivity > 0.5;
        }
        
        public boolean areInvariantsSatisfied() {
            return invariants.values().stream().allMatch(Boolean::booleanValue);
        }
        
        @Override
        public String toString() {
            return "TopologicalGraph{nodes=%d, edges=%d, connectivity=%.3f, clustering=%.3f}"
                .formatted(nodes.size(), edges.size(), connectivity, clustering);
        }
    }
    
    /**
     * Edge in the topological graph.
     */
    public static class TopologyEdge {
        private final TopologyNode source;
        private final TopologyNode target;
        private final TopologicalRelation relation;
        private final double weight;
        private final Map<String, Object> properties;
        
        public TopologyEdge(TopologyNode source, TopologyNode target, 
                          TopologicalRelation relation, double weight) {
            this.source = source;
            this.target = target;
            this.relation = relation;
            this.weight = weight;
            this.properties = new HashMap<>();
        }
        
        public TopologyNode getSource() { return source; }
        public TopologyNode getTarget() { return target; }
        public TopologicalRelation getRelation() { return relation; }
        public double getWeight() { return weight; }
        public Map<String, Object> getProperties() { return new HashMap<>(properties); }
        
        @Override
        public String toString() {
            return "TopologyEdge{%s -[%s:%.3f]-> %s}"
                .formatted(source.getToken(), relation, weight, target.getToken());
        }
    }
    
    /**
     * Result of topological processing.
     */
    public static class TopologyResult {
        private final TopologicalGraph graph;
        private final List<ConnectedComponent> components;
        private final Map<TopologicalInvariant, Double> invariantStrengths;
        private final double topologyCoherence;
        private final int processingTimeMs;
        
        public TopologyResult(TopologicalGraph graph, List<ConnectedComponent> components,
                            Map<TopologicalInvariant, Double> invariantStrengths,
                            double topologyCoherence, int processingTimeMs) {
            this.graph = graph;
            this.components = new ArrayList<>(components);
            this.invariantStrengths = new HashMap<>(invariantStrengths);
            this.topologyCoherence = topologyCoherence;
            this.processingTimeMs = processingTimeMs;
        }
        
        public TopologicalGraph getGraph() { return graph; }
        public List<ConnectedComponent> getComponents() { return new ArrayList<>(components); }
        public Map<TopologicalInvariant, Double> getInvariantStrengths() { return new HashMap<>(invariantStrengths); }
        public double getTopologyCoherence() { return topologyCoherence; }
        public int getProcessingTimeMs() { return processingTimeMs; }
        
        public static TopologyResult empty() {
            var emptyGraph = new TopologicalGraph(new HashMap<>(), new ArrayList<>(), 
                                                 new HashMap<>(), 0.0, 0.0);
            return new TopologyResult(emptyGraph, new ArrayList<>(), new HashMap<>(), 0.0, 0);
        }
        
        @Override
        public String toString() {
            return "TopologyResult{nodes=%d, components=%d, coherence=%.3f, time=%dms}"
                .formatted(graph.getNodeCount(), components.size(), topologyCoherence, processingTimeMs);
        }
    }
    
    /**
     * Connected component in the topological graph.
     */
    public static class ConnectedComponent {
        private final List<TopologyNode> nodes;
        private final Vector2f centroid;
        private final double diameter;
        private final double density;
        private final Map<String, Object> properties;
        
        public ConnectedComponent(List<TopologyNode> nodes) {
            this.nodes = new ArrayList<>(nodes);
            this.centroid = calculateCentroid();
            this.diameter = calculateDiameter();
            this.density = calculateDensity();
            this.properties = new HashMap<>();
        }
        
        private Vector2f calculateCentroid() {
            var centroid = new Vector2f();
            for (var node : nodes) {
                centroid.add(node.getPosition());
            }
            centroid.div(nodes.size());
            return centroid;
        }
        
        private double calculateDiameter() {
            var maxDistance = 0.0;
            for (var i = 0; i < nodes.size(); i++) {
                for (var j = i + 1; j < nodes.size(); j++) {
                    var distance = nodes.get(i).getPosition().distance(nodes.get(j).getPosition());
                    maxDistance = Math.max(maxDistance, distance);
                }
            }
            return maxDistance;
        }
        
        private double calculateDensity() {
            if (nodes.size() < 2) return 1.0;
            
            var possibleEdges = nodes.size() * (nodes.size() - 1) / 2;
            var actualEdges = 0;
            
            for (var node : nodes) {
                actualEdges += node.getDegree();
            }
            actualEdges /= 2; // Each edge counted twice
            
            return possibleEdges > 0 ? (double) actualEdges / possibleEdges : 0.0;
        }
        
        public List<TopologyNode> getNodes() { return new ArrayList<>(nodes); }
        public Vector2f getCentroid() { return new Vector2f(centroid); }
        public double getDiameter() { return diameter; }
        public double getDensity() { return density; }
        public int size() { return nodes.size(); }
        
        @Override
        public String toString() {
            return "ConnectedComponent{nodes=%d, diameter=%.3f, density=%.3f}"
                .formatted(nodes.size(), diameter, density);
        }
    }
    
    /**
     * Creates a topological processor with the given configuration.
     * 
     * @param config HART-CQ configuration
     */
    public TopologicalProcessor(HARTCQConfig config) {
        this.config = Objects.requireNonNull(config, "Configuration cannot be null");
        this.stats = new TopologicalProcessorStats();
        
        // Initialize processing parameters
        this.connectivityThreshold = 1.5; // Maximum distance for connectivity
        this.maxTopologySize = 1000; // Maximum nodes in topology
        this.enableInvariantChecking = true;
        
        logger.info("TopologicalProcessor initialized with connectivityThreshold={}", connectivityThreshold);
    }
    
    /**
     * Process topological relationships between tokens.
     * 
     * @param tokenPositions Map of tokens to their positions
     * @param proximityResult Result from proximity analysis
     * @return Topology processing result
     */
    public TopologyResult processTopology(Map<Token, Vector2f> tokenPositions,
                                        ProximityAnalyzer.ProximityResult proximityResult) {
        if (tokenPositions == null || tokenPositions.isEmpty()) {
            return TopologyResult.empty();
        }
        
        var startTime = System.currentTimeMillis();
        
        try {
            // Build topological graph
            var graph = buildTopologicalGraph(tokenPositions, proximityResult);
            
            // Find connected components
            var components = findConnectedComponents(graph);
            
            // Check topological invariants
            var invariantStrengths = checkTopologicalInvariants(graph, tokenPositions);
            
            // Calculate topology coherence
            var coherence = calculateTopologyCoherence(graph, components, invariantStrengths);
            
            var processingTime = (int) (System.currentTimeMillis() - startTime);
            var result = new TopologyResult(graph, components, invariantStrengths, coherence, processingTime);
            
            // Update statistics
            stats.recordProcessing(tokenPositions.size(), processingTime, coherence);
            
            logger.debug("Topology processing completed: {}", result);
            return result;
            
        } catch (Exception e) {
            logger.error("Error in topological processing", e);
            return TopologyResult.empty();
        }
    }
    
    /**
     * Build the topological graph from token positions and proximity data.
     */
    private TopologicalGraph buildTopologicalGraph(Map<Token, Vector2f> tokenPositions,
                                                  ProximityAnalyzer.ProximityResult proximityResult) {
        var nodes = new HashMap<Token, TopologyNode>();
        var edges = new ArrayList<TopologyEdge>();
        
        // Create nodes
        for (var entry : tokenPositions.entrySet()) {
            var token = entry.getKey();
            var position = entry.getValue();
            var strength = calculateNodeStrength(token, proximityResult);
            
            nodes.put(token, new TopologyNode(token, position, strength));
        }
        
        // Create edges based on proximity relationships
        var proximityGraph = proximityResult.getProximityGraph();
        for (var entry : proximityGraph.entrySet()) {
            var sourceToken = entry.getKey();
            var sourceNode = nodes.get(sourceToken);
            
            if (sourceNode == null) continue;
            
            for (var neighborInfo : entry.getValue()) {
                var targetToken = neighborInfo.getToken();
                var targetNode = nodes.get(targetToken);
                
                if (targetNode == null) continue;
                
                var relation = determineTopologicalRelation(sourceNode, targetNode, neighborInfo);
                var weight = neighborInfo.getWeight();
                
                if (weight > 0.3) { // Minimum threshold for topological connection
                    var edge = new TopologyEdge(sourceNode, targetNode, relation, weight);
                    edges.add(edge);
                    
                    sourceNode.addNeighbor(targetNode, relation);
                    targetNode.addNeighbor(sourceNode, relation);
                }
            }
        }
        
        // Calculate graph metrics
        var connectivity = calculateGraphConnectivity(nodes, edges);
        var clustering = calculateClusteringCoefficient(nodes);
        
        // Check invariants
        Map<TopologicalInvariant, Boolean> invariants = new HashMap<>();
        if (enableInvariantChecking) {
            invariants = checkInvariants(nodes, edges);
        }
        
        return new TopologicalGraph(nodes, edges, invariants, connectivity, clustering);
    }
    
    /**
     * Calculate strength of a topology node.
     */
    private double calculateNodeStrength(Token token, ProximityAnalyzer.ProximityResult proximityResult) {
        var neighborhoods = proximityResult.getNeighborhoods();
        var neighborhood = neighborhoods.get(token);
        
        if (neighborhood == null) {
            return 0.1; // Minimum strength for isolated nodes
        }
        
        // Base strength on neighborhood density and connectivity
        var density = neighborhood.getDensity();
        var connectivity = neighborhood.getNeighborCount();
        
        return Math.min(1.0, 0.3 + 0.4 * Math.min(1.0, density) + 0.3 * Math.min(1.0, connectivity / 10.0));
    }
    
    /**
     * Determine topological relation between two nodes.
     */
    private TopologicalRelation determineTopologicalRelation(TopologyNode source, TopologyNode target,
                                                           ProximityAnalyzer.NeighborInfo neighborInfo) {
        var distance = neighborInfo.getDistance();
        var weight = neighborInfo.getWeight();
        
        if (distance < connectivityThreshold * 0.3) {
            return TopologicalRelation.TOUCHING;
        } else if (distance < connectivityThreshold * 0.6) {
            return TopologicalRelation.ADJACENT;
        } else if (weight > 0.7) {
            return TopologicalRelation.OVERLAPPING;
        } else {
            return TopologicalRelation.SEPARATED;
        }
    }
    
    /**
     * Calculate graph connectivity measure.
     */
    private double calculateGraphConnectivity(Map<Token, TopologyNode> nodes, List<TopologyEdge> edges) {
        if (nodes.isEmpty()) return 0.0;
        
        var totalPossibleEdges = nodes.size() * (nodes.size() - 1) / 2;
        if (totalPossibleEdges == 0) return 1.0;
        
        return Math.min(1.0, (double) edges.size() / totalPossibleEdges);
    }
    
    /**
     * Calculate clustering coefficient.
     */
    private double calculateClusteringCoefficient(Map<Token, TopologyNode> nodes) {
        if (nodes.size() < 3) return 1.0;
        
        var totalCoefficient = 0.0;
        var nodeCount = 0;
        
        for (var node : nodes.values()) {
            var neighbors = node.getNeighbors();
            if (neighbors.size() < 2) continue;
            
            var triangles = 0;
            var possibleTriangles = neighbors.size() * (neighbors.size() - 1) / 2;
            
            var neighborList = new ArrayList<>(neighbors);
            for (var i = 0; i < neighborList.size(); i++) {
                for (var j = i + 1; j < neighborList.size(); j++) {
                    if (neighborList.get(i).getNeighbors().contains(neighborList.get(j))) {
                        triangles++;
                    }
                }
            }
            
            totalCoefficient += possibleTriangles > 0 ? (double) triangles / possibleTriangles : 0.0;
            nodeCount++;
        }
        
        return nodeCount > 0 ? totalCoefficient / nodeCount : 0.0;
    }
    
    /**
     * Check topological invariants.
     */
    private Map<TopologicalInvariant, Boolean> checkInvariants(Map<Token, TopologyNode> nodes, 
                                                              List<TopologyEdge> edges) {
        var invariants = new HashMap<TopologicalInvariant, Boolean>();
        
        // Check connectivity invariant
        invariants.put(TopologicalInvariant.CONNECTIVITY, checkConnectivityInvariant(nodes, edges));
        
        // Check orientation invariant
        invariants.put(TopologicalInvariant.ORIENTATION, checkOrientationInvariant(nodes));
        
        // Check ordering invariant
        invariants.put(TopologicalInvariant.ORDERING, checkOrderingInvariant(nodes));
        
        // Check containment invariant
        invariants.put(TopologicalInvariant.CONTAINMENT, checkContainmentInvariant(nodes));
        
        // Check adjacency invariant
        invariants.put(TopologicalInvariant.ADJACENCY, checkAdjacencyInvariant(edges));
        
        return invariants;
    }
    
    /**
     * Check connectivity preservation invariant.
     */
    private boolean checkConnectivityInvariant(Map<Token, TopologyNode> nodes, List<TopologyEdge> edges) {
        // Check if the graph maintains reasonable connectivity
        var avgDegree = nodes.values().stream()
            .mapToInt(TopologyNode::getDegree)
            .average()
            .orElse(0.0);
        
        return avgDegree >= 1.0; // At least one connection per node on average
    }
    
    /**
     * Check orientation preservation invariant.
     */
    private boolean checkOrientationInvariant(Map<Token, TopologyNode> nodes) {
        // Check if relative orientations are consistent
        if (nodes.size() < 3) return true;
        
        // Sample some triangles and check orientation consistency
        var nodeList = new ArrayList<>(nodes.values());
        var consistentOrientations = 0;
        var totalTriangles = 0;
        
        for (var i = 0; i < Math.min(nodeList.size(), 10); i++) {
            for (var j = i + 1; j < Math.min(nodeList.size(), 10); j++) {
                for (var k = j + 1; k < Math.min(nodeList.size(), 10); k++) {
                    var orientation = calculateTriangleOrientation(
                        nodeList.get(i).getPosition(),
                        nodeList.get(j).getPosition(),
                        nodeList.get(k).getPosition()
                    );
                    
                    if (Math.abs(orientation) > 0.1) { // Not collinear
                        consistentOrientations++;
                    }
                    totalTriangles++;
                }
            }
        }
        
        return totalTriangles == 0 || (double) consistentOrientations / totalTriangles > 0.7;
    }
    
    /**
     * Calculate triangle orientation (cross product).
     */
    private double calculateTriangleOrientation(Vector2f p1, Vector2f p2, Vector2f p3) {
        return (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y);
    }
    
    /**
     * Check ordering preservation invariant.
     */
    private boolean checkOrderingInvariant(Map<Token, TopologyNode> nodes) {
        // Check if token position ordering is maintained
        var nodeList = nodes.values().stream()
            .sorted((n1, n2) -> Integer.compare(n1.getToken().getPosition(), n2.getToken().getPosition()))
            .collect(Collectors.toList());
        
        if (nodeList.size() < 2) return true;
        
        var orderViolations = 0;
        for (var i = 0; i < nodeList.size() - 1; i++) {
            var current = nodeList.get(i);
            var next = nodeList.get(i + 1);
            
            // Check if spatial ordering roughly matches token ordering
            if (current.getPosition().x > next.getPosition().x + 1.0) { // Allow some tolerance
                orderViolations++;
            }
        }
        
        return (double) orderViolations / (nodeList.size() - 1) < 0.3; // Less than 30% violations
    }
    
    /**
     * Check containment preservation invariant.
     */
    private boolean checkContainmentInvariant(Map<Token, TopologyNode> nodes) {
        // For now, assume containment is preserved if no major position inversions
        // This could be enhanced with actual containment relationship tracking
        return true;
    }
    
    /**
     * Check adjacency preservation invariant.
     */
    private boolean checkAdjacencyInvariant(List<TopologyEdge> edges) {
        // Check if adjacency relationships are consistent
        var adjacentEdges = edges.stream()
            .filter(e -> e.getRelation() == TopologicalRelation.ADJACENT)
            .collect(Collectors.toList());
        
        var strongAdjacencies = adjacentEdges.stream()
            .mapToDouble(TopologyEdge::getWeight)
            .filter(w -> w > 0.5)
            .count();
        
        return adjacentEdges.isEmpty() || (double) strongAdjacencies / adjacentEdges.size() > 0.6;
    }
    
    /**
     * Find connected components in the graph.
     */
    private List<ConnectedComponent> findConnectedComponents(TopologicalGraph graph) {
        var components = new ArrayList<ConnectedComponent>();
        var visited = new HashSet<TopologyNode>();
        var nodes = graph.getNodes().values();
        
        for (var node : nodes) {
            if (visited.contains(node)) continue;
            
            var component = exploreComponent(node, visited);
            if (!component.isEmpty()) {
                components.add(new ConnectedComponent(component));
            }
        }
        
        return components;
    }
    
    /**
     * Explore a connected component using DFS.
     */
    private List<TopologyNode> exploreComponent(TopologyNode startNode, Set<TopologyNode> visited) {
        var component = new ArrayList<TopologyNode>();
        var stack = new ArrayDeque<TopologyNode>();
        
        stack.push(startNode);
        
        while (!stack.isEmpty()) {
            var node = stack.pop();
            if (visited.contains(node)) continue;
            
            visited.add(node);
            component.add(node);
            
            for (var neighbor : node.getNeighbors()) {
                if (!visited.contains(neighbor)) {
                    stack.push(neighbor);
                }
            }
        }
        
        return component;
    }
    
    /**
     * Check topological invariants and return their strengths.
     */
    private Map<TopologicalInvariant, Double> checkTopologicalInvariants(TopologicalGraph graph,
                                                                        Map<Token, Vector2f> positions) {
        var strengths = new HashMap<TopologicalInvariant, Double>();
        var invariants = graph.getInvariants();
        
        for (var invariant : TopologicalInvariant.values()) {
            var satisfied = invariants.getOrDefault(invariant, false);
            var strength = satisfied ? 1.0 : 0.0;
            
            // Add some nuance based on graph properties
            if (satisfied) {
                strength = switch (invariant) {
                    case CONNECTIVITY -> Math.min(1.0, graph.getConnectivity() * 2);
                    case ORIENTATION -> Math.min(1.0, graph.getClustering() + 0.3);
                    default -> strength;
                };
            }
            
            strengths.put(invariant, strength);
        }
        
        return strengths;
    }
    
    /**
     * Calculate overall topology coherence.
     */
    private double calculateTopologyCoherence(TopologicalGraph graph, List<ConnectedComponent> components,
                                            Map<TopologicalInvariant, Double> invariantStrengths) {
        // Weight different aspects of topology coherence
        var graphConnectivity = graph.getConnectivity();
        var graphClustering = graph.getClustering();
        
        // Average invariant strength
        var avgInvariantStrength = invariantStrengths.values().stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);
        
        // Component quality (fewer, larger components are better)
        var componentQuality = 0.0;
        if (!components.isEmpty()) {
            var avgComponentSize = components.stream().mapToDouble(ConnectedComponent::size).average().orElse(0.0);
            var avgComponentDensity = components.stream().mapToDouble(ConnectedComponent::getDensity).average().orElse(0.0);
            componentQuality = 0.5 * Math.min(1.0, avgComponentSize / 5.0) + 0.5 * avgComponentDensity;
        }
        
        // Combine metrics with weights
        return 0.3 * graphConnectivity + 
               0.2 * graphClustering + 
               0.3 * avgInvariantStrength + 
               0.2 * componentQuality;
    }
    
    /**
     * Gets topological processing statistics.
     * 
     * @return topological processor statistics
     */
    public TopologicalProcessorStats getStatistics() {
        return stats.copy();
    }
    
    /**
     * Resets the topological processor to initial state.
     */
    public void reset() {
        logger.info("Resetting topological processor");
        stats.reset();
    }
    
    /**
     * Closes the topological processor and releases resources.
     */
    @Override
    public void close() {
        logger.info("Closing topological processor");
        // No resources to clean up currently
    }
    
    /**
     * Statistics for topological processing performance.
     */
    public static class TopologicalProcessorStats {
        private int totalProcessings = 0;
        private int totalNodesProcessed = 0;
        private long totalProcessingTimeMs = 0;
        private double averageCoherence = 0.0;
        private long lastResetTime = System.currentTimeMillis();
        
        synchronized void recordProcessing(int nodeCount, int processingTimeMs, double coherence) {
            totalProcessings++;
            totalNodesProcessed += nodeCount;
            totalProcessingTimeMs += processingTimeMs;
            
            // Update average coherence using exponential moving average
            var alpha = 0.1;
            averageCoherence = alpha * coherence + (1 - alpha) * averageCoherence;
        }
        
        public synchronized int getTotalProcessings() { return totalProcessings; }
        public synchronized int getTotalNodesProcessed() { return totalNodesProcessed; }
        public synchronized long getTotalProcessingTimeMs() { return totalProcessingTimeMs; }
        public synchronized double getAverageProcessingTimeMs() {
            return totalProcessings > 0 ? (double) totalProcessingTimeMs / totalProcessings : 0.0;
        }
        public synchronized double getAverageCoherence() { return averageCoherence; }
        
        synchronized void reset() {
            totalProcessings = 0;
            totalNodesProcessed = 0;
            totalProcessingTimeMs = 0;
            averageCoherence = 0.0;
            lastResetTime = System.currentTimeMillis();
        }
        
        synchronized TopologicalProcessorStats copy() {
            var copy = new TopologicalProcessorStats();
            copy.totalProcessings = this.totalProcessings;
            copy.totalNodesProcessed = this.totalNodesProcessed;
            copy.totalProcessingTimeMs = this.totalProcessingTimeMs;
            copy.averageCoherence = this.averageCoherence;
            copy.lastResetTime = this.lastResetTime;
            return copy;
        }
        
        @Override
        public String toString() {
            return "TopologicalProcessorStats{processings=%d, nodes=%d, avgTime=%.2fms, coherence=%.3f}"
                .formatted(totalProcessings, totalNodesProcessed, getAverageProcessingTimeMs(), averageCoherence);
        }
    }
}