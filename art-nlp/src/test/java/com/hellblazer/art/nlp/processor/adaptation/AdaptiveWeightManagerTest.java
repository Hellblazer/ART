package com.hellblazer.art.nlp.processor.adaptation;

import com.hellblazer.art.nlp.processor.ChannelResult;
import com.hellblazer.art.nlp.processor.ConsensusResult;
import com.hellblazer.art.nlp.processor.adaptation.AdaptiveWeightManager.AdaptationConfig;
import com.hellblazer.art.nlp.processor.adaptation.AdaptiveWeightManager.AdaptationStrategy;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for AdaptiveWeightManager.
 */
public class AdaptiveWeightManagerTest {
    
    private AdaptiveWeightManager manager;
    private AdaptationConfig defaultConfig;
    
    @BeforeEach
    void setUp() {
        defaultConfig = AdaptationConfig.defaultConfig();
        manager = new AdaptiveWeightManager(defaultConfig);
    }
    
    @Test
    void testBasicLearning() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100),
            "entity", ChannelResult.success("entity", 1, 0.8, 120),
            "syntactic", ChannelResult.success("syntactic", 2, 0.6, 80)
        );
        
        var consensusResult = ConsensusResult.create(1, 0.85, "TestConsensus", 
            Map.of("semantic", 0.4, "entity", 0.35, "syntactic", 0.25));
        
        manager.learn(channelResults, consensusResult, "text-classification");
        
        var stats = manager.getAdaptationStats();
        assertThat(stats.totalAdaptations()).isEqualTo(1);
        assertThat(stats.channelStats()).hasSize(3);
        assertThat(stats.knownContexts()).contains("text-classification");
        
        // Verify channel stats
        var semanticStats = stats.channelStats().get("semantic");
        assertThat(semanticStats.channelId()).isEqualTo("semantic");
        assertThat(semanticStats.totalObservations()).isEqualTo(1);
        assertThat(semanticStats.successRate()).isEqualTo(1.0);
        assertThat(semanticStats.currentWeight()).isGreaterThan(0.0);
    }
    
    @Test
    void testAdaptiveWeightGeneration() {
        // First, train the manager with some data
        var trainResults1 = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.95, 100),
            "entity", ChannelResult.success("entity", 1, 0.7, 120)
        );
        var consensus1 = ConsensusResult.create(1, 0.9, "Test", Map.of("semantic", 0.7, "entity", 0.3));
        
        var trainResults2 = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.8, 90),
            "entity", ChannelResult.failed("entity", "Error", 150)
        );
        var consensus2 = ConsensusResult.create(1, 0.8, "Test", Map.of("semantic", 1.0, "entity", 0.0));
        
        manager.learn(trainResults1, consensus1, "context1");
        manager.learn(trainResults2, consensus2, "context1");
        
        // Now get adaptive weights
        var weights = manager.getAdaptiveWeights(Set.of("semantic", "entity"), "context1");
        
        assertThat(weights).hasSize(2);
        assertThat(weights.keySet()).containsExactlyInAnyOrder("semantic", "entity");
        
        // Semantic should have higher weight due to better performance
        assertThat(weights.get("semantic")).isGreaterThan(weights.get("entity"));
        
        // Weights should sum to 1.0 (normalized)
        var totalWeight = weights.values().stream().mapToDouble(Double::doubleValue).sum();
        assertThat(totalWeight).isCloseTo(1.0, within(1e-10));
    }
    
    @Test
    void testContextualLearning() {
        var results = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100),
            "entity", ChannelResult.success("entity", 2, 0.8, 120)
        );
        
        var consensus1 = ConsensusResult.create(1, 0.85, "Test", Map.of("semantic", 0.6, "entity", 0.4));
        var consensus2 = ConsensusResult.create(2, 0.75, "Test", Map.of("semantic", 0.3, "entity", 0.7));
        
        // Learn in different contexts
        manager.learn(results, consensus1, "news");
        manager.learn(results, consensus2, "academic");
        
        var newsWeights = manager.getAdaptiveWeights(Set.of("semantic", "entity"), "news");
        var academicWeights = manager.getAdaptiveWeights(Set.of("semantic", "entity"), "academic");
        
        // Weights should be different for different contexts
        assertThat(newsWeights.get("semantic")).isNotEqualTo(academicWeights.get("semantic"));
        assertThat(newsWeights.get("entity")).isNotEqualTo(academicWeights.get("entity"));
        
        var stats = manager.getAdaptationStats();
        assertThat(stats.knownContexts()).containsExactlyInAnyOrder("news", "academic");
    }
    
    @Test
    void testChannelCorrelations() {
        // Create scenarios where channels agree (high correlation)
        var agreeingResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100),
            "entity", ChannelResult.success("entity", 1, 0.85, 120),  // Same category, similar confidence
            "syntactic", ChannelResult.success("syntactic", 2, 0.4, 80)  // Different category, low confidence
        );
        
        var consensus1 = ConsensusResult.create(1, 0.8, "Test", 
            Map.of("semantic", 0.5, "entity", 0.4, "syntactic", 0.1));
        
        // Learn multiple times to establish correlation patterns
        for (int i = 0; i < 5; i++) {
            manager.learn(agreeingResults, consensus1, "test");
        }
        
        var correlations = manager.getChannelCorrelations();
        
        assertThat(correlations).containsKey("semantic");
        assertThat(correlations).containsKey("entity");
        
        // Semantic and entity should have higher correlation (same category, similar confidence)
        var semanticCorrelations = correlations.get("semantic");
        if (semanticCorrelations != null && semanticCorrelations.containsKey("entity")) {
            var semanticEntityCorr = semanticCorrelations.get("entity");
            assertThat(semanticEntityCorr).isGreaterThan(0.3); // Should show positive correlation
        }
    }
    
    @Test
    void testWeightBounds() {
        // Create a scenario that might produce extreme weights
        var extremeResults = Map.of(
            "good", ChannelResult.success("good", 1, 1.0, 50),
            "bad", ChannelResult.failed("bad", "Always fails", 200)
        );
        
        var consensus = ConsensusResult.create(1, 1.0, "Test", Map.of("good", 1.0, "bad", 0.0));
        
        // Learn multiple times to amplify differences
        for (int i = 0; i < 20; i++) {
            manager.learn(extremeResults, consensus, "extreme");
        }
        
        var weights = manager.getAdaptiveWeights(Set.of("good", "bad"), "extreme");
        
        // Even with extreme performance differences, weights should be bounded
        assertThat(weights.get("good")).isLessThanOrEqualTo(1.0);
        assertThat(weights.get("bad")).isGreaterThanOrEqualTo(0.01); // Minimum threshold
        
        // Total should still be normalized
        var total = weights.values().stream().mapToDouble(Double::doubleValue).sum();
        assertThat(total).isCloseTo(1.0, within(1e-10));
    }
    
    @Test
    void testAdaptationWithNoContext() {
        var results = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.8, 100)
        );
        var consensus = ConsensusResult.create(1, 0.8, "Test", Map.of("semantic", 1.0));
        
        // Learn without specifying context (should use default)
        manager.learn(results, consensus, null);
        
        var weights = manager.getAdaptiveWeights(Set.of("semantic"), null);
        
        assertThat(weights).hasSize(1);
        assertThat(weights.get("semantic")).isEqualTo(1.0);
        
        var stats = manager.getAdaptationStats();
        assertThat(stats.knownContexts()).contains("default");
    }
    
    @Test
    void testProgressiveImprovement() {
        // Simulate a channel that improves over time
        var channelId = "improving";
        
        for (int i = 0; i < 10; i++) {
            var confidence = 0.5 + (i * 0.05); // Gradually improving confidence
            var results = Map.of(
                channelId, ChannelResult.success(channelId, 1, confidence, 100)
            );
            var consensus = ConsensusResult.create(1, confidence, "Test", Map.of(channelId, 1.0));
            
            manager.learn(results, consensus, "improvement");
        }
        
        var stats = manager.getAdaptationStats();
        var channelStats = stats.channelStats().get(channelId);
        
        assertThat(channelStats.totalObservations()).isEqualTo(10);
        assertThat(channelStats.successRate()).isEqualTo(1.0);
        assertThat(channelStats.averageConfidence()).isGreaterThan(0.7); // Should reflect improvement
        assertThat(channelStats.weightTrend()).isIn("INCREASING", "STABLE"); // Depends on exact values
    }
    
    @Test
    void testResetLearning() {
        var results = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.8, 100)
        );
        var consensus = ConsensusResult.create(1, 0.8, "Test", Map.of("semantic", 1.0));
        
        manager.learn(results, consensus, "test");
        
        var statsBefore = manager.getAdaptationStats();
        assertThat(statsBefore.totalAdaptations()).isEqualTo(1);
        assertThat(statsBefore.channelStats()).isNotEmpty();
        
        manager.resetLearning();
        
        var statsAfter = manager.getAdaptationStats();
        assertThat(statsAfter.totalAdaptations()).isEqualTo(0);
        assertThat(statsAfter.channelStats()).isEmpty();
        assertThat(statsAfter.knownContexts()).isEmpty();
    }
    
    @Test
    void testWeightExportImport() {
        // Train some data
        var results = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100),
            "entity", ChannelResult.success("entity", 2, 0.8, 120)
        );
        var consensus = ConsensusResult.create(1, 0.85, "Test", 
            Map.of("semantic", 0.6, "entity", 0.4));
        
        manager.learn(results, consensus, "export-test");
        
        var exported = manager.exportWeights();
        
        assertThat(exported).containsKey("channelHistories");
        assertThat(exported).containsKey("correlations");
        assertThat(exported).containsKey("contextualWeights");
        assertThat(exported).containsKey("totalAdaptations");
        assertThat(exported).containsKey("exportTime");
        
        assertThat(exported.get("totalAdaptations")).isEqualTo(1L);
    }
    
    @Test
    void testConservativeConfiguration() {
        var conservativeConfig = AdaptationConfig.conservativeConfig();
        var conservativeManager = new AdaptiveWeightManager(conservativeConfig);
        
        assertThat(conservativeConfig.learningRate()).isEqualTo(0.05);
        assertThat(conservativeConfig.contextualInfluence()).isEqualTo(0.1);
        assertThat(conservativeConfig.correlationInfluence()).isEqualTo(0.05);
        assertThat(conservativeConfig.enableCorrelationLearning()).isFalse();
        assertThat(conservativeConfig.enableContextualLearning()).isTrue();
        
        var results = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100)
        );
        var consensus = ConsensusResult.create(1, 0.9, "Test", Map.of("semantic", 1.0));
        
        conservativeManager.learn(results, consensus, "test");
        
        var correlations = conservativeManager.getChannelCorrelations();
        assertThat(correlations).isEmpty(); // Correlation learning disabled
    }
    
    @Test
    void testAggressiveConfiguration() {
        var aggressiveConfig = AdaptationConfig.aggressiveConfig();
        
        assertThat(aggressiveConfig.learningRate()).isEqualTo(0.3);
        assertThat(aggressiveConfig.contextualInfluence()).isEqualTo(0.4);
        assertThat(aggressiveConfig.correlationInfluence()).isEqualTo(0.3);
        assertThat(aggressiveConfig.enableCorrelationLearning()).isTrue();
        assertThat(aggressiveConfig.enableContextualLearning()).isTrue();
    }
    
    @Test
    void testConfigurationValidation() {
        // Test invalid learning rate
        assertThatThrownBy(() -> new AdaptationConfig(
            0.0, 1.0, 0.1, 0.1, true, true, AdaptationStrategy.EXPONENTIAL_MOVING_AVERAGE))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Learning rate must be in (0.0, 1.0]");
            
        assertThatThrownBy(() -> new AdaptationConfig(
            1.5, 1.0, 0.1, 0.1, true, true, AdaptationStrategy.EXPONENTIAL_MOVING_AVERAGE))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Learning rate must be in (0.0, 1.0]");
            
        // Test invalid default channel weight
        assertThatThrownBy(() -> new AdaptationConfig(
            0.1, 0.0, 0.1, 0.1, true, true, AdaptationStrategy.EXPONENTIAL_MOVING_AVERAGE))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Default channel weight must be positive");
            
        assertThatThrownBy(() -> new AdaptationConfig(
            0.1, -1.0, 0.1, 0.1, true, true, AdaptationStrategy.EXPONENTIAL_MOVING_AVERAGE))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Default channel weight must be positive");
            
        // Test null strategy
        assertThatThrownBy(() -> new AdaptationConfig(
            0.1, 1.0, 0.1, 0.1, true, true, null))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("adaptationStrategy cannot be null");
    }
    
    @Test
    void testNullInputValidation() {
        assertThatThrownBy(() -> new AdaptiveWeightManager(null))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("config cannot be null");
            
        var results = Map.of("test", ChannelResult.success("test", 1, 0.8, 100));
        var consensus = ConsensusResult.create(1, 0.8, "Test", Map.of("test", 1.0));
        
        assertThatThrownBy(() -> manager.learn(null, consensus, "test"))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("channelResults cannot be null");
            
        assertThatThrownBy(() -> manager.learn(results, null, "test"))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("consensusResult cannot be null");
            
        assertThatThrownBy(() -> manager.getAdaptiveWeights(null, "test"))
            .isInstanceOf(NullPointerException.class)
            .hasMessageContaining("channelIds cannot be null");
    }
    
    @Test
    void testChannelAdaptationStatsRecord() {
        var stats = new AdaptiveWeightManager.ChannelAdaptationStats(
            "test", 10, 0.8, 0.75, 1.2, "INCREASING"
        );
        
        assertThat(stats.channelId()).isEqualTo("test");
        assertThat(stats.totalObservations()).isEqualTo(10);
        assertThat(stats.successRate()).isEqualTo(0.8);
        assertThat(stats.averageConfidence()).isEqualTo(0.75);
        assertThat(stats.currentWeight()).isEqualTo(1.2);
        assertThat(stats.weightTrend()).isEqualTo("INCREASING");
    }
    
    @Test
    void testAdaptationStatsRecord() {
        var channelStats = Map.of(
            "test", new AdaptiveWeightManager.ChannelAdaptationStats("test", 5, 1.0, 0.9, 1.1, "STABLE")
        );
        var correlations = Map.of("ch1", Map.of("ch2", 0.8));
        var contexts = Set.of("ctx1", "ctx2");
        
        var stats = new AdaptiveWeightManager.AdaptationStats(100, channelStats, correlations, contexts);
        
        assertThat(stats.totalAdaptations()).isEqualTo(100);
        assertThat(stats.channelStats()).hasSize(1);
        assertThat(stats.channelCorrelations()).containsKey("ch1");
        assertThat(stats.knownContexts()).hasSize(2);
    }
    
    @Test
    void testUnknownChannelWeights() {
        // Request weights for channels that haven't been seen before
        var weights = manager.getAdaptiveWeights(Set.of("unknown1", "unknown2"), "new-context");
        
        assertThat(weights).hasSize(2);
        
        // Should use default weights and normalize
        assertThat(weights.get("unknown1")).isEqualTo(0.5);
        assertThat(weights.get("unknown2")).isEqualTo(0.5);
        
        var total = weights.values().stream().mapToDouble(Double::doubleValue).sum();
        assertThat(total).isCloseTo(1.0, within(1e-10));
    }
}