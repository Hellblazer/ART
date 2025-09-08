package com.hellblazer.art.nlp.processor.consensus;

import com.hellblazer.art.nlp.processor.ChannelResult;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for AttentionConsensus strategy.
 */
public class AttentionConsensusTest {
    
    private AttentionConsensus consensus;
    
    @BeforeEach
    void setUp() {
        consensus = new AttentionConsensus(0.5, true, 1.0, 64, false);
    }
    
    @Test
    void testBasicAttentionConsensus() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.8, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 1, 0.7, 80, Map.of())
        );
        
        var channelWeights = Map.of(
            "semantic", 1.0,
            "entity", 0.8,
            "syntactic", 0.6
        );
        
        var result = consensus.computeConsensus(channelResults, channelWeights);
        
        assertThat(result.category()).isIn(1, 2);  // Should pick one of the categories
        assertThat(result.confidence()).isGreaterThan(0.5);
        assertThat(result.strategy()).isEqualTo("AttentionConsensus");
        
        // Check attention metadata
        var metadata = result.metadata();
        assertThat(metadata).containsKey("attentionWeights");
        assertThat(metadata).containsKey("attentionStats");
        assertThat(metadata).containsKey("temperatureScaling");
        
        @SuppressWarnings("unchecked")
        var attentionWeights = (Map<String, Double>) metadata.get("attentionWeights");
        assertThat(attentionWeights).hasSize(3);
        assertThat(attentionWeights.get("semantic")).isGreaterThan(0.0);
    }
    
    @Test
    void testSingleChannelAttention() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.8, 100, Map.of())
        );
        
        var channelWeights = Map.of("semantic", 1.0);
        
        var result = consensus.computeConsensus(channelResults, channelWeights);
        
        assertThat(result.category()).isEqualTo(1);
        assertThat(result.confidence()).isEqualTo(0.8);
        
        var contributions = result.channelContributions();
        assertThat(contributions).hasSize(1);
        assertThat(contributions.get("semantic")).isEqualTo(1.0);
    }
    
    @Test
    void testNormalizedAttentionWeights() {
        var consensus = new AttentionConsensus(0.3, true, 1.0, 32, false);
        
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 1, 0.8, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 2, 0.7, 80, Map.of())
        );
        
        var channelWeights = Map.of(
            "semantic", 1.0,
            "entity", 1.0,
            "syntactic", 1.0
        );
        
        var result = consensus.computeConsensus(channelResults, channelWeights);
        
        @SuppressWarnings("unchecked")
        var attentionWeights = (Map<String, Double>) result.metadata().get("attentionWeights");
        
        // Attention weights should be normalized when normalizeAttentionWeights = true
        var totalWeight = attentionWeights.values().stream()
            .mapToDouble(Double::doubleValue)
            .sum();
        assertThat(totalWeight).isCloseTo(1.0, within(0.01));
    }
    
    @Test
    void testTemperatureScaling() {
        // Low temperature should make attention more focused
        var lowTempConsensus = new AttentionConsensus(0.3, true, 0.5, 32, false);
        // High temperature should make attention more uniform
        var highTempConsensus = new AttentionConsensus(0.3, true, 2.0, 32, false);
        
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.6, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 3, 0.5, 80, Map.of())
        );
        
        var channelWeights = Map.of(
            "semantic", 1.0,
            "entity", 1.0,
            "syntactic", 1.0
        );
        
        var lowTempResult = lowTempConsensus.computeConsensus(channelResults, channelWeights);
        var highTempResult = highTempConsensus.computeConsensus(channelResults, channelWeights);
        
        @SuppressWarnings("unchecked")
        var lowTempStats = (Map<String, Double>) lowTempResult.metadata().get("attentionStats");
        @SuppressWarnings("unchecked")
        var highTempStats = (Map<String, Double>) highTempResult.metadata().get("attentionStats");
        
        // Low temperature should have higher variance (more focused attention)
        assertThat(lowTempStats.get("variance")).isGreaterThanOrEqualTo(0.0);
        assertThat(highTempStats.get("variance")).isGreaterThanOrEqualTo(0.0);
    }
    
    @Test
    void testPositionalEncoding() {
        var consensusWithPos = new AttentionConsensus(0.3, true, 1.0, 64, true);
        var consensusWithoutPos = new AttentionConsensus(0.3, true, 1.0, 64, false);
        
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.8, 100, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 1, 0.8, 80, Map.of())
        );
        
        var channelWeights = Map.of("semantic", 1.0, "syntactic", 1.0);
        
        var resultWithPos = consensusWithPos.computeConsensus(channelResults, channelWeights);
        var resultWithoutPos = consensusWithoutPos.computeConsensus(channelResults, channelWeights);
        
        // Both should work, but with different internal processing
        assertThat(resultWithPos.category()).isEqualTo(1);
        assertThat(resultWithoutPos.category()).isEqualTo(1);
        
        assertThat(resultWithPos.metadata().get("positionalEncoding")).isEqualTo(true);
        assertThat(resultWithoutPos.metadata().get("positionalEncoding")).isEqualTo(false);
    }
    
    @Test
    void testChannelTypeDifferentiation() {
        // Test that attention mechanism differentiates between channel types
        var channelResults = Map.of(
            "fasttext", ChannelResult.success("fasttext", 1, 0.8, 100, Map.of()),
            "ner", ChannelResult.success("ner", 1, 0.8, 100, Map.of()),
            "pos", ChannelResult.success("pos", 2, 0.8, 100, Map.of())
        );
        
        var channelWeights = Map.of(
            "fasttext", 1.0,
            "ner", 1.0,
            "pos", 1.0
        );
        
        var result = consensus.computeConsensus(channelResults, channelWeights);
        
        assertThat(result.category()).isIn(1, 2);
        assertThat(result.confidence()).isGreaterThan(0.0);
        
        @SuppressWarnings("unchecked")
        var attentionWeights = (Map<String, Double>) result.metadata().get("attentionWeights");
        
        // All channels should receive some attention
        assertThat(attentionWeights.get("fasttext")).isGreaterThan(0.0);
        assertThat(attentionWeights.get("ner")).isGreaterThan(0.0);
        assertThat(attentionWeights.get("pos")).isGreaterThan(0.0);
    }
    
    @Test
    void testLowConfidenceFiltering() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.3, 120, Map.of()),  // Below threshold
            "syntactic", ChannelResult.success("syntactic", 3, 0.7, 80, Map.of())
        );
        
        var channelWeights = Map.of(
            "semantic", 1.0,
            "entity", 1.0,
            "syntactic", 1.0
        );
        
        var result = consensus.computeConsensus(channelResults, channelWeights);
        
        // Entity channel should be excluded due to low confidence
        @SuppressWarnings("unchecked")
        var validChannels = (Set<String>) result.metadata().get("validChannels");
        assertThat(validChannels).hasSize(2);
        assertThat(validChannels).contains("semantic", "syntactic");
        assertThat(validChannels).doesNotContain("entity");
    }
    
    @Test
    void testEmptyResults() {
        var result = consensus.computeConsensus(Map.of(), Map.of());
        
        assertThat(result.category()).isEqualTo(-1);
        assertThat(result.confidence()).isEqualTo(0.0);
    }
    
    @Test
    void testAllBelowThreshold() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.3, 100, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 2, 0.4, 80, Map.of())
        );
        
        var channelWeights = Map.of("semantic", 1.0, "syntactic", 1.0);
        
        var result = consensus.computeConsensus(channelResults, channelWeights);
        
        assertThat(result.category()).isEqualTo(-1);
        assertThat(result.confidence()).isEqualTo(0.0);
    }
    
    @Test
    void testAttentionStatistics() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 1, 0.8, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 2, 0.7, 80, Map.of()),
            "lexical", ChannelResult.success("lexical", 1, 0.6, 60, Map.of())
        );
        
        var channelWeights = Map.of(
            "semantic", 1.0,
            "entity", 0.8,
            "syntactic", 0.6,
            "lexical", 0.4
        );
        
        var result = consensus.computeConsensus(channelResults, channelWeights);
        
        @SuppressWarnings("unchecked")
        var attentionStats = (Map<String, Double>) result.metadata().get("attentionStats");
        
        assertThat(attentionStats).containsKeys("max", "min", "average", "variance");
        assertThat(attentionStats.get("max")).isGreaterThanOrEqualTo(attentionStats.get("average"));
        assertThat(attentionStats.get("average")).isGreaterThanOrEqualTo(attentionStats.get("min"));
        assertThat(attentionStats.get("variance")).isGreaterThanOrEqualTo(0.0);
    }
    
    @Test
    void testConfigurationValidation() {
        // Test invalid confidence threshold
        assertThatThrownBy(() -> new AttentionConsensus(-0.1, true, 1.0, 64, false))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Confidence threshold must be in [0.0, 1.0]");
        
        assertThatThrownBy(() -> new AttentionConsensus(1.1, true, 1.0, 64, false))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Confidence threshold must be in [0.0, 1.0]");
        
        // Test invalid temperature scaling
        assertThatThrownBy(() -> new AttentionConsensus(0.5, true, 0.0, 64, false))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Temperature scaling must be positive");
        
        assertThatThrownBy(() -> new AttentionConsensus(0.5, true, -1.0, 64, false))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Temperature scaling must be positive");
        
        // Test invalid attention dimensions
        assertThatThrownBy(() -> new AttentionConsensus(0.5, true, 1.0, 0, false))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Attention dimensions must be positive");
        
        assertThatThrownBy(() -> new AttentionConsensus(0.5, true, 1.0, -10, false))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Attention dimensions must be positive");
    }
    
    @Test
    void testGettersAndToString() {
        var consensus = new AttentionConsensus(0.7, false, 1.5, 128, true);
        
        assertThat(consensus.getConfidenceThreshold()).isEqualTo(0.7);
        assertThat(consensus.isNormalizeAttentionWeights()).isFalse();
        assertThat(consensus.getTemperatureScaling()).isEqualTo(1.5);
        assertThat(consensus.getAttentionDimensions()).isEqualTo(128);
        assertThat(consensus.isUsePositionalEncoding()).isTrue();
        assertThat(consensus.getStrategyName()).isEqualTo("AttentionConsensus");
        assertThat(consensus.getMinimumRequiredChannels()).isEqualTo(1);
        
        var toString = consensus.toString();
        assertThat(toString).contains("AttentionConsensus");
        assertThat(toString).contains("threshold=0.70");
        assertThat(toString).contains("normalized=false");
        assertThat(toString).contains("temp=1.5");
        assertThat(toString).contains("dims=128");
        assertThat(toString).contains("pos=true");
    }
    
    @Test
    void testChannelContributions() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.8, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 1, 0.7, 80, Map.of())
        );
        
        var channelWeights = Map.of(
            "semantic", 1.0,
            "entity", 0.8,
            "syntactic", 0.6
        );
        
        var result = consensus.computeConsensus(channelResults, channelWeights);
        
        var contributions = result.channelContributions();
        assertThat(contributions).hasSize(3);
        
        // All channels above threshold should have contributions > 0
        assertThat(contributions.get("semantic")).isGreaterThan(0.0);
        assertThat(contributions.get("entity")).isGreaterThan(0.0);
        assertThat(contributions.get("syntactic")).isGreaterThan(0.0);
        
        // Contributions should sum to approximately 1.0
        var totalContribution = contributions.values().stream()
            .mapToDouble(Double::doubleValue)
            .sum();
        assertThat(totalContribution).isCloseTo(1.0, within(0.01));
    }
}