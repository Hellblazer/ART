package com.hellblazer.art.nlp.processor.consensus;

import com.hellblazer.art.nlp.processor.ChannelResult;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.Map;
import java.util.HashMap;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for HierarchicalConsensus strategy.
 */
public class HierarchicalConsensusTest {
    
    private HierarchicalConsensus consensus;
    
    @BeforeEach
    void setUp() {
        // Create with custom tier configuration
        var tiers = Map.of(
            "semantic", 3,
            "entity", 2,
            "syntactic", 1,
            "lexical", 0
        );
        consensus = new HierarchicalConsensus(tiers, 0.6, true, 2.0);
    }
    
    @Test
    void testHighTierDecision() {
        // High-tier channel (semantic) should dominate
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.8, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 3, 0.7, 80, Map.of())
        );
        
        var channelWeights = Map.of(
            "semantic", 1.0,
            "entity", 1.0,
            "syntactic", 1.0
        );
        
        var result = consensus.computeConsensus(channelResults, channelWeights);
        
        assertThat(result.category()).isEqualTo(1);  // Semantic channel's decision
        assertThat(result.confidence()).isGreaterThan(0.8);
        assertThat(result.metadata()).containsKey("decidingTier");
        assertThat(result.metadata().get("decidingTier")).isEqualTo(3);
        assertThat(result.metadata()).containsEntry("highTierDecision", true);
    }
    
    @Test
    void testLowConfidenceHighTier() {
        // High-tier channel with low confidence should fall back to lower tiers
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.5, 100, Map.of()), // Below threshold
            "entity", ChannelResult.success("entity", 2, 0.8, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 2, 0.7, 80, Map.of())
        );
        
        var channelWeights = Map.of(
            "semantic", 1.0,
            "entity", 1.0,
            "syntactic", 1.0
        );
        
        var result = consensus.computeConsensus(channelResults, channelWeights);
        
        // Should use tier 2 decision (entity channel)
        assertThat(result.category()).isEqualTo(2);
        assertThat(result.metadata().get("decidingTier")).isEqualTo(2);
        assertThat(result.metadata()).containsEntry("highTierDecision", false);
    }
    
    @Test
    void testTierWeightMultiplier() {
        // Test that higher tiers get exponentially more weight
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.7, 100, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 2, 0.8, 80, Map.of()) // Higher confidence but lower tier
        );
        
        var channelWeights = Map.of(
            "semantic", 1.0,
            "syntactic", 1.0
        );
        
        var result = consensus.computeConsensus(channelResults, channelWeights);
        
        // Semantic should win despite lower confidence due to tier multiplier
        assertThat(result.category()).isEqualTo(1);
        assertThat(result.metadata().get("decidingTier")).isEqualTo(3);
    }
    
    @Test
    void testNoHighTierAgreementRequired() {
        // Test with requireHighTierAgreement = false
        var tiers = Map.of("semantic", 2, "syntactic", 1);
        var flexibleConsensus = new HierarchicalConsensus(tiers, 0.6, false, 2.0);
        
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.5, 100, Map.of()), // Below threshold
            "syntactic", ChannelResult.success("syntactic", 2, 0.8, 80, Map.of())
        );
        
        var channelWeights = Map.of(
            "semantic", 1.0,
            "syntactic", 1.0
        );
        
        var result = flexibleConsensus.computeConsensus(channelResults, channelWeights);
        
        assertThat(result.category()).isEqualTo(2);  // Lower tier decision accepted
        assertThat(result.confidence()).isGreaterThan(0.6);
    }
    
    @Test
    void testUnknownChannelDefaultTier() {
        // Unknown channels should be assigned to tier 0
        var channelResults = Map.of(
            "unknown_channel", ChannelResult.success("unknown_channel", 1, 0.8, 100, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 2, 0.7, 80, Map.of())
        );
        
        var channelWeights = Map.of(
            "unknown_channel", 1.0,
            "syntactic", 1.0
        );
        
        var result = consensus.computeConsensus(channelResults, channelWeights);
        
        // Syntactic (tier 1) should beat unknown (tier 0)
        assertThat(result.category()).isEqualTo(2);
        assertThat(result.metadata().get("decidingTier")).isEqualTo(1);
    }
    
    @Test
    void testEmptyChannelResults() {
        var result = consensus.computeConsensus(Map.of(), Map.of());
        
        assertThat(result.category()).isEqualTo(-1);
        assertThat(result.confidence()).isEqualTo(0.0);
    }
    
    @Test
    void testAllChannelsBelowThreshold() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.5, 100, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 2, 0.4, 80, Map.of())
        );
        
        var channelWeights = Map.of(
            "semantic", 1.0,
            "syntactic", 1.0
        );
        
        var result = consensus.computeConsensus(channelResults, channelWeights);
        
        assertThat(result.category()).isEqualTo(-1);
        assertThat(result.confidence()).isEqualTo(0.0);
    }
    
    @Test
    void testMetadataCompleteness() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 1, 0.8, 120, Map.of())
        );
        
        var channelWeights = Map.of(
            "semantic", 1.0,
            "entity", 0.8
        );
        
        var result = consensus.computeConsensus(channelResults, channelWeights);
        
        var metadata = result.metadata();
        assertThat(metadata).containsKeys(
            "decidingTier", "highTierDecision", "tierGroups", 
            "maxTier", "tierWeightMultiplier", "channelTiers"
        );
        
        assertThat(metadata.get("maxTier")).isEqualTo(3);
        assertThat(metadata.get("tierWeightMultiplier")).isEqualTo(2.0);
    }
    
    @Test
    void testConfigurationValidation() {
        // Test invalid confidence threshold
        assertThatThrownBy(() -> new HierarchicalConsensus(Map.of(), -0.1, true, 2.0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Confidence threshold must be in [0.0, 1.0]");
        
        assertThatThrownBy(() -> new HierarchicalConsensus(Map.of(), 1.1, true, 2.0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Confidence threshold must be in [0.0, 1.0]");
        
        // Test invalid tier weight multiplier
        assertThatThrownBy(() -> new HierarchicalConsensus(Map.of(), 0.6, true, 0.0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Tier weight multiplier must be positive");
        
        assertThatThrownBy(() -> new HierarchicalConsensus(Map.of(), 0.6, true, -1.0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Tier weight multiplier must be positive");
    }
    
    @Test
    void testGettersAndToString() {
        var tiers = Map.of("semantic", 2, "syntactic", 1);
        var consensus = new HierarchicalConsensus(tiers, 0.7, true, 1.5);
        
        assertThat(consensus.getChannelTiers()).isEqualTo(tiers);
        assertThat(consensus.getConfidenceThreshold()).isEqualTo(0.7);
        assertThat(consensus.isRequireHighTierAgreement()).isTrue();
        assertThat(consensus.getTierWeightMultiplier()).isEqualTo(1.5);
        assertThat(consensus.getStrategyName()).isEqualTo("HierarchicalConsensus");
        assertThat(consensus.getMinimumRequiredChannels()).isEqualTo(1);
        
        var toString = consensus.toString();
        assertThat(toString).contains("HierarchicalConsensus");
        assertThat(toString).contains("threshold=0.70");
        assertThat(toString).contains("requireHighTier=true");
        assertThat(toString).contains("multiplier=1.5");
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
        assertThat(contributions.get("semantic")).isGreaterThan(0.0);
        
        // Contributions should sum to approximately 1.0
        var totalContribution = contributions.values().stream()
            .mapToDouble(Double::doubleValue)
            .sum();
        assertThat(totalContribution).isCloseTo(1.0, within(0.01));
    }
}