package com.hellblazer.art.nlp.processor.fusion;

import com.hellblazer.art.nlp.processor.ChannelResult;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.Map;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for AttentionFusion strategy.
 */
public class AttentionFusionTest {
    
    private AttentionFusion fusion;
    
    @BeforeEach
    void setUp() {
        fusion = new AttentionFusion(32, 2, 1.0, false, true, 128);
    }
    
    @Test
    void testBasicAttentionFusion() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.8, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 1, 0.7, 80, Map.of())
        );
        
        var result = fusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isLessThanOrEqualTo(128);
        assertThat(result.dimension()).isGreaterThan(0);
    }
    
    @Test
    void testSingleChannelFusion() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.8, 100, Map.of())
        );
        
        var result = fusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isLessThanOrEqualTo(128);
        assertThat(result.dimension()).isGreaterThan(0);
    }
    
    @Test
    void testMultipleAttentionHeads() {
        var multiHeadFusion = new AttentionFusion(64, 8, 1.0, false, true, 256);
        
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.8, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 1, 0.7, 80, Map.of()),
            "lexical", ChannelResult.success("lexical", 3, 0.6, 90, Map.of())
        );
        
        var result = multiHeadFusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isLessThanOrEqualTo(256);
        assertThat(result.dimension()).isGreaterThan(0);
    }
    
    @Test
    void testPositionalEncoding() {
        var fusionWithPos = new AttentionFusion(32, 4, 1.0, true, true, 128);
        var fusionWithoutPos = new AttentionFusion(32, 4, 1.0, false, true, 128);
        
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.8, 100, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 2, 0.7, 80, Map.of())
        );
        
        var resultWithPos = fusionWithPos.fuseFeatures(channelResults);
        var resultWithoutPos = fusionWithoutPos.fuseFeatures(channelResults);
        
        assertThat(resultWithPos).isNotNull();
        assertThat(resultWithoutPos).isNotNull();
        
        // Results should be different due to positional encoding
        assertThat(resultWithPos.dimension()).isEqualTo(resultWithoutPos.dimension());
        
        // At least some values should be different
        var different = false;
        for (var i = 0; i < resultWithPos.dimension(); i++) {
            if (Math.abs(resultWithPos.get(i) - resultWithoutPos.get(i)) > 1e-6) {
                different = true;
                break;
            }
        }
        assertThat(different).isTrue();
    }
    
    @Test
    void testTemperatureScaling() {
        var lowTempFusion = new AttentionFusion(32, 2, 0.5, false, true, 128);
        var highTempFusion = new AttentionFusion(32, 2, 2.0, false, true, 128);
        
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.6, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 1, 0.8, 80, Map.of())
        );
        
        var lowTempResult = lowTempFusion.fuseFeatures(channelResults);
        var highTempResult = highTempFusion.fuseFeatures(channelResults);
        
        assertThat(lowTempResult).isNotNull();
        assertThat(highTempResult).isNotNull();
        assertThat(lowTempResult.dimension()).isEqualTo(highTempResult.dimension());
        
        // Temperature scaling should affect the fusion results
        // Low temperature should create more focused attention
    }
    
    @Test
    void testWithoutNormalization() {
        var unnormalizedFusion = new AttentionFusion(32, 2, 1.0, false, false, 128);
        
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 2, 0.8, 80, Map.of())
        );
        
        var result = unnormalizedFusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isLessThanOrEqualTo(128);
    }
    
    @Test
    void testEmptyChannelResults() {
        var result = fusion.fuseFeatures(Map.of());
        
        assertThat(result).isNull();
    }
    
    @Test
    void testFailedChannelResults() {
        var channelResults = Map.of(
            "semantic", ChannelResult.failed("semantic", "Error", 100),
            "entity", ChannelResult.failed("entity", "Error", 120)
        );
        
        var result = fusion.fuseFeatures(channelResults);
        
        assertThat(result).isNull();
    }
    
    @Test
    void testMixedSuccessFailure() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.failed("entity", "Error", 120),
            "syntactic", ChannelResult.success("syntactic", 2, 0.7, 80, Map.of())
        );
        
        var result = fusion.fuseFeatures(channelResults);
        
        // Should work with partial success
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isGreaterThan(0);
    }
    
    @Test
    void testDifferentChannelTypes() {
        var channelResults = Map.of(
            "fasttext", ChannelResult.success("fasttext", 1, 0.9, 100, Map.of()),
            "ner", ChannelResult.success("ner", 2, 0.8, 120, Map.of()),
            "pos", ChannelResult.success("pos", 1, 0.7, 80, Map.of()),
            "unknown", ChannelResult.success("unknown", 3, 0.6, 90, Map.of())
        );
        
        var result = fusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isGreaterThan(0);
        
        // Should handle different channel types appropriately
    }
    
    @Test
    void testLargeNumberOfChannels() {
        var channelResults = Map.of(
            "semantic1", ChannelResult.success("semantic1", 1, 0.9, 100, Map.of()),
            "semantic2", ChannelResult.success("semantic2", 2, 0.8, 110, Map.of()),
            "entity1", ChannelResult.success("entity1", 1, 0.7, 120, Map.of()),
            "entity2", ChannelResult.success("entity2", 3, 0.75, 130, Map.of()),
            "syntactic1", ChannelResult.success("syntactic1", 2, 0.6, 80, Map.of()),
            "syntactic2", ChannelResult.success("syntactic2", 1, 0.65, 90, Map.of())
        );
        
        var result = fusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isLessThanOrEqualTo(128);
        
        // Should handle many channels effectively
        assertThat(result.dimension()).isGreaterThan(0);
    }
    
    @Test
    void testConsistentOutput() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.8, 120, Map.of())
        );
        
        var result1 = fusion.fuseFeatures(channelResults);
        var result2 = fusion.fuseFeatures(channelResults);
        
        assertThat(result1).isNotNull();
        assertThat(result2).isNotNull();
        assertThat(result1.dimension()).isEqualTo(result2.dimension());
        
        // Results should be consistent (same input → same output)
        for (var i = 0; i < result1.dimension(); i++) {
            assertThat(result1.get(i)).isCloseTo(result2.get(i), within(1e-6));
        }
    }
    
    @Test
    void testConfigurationValidation() {
        // Test invalid attention dimensions
        assertThatThrownBy(() -> new AttentionFusion(0, 2, 1.0, false, true, 128))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Attention dimensions must be positive");
        
        assertThatThrownBy(() -> new AttentionFusion(-1, 2, 1.0, false, true, 128))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Attention dimensions must be positive");
        
        // Test invalid number of attention heads
        assertThatThrownBy(() -> new AttentionFusion(32, 0, 1.0, false, true, 128))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Number of attention heads must be positive");
        
        assertThatThrownBy(() -> new AttentionFusion(32, -1, 1.0, false, true, 128))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Number of attention heads must be positive");
        
        // Test invalid temperature scaling
        assertThatThrownBy(() -> new AttentionFusion(32, 2, 0.0, false, true, 128))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Temperature scaling must be positive");
        
        assertThatThrownBy(() -> new AttentionFusion(32, 2, -1.0, false, true, 128))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Temperature scaling must be positive");
        
        // Test invalid max feature dimensions
        assertThatThrownBy(() -> new AttentionFusion(32, 2, 1.0, false, true, 0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Max feature dimensions must be positive");
        
        assertThatThrownBy(() -> new AttentionFusion(32, 2, 1.0, false, true, -10))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Max feature dimensions must be positive");
    }
    
    @Test
    void testGettersAndProperties() {
        var customFusion = new AttentionFusion(48, 6, 1.5, true, false, 200);
        
        assertThat(customFusion.getAttentionDimensions()).isEqualTo(48);
        assertThat(customFusion.getNumAttentionHeads()).isEqualTo(6);
        assertThat(customFusion.getTemperatureScaling()).isEqualTo(1.5);
        assertThat(customFusion.isUsePositionalEncoding()).isTrue();
        assertThat(customFusion.isNormalizeOutputs()).isFalse();
        assertThat(customFusion.getMaxFeatureDimensions()).isEqualTo(200);
        
        assertThat(customFusion.getStrategyName()).isEqualTo("AttentionFusion");
        assertThat(customFusion.getOutputDimension()).isEqualTo(200);
        assertThat(customFusion.getMinimumRequiredChannels()).isEqualTo(1);
    }
    
    @Test
    void testToString() {
        var customFusion = new AttentionFusion(48, 6, 1.5, true, false, 200);
        
        var toString = customFusion.toString();
        assertThat(toString).contains("AttentionFusion");
        assertThat(toString).contains("attentionDims=48");
        assertThat(toString).contains("heads=6");
        assertThat(toString).contains("temp=1.5");
        assertThat(toString).contains("pos=true");
        assertThat(toString).contains("norm=false");
        assertThat(toString).contains("maxDims=200");
    }
    
    @Test
    void testVaryingConfidenceLevels() {
        // Test with channels having very different confidence levels
        var channelResults = Map.of(
            "high_conf", ChannelResult.success("high_conf", 1, 0.95, 100, Map.of()),
            "med_conf", ChannelResult.success("med_conf", 2, 0.6, 120, Map.of()),
            "low_conf", ChannelResult.success("low_conf", 1, 0.3, 80, Map.of())
        );
        
        var result = fusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isGreaterThan(0);
        
        // Attention mechanism should weight high-confidence channels more heavily
        // This is implicit in the result - we can't directly test the attention weights
        // without exposing them, but the fusion should complete successfully
    }
    
    @Test
    void testSingleHeadAttention() {
        var singleHeadFusion = new AttentionFusion(32, 1, 1.0, false, true, 128);
        
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.8, 120, Map.of())
        );
        
        var result = singleHeadFusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isGreaterThan(0);
        assertThat(result.dimension()).isLessThanOrEqualTo(128);
    }
    
    @Test
    void testSmallMaxDimensions() {
        var smallFusion = new AttentionFusion(16, 2, 1.0, false, true, 32);
        
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.8, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 1, 0.7, 80, Map.of())
        );
        
        var result = smallFusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isLessThanOrEqualTo(32);
        assertThat(result.dimension()).isGreaterThan(0);
    }
}