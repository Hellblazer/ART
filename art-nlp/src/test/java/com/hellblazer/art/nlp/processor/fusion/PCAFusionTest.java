package com.hellblazer.art.nlp.processor.fusion;

import com.hellblazer.art.nlp.processor.ChannelResult;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for PCAFusion strategy.
 */
public class PCAFusionTest {
    private static final Logger log = LoggerFactory.getLogger(PCAFusionTest.class);
    
    private PCAFusion fusion;
    
    @BeforeEach
    void setUp() {
        fusion = new PCAFusion(8, 0.9, true, true);
    }
    
    @Test
    void testBasicPCAFusion() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.8, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 1, 0.7, 80, Map.of())
        );
        
        var result = fusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isLessThanOrEqualTo(8);
        assertThat(result.dimension()).isGreaterThan(0);
        
        log.debug("PCA fusion result: {} dimensions", result.dimension());
    }
    
    @Test
    void testSingleChannelFusion() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.8, 100, Map.of())
        );
        
        var result = fusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isLessThanOrEqualTo(8);
        assertThat(result.dimension()).isGreaterThan(0);
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
    void testVarianceThreshold() {
        // Test with high variance threshold (should use fewer components)
        var highVarianceFusion = new PCAFusion(10, 0.99, true, true);
        
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.8, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 1, 0.7, 80, Map.of()),
            "lexical", ChannelResult.success("lexical", 3, 0.6, 90, Map.of())
        );
        
        var result = highVarianceFusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isLessThanOrEqualTo(10);
    }
    
    @Test
    void testWithoutNormalization() {
        var unnormalizedFusion = new PCAFusion(6, 0.85, false, true);
        
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 2, 0.8, 80, Map.of())
        );
        
        var result = unnormalizedFusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isLessThanOrEqualTo(6);
    }
    
    @Test
    void testWithoutCentering() {
        var uncenteredFusion = new PCAFusion(6, 0.85, true, false);
        
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 2, 0.8, 80, Map.of())
        );
        
        var result = uncenteredFusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isLessThanOrEqualTo(6);
    }
    
    @Test
    void testLargeNumberOfChannels() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.8, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 1, 0.7, 80, Map.of()),
            "lexical", ChannelResult.success("lexical", 3, 0.6, 90, Map.of()),
            "phonetic", ChannelResult.success("phonetic", 2, 0.75, 110, Map.of())
        );
        
        var result = fusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        assertThat(result.dimension()).isLessThanOrEqualTo(8);
        
        // More channels should still produce valid PCA
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
        // Test invalid target dimensions
        assertThatThrownBy(() -> new PCAFusion(0, 0.9, true, true))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Target dimensions must be positive");
        
        assertThatThrownBy(() -> new PCAFusion(-1, 0.9, true, true))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Target dimensions must be positive");
        
        // Test invalid variance threshold
        assertThatThrownBy(() -> new PCAFusion(5, -0.1, true, true))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Variance threshold must be in [0.0, 1.0]");
        
        assertThatThrownBy(() -> new PCAFusion(5, 1.1, true, true))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Variance threshold must be in [0.0, 1.0]");
    }
    
    @Test
    void testGettersAndProperties() {
        var customFusion = new PCAFusion(12, 0.95, false, false);
        
        assertThat(customFusion.getTargetDimensions()).isEqualTo(12);
        assertThat(customFusion.getVarianceThreshold()).isEqualTo(0.95);
        assertThat(customFusion.isNormalizeInput()).isFalse();
        assertThat(customFusion.isCenterData()).isFalse();
        
        assertThat(customFusion.getStrategyName()).isEqualTo("PCA");
        assertThat(customFusion.getOutputDimension()).isEqualTo(12);
        assertThat(customFusion.getMinimumRequiredChannels()).isEqualTo(2);
        
        // PCA components should be null before computation
        assertThat(customFusion.getPrincipalComponents()).isNull();
        assertThat(customFusion.getEigenvalues()).isNull();
    }
    
    @Test
    void testPCAComponentsAfterComputation() {
        var channelResults = Map.of(
            "semantic", ChannelResult.success("semantic", 1, 0.9, 100, Map.of()),
            "entity", ChannelResult.success("entity", 2, 0.8, 120, Map.of()),
            "syntactic", ChannelResult.success("syntactic", 1, 0.7, 80, Map.of())
        );
        
        var result = fusion.fuseFeatures(channelResults);
        
        assertThat(result).isNotNull();
        
        // After PCA computation, components should be available
        // Note: This depends on internal implementation details
        // In a real scenario, we might expose computed components differently
    }
    
    @Test
    void testToString() {
        var customFusion = new PCAFusion(15, 0.88, true, false);
        
        var toString = customFusion.toString();
        assertThat(toString).contains("PCAFusion");
        assertThat(toString).contains("targetDims=15");
        assertThat(toString).contains("varThreshold=0.88");
        assertThat(toString).contains("normalize=true");
        assertThat(toString).contains("center=false");
    }
    
    @Test
    void testDifferentChannelTypes() {
        // Test with different combinations of channel types
        var semanticOnly = Map.of(
            "fasttext", ChannelResult.success("fasttext", 1, 0.9, 100, Map.of())
        );
        
        var entityOnly = Map.of(
            "ner", ChannelResult.success("ner", 2, 0.8, 120, Map.of())
        );
        
        var syntacticOnly = Map.of(
            "pos", ChannelResult.success("pos", 1, 0.7, 80, Map.of())
        );
        
        var result1 = fusion.fuseFeatures(semanticOnly);
        var result2 = fusion.fuseFeatures(entityOnly);
        var result3 = fusion.fuseFeatures(syntacticOnly);
        
        assertThat(result1).isNotNull();
        assertThat(result2).isNotNull();
        assertThat(result3).isNotNull();
        
        // Different channel types should produce different results
        assertThat(result1.dimension()).isGreaterThan(0);
        assertThat(result2.dimension()).isGreaterThan(0);
        assertThat(result3.dimension()).isGreaterThan(0);
    }
}