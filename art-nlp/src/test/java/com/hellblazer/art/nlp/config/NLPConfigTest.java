package com.hellblazer.art.nlp.config;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.assertj.core.api.Assertions.*;

/**
 * Tests for NLPConfig builder and configuration classes.
 * 
 * Validates complete TARGET_VISION.md API compliance.
 */
@DisplayName("NLPConfig Tests")
class NLPConfigTest {

    @Test
    @DisplayName("Should create config with TARGET_VISION.md usage pattern")
    void shouldCreateConfigWithTargetVisionUsage() {
        // This is the exact usage pattern from TARGET_VISION.md
        var config = new NLPConfig.Builder()
            .withSemanticVigilance(0.85)
            .withFastTextModel("models/cc.en.300.vec.gz")
            .withContextWindowSize(200)
            .withThreadPoolSize(8)
            .enableAllChannels()
            .build();
        
        assertThat(config).isNotNull();
        
        // Verify semantic channel configuration
        assertThat(config.getSemanticConfig().getVigilance()).isEqualTo(0.85);
        assertThat(config.getSemanticConfig().getModelPath()).isEqualTo("models/cc.en.300.vec.gz");
        assertThat(config.isChannelEnabled("semantic")).isTrue();
        
        // Verify context configuration
        assertThat(config.getContextConfig().getWindowSize()).isEqualTo(200);
        assertThat(config.isChannelEnabled("context")).isTrue();
        
        // Verify performance configuration  
        assertThat(config.getPerformanceConfig().getThreadPoolSize()).isEqualTo(8);
        
        // Verify all channels enabled
        assertThat(config.isChannelEnabled("semantic")).isTrue();
        assertThat(config.isChannelEnabled("syntactic")).isTrue();
        assertThat(config.isChannelEnabled("context")).isTrue();
        assertThat(config.isChannelEnabled("entity")).isTrue();
        assertThat(config.isChannelEnabled("sentiment")).isTrue();
    }

    @Test
    @DisplayName("Should have correct TARGET_VISION.md default vigilance values")
    void shouldHaveCorrectDefaultVigilanceValues() {
        var config = NLPConfig.builder().build();
        
        // These are the exact vigilance values from TARGET_VISION.md
        assertThat(config.getSemanticConfig().getVigilance()).isEqualTo(0.85);  // range 0.70-0.95
        assertThat(config.getSyntacticConfig().getVigilance()).isEqualTo(0.75); // range 0.70-0.85
        assertThat(config.getContextConfig().getVigilance()).isEqualTo(0.85);   // range 0.80-0.95
        assertThat(config.getEntityConfig().getVigilance()).isEqualTo(0.80);    // range 0.75-0.85
        assertThat(config.getSentimentConfig().getVigilance()).isEqualTo(0.50); // range 0.40-0.70
    }

    @Test
    @DisplayName("Should support individual vigilance configuration")
    void shouldSupportIndividualVigilanceConfiguration() {
        var config = new NLPConfig.Builder()
            .withSemanticVigilance(0.90)
            .withSyntacticVigilance(0.80)
            .withContextVigilance(0.90)
            .withEntityVigilance(0.75)
            .withSentimentVigilance(0.60)
            .build();
        
        assertThat(config.getSemanticConfig().getVigilance()).isEqualTo(0.90);
        assertThat(config.getSyntacticConfig().getVigilance()).isEqualTo(0.80);
        assertThat(config.getContextConfig().getVigilance()).isEqualTo(0.90);
        assertThat(config.getEntityConfig().getVigilance()).isEqualTo(0.75);
        assertThat(config.getSentimentConfig().getVigilance()).isEqualTo(0.60);
    }

    @Test
    @DisplayName("Should support channel enable/disable operations")
    void shouldSupportChannelEnableDisableOperations() {
        var config = new NLPConfig.Builder()
            .disableAllChannels()
            .enableChannel("semantic")
            .enableChannel("entity")
            .build();
        
        assertThat(config.isChannelEnabled("semantic")).isTrue();
        assertThat(config.isChannelEnabled("syntactic")).isFalse();
        assertThat(config.isChannelEnabled("context")).isFalse();
        assertThat(config.isChannelEnabled("entity")).isTrue();
        assertThat(config.isChannelEnabled("sentiment")).isFalse();
    }

    @Test
    @DisplayName("Should handle unknown channel names gracefully")
    void shouldHandleUnknownChannelNamesGracefully() {
        var config = NLPConfig.builder().build();
        
        assertThat(config.isChannelEnabled("unknown")).isFalse();
        assertThat(config.isChannelEnabled("")).isFalse();
        assertThat(config.isChannelEnabled(null)).isFalse();
    }

    @Test
    @DisplayName("Should provide access to all configuration objects")
    void shouldProvideAccessToAllConfigurationObjects() {
        var config = NLPConfig.builder().build();
        
        assertThat(config.getSemanticConfig()).isNotNull();
        assertThat(config.getSyntacticConfig()).isNotNull();
        assertThat(config.getContextConfig()).isNotNull();
        assertThat(config.getEntityConfig()).isNotNull();
        assertThat(config.getSentimentConfig()).isNotNull();
        assertThat(config.getPerformanceConfig()).isNotNull();
    }

    @Test
    @DisplayName("Should validate vigilance parameter ranges")
    void shouldValidateVigilanceParameterRanges() {
        var builder = new NLPConfig.Builder();
        
        // Test valid ranges
        assertThatNoException().isThrownBy(() -> builder.withSemanticVigilance(0.85));
        assertThatNoException().isThrownBy(() -> builder.withSyntacticVigilance(0.75));
        
        // Test invalid ranges  
        var config = builder.build();
        assertThatThrownBy(() -> config.getSemanticConfig().setVigilance(-0.1))
            .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> config.getSemanticConfig().setVigilance(1.1))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("Should have meaningful toString representations")
    void shouldHaveMeaningfulToStringRepresentations() {
        var config = NLPConfig.builder().build();
        
        assertThat(config.getSemanticConfig().toString()).contains("SemanticConfig");
        assertThat(config.getSyntacticConfig().toString()).contains("SyntacticConfig");
        assertThat(config.getContextConfig().toString()).contains("ContextConfig");
        assertThat(config.getEntityConfig().toString()).contains("EntityConfig");
        assertThat(config.getSentimentConfig().toString()).contains("SentimentConfig");
        assertThat(config.getPerformanceConfig().toString()).contains("PerformanceConfig");
    }

    @Test
    @DisplayName("Should support fluent builder chaining")
    void shouldSupportFluentBuilderChaining() {
        var config = NLPConfig.builder()
            .withSemanticVigilance(0.85)
            .withSyntacticVigilance(0.75)
            .withContextVigilance(0.85)
            .withEntityVigilance(0.80)
            .withSentimentVigilance(0.50)
            .withFastTextModel("custom-model.bin")
            .withContextWindowSize(300)
            .withThreadPoolSize(16)
            .enableAllChannels()
            .disableChannel("sentiment")
            .build();
        
        assertThat(config).isNotNull();
        assertThat(config.isChannelEnabled("semantic")).isTrue();
        assertThat(config.isChannelEnabled("sentiment")).isFalse();
    }
}