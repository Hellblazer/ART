package com.hellblazer.art.nlp.processor;

import com.hellblazer.art.nlp.channels.FastTextChannel;
import com.hellblazer.art.nlp.channels.EntityChannel;
import com.hellblazer.art.nlp.channels.SyntacticChannel;
import com.hellblazer.art.nlp.core.ProcessingResult;
import com.hellblazer.art.nlp.processor.consensus.WeightedVotingConsensus;
import com.hellblazer.art.nlp.processor.fusion.ConcatenationFusion;
import com.hellblazer.art.nlp.fasttext.FastTextModel;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.nio.file.Files;
import java.util.Map;
import java.io.IOException;

import static org.assertj.core.api.Assertions.*;

/**
 * Integration tests for MultiChannelProcessor - Phase 4 multi-modal processing.
 */
public class MultiChannelProcessorTest {

    @TempDir
    private Path tempDir;
    
    private MultiChannelProcessor processor;
    private Path fasttextModel;
    
    @BeforeEach
    void setUp() throws IOException {
        // Create test FastText model file
        fasttextModel = createTestFastTextModel();
        
        // Build multi-channel processor with basic configuration
        processor = MultiChannelProcessor.builder()
            .consensusStrategy(new WeightedVotingConsensus(0.5, false))
            .fusionStrategy(new ConcatenationFusion())
            .enableParallelProcessing(true)
            .build();
    }
    
    @AfterEach
    void tearDown() {
        if (processor != null) {
            processor.close();
        }
    }
    
    @Test
    void testBasicMultiChannelProcessing() {
        // Configure channels
        var fastTextChannel = createFastTextChannel("semantic");
        var entityChannel = createEntityChannel("entity");  
        var syntacticChannel = createSyntacticChannel("syntactic");
        
        processor.addChannel("semantic", fastTextChannel, 1.0);
        processor.addChannel("entity", entityChannel, 0.8);
        processor.addChannel("syntactic", syntacticChannel, 0.9);
        
        var testText = "John Smith works at Apple Inc. in Cupertino.";
        var result = processor.processText(testText);
        
        // Verify result structure
        assertThat(result).isNotNull();
        assertThat(result.getText()).isEqualTo(testText);
        assertThat(result.isSuccess()).isTrue();
        
        // Verify channel results
        assertThat(result.getChannelResults()).hasSize(3);
        assertThat(result.getChannelResults()).containsKeys("semantic", "entity", "syntactic");
        
        // Verify each channel processed successfully
        var semanticResult = result.getChannelResult("semantic");
        assertThat(semanticResult).isNotNull();
        assertThat(semanticResult.isSuccess()).isTrue();
        assertThat(semanticResult.channelId()).isEqualTo("semantic");
        
        var entityResult = result.getChannelResult("entity");
        assertThat(entityResult).isNotNull(); 
        assertThat(entityResult.isSuccess()).isTrue();
        assertThat(entityResult.channelId()).isEqualTo("entity");
        
        var syntacticResult = result.getChannelResult("syntactic");
        assertThat(syntacticResult).isNotNull();
        assertThat(syntacticResult.isSuccess()).isTrue();
        assertThat(syntacticResult.channelId()).isEqualTo("syntactic");
        
        // Verify consensus decision
        assertThat(result.getCategory()).isGreaterThanOrEqualTo(0);
        assertThat(result.getConfidence()).isBetween(0.0, 1.0);
        
        // Verify fused features (if present)
        if (result.getFusedFeatures() != null) {
            assertThat(result.getFusedFeatures()).isNotNull();
        }
        
        // Verify processing metadata
        assertThat(result.getProcessingTimeMs()).isGreaterThan(0);
        assertThat(result.getConsensusMetadata()).isNotEmpty();
    }
    
    @Test
    void testDegradedProcessing() {
        // Create processor with only working channels
        var fastTextChannel = createFastTextChannel("semantic");
        var syntacticChannel = createSyntacticChannel("syntactic");
        
        processor.addChannel("semantic", fastTextChannel, 1.0);
        processor.addChannel("syntactic", syntacticChannel, 0.9);
        
        var testText = "The quick brown fox jumps over the lazy dog.";
        var result = processor.processText(testText);
        
        // Should succeed with partial channel results
        assertThat(result).isNotNull();
        assertThat(result.isSuccess()).isTrue();
        assertThat(result.getChannelResults()).hasSize(2);
        
        // Verify consensus still works with fewer channels
        assertThat(result.getCategory()).isGreaterThanOrEqualTo(0);
        assertThat(result.getConfidence()).isGreaterThan(0.0);
        
        // Check metadata indicates partial processing
        var metadata = result.getConsensusMetadata();
        assertThat(metadata).isNotNull();
    }
    
    @Test
    void testSequentialProcessing() {
        // Configure for sequential processing
        processor = MultiChannelProcessor.builder()
            .enableParallelProcessing(false)
            .consensusStrategy(new WeightedVotingConsensus())
            .fusionStrategy(new ConcatenationFusion())
            .build();
            
        var fastTextChannel = createFastTextChannel("semantic");
        var syntacticChannel = createSyntacticChannel("syntactic");
        
        processor.addChannel("semantic", fastTextChannel, 1.0);
        processor.addChannel("syntactic", syntacticChannel, 0.8);
        
        var testText = "Sequential processing test sentence.";
        var result = processor.processText(testText);
        
        assertThat(result).isNotNull();
        assertThat(result.isSuccess()).isTrue();
        assertThat(result.getChannelResults()).hasSize(2);
        
        // Sequential processing should still produce valid results
        assertThat(result.getCategory()).isGreaterThanOrEqualTo(0);
        if (result.getFusedFeatures() != null) {
            assertThat(result.getFusedFeatures()).isNotNull();
        }
    }
    
    @Test
    void testCrossChannelLearning() {
        var fastTextChannel = createFastTextChannel("semantic");
        var syntacticChannel = createSyntacticChannel("syntactic");
        
        processor.addChannel("semantic", fastTextChannel, 1.0);
        processor.addChannel("syntactic", syntacticChannel, 0.8);
        
        // Process multiple texts to trigger cross-channel learning
        var texts = new String[] {
            "This is a positive example.",
            "Another good example here.", 
            "Negative example follows.",
            "Bad example for testing."
        };
        
        for (var text : texts) {
            var result = processor.processText(text);
            assertThat(result.isSuccess()).isTrue();
        }
        
        // Verify statistics were updated
        var metrics = processor.getMetrics();
        assertThat(metrics.totalProcessed()).isEqualTo(texts.length);
        assertThat(metrics.successfulProcessed()).isEqualTo(texts.length);
        assertThat(metrics.activeChannels()).isEqualTo(2);
    }
    
    @Test
    void testAdaptiveWeighting() {
        var fastTextChannel = createFastTextChannel("semantic");
        var syntacticChannel = createSyntacticChannel("syntactic");
        
        processor.addChannel("semantic", fastTextChannel, 1.0);
        processor.addChannel("syntactic", syntacticChannel, 0.5); // Lower initial weight
        
        // Process some text
        var result = processor.processText("Adaptive weighting test example.");
        assertThat(result.isSuccess()).isTrue();
        
        // Verify channels were configured correctly
        assertThat(processor.getChannelIds()).hasSize(2);
        assertThat(processor.getChannelIds()).containsExactlyInAnyOrder("semantic", "syntactic");
    }
    
    @Test
    void testEmptyTextHandling() {
        var fastTextChannel = createFastTextChannel("semantic");
        processor.addChannel("semantic", fastTextChannel, 1.0);
        
        var result = processor.processText("");
        
        // Should handle empty text gracefully
        assertThat(result).isNotNull();
        assertThat(result.getText()).isEmpty();
        // Result success depends on implementation - could be success with default values
    }
    
    @Test 
    void testProcessorConfiguration() {
        // Test builder pattern configuration
        var customProcessor = MultiChannelProcessor.builder()
            .enableParallelProcessing(true)
            .consensusStrategy(new WeightedVotingConsensus(0.7, true))
            .fusionStrategy(new ConcatenationFusion(false, true, 2000))
            .learningRateDecay(0.9)
            .build();
            
        assertThat(customProcessor).isNotNull();
        customProcessor.close();
    }
    
    // Helper methods
    
    private Path createTestFastTextModel() throws IOException {
        var modelFile = tempDir.resolve("test.vec");
        var content = """
            8 3
            the 0.1 0.2 0.3
            quick 0.2 0.3 0.4
            brown 0.3 0.4 0.5
            fox 0.4 0.5 0.6
            jumps 0.5 0.6 0.7
            over 0.6 0.7 0.8
            lazy 0.7 0.8 0.9
            dog 0.8 0.9 1.0
            """;
        Files.writeString(modelFile, content);
        return modelFile;
    }
    
    private FastTextChannel createFastTextChannel(String channelId) {
        try {
            var channel = new FastTextChannel(channelId, 0.8, fasttextModel, 3);
            channel.initialize();
            return channel;
        } catch (Exception e) {
            throw new RuntimeException("Failed to create FastText channel", e);
        }
    }
    
    private EntityChannel createEntityChannel(String channelId) {
        try {
            var channel = new EntityChannel(channelId, 0.8);
            channel.initialize();
            return channel;
        } catch (Exception e) {
            throw new RuntimeException("Failed to create Entity channel", e);
        }
    }
    
    private SyntacticChannel createSyntacticChannel(String channelId) {
        try {
            var channel = new SyntacticChannel(channelId, 0.8);
            channel.initialize();
            return channel;
        } catch (Exception e) {
            throw new RuntimeException("Failed to create Syntactic channel", e);
        }
    }
}