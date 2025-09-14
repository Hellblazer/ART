package com.hellblazer.art.hartcq;

import com.hellblazer.art.hartcq.integration.HARTCQ;
import com.hellblazer.art.hartcq.integration.ProcessingResult;
import org.junit.jupiter.api.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import static org.assertj.core.api.Assertions.*;

/**
 * Basic integration tests for the consolidated HART-CQ system.
 * Validates that the refactoring worked and the API is functional.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class HARTCQIntegrationTest {
    private static final Logger log = LoggerFactory.getLogger(HARTCQIntegrationTest.class);

    private HARTCQ hartcq;

    @BeforeEach
    void setUp() {
        hartcq = new HARTCQ();
    }

    @Test
    @Order(1)
    @DisplayName("Basic text processing should work")
    void testBasicProcessing() {
        var result = hartcq.process("Hello world, how are you today?");

        assertThat(result).isNotNull();
        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getOutput()).isNotEmpty();
        assertThat(result.getConfidence()).isGreaterThanOrEqualTo(0);

        log.info("Basic processing result: {}", result.getOutput());
    }

    @Test
    @Order(2)
    @DisplayName("Batch processing should handle multiple texts")
    void testBatchProcessing() {
        var texts = List.of(
            "First sentence.",
            "Second sentence.",
            "Third sentence."
        );

        var results = hartcq.processBatch(texts);

        assertThat(results).hasSize(texts.size());
        assertThat(results).allMatch(r -> r.isSuccessful());
        
        log.info("Batch processing completed successfully");
    }

    @Test
    @Order(3)
    @DisplayName("Empty input should be handled gracefully")
    void testEmptyInput() {
        var result = hartcq.process("");

        assertThat(result).isNotNull();
        // Don't assert success/failure - just that we get a result
        
        log.info("Empty input handled: successful={}", result.isSuccessful());
    }

    @Test
    @Order(4)
    @DisplayName("Whitespace input should be handled gracefully")
    void testWhitespaceInput() {
        var result = hartcq.process("   \t\n   ");

        assertThat(result).isNotNull();
        // Don't assert success/failure - just that we get a result
        
        log.info("Whitespace input handled: successful={}", result.isSuccessful());
    }

    @Test
    @Order(5)
    @DisplayName("Processing result contains expected metadata")
    void testResultMetadata() {
        var result = hartcq.process("Test sentence.");

        assertThat(result.getInput()).isNotNull();
        assertThat(result.getOutput()).isNotNull();
        assertThat(result.getProcessingTime()).isNotNull();
        assertThat(result.getTimestamp()).isNotNull();
        assertThat(result.getMetadata()).isNotNull();
        
        log.info("Result metadata verified: processing time = {}", result.getProcessingTime());
    }
}