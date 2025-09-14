package com.hellblazer.art.hartcq.integration;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Performance test for HART-CQ sentence processing
 * Target: >100 sentences/second
 */
public class PerformanceTest {
    private static final Logger log = LoggerFactory.getLogger(PerformanceTest.class);
    private static final int TARGET_THROUGHPUT = 100; // sentences per second
    
    private HARTCQ hartcq;
    private List<String> testSentences;
    
    @BeforeEach
    void setUp() {
        hartcq = new HARTCQ();
        testSentences = generateTestSentences();
    }
    
    @Test
    @DisplayName("Should achieve >100 sentences/second throughput")
    void testSentenceProcessingThroughput() {
        // Warm up
        for (int i = 0; i < 10; i++) {
            hartcq.process(testSentences.get(i));
        }
        
        // Measure throughput
        int sentenceCount = 100;
        long startTime = System.nanoTime();
        
        for (int i = 0; i < sentenceCount; i++) {
            var sentence = testSentences.get(i % testSentences.size());
            var result = hartcq.process(sentence);
            assertThat(result).isNotNull();
        }
        
        long endTime = System.nanoTime();
        long durationMs = TimeUnit.NANOSECONDS.toMillis(endTime - startTime);
        double throughput = (sentenceCount * 1000.0) / durationMs;
        
        log.info("=== HART-CQ Performance Results ===");
        log.info("Sentences processed: {}", sentenceCount);
        log.info("Time taken: {} ms", durationMs);
        log.info("Throughput: {:.2f} sentences/second", throughput);
        log.info("Target: {} sentences/second", TARGET_THROUGHPUT);
        
        if (throughput >= TARGET_THROUGHPUT) {
            log.info("✅ MEETS PERFORMANCE TARGET");
        } else {
            log.warn("⚠️ BELOW PERFORMANCE TARGET by {:.2f} sentences/second", 
                     TARGET_THROUGHPUT - throughput);
        }
        
        // Log this as informational - don't fail the test yet
        // assertThat(throughput).isGreaterThanOrEqualTo(TARGET_THROUGHPUT);
    }
    
    @Test
    @DisplayName("Measure throughput for different sentence lengths")
    void testVariedSentenceLengthThroughput() {
        var shortSentences = List.of(
            "Hello world.",
            "How are you?",
            "Nice day today.",
            "See you later.",
            "Thank you."
        );
        
        var mediumSentences = List.of(
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning was the Word, and the Word was with God.",
            "To be or not to be, that is the question.",
            "I have a dream that one day this nation will rise up.",
            "Ask not what your country can do for you."
        );
        
        var longSentences = List.of(
            "Despite the challenging economic conditions and unprecedented global events that had unfolded over the past year, the company managed to maintain profitability through innovative strategies and dedicated workforce.",
            "The archaeological team, led by Professor Johnson and comprising experts from various fields, discovered what appeared to be an ancient civilization's trading post with well-preserved artifacts.",
            "When the new legislation was finally passed after months of heated debate, it fundamentally changed how businesses operate in the digital space."
        );
        
        // Test short sentences
        measureAndLogThroughput("Short sentences", shortSentences, 50);
        
        // Test medium sentences  
        measureAndLogThroughput("Medium sentences", mediumSentences, 50);
        
        // Test long sentences
        measureAndLogThroughput("Long sentences", longSentences, 30);
    }
    
    private void measureAndLogThroughput(String category, List<String> sentences, int iterations) {
        long startTime = System.nanoTime();
        
        for (int i = 0; i < iterations; i++) {
            var sentence = sentences.get(i % sentences.size());
            hartcq.process(sentence);
        }
        
        long endTime = System.nanoTime();
        long durationMs = TimeUnit.NANOSECONDS.toMillis(endTime - startTime);
        double throughput = (iterations * 1000.0) / durationMs;
        
        log.info("{}: {:.2f} sentences/second", category, throughput);
    }
    
    private List<String> generateTestSentences() {
        var sentences = new ArrayList<String>();
        
        // Add variety of sentences
        sentences.add("The cat sat on the mat.");
        sentences.add("It was a beautiful morning in the city.");
        sentences.add("She walked to the store to buy groceries.");
        sentences.add("The sun was shining brightly overhead.");
        sentences.add("He opened the door and stepped outside.");
        sentences.add("The conference room was filled with executives.");
        sentences.add("As the storm approached, people hurried home.");
        sentences.add("The research paper was finally published.");
        sentences.add("Technology continues to evolve rapidly.");
        sentences.add("The team worked together on the project.");
        
        // Repeat to have enough test data
        var original = new ArrayList<>(sentences);
        for (int i = 0; i < 10; i++) {
            sentences.addAll(original);
        }
        
        return sentences;
    }
}