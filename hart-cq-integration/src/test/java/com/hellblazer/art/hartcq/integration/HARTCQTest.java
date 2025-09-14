package com.hellblazer.art.hartcq.integration;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Basic test suite for HART-CQ system after refactoring.
 * Validates core functionality works with the new consolidated API.
 */
@DisplayName("HART-CQ System Tests")
public class HARTCQTest {
    
    private HARTCQ hartcq;
    
    @BeforeEach
    void setUp() {
        hartcq = new HARTCQ();
    }
    
    @Test
    @DisplayName("Basic Processing - Question Input")
    void testQuestionProcessing() {
        var input = "What is the weather today?";
        var response = hartcq.process(input);
        
        assertThat(response).isNotNull();
        assertThat(response.getOutput()).isNotNull();
        System.out.println("Question Response: " + response.getOutput());
    }
    
    @Test
    @DisplayName("Basic Processing - Statement Input")
    void testStatementProcessing() {
        var input = "The sky is blue and beautiful.";
        var response = hartcq.process(input);
        
        assertThat(response).isNotNull();
        assertThat(response.getOutput()).isNotNull();
        System.out.println("Statement Response: " + response.getOutput());
    }
    
    @Test
    @DisplayName("Empty Input Handling")
    void testEmptyInput() {
        var response = hartcq.process("");
        
        assertThat(response).isNotNull();
        System.out.println("Empty Input Response: " + response.getOutput());
    }
}