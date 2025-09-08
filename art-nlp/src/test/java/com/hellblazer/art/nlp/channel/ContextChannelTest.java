package com.hellblazer.art.nlp.channel;

import com.hellblazer.art.nlp.channels.ContextChannel;
import com.hellblazer.art.nlp.config.ChannelConfig;
import com.hellblazer.art.nlp.core.Entity;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("ContextChannel Tests")
class ContextChannelTest {
    
    private ContextChannel contextChannel;
    
    @BeforeEach
    void setUp() {
        contextChannel = new ContextChannel();
    }
    
    @Test
    @DisplayName("Should initialize with default configuration")
    void testDefaultInitialization() {
        assertNotNull(contextChannel);
        assertEquals("context", contextChannel.getChannelName());
        assertEquals(0.5, contextChannel.getVigilance(), 0.001);
        // Note: Learning rate is not exposed in BaseChannel API
        assertEquals(0.2, contextChannel.getNeighbourhoodRadius(), 0.001);
    }
    
    @Test
    @DisplayName("Should initialize with custom configuration")
    void testCustomInitialization() {
        var config = ChannelConfig.builder()
            .channelName("custom_context")
            .vigilance(0.7)
            .learningRate(0.6)
            .maxTokensPerInput(150)
            .build();
        
        var customChannel = new ContextChannel(config, 0.3);
        
        assertEquals("custom_context", customChannel.getChannelName());
        assertEquals(0.7, customChannel.getVigilance(), 0.001);
        // Note: Learning rate is not exposed in BaseChannel API
        assertEquals(0.3, customChannel.getNeighbourhoodRadius(), 0.001);
    }
    
    @Test
    @DisplayName("Should have working channel properties")
    void testChannelProperties() {
        assertNotNull(contextChannel);
        assertEquals("context", contextChannel.getChannelName());
        assertEquals(0.5, contextChannel.getVigilance(), 0.001);
        assertEquals(0.2, contextChannel.getNeighbourhoodRadius(), 0.001);
        
        // Test category count starts at 0
        assertEquals(0, contextChannel.getCategoryCount());
    }
    
    @Test
    @DisplayName("Should handle reset operation")
    void testReset() {
        // Test reset functionality
        contextChannel.reset();
        assertEquals(0, contextChannel.getCategoryCount());
        
        // Channel properties should remain unchanged after reset
        assertEquals("context", contextChannel.getChannelName());
        assertEquals(0.5, contextChannel.getVigilance(), 0.001);
    }
    
    @Test
    @DisplayName("Should have neighbourhood radius property")
    void testNeighbourhoodRadius() {
        // Test default neighbourhood radius
        assertEquals(0.2, contextChannel.getNeighbourhoodRadius(), 0.001);
        
        // Test custom neighbourhood radius
        var config = ChannelConfig.builder()
            .channelName("test_context")
            .vigilance(0.6)
            .build();
        var customChannel = new ContextChannel(config, 0.5);
        assertEquals(0.5, customChannel.getNeighbourhoodRadius(), 0.001);
    }
    
    @Test
    @DisplayName("Should get neighbours for categories")
    void testNeighbourRetrieval() {
        // Test getting neighbours for a non-existent category
        var neighbours = contextChannel.getNeighbours(999);
        assertNotNull(neighbours);
        assertTrue(neighbours.isEmpty()); // Should be empty for non-existent category
    }
    
    @Test
    @DisplayName("Should handle configuration builder properly")
    void testConfigurationBuilder() {
        var config = ChannelConfig.builder()
            .channelName("test_context")
            .vigilance(0.8)
            .learningRate(0.4)
            .maxTokensPerInput(200)
            .build();
            
        assertNotNull(config);
        
        var channel = new ContextChannel(config, 0.3);
        assertEquals("test_context", channel.getChannelName());
        assertEquals(0.8, channel.getVigilance(), 0.001);
        assertEquals(0.3, channel.getNeighbourhoodRadius(), 0.001);
    }
}