package com.hellblazer.art.hartcq.core;

import com.hellblazer.art.hartcq.Token;
import com.hellblazer.art.hartcq.core.channels.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Clock;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * Multi-channel processor that runs 6 parallel channels for HART-CQ.
 * Each channel processes the token window independently and in parallel.
 */
public class MultiChannelProcessor {
    private static final Logger logger = LoggerFactory.getLogger(MultiChannelProcessor.class);
    
    private final List<Channel> channels;
    private final ExecutorService executor;
    private final int totalOutputDimension;
    private final Clock clock;

    public MultiChannelProcessor() {
        this(Runtime.getRuntime().availableProcessors(), Clock.systemUTC());
    }

    public MultiChannelProcessor(int threadPoolSize) {
        this(threadPoolSize, Clock.systemUTC());
    }

    public MultiChannelProcessor(int threadPoolSize, Clock clock) {
        this.channels = new ArrayList<>();
        this.executor = Executors.newFixedThreadPool(threadPoolSize);
        this.clock = clock;

        // Initialize all 6 channels
        initializeChannels();

        // Calculate total output dimension
        this.totalOutputDimension = channels.stream()
            .mapToInt(Channel::getOutputDimension)
            .sum();

        logger.info("Initialized MultiChannelProcessor with {} channels, total dimension: {}",
                   channels.size(), totalOutputDimension);
    }

    private void initializeChannels() {
        // CRITICAL: Positional encoding channel - provides position awareness using sinusoidal encoding
        channels.add(new PositionalChannel());

        // CRITICAL: Word channel for COMPREHENSION ONLY - Word2Vec integration, never for generation
        channels.add(new WordChannel());

        // Context tracking channel - maintains historical information across windows (last 5 windows)
        channels.add(new ContextChannel());

        // Structural analysis channel - analyzes sentence structure and grammatical patterns
        channels.add(new StructuralChannel());

        // Semantic meaning channel - extracts semantic features and topic coherence
        channels.add(new SemanticChannel());

        // Temporal sequence channel - tracks time-based patterns and sequence ordering
        channels.add(new TemporalChannel(clock));
    }
    
    /**
     * Processes a token window through all channels in parallel.
     * @param tokens The token window to process
     * @return Combined feature vector from all channels
     */
    public float[] processWindow(Token[] tokens) {
        var futures = new ArrayList<Future<ChannelResult>>();
        
        // Submit all channels for parallel processing
        for (Channel channel : channels) {
            futures.add(executor.submit(() -> {
                var startTime = System.nanoTime();
                var result = channel.processWindow(tokens);
                var duration = System.nanoTime() - startTime;
                return new ChannelResult(channel.getName(), result, duration);
            }));
        }
        
        // Collect results
        var combinedOutput = new float[totalOutputDimension];
        int offset = 0;
        
        for (int i = 0; i < futures.size(); i++) {
            try {
                var result = futures.get(i).get(1, TimeUnit.SECONDS);
                var channelOutput = result.output();
                
                // Copy channel output to combined vector
                System.arraycopy(channelOutput, 0, combinedOutput, offset, channelOutput.length);
                offset += channelOutput.length;
                
                if (logger.isDebugEnabled()) {
                    logger.debug("Channel {} processed in {} ns", 
                               result.channelName(), result.processingTimeNanos());
                }
            } catch (Exception e) {
                logger.error("Error processing channel {}", channels.get(i).getName(), e);
                // Fill with zeros on error
                offset += channels.get(i).getOutputDimension();
            }
        }
        
        return combinedOutput;
    }
    
    /**
     * Processes multiple windows in batch.
     * @param windowBatch List of token windows
     * @return List of combined feature vectors
     */
    public List<float[]> processBatch(List<Token[]> windowBatch) {
        var results = new ArrayList<float[]>();
        
        for (Token[] window : windowBatch) {
            results.add(processWindow(window));
        }
        
        return results;
    }
    
    /**
     * Resets all channels to their initial state.
     */
    public void resetChannels() {
        for (Channel channel : channels) {
            channel.reset();
        }
        logger.info("All channels reset");
    }
    
    /**
     * Gets information about all channels.
     * @return List of channel information
     */
    public List<ChannelInfo> getChannelInfo() {
        var info = new ArrayList<ChannelInfo>();
        
        for (Channel channel : channels) {
            info.add(new ChannelInfo(
                channel.getName(),
                channel.getOutputDimension(),
                channel.isDeterministic()
            ));
        }
        
        return info;
    }
    
    /**
     * Gets the total output dimension across all channels.
     * @return Total dimension
     */
    public int getTotalOutputDimension() {
        return totalOutputDimension;
    }
    
    /**
     * Shuts down the processor and releases resources.
     */
    public void shutdown() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
    
    /**
     * Result from a single channel processing.
     */
    private record ChannelResult(String channelName, float[] output, long processingTimeNanos) {}
    
    /**
     * Information about a channel.
     */
    public record ChannelInfo(String name, int outputDimension, boolean isDeterministic) {}
}