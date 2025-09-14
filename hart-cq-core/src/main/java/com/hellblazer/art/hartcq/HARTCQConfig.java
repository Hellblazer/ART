package com.hellblazer.art.hartcq;

import java.util.Objects;
import java.util.concurrent.TimeUnit;

/**
 * Configuration class for HART-CQ system parameters.
 * Contains window size configuration, channel configurations,
 * performance tuning parameters, and template settings.
 */
public class HARTCQConfig {
    
    // Window Configuration
    private final int windowSize;
    private final int windowOverlap;
    private final boolean enableSlidingWindow;
    
    // Channel Configuration
    private final ChannelConfig channelConfig;
    
    // Performance Tuning
    private final PerformanceConfig performanceConfig;
    
    // Template Settings
    private final TemplateConfig templateConfig;
    
    /**
     * Creates HARTCQConfig with default settings.
     */
    public HARTCQConfig() {
        this(new Builder());
    }
    
    private HARTCQConfig(Builder builder) {
        this.windowSize = builder.windowSize;
        this.windowOverlap = builder.windowOverlap;
        this.enableSlidingWindow = builder.enableSlidingWindow;

        // Initialize nested configs if not already set
        this.channelConfig = builder.channelConfig != null ? builder.channelConfig :
            new ChannelConfig(builder);

        this.performanceConfig = builder.performanceConfig != null ? builder.performanceConfig :
            new PerformanceConfig(builder);

        this.templateConfig = builder.templateConfig != null ? builder.templateConfig :
            new TemplateConfig(builder);
    }
    
    /**
     * Gets the window size for token processing.
     * @return window size (typically 20 tokens)
     */
    public int getWindowSize() {
        return windowSize;
    }
    
    /**
     * Gets the window overlap size.
     * @return number of tokens that overlap between consecutive windows
     */
    public int getWindowOverlap() {
        return windowOverlap;
    }
    
    /**
     * Checks if sliding window mode is enabled.
     * @return true if sliding window is enabled
     */
    public boolean isEnableSlidingWindow() {
        return enableSlidingWindow;
    }
    
    /**
     * Gets the channel configuration.
     * @return channel configuration
     */
    public ChannelConfig getChannelConfig() {
        return channelConfig;
    }
    
    /**
     * Gets the performance configuration.
     * @return performance configuration
     */
    public PerformanceConfig getPerformanceConfig() {
        return performanceConfig;
    }
    
    /**
     * Gets the template configuration.
     * @return template configuration
     */
    public TemplateConfig getTemplateConfig() {
        return templateConfig;
    }
    
    /**
     * Channel configuration for HART-CQ processing.
     */
    public static class ChannelConfig {
        private final boolean enablePositionalChannel;
        private final boolean enableSyntaxChannel;
        private final boolean enableSemanticChannel;
        private final double channelWeightPositional;
        private final double channelWeightSyntax;
        private final double channelWeightSemantic;
        private final int maxChannelActivations;
        
        private ChannelConfig(Builder builder) {
            this.enablePositionalChannel = builder.enablePositionalChannel;
            this.enableSyntaxChannel = builder.enableSyntaxChannel;
            this.enableSemanticChannel = builder.enableSemanticChannel;
            this.channelWeightPositional = builder.channelWeightPositional;
            this.channelWeightSyntax = builder.channelWeightSyntax;
            this.channelWeightSemantic = builder.channelWeightSemantic;
            this.maxChannelActivations = builder.maxChannelActivations;
        }
        
        public boolean isEnablePositionalChannel() { return enablePositionalChannel; }
        public boolean isEnableSyntaxChannel() { return enableSyntaxChannel; }
        public boolean isEnableSemanticChannel() { return enableSemanticChannel; }
        public double getChannelWeightPositional() { return channelWeightPositional; }
        public double getChannelWeightSyntax() { return channelWeightSyntax; }
        public double getChannelWeightSemantic() { return channelWeightSemantic; }
        public int getMaxChannelActivations() { return maxChannelActivations; }
    }
    
    /**
     * Performance configuration for optimization and tuning.
     */
    public static class PerformanceConfig {
        private final int maxConcurrentProcessors;
        private final int queueCapacity;
        private final long processingTimeoutMs;
        private final int targetThroughputSentencesPerSecond;
        private final boolean enablePerformanceMonitoring;
        private final int performanceReportIntervalSeconds;
        private final boolean enableAdaptiveThrottling;
        private final double cpuUsageThreshold;
        private final double memoryUsageThreshold;
        
        private PerformanceConfig(Builder builder) {
            this.maxConcurrentProcessors = builder.maxConcurrentProcessors;
            this.queueCapacity = builder.queueCapacity;
            this.processingTimeoutMs = builder.processingTimeoutMs;
            this.targetThroughputSentencesPerSecond = builder.targetThroughputSentencesPerSecond;
            this.enablePerformanceMonitoring = builder.enablePerformanceMonitoring;
            this.performanceReportIntervalSeconds = builder.performanceReportIntervalSeconds;
            this.enableAdaptiveThrottling = builder.enableAdaptiveThrottling;
            this.cpuUsageThreshold = builder.cpuUsageThreshold;
            this.memoryUsageThreshold = builder.memoryUsageThreshold;
        }
        
        public int getMaxConcurrentProcessors() { return maxConcurrentProcessors; }
        public int getQueueCapacity() { return queueCapacity; }
        public long getProcessingTimeoutMs() { return processingTimeoutMs; }
        public int getTargetThroughputSentencesPerSecond() { return targetThroughputSentencesPerSecond; }
        public boolean isEnablePerformanceMonitoring() { return enablePerformanceMonitoring; }
        public int getPerformanceReportIntervalSeconds() { return performanceReportIntervalSeconds; }
        public boolean isEnableAdaptiveThrottling() { return enableAdaptiveThrottling; }
        public double getCpuUsageThreshold() { return cpuUsageThreshold; }
        public double getMemoryUsageThreshold() { return memoryUsageThreshold; }
    }
    
    /**
     * Template configuration for pattern matching and recognition.
     */
    public static class TemplateConfig {
        private final double vigilanceParameter;
        private final double learningRate;
        private final int maxTemplates;
        private final double matchThreshold;
        private final boolean enableTemplateDecay;
        private final double decayRate;
        private final int templateLifetimeSeconds;
        private final boolean enableDynamicTemplates;
        
        private TemplateConfig(Builder builder) {
            this.vigilanceParameter = builder.vigilanceParameter;
            this.learningRate = builder.learningRate;
            this.maxTemplates = builder.maxTemplates;
            this.matchThreshold = builder.matchThreshold;
            this.enableTemplateDecay = builder.enableTemplateDecay;
            this.decayRate = builder.decayRate;
            this.templateLifetimeSeconds = builder.templateLifetimeSeconds;
            this.enableDynamicTemplates = builder.enableDynamicTemplates;
        }
        
        public double getVigilanceParameter() { return vigilanceParameter; }
        public double getLearningRate() { return learningRate; }
        public int getMaxTemplates() { return maxTemplates; }
        public double getMatchThreshold() { return matchThreshold; }
        public boolean isEnableTemplateDecay() { return enableTemplateDecay; }
        public double getDecayRate() { return decayRate; }
        public int getTemplateLifetimeSeconds() { return templateLifetimeSeconds; }
        public boolean isEnableDynamicTemplates() { return enableDynamicTemplates; }
    }
    
    /**
     * Builder for creating HARTCQConfig instances.
     */
    public static class Builder {
        // Window Configuration defaults
        private int windowSize = 20;
        private int windowOverlap = 5;
        private boolean enableSlidingWindow = true;
        
        // Channel Configuration defaults
        private boolean enablePositionalChannel = true;
        private boolean enableSyntaxChannel = true;
        private boolean enableSemanticChannel = true;
        private double channelWeightPositional = 0.33;
        private double channelWeightSyntax = 0.33;
        private double channelWeightSemantic = 0.34;
        private int maxChannelActivations = 10;
        
        // Performance Configuration defaults
        private int maxConcurrentProcessors = Runtime.getRuntime().availableProcessors();
        private int queueCapacity = 1000;
        private long processingTimeoutMs = 5000;
        private int targetThroughputSentencesPerSecond = 100;
        private boolean enablePerformanceMonitoring = true;
        private int performanceReportIntervalSeconds = 30;
        private boolean enableAdaptiveThrottling = true;
        private double cpuUsageThreshold = 0.85;
        private double memoryUsageThreshold = 0.90;
        
        // Template Configuration defaults
        private double vigilanceParameter = 0.7;
        private double learningRate = 0.1;
        private int maxTemplates = 100;
        private double matchThreshold = 0.6;
        private boolean enableTemplateDecay = true;
        private double decayRate = 0.01;
        private int templateLifetimeSeconds = 3600; // 1 hour
        private boolean enableDynamicTemplates = true;
        
        private ChannelConfig channelConfig;
        private PerformanceConfig performanceConfig;
        private TemplateConfig templateConfig;
        
        public Builder windowSize(int windowSize) {
            this.windowSize = Math.max(1, windowSize);
            return this;
        }
        
        public Builder windowOverlap(int windowOverlap) {
            this.windowOverlap = Math.max(0, windowOverlap);
            return this;
        }
        
        public Builder enableSlidingWindow(boolean enable) {
            this.enableSlidingWindow = enable;
            return this;
        }
        
        public Builder enablePositionalChannel(boolean enable) {
            this.enablePositionalChannel = enable;
            return this;
        }
        
        public Builder enableSyntaxChannel(boolean enable) {
            this.enableSyntaxChannel = enable;
            return this;
        }
        
        public Builder enableSemanticChannel(boolean enable) {
            this.enableSemanticChannel = enable;
            return this;
        }
        
        public Builder channelWeights(double positional, double syntax, double semantic) {
            var total = positional + syntax + semantic;
            if (total > 0) {
                this.channelWeightPositional = positional / total;
                this.channelWeightSyntax = syntax / total;
                this.channelWeightSemantic = semantic / total;
            }
            return this;
        }
        
        public Builder maxConcurrentProcessors(int processors) {
            this.maxConcurrentProcessors = Math.max(1, processors);
            return this;
        }
        
        public Builder queueCapacity(int capacity) {
            this.queueCapacity = Math.max(1, capacity);
            return this;
        }
        
        public Builder processingTimeout(long timeout, TimeUnit unit) {
            this.processingTimeoutMs = unit.toMillis(timeout);
            return this;
        }
        
        public Builder targetThroughput(int sentencesPerSecond) {
            this.targetThroughputSentencesPerSecond = Math.max(1, sentencesPerSecond);
            return this;
        }
        
        public Builder vigilanceParameter(double vigilance) {
            this.vigilanceParameter = Math.max(0.0, Math.min(1.0, vigilance));
            return this;
        }
        
        public Builder learningRate(double rate) {
            this.learningRate = Math.max(0.0, Math.min(1.0, rate));
            return this;
        }
        
        public Builder maxTemplates(int maxTemplates) {
            this.maxTemplates = Math.max(1, maxTemplates);
            return this;
        }
        
        public Builder enablePerformanceMonitoring(boolean enable) {
            this.enablePerformanceMonitoring = enable;
            return this;
        }

        public Builder enableAdaptiveThrottling(boolean enable) {
            this.enableAdaptiveThrottling = enable;
            return this;
        }

        public Builder enableTemplateDecay(boolean enable) {
            this.enableTemplateDecay = enable;
            return this;
        }

        public HARTCQConfig build() {
            // Build nested configurations
            this.channelConfig = new ChannelConfig(this);
            this.performanceConfig = new PerformanceConfig(this);
            this.templateConfig = new TemplateConfig(this);
            
            return new HARTCQConfig(this);
        }
    }
    
    /**
     * Creates a configuration optimized for high throughput processing.
     * @return high-performance configuration
     */
    public static HARTCQConfig forHighThroughput() {
        return new Builder()
            .maxConcurrentProcessors(Runtime.getRuntime().availableProcessors() * 2)
            .queueCapacity(5000)
            .targetThroughput(200)
            .processingTimeout(1000, TimeUnit.MILLISECONDS)
            .enableAdaptiveThrottling(true)
            .build();
    }
    
    /**
     * Creates a configuration optimized for low latency processing.
     * @return low-latency configuration
     */
    public static HARTCQConfig forLowLatency() {
        return new Builder()
            .maxConcurrentProcessors(Runtime.getRuntime().availableProcessors())
            .queueCapacity(100)
            .targetThroughput(50)
            .processingTimeout(500, TimeUnit.MILLISECONDS)
            .windowSize(10)
            .build();
    }
    
    /**
     * Creates a configuration for memory-constrained environments.
     * @return memory-optimized configuration
     */
    public static HARTCQConfig forMemoryConstrained() {
        return new Builder()
            .maxConcurrentProcessors(2)
            .queueCapacity(50)
            .maxTemplates(25)
            .enableTemplateDecay(true)
            .enablePerformanceMonitoring(false)
            .build();
    }
    
    @Override
    public String toString() {
        return String.format("HARTCQConfig[windowSize=%d, processors=%d, throughput=%d/sec, vigilance=%.2f]",
                           windowSize, 
                           performanceConfig.getMaxConcurrentProcessors(),
                           performanceConfig.getTargetThroughputSentencesPerSecond(),
                           templateConfig.getVigilanceParameter());
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        var config = (HARTCQConfig) o;
        return windowSize == config.windowSize &&
               windowOverlap == config.windowOverlap &&
               enableSlidingWindow == config.enableSlidingWindow &&
               Objects.equals(channelConfig, config.channelConfig) &&
               Objects.equals(performanceConfig, config.performanceConfig) &&
               Objects.equals(templateConfig, config.templateConfig);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(windowSize, windowOverlap, enableSlidingWindow, 
                          channelConfig, performanceConfig, templateConfig);
    }
}