package com.hellblazer.art.hartcq.integration;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;

/**
 * Result of HART-CQ text processing operation.
 * Contains the processed output along with metadata about the processing.
 */
public class ProcessingResult {
    private final String input;
    private final String output;
    private final boolean successful;
    private final String errorMessage;
    private final Duration processingTime;
    private final Instant timestamp;
    private final Map<String, Object> metadata;
    private final double confidence;
    private final int tokensProcessed;

    private ProcessingResult(Builder builder) {
        this.input = builder.input;
        this.output = builder.output;
        this.successful = builder.successful;
        this.errorMessage = builder.errorMessage;
        this.processingTime = builder.processingTime;
        this.timestamp = builder.timestamp;
        this.metadata = Map.copyOf(builder.metadata);
        this.confidence = builder.confidence;
        this.tokensProcessed = builder.tokensProcessed;
    }

    public String getInput() { return input; }
    public String getOutput() { return output; }
    public boolean isSuccessful() { return successful; }
    public String getErrorMessage() { return errorMessage; }
    public Duration getProcessingTime() { return processingTime; }
    public Instant getTimestamp() { return timestamp; }
    public Map<String, Object> getMetadata() { return metadata; }
    public double getConfidence() { return confidence; }
    public int getTokensProcessed() { return tokensProcessed; }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String input;
        private String output;
        private boolean successful = true;
        private String errorMessage;
        private Duration processingTime = Duration.ZERO;
        private Instant timestamp = Instant.now();
        private Map<String, Object> metadata = Map.of();
        private double confidence = 1.0;
        private int tokensProcessed = 0;

        public Builder input(String input) {
            this.input = input;
            return this;
        }

        public Builder output(String output) {
            this.output = output;
            return this;
        }

        public Builder successful(boolean successful) {
            this.successful = successful;
            return this;
        }

        public Builder errorMessage(String errorMessage) {
            this.errorMessage = errorMessage;
            this.successful = false;
            return this;
        }

        public Builder processingTime(Duration processingTime) {
            this.processingTime = processingTime;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public Builder confidence(double confidence) {
            this.confidence = confidence;
            return this;
        }

        public Builder tokensProcessed(int tokensProcessed) {
            this.tokensProcessed = tokensProcessed;
            return this;
        }

        public ProcessingResult build() {
            return new ProcessingResult(this);
        }
    }

    @Override
    public String toString() {
        return "ProcessingResult{" +
                "input='" + input + '\'' +
                ", output='" + output + '\'' +
                ", successful=" + successful +
                ", processingTime=" + processingTime +
                ", confidence=" + confidence +
                ", tokensProcessed=" + tokensProcessed +
                '}';
    }
}