package com.hellblazer.art.hartcq;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import com.hellblazer.art.hartcq.integration.ProcessingResult;

/**
 * Task for processing in competitive queue.
 */
public class ProcessingTask {
    private final String input;
    private final Map<String, String> variables;
    private final CompletableFuture<ProcessingResult> resultFuture;
    private final long creationTime;

    public ProcessingTask(String input, Map<String, String> variables) {
        this.input = input;
        this.variables = variables;
        this.resultFuture = new CompletableFuture<>();
        this.creationTime = System.nanoTime();
    }

    public String getInput() {
        return input;
    }

    public Map<String, String> getVariables() {
        return variables;
    }

    public CompletableFuture<ProcessingResult> getResultFuture() {
        return resultFuture;
    }

    public long getCreationTime() {
        return creationTime;
    }

    public void complete(ProcessingResult result) {
        resultFuture.complete(result);
    }

    public void completeExceptionally(Throwable ex) {
        resultFuture.completeExceptionally(ex);
    }
}