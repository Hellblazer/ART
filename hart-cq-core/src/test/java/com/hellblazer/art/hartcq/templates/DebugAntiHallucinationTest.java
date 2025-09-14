package com.hellblazer.art.hartcq.templates;

import org.junit.jupiter.api.Test;
import java.util.Map;

public class DebugAntiHallucinationTest {

    @Test
    void debugGetAllPossibleOutputs() {
        var templateManager = new TemplateManager();
        var input = "Help me";
        var variables = Map.<String, String>of("context", "general");

        System.out.println("Testing input: '" + input + "' with variables: " + variables);

        // Try to get all possible outputs
        var allResults = templateManager.getAllPossibleOutputs(input, variables);

        System.out.println("Number of results: " + allResults.size());

        for (var result : allResults) {
            System.out.println("Result: " + result);
            if (result.successful()) {
                System.out.println("  Template: " + result.template().id());
                System.out.println("  Output: " + result.output());
            } else {
                System.out.println("  Errors: " + result.errors());
            }
        }

        // Also test processInput
        var processResult = templateManager.processInput(input, variables);
        System.out.println("\nProcessInput result: " + processResult);
    }
}