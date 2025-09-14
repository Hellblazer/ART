package com.hellblazer.art.hartcq.templates;

import org.junit.jupiter.api.Test;
import java.util.*;

public class DebugTemplateMatching {

    @Test
    void debugTemplateMatching() {
        var templateManager = new TemplateManager();

        // Test inputs from HARTCQValidation
        var testInputs = List.of(
            "Hello world", "What is this?", "How are you?",
            "Please help", "Thank you", "Goodbye",
            "What is", "How does", "Why do",
            "The cat is", "A dog runs", "Birds fly",
            "I understand", "You said", "They believe",
            "Next we", "Then you", "After that"
        );

        // Provide all variables
        var variables = new HashMap<String, String>();
        variables.put("SUBJECT", "User");
        variables.put("TOPIC", "testing");
        variables.put("USER", "Tester");
        variables.put("SYSTEM", "HART-CQ");
        variables.put("PROPERTY", "status");
        variables.put("ENTITY", "system");
        variables.put("ACTION", "process");
        variables.put("OBJECT", "data");
        variables.put("PARAMETER", "default");
        variables.put("REASON", "testing");
        variables.put("LOCATION", "here");
        variables.put("TIME", "now");
        variables.put("STATE", "active");
        variables.put("ERROR", "none");
        variables.put("REQUEST", "help");
        variables.put("CLARIFICATION", "details");
        variables.put("RECOMMENDATION", "proceed");
        variables.put("MESSAGE", "understood");
        variables.put("RESULT", "success");
        variables.put("STEP", "next");

        System.out.println("Total templates in repository: " + templateManager.getTemplateCount());
        System.out.println("Available categories: " + templateManager.getAvailableCategories());

        var matchedTemplates = new HashSet<String>();
        var unmatchedInputs = new ArrayList<String>();

        for (var input : testInputs) {
            var result = templateManager.processInput(input, variables);
            if (result.successful() && result.template() != null) {
                matchedTemplates.add(result.template().id());
                System.out.println("Input: '" + input + "' -> Template: " + result.template().id());
            } else {
                unmatchedInputs.add(input);
                System.out.println("Input: '" + input + "' -> NO MATCH");
            }
        }

        System.out.println("\n=== SUMMARY ===");
        System.out.println("Total unique templates matched: " + matchedTemplates.size());
        System.out.println("Matched template IDs: " + matchedTemplates);
        System.out.println("Unmatched inputs: " + unmatchedInputs);

        // Show all available templates
        System.out.println("\n=== ALL AVAILABLE TEMPLATES ===");
        for (var category : templateManager.getAvailableCategories()) {
            var templates = templateManager.getTemplatesInCategory(category);
            System.out.println("Category " + category + ": " + templates.size() + " templates");
            for (var template : templates) {
                System.out.println("  - " + template.id());
            }
        }
    }
}