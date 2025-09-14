package com.hellblazer.art.hartcq.templates;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.*;

public class DebugTemplateTest {

    @Test
    void debugTemplateRepository() {
        var repo = new TemplateRepository();
        var templates = repo.getAllTemplates();

        System.out.println("Total templates: " + templates.size());

        // Check specific categories
        for (var template : templates) {
            System.out.println("Template ID: " + template.id() + ", Category: " + template.category());
        }

        assertThat(templates).isNotEmpty();
    }

    @Test
    void debugTemplateManager() {
        var manager = new TemplateManager();

        // Try a simple input
        var result = manager.processInput("Hello", java.util.Map.of());

        System.out.println("Result successful: " + result.successful());
        if (result.successful()) {
            System.out.println("Template: " + result.template().id());
            System.out.println("Output: " + result.output());
        } else {
            System.out.println("Errors: " + result.errors());
        }

        // Check if repository has templates
        var stats = manager.getProcessingStats();
        System.out.println("Template count: " + stats.templateCount());

        assertThat(stats.templateCount()).isGreaterThan(0);
    }
}