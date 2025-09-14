package com.hellblazer.art.hartcq.spatial;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TemplateSystemTest {
    
    private TemplateLibrary library;
    
    @BeforeEach
    void setUp() {
        library = new TemplateLibrary();
    }
    
    @Test
    void testMinimumTemplateCount() {
        // Verify we have at least 25 templates as required
        assertThat(library.validateMinimumTemplates()).isTrue();
        assertThat(library.getTemplateCount()).isGreaterThanOrEqualTo(25);
    }
    
    @Test
    void testTemplateGeneration() {
        var template = library.getTemplate("STMT_001");
        assertThat(template).isNotNull();
        
        Map<String, String> slots = new HashMap<>();
        slots.put("subject", "cat");
        slots.put("verb", "chased");
        slots.put("object", "mouse");
        
        var result = template.generate(slots);
        assertThat(result).isEqualTo("The cat chased the mouse.");
    }
    
    @Test
    void testOptionalSlots() {
        var template = library.getTemplate("STMT_003");
        assertThat(template).isNotNull();
        
        Map<String, String> slots = new HashMap<>();
        slots.put("adj1", "quick");
        slots.put("subj", "fox");
        slots.put("verb", "jumped");
        slots.put("prep", "over");
        slots.put("obj", "fence");
        // adj2 is optional, not provided
        
        var result = template.generate(slots);
        assertThat(result).contains("quick fox jumped over");
    }
    
    @Test
    void testRequiredSlotValidation() {
        var template = library.getTemplate("STMT_001");
        
        Map<String, String> slots = new HashMap<>();
        slots.put("subject", "cat");
        // Missing required slots: verb and object
        
        assertThatThrownBy(() -> template.generate(slots))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Required slot");
    }
    
    @Test
    void testTemplatesByType() {
        var statements = library.getTemplatesByType(Template.TemplateType.STATEMENT);
        assertThat(statements).isNotEmpty();
        
        var questions = library.getTemplatesByType(Template.TemplateType.QUESTION);
        assertThat(questions).isNotEmpty();
        
        var commands = library.getTemplatesByType(Template.TemplateType.COMMAND);
        assertThat(commands).isNotEmpty();
        
        var descriptions = library.getTemplatesByType(Template.TemplateType.DESCRIPTION);
        assertThat(descriptions).isNotEmpty();
        
        var narratives = library.getTemplatesByType(Template.TemplateType.NARRATIVE);
        assertThat(narratives).isNotEmpty();
        
        var dialogues = library.getTemplatesByType(Template.TemplateType.DIALOGUE);
        assertThat(dialogues).isNotEmpty();
        
        var technical = library.getTemplatesByType(Template.TemplateType.TECHNICAL);
        assertThat(technical).isNotEmpty();
    }
    
    @Test
    void testRandomTemplateSelection() {
        var randomStatement = library.getRandomTemplate(Template.TemplateType.STATEMENT);
        assertThat(randomStatement).isNotNull();
        assertThat(randomStatement.getType()).isEqualTo(Template.TemplateType.STATEMENT);
    }
    
    @Test
    void testComplexTemplateGeneration() {
        var template = library.getTemplate("CMPLX_001");
        assertThat(template).isNotNull();
        
        Map<String, String> slots = new HashMap<>();
        slots.put("condition", "temperature");
        slots.put("verb", "rises");
        slots.put("result", "ice");
        slots.put("action", "melt");
        
        var result = template.generate(slots);
        assertThat(result).isEqualTo("If the temperature rises, then the ice will melt.");
    }
    
    @Test
    void testDeterministicGeneration() {
        // Test that same input produces same output (deterministic)
        var template = library.getTemplate("TECH_003");
        
        Map<String, String> slots = new HashMap<>();
        slots.put("component", "processor");
        slots.put("number", "8");
        slots.put("units", "cores");
        
        var result1 = template.generate(slots);
        var result2 = template.generate(slots);
        
        assertThat(result1).isEqualTo(result2);
        assertThat(result1).isEqualTo("The processor has 8 cores.");
    }
}