package com.hellblazer.art.hartcq.spatial;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Library of predefined templates for HART-CQ text generation.
 * Contains 25+ templates for various text patterns to ensure bounded generation.
 * CRITICAL: This prevents hallucination by limiting output to predefined structures.
 */
public class TemplateLibrary {
    
    private final Map<String, Template> templates;
    private final Map<Template.TemplateType, List<Template>> templatesByType;
    
    public TemplateLibrary() {
        this.templates = new ConcurrentHashMap<>();
        this.templatesByType = new ConcurrentHashMap<>();
        initializeTemplates();
    }
    
    /**
     * Initializes the library with 25+ predefined templates.
     */
    private void initializeTemplates() {
        // Basic statement templates
        addTemplate(new Template("STMT_001", "Simple Subject-Verb-Object",
            "The {subject:NOUN} {verb:VERB} the {object:NOUN}.",
            Template.TemplateType.STATEMENT));
            
        addTemplate(new Template("STMT_002", "Subject-Verb with Adjective",
            "The {adjective:ADJECTIVE} {subject:NOUN} {verb:VERB}.",
            Template.TemplateType.STATEMENT));
            
        addTemplate(new Template("STMT_003", "Complex Statement",
            "The {adj1:ADJECTIVE} {subj:NOUN} {verb:VERB} {prep:PREPOSITION} the {adj2:ADJECTIVE?} {obj:NOUN}.",
            Template.TemplateType.STATEMENT));
            
        // Question templates
        addTemplate(new Template("QUES_001", "Simple What Question",
            "What {verb:VERB} the {noun:NOUN}?",
            Template.TemplateType.QUESTION));
            
        addTemplate(new Template("QUES_002", "Who Question",
            "Who {verb:VERB} the {object:NOUN}?",
            Template.TemplateType.QUESTION));
            
        addTemplate(new Template("QUES_003", "Where Question",
            "Where {aux:VERB} the {subject:NOUN} {verb:VERB}?",
            Template.TemplateType.QUESTION));
            
        addTemplate(new Template("QUES_004", "When Question",
            "When {aux:VERB} the {subject:NOUN} {verb:VERB} the {object:NOUN}?",
            Template.TemplateType.QUESTION));
            
        // Command templates
        addTemplate(new Template("CMD_001", "Simple Command",
            "{verb:VERB} the {object:NOUN}.",
            Template.TemplateType.COMMAND));
            
        addTemplate(new Template("CMD_002", "Polite Command",
            "Please {verb:VERB} the {object:NOUN}.",
            Template.TemplateType.COMMAND));
            
        addTemplate(new Template("CMD_003", "Complex Command",
            "{verb1:VERB} the {object1:NOUN} and {verb2:VERB} the {object2:NOUN}.",
            Template.TemplateType.COMMAND));
            
        // Description templates
        addTemplate(new Template("DESC_001", "Simple Description",
            "It is {adjective:ADJECTIVE}.",
            Template.TemplateType.DESCRIPTION));
            
        addTemplate(new Template("DESC_002", "Object Description",
            "The {noun:NOUN} is {adj1:ADJECTIVE} and {adj2:ADJECTIVE}.",
            Template.TemplateType.DESCRIPTION));
            
        addTemplate(new Template("DESC_003", "Comparative Description",
            "The {noun1:NOUN} is more {adjective:ADJECTIVE} than the {noun2:NOUN}.",
            Template.TemplateType.DESCRIPTION));
            
        // Narrative templates
        addTemplate(new Template("NARR_001", "Simple Past Narrative",
            "Yesterday, the {subject:NOUN} {verb:VERB} the {object:NOUN}.",
            Template.TemplateType.NARRATIVE));
            
        addTemplate(new Template("NARR_002", "Sequential Narrative",
            "First, the {subj:NOUN} {verb1:VERB}. Then, it {verb2:VERB} the {obj:NOUN}.",
            Template.TemplateType.NARRATIVE));
            
        addTemplate(new Template("NARR_003", "Causal Narrative",
            "Because the {subj1:NOUN} {verb1:VERB}, the {subj2:NOUN} {verb2:VERB}.",
            Template.TemplateType.NARRATIVE));
            
        // Dialogue templates
        addTemplate(new Template("DIAL_001", "Simple Dialogue",
            "\"{greeting:ANY},\" said {speaker:PROPER_NOUN}.",
            Template.TemplateType.DIALOGUE));
            
        addTemplate(new Template("DIAL_002", "Question Dialogue",
            "\"{question:ANY}?\" asked {speaker:PROPER_NOUN}.",
            Template.TemplateType.DIALOGUE));
            
        addTemplate(new Template("DIAL_003", "Response Dialogue",
            "\"{response:ANY},\" replied {speaker:PROPER_NOUN}.",
            Template.TemplateType.DIALOGUE));
            
        // Technical templates
        addTemplate(new Template("TECH_001", "Definition Template",
            "A {term:NOUN} is a {category:NOUN} that {definition:ANY}.",
            Template.TemplateType.TECHNICAL));
            
        addTemplate(new Template("TECH_002", "Process Template",
            "To {goal:VERB}, first {step1:VERB}, then {step2:VERB}.",
            Template.TemplateType.TECHNICAL));
            
        addTemplate(new Template("TECH_003", "Specification Template",
            "The {component:NOUN} has {number:NUMBER} {units:NOUN}.",
            Template.TemplateType.TECHNICAL));
            
        addTemplate(new Template("TECH_004", "Comparison Template",
            "While {option1:NOUN} {verb1:VERB}, {option2:NOUN} {verb2:VERB}.",
            Template.TemplateType.TECHNICAL));
            
        // Additional complex templates
        addTemplate(new Template("CMPLX_001", "Conditional Statement",
            "If the {condition:NOUN} {verb:VERB}, then the {result:NOUN} will {action:VERB}.",
            Template.TemplateType.STATEMENT));
            
        addTemplate(new Template("CMPLX_002", "List Template",
            "The {category:NOUN} includes {item1:NOUN}, {item2:NOUN}, and {item3:NOUN}.",
            Template.TemplateType.DESCRIPTION));
            
        addTemplate(new Template("CMPLX_003", "Temporal Sequence",
            "Before the {event1:NOUN} {verb1:VERB}, the {event2:NOUN} must {verb2:VERB}.",
            Template.TemplateType.NARRATIVE));
            
        addTemplate(new Template("CMPLX_004", "Spatial Relation",
            "The {object1:NOUN} is {preposition:PREPOSITION} the {object2:NOUN}.",
            Template.TemplateType.DESCRIPTION));
            
        addTemplate(new Template("CMPLX_005", "Purpose Statement",
            "The {subject:NOUN} {verb:VERB} in order to {purpose:VERB} the {goal:NOUN}.",
            Template.TemplateType.STATEMENT));
    }
    
    private void addTemplate(Template template) {
        templates.put(template.getId(), template);
        templatesByType.computeIfAbsent(template.getType(), k -> new ArrayList<>()).add(template);
    }
    
    /**
     * Gets a template by ID.
     */
    public Template getTemplate(String id) {
        return templates.get(id);
    }
    
    /**
     * Gets all templates of a specific type.
     */
    public List<Template> getTemplatesByType(Template.TemplateType type) {
        return templatesByType.getOrDefault(type, Collections.emptyList());
    }
    
    /**
     * Gets a random template of a specific type.
     */
    public Template getRandomTemplate(Template.TemplateType type) {
        var templates = getTemplatesByType(type);
        if (templates.isEmpty()) {
            return null;
        }
        return templates.get(new Random().nextInt(templates.size()));
    }
    
    /**
     * Gets all available templates.
     */
    public Collection<Template> getAllTemplates() {
        return templates.values();
    }
    
    /**
     * Gets the total number of templates.
     */
    public int getTemplateCount() {
        return templates.size();
    }
    
    /**
     * Validates that the library has at least 25 templates.
     */
    public boolean validateMinimumTemplates() {
        return templates.size() >= 25;
    }
}