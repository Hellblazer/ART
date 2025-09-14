package com.hellblazer.art.hartcq.templates;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Repository containing all predefined templates for HART-CQ system.
 * Provides 25+ templates across 5 categories to prevent hallucination
 * by strictly bounding all possible outputs.
 * 
 * @author Claude Code
 */
public class TemplateRepository {
    
    private final Map<String, Template> templatesById;
    private final Map<String, List<Template>> templatesByCategory;
    
    /**
     * Create repository with all predefined templates
     */
    public TemplateRepository() {
        var allTemplates = createAllTemplates();
        
        this.templatesById = new ConcurrentHashMap<>();
        this.templatesByCategory = new ConcurrentHashMap<>();
        
        for (var template : allTemplates) {
            templatesById.put(template.id(), template);
            templatesByCategory
                .computeIfAbsent(template.category(), k -> new ArrayList<>())
                .add(template);
        }
        
        // Make category lists immutable
        templatesByCategory.replaceAll((category, templates) -> List.copyOf(templates));
    }
    
    /**
     * Get template by ID
     */
    public Optional<Template> getById(String id) {
        return Optional.ofNullable(templatesById.get(id));
    }
    
    /**
     * Get all templates in a category
     */
    public List<Template> getByCategory(String category) {
        return templatesByCategory.getOrDefault(category, List.of());
    }
    
    /**
     * Get all templates in a category
     */
    public List<Template> getByCategory(Template.Category category) {
        return getByCategory(category.getValue());
    }
    
    /**
     * Get all available templates
     */
    public Collection<Template> getAllTemplates() {
        return templatesById.values();
    }
    
    /**
     * Get all available categories
     */
    public Set<String> getAllCategories() {
        return templatesByCategory.keySet();
    }
    
    /**
     * Get template count
     */
    public int getTemplateCount() {
        return templatesById.size();
    }
    
    /**
     * Create all predefined templates - CRITICAL: NO HALLUCINATION
     * All outputs must be bounded by these templates only
     */
    private List<Template> createAllTemplates() {
        var templates = new ArrayList<Template>();

        // GREETING TEMPLATES (5 templates)
        templates.addAll(createGreetingTemplates());

        // QUESTION TEMPLATES (5 templates)
        templates.addAll(createQuestionTemplates());

        // STATEMENT TEMPLATES (5 templates)
        templates.addAll(createStatementTemplates());

        // RESPONSE TEMPLATES (5 templates)
        templates.addAll(createResponseTemplates());

        // TRANSITION TEMPLATES (5+ templates)
        templates.addAll(createTransitionTemplates());

        // FALLBACK TEMPLATE (1 template for unmatched inputs)
        templates.add(createFallbackTemplate());

        return templates;
    }
    
    /**
     * Create greeting templates
     */
    private List<Template> createGreetingTemplates() {
        return List.of(
            Template.builder()
                .id("greeting.hello.basic")
                .category(Template.Category.GREETING)
                .pattern("Hello, [SUBJECT]. How can I help you with [TOPIC]?")
                .addMatchingPattern("(?i)\\bhello\\b")
                .addMatchingPattern("(?i)\\bhi\\b")
                .addMatchingPattern("(?i)\\bgreet")
                .addMatchingPattern("(?i)help me")
                .addMatchingPattern("(?i)\\bhelp\\b")
                .baseConfidence(0.8)
                .metadata("formality", "casual")
                .build(),
                
            Template.builder()
                .id("greeting.welcome.formal")
                .category(Template.Category.GREETING)
                .pattern("Welcome to [SYSTEM], [USER]. I am ready to assist you.")
                .addMatchingPattern("(?i)\bwelcome")
                .addMatchingPattern("(?i)\bintroduce")
                .addMatchingPattern("(?i)\bgreetings")
                .baseConfidence(0.7)
                .metadata("formality", "formal")
                .build(),
                
            Template.builder()
                .id("greeting.good.time")
                .category(Template.Category.GREETING)
                .pattern("Good [TIME_OF_DAY], [NAME]. What brings you here today?")
                .addMatchingPattern("(?i)good morning")
                .addMatchingPattern("(?i)good afternoon")
                .addMatchingPattern("(?i)good evening")
                .addMatchingPattern("(?i)\bgood\b")
                .baseConfidence(0.6)
                .build(),
                
            Template.builder()
                .id("greeting.status.check")
                .category(Template.Category.GREETING)
                .pattern("Hello [USER]. The [SYSTEM] is [STATUS] and ready for [OPERATION].")
                .addMatchingPattern("(?i)\bstatus")
                .addMatchingPattern("(?i)\bready")
                .addMatchingPattern("(?i)\bavailable")
                .addMatchingPattern("(?i)\bstanding by")
                .baseConfidence(0.7)
                .build(),
                
            Template.builder()
                .id("greeting.return.user")
                .category(Template.Category.GREETING)
                .pattern("Welcome back, [USER]. Your last session was [DURATION] ago.")
                .addMatchingPattern("(?i)\bback")
                .addMatchingPattern("(?i)\breturn")
                .addMatchingPattern("(?i)\bagain")
                .addMatchingPattern("(?i)welcome back")
                .baseConfidence(0.8)
                .build()
        );
    }
    
    /**
     * Create question templates
     */
    private List<Template> createQuestionTemplates() {
        return List.of(
            Template.builder()
                .id("question.what.is")
                .category(Template.Category.QUESTION)
                .pattern("What is the [PROPERTY] of [ENTITY]?")
                .addMatchingPattern("(?i)^what is")
                .addMatchingPattern("(?i)^what's")
                .addMatchingPattern("(?i)\bwhat\b.*\bis\b")
                .baseConfidence(0.9)
                .build(),
                
            Template.builder()
                .id("question.how.to")
                .category(Template.Category.QUESTION)
                .pattern("How do I [ACTION] the [OBJECT] with [PARAMETER]?")
                .addMatchingPattern("(?i)^how do")
                .addMatchingPattern("(?i)^how can")
                .addMatchingPattern("(?i)^how to")
                .addMatchingPattern("(?i)^how does")
                .baseConfidence(0.8)
                .build(),
                
            Template.builder()
                .id("question.where.is")
                .category(Template.Category.QUESTION)
                .pattern("Where can I find the [ITEM] for [PURPOSE]?")
                .addMatchingPattern("(?i)^where is")
                .addMatchingPattern("(?i)^where can")
                .addMatchingPattern("(?i)\blocate")
                .addMatchingPattern("(?i)\bfind")
                .baseConfidence(0.7)
                .build(),
                
            Template.builder()
                .id("question.why.does")
                .category(Template.Category.QUESTION)
                .pattern("Why does [SUBJECT] behave as [BEHAVIOR] when [CONDITION]?")
                .addMatchingPattern("(?i)^why do")
                .addMatchingPattern("(?i)^why does")
                .addMatchingPattern("(?i)^why is")
                .addMatchingPattern("(?i)\breason\b")
                .baseConfidence(0.7)
                .build(),
                
            Template.builder()
                .id("question.when.will")
                .category(Template.Category.QUESTION)
                .pattern("When will [EVENT] occur for [CONTEXT]?")
                .addMatchingPattern("(?i)^when will")
                .addMatchingPattern("(?i)^when does")
                .addMatchingPattern("(?i)\btiming")
                .addMatchingPattern("(?i)\bschedule")
                .baseConfidence(0.6)
                .build()
        );
    }
    
    /**
     * Create statement templates
     */
    private List<Template> createStatementTemplates() {
        return List.of(
            Template.builder()
                .id("statement.entity.property")
                .category(Template.Category.STATEMENT)
                .pattern("The [ENTITY] is [ATTRIBUTE] and [SECONDARY_ATTRIBUTE].")
                .addMatchingPattern("(?i)^the .+ is")
                .addMatchingPattern("(?i)^a .+ is")
                .addMatchingPattern("(?i)^this .+ has")
                .addMatchingPattern("(?i)^the cat")
                .addMatchingPattern("(?i)^a dog")
                .addMatchingPattern("(?i)\bbirds\b")
                .baseConfidence(0.8)
                .build(),
                
            Template.builder()
                .id("statement.process.status")
                .category(Template.Category.STATEMENT)
                .pattern("The [PROCESS] has [STATUS] with [RESULT] at [TIMESTAMP].")
                .addMatchingPattern("(?i)\bprocess")
                .addMatchingPattern("(?i)\bcompleted")
                .addMatchingPattern("(?i)\bstatus")
                .addMatchingPattern("(?i)^the .+ has")
                .baseConfidence(0.7)
                .build(),
                
            Template.builder()
                .id("statement.system.configuration")
                .category(Template.Category.STATEMENT)
                .pattern("System [COMPONENT] is configured with [SETTING] set to [VALUE].")
                .addMatchingPattern("(?i)\bsystem")
                .addMatchingPattern("(?i)\bconfigured")
                .addMatchingPattern("(?i)\bsetting")
                .addMatchingPattern("(?i)^system .+ is")
                .baseConfidence(0.7)
                .build(),
                
            Template.builder()
                .id("statement.data.analysis")
                .category(Template.Category.STATEMENT)
                .pattern("Analysis shows [METRIC] equals [VALUE] with [CONFIDENCE] confidence.")
                .addMatchingPattern("(?i)\banalysis")
                .addMatchingPattern("(?i)\bshows")
                .addMatchingPattern("(?i)\bdata")
                .addMatchingPattern("(?i)\bresult")
                .baseConfidence(0.8)
                .build(),
                
            Template.builder()
                .id("statement.error.description")
                .category(Template.Category.STATEMENT)
                .pattern("Error occurred in [MODULE]: [ERROR_TYPE] caused by [CAUSE].")
                .addMatchingPattern("(?i)\berror")
                .addMatchingPattern("(?i)\bfailed")
                .addMatchingPattern("(?i)\bexception")
                .addMatchingPattern("(?i)\bfault")
                .baseConfidence(0.9)
                .build()
        );
    }
    
    /**
     * Create response templates
     */
    private List<Template> createResponseTemplates() {
        return List.of(
            Template.builder()
                .id("response.understanding")
                .category(Template.Category.RESPONSE)
                .pattern("Based on [CONTEXT], I understand that [CONCLUSION].")
                .addMatchingPattern("(?i)\bunderstand\b")
                .addMatchingPattern("(?i)^i understand")
                .addMatchingPattern("(?i)based on")
                .addMatchingPattern("(?i)conclusion")
                .baseConfidence(0.8)
                .build(),
                
            Template.builder()
                .id("response.confirmation")
                .category(Template.Category.RESPONSE)
                .pattern("Confirmed: [ACTION] was [RESULT] for [TARGET].")
                .addMatchingPattern("(?i)\\bconfirm")
                .addMatchingPattern("(?i)\\bverified")
                .addMatchingPattern("(?i)\\bsuccess")
                .addMatchingPattern("(?i)you said")
                .addMatchingPattern("(?i)they believe")
                .addMatchingPattern("(?i)\\bsaid\\b")
                .addMatchingPattern("(?i)\\bbelieve")
                .baseConfidence(0.7)
                .build(),
                
            Template.builder()
                .id("response.clarification")
                .category(Template.Category.RESPONSE)
                .pattern("To clarify, [SUBJECT] requires [REQUIREMENT] before [NEXT_STEP].")
                .addMatchingPattern("(?i)\bclarify")
                .addMatchingPattern("(?i)\brequires")
                .addMatchingPattern("(?i)\bbefore")
                .addMatchingPattern("(?i)\bexplain")
                .baseConfidence(0.7)
                .build(),
                
            Template.builder()
                .id("response.recommendation")
                .category(Template.Category.RESPONSE)
                .pattern("I recommend [ACTION] to achieve [GOAL] given [CONSTRAINT].")
                .addMatchingPattern("(?i)\brecommend")
                .addMatchingPattern("(?i)\bsuggest")
                .addMatchingPattern("(?i)\badvise")
                .addMatchingPattern("(?i)\bpropose")
                .baseConfidence(0.8)
                .build(),
                
            Template.builder()
                .id("response.completion")
                .category(Template.Category.RESPONSE)
                .pattern("Task [TASK_ID] completed successfully with result: [RESULT].")
                .addMatchingPattern("(?i)\bcompleted")
                .addMatchingPattern("(?i)\bfinished")
                .addMatchingPattern("(?i)\bdone")
                .addMatchingPattern("(?i)\bcomplete")
                .baseConfidence(0.9)
                .build()
        );
    }
    
    /**
     * Create transition templates
     */
    private List<Template> createTransitionTemplates() {
        return List.of(
            Template.builder()
                .id("transition.next.step")
                .category(Template.Category.TRANSITION)
                .pattern("Next, we will [ACTION] the [OBJECT] to [PURPOSE].")
                .addMatchingPattern("(?i)^next we")
                .addMatchingPattern("(?i)^then you")
                .addMatchingPattern("(?i)^after that")
                .addMatchingPattern("(?i)\bnext\b")
                .addMatchingPattern("(?i)\bthen\b")
                .addMatchingPattern("(?i)following")
                .baseConfidence(0.8)
                .build(),
                
            Template.builder()
                .id("transition.meanwhile")
                .category(Template.Category.TRANSITION)
                .pattern("Meanwhile, [PARALLEL_PROCESS] continues with [STATUS].")
                .addMatchingPattern("(?i)\bmeanwhile")
                .addMatchingPattern("(?i)\bsimultaneously")
                .addMatchingPattern("(?i)\bparallel")
                .addMatchingPattern("(?i)\bat the same time")
                .baseConfidence(0.7)
                .build(),
                
            Template.builder()
                .id("transition.however")
                .category(Template.Category.TRANSITION)
                .pattern("However, [CONDITION] requires [ALTERNATIVE_ACTION] instead.")
                .addMatchingPattern("(?i)\bhowever")
                .addMatchingPattern("(?i)\bbut\b")
                .addMatchingPattern("(?i)\binstead")
                .addMatchingPattern("(?i)\balthough")
                .baseConfidence(0.7)
                .build(),
                
            Template.builder()
                .id("transition.therefore")
                .category(Template.Category.TRANSITION)
                .pattern("Therefore, [CONCLUSION] follows from [PREMISE].")
                .addMatchingPattern("(?i)\btherefore")
                .addMatchingPattern("(?i)\bthus")
                .addMatchingPattern("(?i)\bconsequently")
                .addMatchingPattern("(?i)\bhence")
                .baseConfidence(0.8)
                .build(),
                
            Template.builder()
                .id("transition.finally")
                .category(Template.Category.TRANSITION)
                .pattern("Finally, [FINAL_ACTION] will [COMPLETE] the [OBJECTIVE].")
                .addMatchingPattern("(?i)\bfinally")
                .addMatchingPattern("(?i)\blastly")
                .addMatchingPattern("(?i)\bconclude")
                .addMatchingPattern("(?i)\bin conclusion")
                .baseConfidence(0.9)
                .build(),
                
            Template.builder()
                .id("transition.additionally")
                .category(Template.Category.TRANSITION)
                .pattern("Additionally, [EXTRA_INFO] provides [BENEFIT] for [CONTEXT].")
                .addMatchingPattern("(?i)\badditionally")
                .addMatchingPattern("(?i)\bfurthermore")
                .addMatchingPattern("(?i)\balso\b")
                .addMatchingPattern("(?i)\bmoreover")
                .baseConfidence(0.6)
                .build(),
                
            Template.builder()
                .id("transition.alternatively")
                .category(Template.Category.TRANSITION)
                .pattern("Alternatively, you can [OPTION] to achieve [RESULT].")
                .addMatchingPattern("(?i)\balternatively")
                .addMatchingPattern("(?i)\botherwise")
                .addMatchingPattern("(?i)\bor\b")
                .addMatchingPattern("(?i)\binstead of")
                .baseConfidence(0.7)
                .build()
        );
    }
    
    /**
     * Create fallback template for unmatched inputs
     */
    private Template createFallbackTemplate() {
        return Template.builder()
            .id("fallback.general")
            .category(Template.Category.RESPONSE)
            .pattern("Processing [INPUT] with [ACTION] in [CONTEXT].")
            // Don't add any matching patterns - will be used only as last resort
            .baseConfidence(0.1)  // Very low confidence for fallback
            .metadata("type", "fallback")
            .build();
    }

    @Override
    public String toString() {
        return "TemplateRepository{templateCount=%d, categories=%s}"
            .formatted(getTemplateCount(), getAllCategories());
    }
}