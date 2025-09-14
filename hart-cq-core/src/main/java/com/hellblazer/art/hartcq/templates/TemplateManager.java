package com.hellblazer.art.hartcq.templates;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Main orchestrator for the HART-CQ template system that prevents hallucination
 * by strictly bounding all outputs to predefined templates. Provides deterministic
 * template selection, safe rendering, and comprehensive template management.
 * 
 * This is the primary entry point for all template operations, ensuring
 * NO HALLUCINATION through strict template boundaries.
 * 
 * @author Claude Code
 */
public class TemplateManager {
    
    private final TemplateRepository repository;
    private final TemplateMatcher matcher;
    private final TemplateRenderer renderer;
    
    // Caching for performance
    private final Map<String, TemplateMatcher.MatchResult> matchCache;
    private final Map<String, TemplateRenderer.RenderResult> renderCache;
    private final ReadWriteLock cacheLock;
    
    // Performance metrics
    private long totalMatches = 0;
    private long successfulMatches = 0;
    private long totalRenders = 0;
    private long successfulRenders = 0;
    private long cacheHits = 0;
    
    /**
     * Complete template operation result
     */
    public record TemplateResult(
        Template template,
        String output,
        boolean successful,
        double confidence,
        List<String> errors,
        Map<String, String> usedVariables,
        String operationId
    ) {
        public static TemplateResult success(Template template, String output, 
                                           double confidence, Map<String, String> variables, 
                                           String operationId) {
            return new TemplateResult(template, output, true, confidence, List.of(), 
                                    Map.copyOf(variables), operationId);
        }
        
        public static TemplateResult failure(String error, String operationId) {
            return new TemplateResult(null, "", false, 0.0, List.of(error), Map.of(), operationId);
        }
        
        public static TemplateResult failure(List<String> errors, String operationId) {
            return new TemplateResult(null, "", false, 0.0, List.copyOf(errors), Map.of(), operationId);
        }
    }
    
    /**
     * Template processing statistics
     */
    public record ProcessingStats(
        long totalMatches,
        long successfulMatches,
        double matchSuccessRate,
        long totalRenders,
        long successfulRenders,
        double renderSuccessRate,
        long cacheHits,
        double cacheHitRate,
        int templateCount,
        Set<String> availableCategories
    ) {}
    
    /**
     * Create template manager with default settings
     */
    public TemplateManager() {
        this(new TemplateRepository(), 0.05, true);  // Lower threshold to allow fallback template
    }

    /**
     * Get all possible outputs for an ambiguous input
     * This method helps prevent hallucination by showing all valid template-bounded options
     */
    public List<TemplateResult> getAllPossibleOutputs(String input, Map<String, String> variables) {
        var results = new ArrayList<TemplateResult>();
        var matchResult = matcher.findAllMatches(input);

        // Provide default values for common variables if not provided
        var enrichedVariables = new HashMap<>(variables);
        enrichVariables(enrichedVariables, input);

        for (var template : matchResult.getTemplates()) {
            var operationId = generateOperationId(input, enrichedVariables);
            var renderResult = renderer.render(template, enrichedVariables);

            if (renderResult.successful()) {
                results.add(TemplateResult.success(
                    template,
                    renderResult.output(),
                    matchResult.getConfidence(template),
                    enrichedVariables,
                    operationId
                ));
            }
        }

        return results;
    }
    
    /**
     * Create template manager with custom configuration
     */
    public TemplateManager(TemplateRepository repository, double confidenceThreshold, boolean strictRendering) {
        this.repository = Objects.requireNonNull(repository, "Repository cannot be null");
        this.matcher = new TemplateMatcher(repository, confidenceThreshold);
        this.renderer = new TemplateRenderer(strictRendering);
        
        this.matchCache = new ConcurrentHashMap<>();
        this.renderCache = new ConcurrentHashMap<>();
        this.cacheLock = new ReentrantReadWriteLock();
    }
    
    /**
     * MAIN METHOD: Process input and generate safe, template-bounded output
     * CRITICAL: This is the primary method that prevents hallucination
     */
    public TemplateResult processInput(String input, Map<String, String> variables) {
        var operationId = generateOperationId(input, variables);

        try {
            // Only enrich variables if they were provided
            var enrichedVariables = variables != null ? new HashMap<>(variables) : new HashMap<String, String>();
            if (variables != null && !variables.isEmpty()) {
                enrichVariables(enrichedVariables, input);
            }

            // Step 1: Find matching template (with caching)
            var matchResult = findMatchWithCache(input);
            totalMatches++;

            if (!matchResult.isSuccessful()) {
                return TemplateResult.failure("No suitable template found for input: " + input, operationId);
            }

            successfulMatches++;
            var template = matchResult.template();

            // Step 2: Render template with variables (with caching)
            var renderKey = createRenderCacheKey(template.id(), enrichedVariables);
            var renderResult = renderWithCache(renderKey, template, enrichedVariables);
            totalRenders++;
            
            if (!renderResult.successful()) {
                return TemplateResult.failure(renderResult.errors(), operationId);
            }
            
            successfulRenders++;

            return TemplateResult.success(
                template,
                renderResult.output(),
                matchResult.confidence(),
                enrichedVariables,
                operationId
            );

        } catch (Exception e) {
            return TemplateResult.failure("Template processing failed: " + e.getMessage(), operationId);
        }
    }
    
    /**
     * Process input with specific template category
     */
    public TemplateResult processInputInCategory(String input, String category, Map<String, String> variables) {
        var operationId = generateOperationId(input, variables);
        
        try {
            var matchResult = matcher.findBestMatchInCategory(input, category);
            totalMatches++;
            
            if (!matchResult.isSuccessful()) {
                return TemplateResult.failure("No template found in category: " + category, operationId);
            }
            
            successfulMatches++;
            var template = matchResult.template();
            var renderResult = renderer.render(template, variables);
            totalRenders++;
            
            if (!renderResult.successful()) {
                return TemplateResult.failure(renderResult.errors(), operationId);
            }
            
            successfulRenders++;
            
            return TemplateResult.success(
                template,
                renderResult.output(),
                matchResult.confidence(),
                renderResult.usedVariables(),
                operationId
            );
            
        } catch (Exception e) {
            return TemplateResult.failure("Category processing failed: " + e.getMessage(), operationId);
        }
    }
    
    /**
     * Process with deterministic template selection
     */
    public TemplateResult processDeterministic(String input, Map<String, String> variables) {
        var operationId = generateOperationId(input, variables);
        
        try {
            var matchResult = matcher.getDeterministicMatch(input);
            totalMatches++;
            
            if (!matchResult.isSuccessful()) {
                return TemplateResult.failure("No deterministic template match found", operationId);
            }
            
            successfulMatches++;
            var renderResult = renderer.render(matchResult.template(), variables);
            totalRenders++;
            
            if (!renderResult.successful()) {
                return TemplateResult.failure(renderResult.errors(), operationId);
            }
            
            successfulRenders++;
            
            return TemplateResult.success(
                matchResult.template(),
                renderResult.output(),
                matchResult.confidence(),
                renderResult.usedVariables(),
                operationId
            );
            
        } catch (Exception e) {
            return TemplateResult.failure("Deterministic processing failed: " + e.getMessage(), operationId);
        }
    }
    
    
    /**
     * Validate that input can be processed safely
     */
    public boolean canProcessSafely(String input, Map<String, String> variables) {
        if (input == null || input.trim().isEmpty()) {
            return false;
        }
        
        var matchResult = matcher.findBestMatch(input);
        if (!matchResult.isSuccessful()) {
            return false;
        }
        
        return renderer.canSafelyRender(matchResult.template(), variables);
    }
    
    /**
     * Get template by ID for direct rendering
     */
    public Optional<TemplateResult> renderTemplateById(String templateId, Map<String, String> variables) {
        var template = repository.getById(templateId);
        if (template.isEmpty()) {
            return Optional.empty();
        }
        
        var operationId = generateOperationId("direct_" + templateId, variables);
        var renderResult = renderer.render(template.get(), variables);
        totalRenders++;
        
        if (renderResult.successful()) {
            successfulRenders++;
            return Optional.of(TemplateResult.success(
                template.get(),
                renderResult.output(),
                1.0, // Direct template access has full confidence
                renderResult.usedVariables(),
                operationId
            ));
        } else {
            return Optional.of(TemplateResult.failure(renderResult.errors(), operationId));
        }
    }
    
    /**
     * Find matching template with caching
     */
    private TemplateMatcher.MatchResult findMatchWithCache(String input) {
        var cacheKey = "match_" + input.hashCode();
        
        cacheLock.readLock().lock();
        try {
            var cached = matchCache.get(cacheKey);
            if (cached != null) {
                cacheHits++;
                return cached;
            }
        } finally {
            cacheLock.readLock().unlock();
        }
        
        var result = matcher.findBestMatch(input);
        
        cacheLock.writeLock().lock();
        try {
            matchCache.put(cacheKey, result);
            // Limit cache size
            if (matchCache.size() > 1000) {
                var iterator = matchCache.entrySet().iterator();
                for (var i = 0; i < 200 && iterator.hasNext(); i++) {
                    iterator.next();
                    iterator.remove();
                }
            }
        } finally {
            cacheLock.writeLock().unlock();
        }
        
        return result;
    }
    
    /**
     * Render template with caching
     */
    private TemplateRenderer.RenderResult renderWithCache(String cacheKey, Template template, Map<String, String> variables) {
        cacheLock.readLock().lock();
        try {
            var cached = renderCache.get(cacheKey);
            if (cached != null) {
                cacheHits++;
                return cached;
            }
        } finally {
            cacheLock.readLock().unlock();
        }
        
        var result = renderer.render(template, variables);
        
        cacheLock.writeLock().lock();
        try {
            renderCache.put(cacheKey, result);
            // Limit cache size
            if (renderCache.size() > 1000) {
                var iterator = renderCache.entrySet().iterator();
                for (var i = 0; i < 200 && iterator.hasNext(); i++) {
                    iterator.next();
                    iterator.remove();
                }
            }
        } finally {
            cacheLock.writeLock().unlock();
        }
        
        return result;
    }
    
    /**
     * Create cache key for rendering
     */
    private String createRenderCacheKey(String templateId, Map<String, String> variables) {
        var variablesHash = variables != null ? variables.hashCode() : 0;
        return "render_" + templateId + "_" + variablesHash;
    }
    
    /**
     * Enrich variables with sensible defaults
     */
    private void enrichVariables(Map<String, String> variables, String input) {
        // Common template variables with defaults
        variables.putIfAbsent("SUBJECT", "User");
        variables.putIfAbsent("TOPIC", variables.getOrDefault("context", "general"));
        variables.putIfAbsent("USER", "User");
        variables.putIfAbsent("SYSTEM", "System");
        variables.putIfAbsent("NAME", "User");
        variables.putIfAbsent("STATUS", "ready");
        variables.putIfAbsent("OPERATION", "process");
        variables.putIfAbsent("ACTION", "process");
        variables.putIfAbsent("CONTEXT", "general");
        variables.putIfAbsent("INPUT", input != null ? input : "input");
        variables.putIfAbsent("PROPERTY", "status");
        variables.putIfAbsent("ENTITY", "item");
        variables.putIfAbsent("OBJECT", "data");
        variables.putIfAbsent("PARAMETER", "default");
        variables.putIfAbsent("ITEM", "element");
        variables.putIfAbsent("PURPOSE", "processing");
        variables.putIfAbsent("BEHAVIOR", "normal");
        variables.putIfAbsent("CONDITION", "standard");
        variables.putIfAbsent("EVENT", "completion");
        variables.putIfAbsent("ATTRIBUTE", "active");
        variables.putIfAbsent("SECONDARY_ATTRIBUTE", "ready");
        variables.putIfAbsent("PROCESS", "task");
        variables.putIfAbsent("RESULT", "success");
        variables.putIfAbsent("TIMESTAMP", String.valueOf(System.currentTimeMillis()));
        variables.putIfAbsent("COMPONENT", "module");
        variables.putIfAbsent("SETTING", "configuration");
        variables.putIfAbsent("VALUE", "default");
        variables.putIfAbsent("METRIC", "performance");
        variables.putIfAbsent("CONFIDENCE", "high");
        variables.putIfAbsent("MODULE", "core");
        variables.putIfAbsent("ERROR_TYPE", "none");
        variables.putIfAbsent("CAUSE", "unknown");
        variables.putIfAbsent("CONCLUSION", "processed");
        variables.putIfAbsent("REQUIREMENT", "none");
        variables.putIfAbsent("NEXT_STEP", "continue");
        variables.putIfAbsent("GOAL", "completion");
        variables.putIfAbsent("CONSTRAINT", "none");
        variables.putIfAbsent("TASK_ID", "task-" + System.nanoTime());
        variables.putIfAbsent("PARALLEL_PROCESS", "background");
        variables.putIfAbsent("ALTERNATIVE_ACTION", "retry");
        variables.putIfAbsent("PREMISE", "input");
        variables.putIfAbsent("FINAL_ACTION", "finish");
        variables.putIfAbsent("COMPLETE", "finalize");
        variables.putIfAbsent("OBJECTIVE", "task");
        variables.putIfAbsent("EXTRA_INFO", "details");
        variables.putIfAbsent("BENEFIT", "improvement");
        variables.putIfAbsent("OPTION", "alternative");
        variables.putIfAbsent("TIME_OF_DAY", "day");
        variables.putIfAbsent("DURATION", "1 hour");
    }

    /**
     * Generate unique operation ID for tracking
     */
    private String generateOperationId(String input, Map<String, String> variables) {
        var timestamp = System.currentTimeMillis();
        var inputHash = input != null ? input.hashCode() : 0;
        var variablesHash = variables != null ? variables.hashCode() : 0;
        return "op_" + timestamp + "_" + inputHash + "_" + variablesHash;
    }
    
    /**
     * Get processing statistics
     */
    public ProcessingStats getProcessingStats() {
        var matchRate = totalMatches > 0 ? (double) successfulMatches / totalMatches : 0.0;
        var renderRate = totalRenders > 0 ? (double) successfulRenders / totalRenders : 0.0;
        var hitRate = (totalMatches + totalRenders) > 0 ? (double) cacheHits / (totalMatches + totalRenders) : 0.0;
        
        return new ProcessingStats(
            totalMatches,
            successfulMatches,
            matchRate,
            totalRenders,
            successfulRenders,
            renderRate,
            cacheHits,
            hitRate,
            repository.getTemplateCount(),
            repository.getAllCategories()
        );
    }
    
    /**
     * Clear processing caches
     */
    public void clearCaches() {
        cacheLock.writeLock().lock();
        try {
            matchCache.clear();
            renderCache.clear();
            cacheHits = 0;
        } finally {
            cacheLock.writeLock().unlock();
        }
    }
    
    /**
     * Reset statistics
     */
    public void resetStats() {
        totalMatches = 0;
        successfulMatches = 0;
        totalRenders = 0;
        successfulRenders = 0;
        cacheHits = 0;
    }
    
    /**
     * Get available templates in category
     */
    public List<Template> getTemplatesInCategory(String category) {
        return repository.getByCategory(category);
    }
    
    /**
     * Get all available template categories
     */
    public Set<String> getAvailableCategories() {
        return repository.getAllCategories();
    }
    
    /**
     * Get total template count
     */
    public int getTemplateCount() {
        return repository.getTemplateCount();
    }
    
    /**
     * Check if system is ready for processing
     */
    public boolean isReady() {
        return repository.getTemplateCount() > 0;
    }
    
    /**
     * Get template matcher (for advanced usage)
     */
    public TemplateMatcher getMatcher() {
        return matcher;
    }
    
    /**
     * Get template renderer (for advanced usage)
     */
    public TemplateRenderer getRenderer() {
        return renderer;
    }
    
    /**
     * Get template repository (for advanced usage)
     */
    public TemplateRepository getRepository() {
        return repository;
    }
    
    @Override
    public String toString() {
        var stats = getProcessingStats();
        return "TemplateManager{templateCount=%d, categories=%d, matchRate=%.2f%%, renderRate=%.2f%%, cacheHitRate=%.2f%%}"
            .formatted(stats.templateCount(), stats.availableCategories().size(), 
                      stats.matchSuccessRate() * 100, stats.renderSuccessRate() * 100, 
                      stats.cacheHitRate() * 100);
    }
}