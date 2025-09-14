package com.hellblazer.art.hartcq.templates;

import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.util.function.Function;

/**
 * Safe template renderer that performs variable substitution and constraint validation
 * to prevent hallucination. All outputs are strictly bounded by template patterns
 * with validated variable substitution.
 * 
 * @author Claude Code
 */
public class TemplateRenderer {
    
    private final Map<String, Function<String, String>> validators;
    private final Map<String, Function<String, String>> transformers;
    private final boolean strictMode;
    
    /**
     * Rendering result with the generated output and validation status
     */
    public record RenderResult(
        String output,
        boolean successful,
        List<String> errors,
        Map<String, String> usedVariables
    ) {
        public static RenderResult success(String output, Map<String, String> variables) {
            return new RenderResult(output, true, List.of(), Map.copyOf(variables));
        }
        
        public static RenderResult failure(String error) {
            return new RenderResult("", false, List.of(error), Map.of());
        }
        
        public static RenderResult failure(List<String> errors) {
            return new RenderResult("", false, List.copyOf(errors), Map.of());
        }
    }
    
    /**
     * Variable validation constraints
     */
    public enum ValidationRule {
        NO_HTML(input -> input.replaceAll("<[^>]*>", "")),
        NO_SCRIPT(input -> input.replaceAll("(?i)<script[^>]*>.*?</script>", "")),
        ALPHANUMERIC_ONLY(input -> input.replaceAll("[^a-zA-Z0-9\\s]", "")),
        NO_SPECIAL_CHARS(input -> input.replaceAll("[^a-zA-Z0-9\\s\\-_.]", "")),
        MAX_LENGTH_50(input -> input.length() > 50 ? input.substring(0, 50) : input),
        MAX_LENGTH_100(input -> input.length() > 100 ? input.substring(0, 100) : input),
        TRIM_WHITESPACE(String::trim),
        CAPITALIZE(input -> input.isEmpty() ? input : 
                   Character.toUpperCase(input.charAt(0)) + input.substring(1).toLowerCase());
        
        private final Function<String, String> transformer;
        
        ValidationRule(Function<String, String> transformer) {
            this.transformer = transformer;
        }
        
        public String apply(String input) {
            return transformer.apply(input == null ? "" : input);
        }
    }
    
    private static final Pattern VARIABLE_PATTERN = Pattern.compile("\\[([A-Z_]+)\\]");
    
    /**
     * Create renderer with strict mode enabled (default)
     */
    public TemplateRenderer() {
        this(true);
    }
    
    /**
     * Create renderer with configurable strict mode
     */
    public TemplateRenderer(boolean strictMode) {
        this.strictMode = strictMode;
        this.validators = new HashMap<>();
        this.transformers = new HashMap<>();
        initializeDefaultValidators();
    }
    
    /**
     * Render template with provided variables
     * CRITICAL: Prevents hallucination through strict template boundaries
     */
    public RenderResult render(Template template, Map<String, String> variables) {
        if (template == null) {
            return RenderResult.failure("Template cannot be null");
        }
        
        var errors = new ArrayList<String>();
        var safeVariables = new HashMap<String, String>();
        var providedVariables = variables != null ? variables : Map.<String, String>of();
        
        // Validate template can be rendered with provided variables
        if (!template.canRender(providedVariables)) {
            var missing = new ArrayList<>(template.getRequiredVariables());
            missing.removeAll(providedVariables.keySet());
            errors.add("Missing required variables: " + missing);
        }
        
        // Validate and transform each variable
        for (var requiredVar : template.getRequiredVariables()) {
            var rawValue = providedVariables.get(requiredVar);
            if (rawValue == null || rawValue.trim().isEmpty()) {
                if (strictMode) {
                    errors.add("Variable [" + requiredVar + "] cannot be null or empty");
                    continue;
                } else {
                    rawValue = "[MISSING_" + requiredVar + "]";
                }
            }
            
            // Apply validation and transformation
            var safeValue = validateAndTransformVariable(requiredVar, rawValue);
            safeVariables.put(requiredVar, safeValue);
        }
        
        if (!errors.isEmpty()) {
            return RenderResult.failure(errors);
        }
        
        // Perform template substitution
        try {
            var output = substituteVariables(template.pattern(), safeVariables);
            
            // Final validation of output
            var finalValidationErrors = validateOutput(output);
            if (!finalValidationErrors.isEmpty()) {
                return RenderResult.failure(finalValidationErrors);
            }
            
            return RenderResult.success(output, safeVariables);
            
        } catch (Exception e) {
            return RenderResult.failure("Template rendering failed: " + e.getMessage());
        }
    }
    
    /**
     * Render template with single variable (convenience method)
     */
    public RenderResult render(Template template, String variableName, String value) {
        return render(template, Map.of(variableName, value));
    }
    
    /**
     * Add custom validator for a variable name
     */
    public TemplateRenderer addValidator(String variableName, Function<String, String> validator) {
        validators.put(variableName.toUpperCase(), validator);
        return this;
    }
    
    /**
     * Add custom transformer for a variable name
     */
    public TemplateRenderer addTransformer(String variableName, Function<String, String> transformer) {
        transformers.put(variableName.toUpperCase(), transformer);
        return this;
    }
    
    /**
     * Add validation rule for a variable
     */
    public TemplateRenderer addValidationRule(String variableName, ValidationRule rule) {
        return addValidator(variableName, rule::apply);
    }
    
    /**
     * Validate and transform a single variable value
     */
    private String validateAndTransformVariable(String variableName, String value) {
        var processedValue = value;
        
        // Apply custom validator if present
        var validator = validators.get(variableName.toUpperCase());
        if (validator != null) {
            processedValue = validator.apply(processedValue);
        }
        
        // Apply custom transformer if present
        var transformer = transformers.get(variableName.toUpperCase());
        if (transformer != null) {
            processedValue = transformer.apply(processedValue);
        }
        
        // Apply default safety transformations
        processedValue = applySafetyTransformations(processedValue);
        
        return processedValue;
    }
    
    /**
     * Apply default safety transformations to prevent injection attacks
     */
    private String applySafetyTransformations(String value) {
        if (value == null) {
            return "";
        }
        
        var safe = value;
        
        // Remove HTML tags
        safe = ValidationRule.NO_HTML.apply(safe);
        
        // Remove script tags
        safe = ValidationRule.NO_SCRIPT.apply(safe);
        
        // Trim whitespace
        safe = ValidationRule.TRIM_WHITESPACE.apply(safe);
        
        // Limit length to reasonable maximum
        if (safe.length() > 200) {
            safe = safe.substring(0, 200) + "...";
        }
        
        return safe;
    }
    
    /**
     * Substitute variables in template pattern
     */
    private String substituteVariables(String pattern, Map<String, String> variables) {
        var result = pattern;
        
        var matcher = VARIABLE_PATTERN.matcher(pattern);
        var processedVariables = new HashSet<String>();
        
        while (matcher.find()) {
            var variableName = matcher.group(1);
            
            if (processedVariables.contains(variableName)) {
                continue; // Already processed this variable
            }
            
            var value = variables.get(variableName);
            if (value != null) {
                result = result.replaceAll("\\[" + Pattern.quote(variableName) + "\\]", 
                                         Matcher.quoteReplacement(value));
                processedVariables.add(variableName);
            }
        }
        
        return result;
    }
    
    /**
     * Validate final output for safety
     */
    private List<String> validateOutput(String output) {
        var errors = new ArrayList<String>();
        
        if (output == null || output.trim().isEmpty()) {
            errors.add("Rendered output cannot be empty");
            return errors;
        }
        
        // Check for unsubstituted variables
        var matcher = VARIABLE_PATTERN.matcher(output);
        if (matcher.find()) {
            errors.add("Template contains unsubstituted variables: " + matcher.group(0));
        }
        
        // Check output length
        if (output.length() > 1000) {
            errors.add("Rendered output exceeds maximum length (1000 characters)");
        }
        
        // Additional safety checks can be added here
        
        return errors;
    }
    
    /**
     * Initialize default validators for common variable types
     */
    private void initializeDefaultValidators() {
        // Common variable type validators
        addValidator("USER", ValidationRule.NO_SPECIAL_CHARS::apply);
        addValidator("NAME", ValidationRule.NO_SPECIAL_CHARS::apply);
        addValidator("SUBJECT", ValidationRule.NO_HTML::apply);
        addValidator("TOPIC", ValidationRule.NO_HTML::apply);
        addValidator("ENTITY", ValidationRule.NO_SPECIAL_CHARS::apply);
        addValidator("OBJECT", ValidationRule.NO_SPECIAL_CHARS::apply);
        addValidator("ACTION", ValidationRule.ALPHANUMERIC_ONLY::apply);
        addValidator("STATUS", ValidationRule.ALPHANUMERIC_ONLY::apply);
        addValidator("SYSTEM", ValidationRule.NO_SPECIAL_CHARS::apply);
        addValidator("MODULE", ValidationRule.NO_SPECIAL_CHARS::apply);
        addValidator("PROCESS", ValidationRule.NO_SPECIAL_CHARS::apply);
        
        // Add length limits to text fields
        addValidator("CONTEXT", input -> ValidationRule.MAX_LENGTH_100.apply(
                                ValidationRule.NO_HTML.apply(input)));
        addValidator("DESCRIPTION", input -> ValidationRule.MAX_LENGTH_100.apply(
                                    ValidationRule.NO_HTML.apply(input)));
        addValidator("CONCLUSION", input -> ValidationRule.MAX_LENGTH_100.apply(
                                   ValidationRule.NO_HTML.apply(input)));
    }
    
    /**
     * Check if a template can be safely rendered
     */
    public boolean canSafelyRender(Template template, Map<String, String> variables) {
        var result = render(template, variables);
        return result.successful();
    }
    
    /**
     * Get list of required variables that are missing from provided variables
     */
    public List<String> getMissingVariables(Template template, Map<String, String> variables) {
        var provided = variables != null ? variables.keySet() : Set.<String>of();
        var required = template.getRequiredVariables();
        
        return required.stream()
            .filter(var -> !provided.contains(var))
            .sorted()
            .toList();
    }
    
    /**
     * Create a new renderer with additional validators
     */
    public TemplateRenderer withValidators(Map<String, Function<String, String>> additionalValidators) {
        var newRenderer = new TemplateRenderer(strictMode);
        newRenderer.validators.putAll(this.validators);
        newRenderer.validators.putAll(additionalValidators);
        newRenderer.transformers.putAll(this.transformers);
        return newRenderer;
    }
    
    public boolean isStrictMode() {
        return strictMode;
    }
    
    @Override
    public String toString() {
        return "TemplateRenderer{strictMode=%s, validatorCount=%d, transformerCount=%d}"
            .formatted(strictMode, validators.size(), transformers.size());
    }
}