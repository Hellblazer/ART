package com.hellblazer.art.hartcq.spatial;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Template for bounded text generation in HART-CQ.
 * Templates ensure deterministic output and prevent hallucination.
 */
public class Template {
    private final String id;
    private final String name;
    private final String pattern;
    private final List<Slot> slots;
    private final TemplateType type;
    
    public enum TemplateType {
        STATEMENT,
        QUESTION,
        COMMAND,
        EXCLAMATION,
        DESCRIPTION,
        NARRATIVE,
        DIALOGUE,
        TECHNICAL
    }
    
    public static class Slot {
        private final String name;
        private final SlotType type;
        private final boolean required;
        private final String defaultValue;
        private final Pattern validation;
        
        public enum SlotType {
            NOUN,
            VERB,
            ADJECTIVE,
            ADVERB,
            NUMBER,
            PROPER_NOUN,
            PRONOUN,
            PREPOSITION,
            CONJUNCTION,
            ARTICLE,
            ANY
        }
        
        public Slot(String name, SlotType type, boolean required, String defaultValue) {
            this.name = name;
            this.type = type;
            this.required = required;
            this.defaultValue = defaultValue;
            this.validation = createValidationPattern(type);
        }
        
        private Pattern createValidationPattern(SlotType type) {
            return switch (type) {
                case NUMBER -> Pattern.compile("\\d+(\\.\\d+)?");
                case PROPER_NOUN -> Pattern.compile("[A-Z][a-z]+( [A-Z][a-z]+)*");  // Allow multi-word proper nouns
                case ARTICLE -> Pattern.compile("(a|an|the|A|An|The)");
                case ANY -> Pattern.compile(".*");  // ANY type accepts anything
                default -> Pattern.compile("[\\w\\s]+");  // Allow spaces in default pattern for multi-word values
            };
        }
        
        public boolean validate(String value) {
            if (value == null || value.isEmpty()) {
                return !required;
            }
            return validation.matcher(value).matches();
        }
        
        public String getName() { return name; }
        public SlotType getType() { return type; }
        public boolean isRequired() { return required; }
        public String getDefaultValue() { return defaultValue; }
    }
    
    public Template(String id, String name, String pattern, TemplateType type) {
        this.id = id;
        this.name = name;
        this.pattern = pattern;
        this.type = type;
        this.slots = extractSlots(pattern);
    }
    
    private List<Slot> extractSlots(String pattern) {
        var slots = new ArrayList<Slot>();
        var matcher = Pattern.compile("\\{(\\w+):(\\w+)(\\?)?(?:=([^}]+))?\\}").matcher(pattern);
        
        while (matcher.find()) {
            var slotName = matcher.group(1);
            var slotType = Slot.SlotType.valueOf(matcher.group(2).toUpperCase());
            var optional = matcher.group(3) != null;
            var defaultValue = matcher.group(4);
            
            slots.add(new Slot(slotName, slotType, !optional, defaultValue));
        }
        
        return slots;
    }
    
    /**
     * Generates text from the template with given slot values.
     */
    public String generate(java.util.Map<String, String> slotValues) {
        var result = pattern;
        
        for (Slot slot : slots) {
            var value = slotValues.getOrDefault(slot.getName(), slot.getDefaultValue());
            
            if (value == null && slot.isRequired()) {
                throw new IllegalArgumentException("Required slot " + slot.getName() + " has no value");
            }
            
            if (value != null && !slot.validate(value)) {
                throw new IllegalArgumentException("Invalid value for slot " + slot.getName() + ": " + value);
            }
            
            if (value != null) {
                var slotPattern = "\\{" + slot.getName() + ":[^}]+\\}";
                result = result.replaceAll(slotPattern, value);
            }
        }
        
        return result;
    }
    
    public String getId() { return id; }
    public String getName() { return name; }
    public String getPattern() { return pattern; }
    public List<Slot> getSlots() { return slots; }
    public TemplateType getType() { return type; }
}