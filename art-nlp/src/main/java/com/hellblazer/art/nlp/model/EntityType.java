package com.hellblazer.art.nlp.model;

/**
 * Enumeration of entity types for Named Entity Recognition (NER).
 * Based on common NLP entity classification standards.
 */
public enum EntityType {
    
    // Person-related entities
    PERSON("Person names and identifiers"),
    ORGANIZATION("Companies, institutions, agencies"),
    LOCATION("Geographic locations, places"),
    
    // Temporal entities
    DATE("Dates and date ranges"),
    TIME("Time expressions"),
    DURATION("Time durations and periods"),
    
    // Numeric entities  
    MONEY("Monetary values and currencies"),
    PERCENT("Percentage values"),
    NUMBER("Numeric quantities"),
    ORDINAL("Ordinal numbers (first, second, etc.)"),
    QUANTITY("Measurements and quantities"),
    
    // Technical entities
    EMAIL("Email addresses"),
    URL("Web URLs and URIs"),
    PHONE("Phone numbers"),
    
    // Document entities
    TITLE("Document or work titles"),
    AUTHOR("Author names"),
    PUBLICATION("Publication names"),
    
    // Product entities
    PRODUCT("Product names and models"),
    BRAND("Brand names"),
    SERVICE("Service names"),
    
    // Geographic entities (more specific)
    COUNTRY("Country names"),
    STATE("State or province names"),
    CITY("City names"),
    ADDRESS("Street addresses"),
    
    // Miscellaneous
    EVENT("Event names"),
    LANGUAGE("Language names"),
    NATIONALITY("Nationality or ethnic group"),
    RELIGION("Religious groups or beliefs"),
    
    // Unknown or unclassified
    UNKNOWN("Unclassified entity type"),
    OTHER("Other entity type not covered above");
    
    private final String description;
    
    EntityType(String description) {
        this.description = description;
    }
    
    /**
     * Get human-readable description of the entity type.
     * 
     * @return Description string
     */
    public String getDescription() {
        return description;
    }
    
    /**
     * Check if this is a person-related entity.
     * 
     * @return true if person-related
     */
    public boolean isPersonRelated() {
        return this == PERSON || this == AUTHOR;
    }
    
    /**
     * Check if this is a location-related entity.
     * 
     * @return true if location-related  
     */
    public boolean isLocationRelated() {
        return this == LOCATION || this == COUNTRY || 
               this == STATE || this == CITY || this == ADDRESS;
    }
    
    /**
     * Check if this is a temporal entity.
     * 
     * @return true if temporal
     */
    public boolean isTemporal() {
        return this == DATE || this == TIME || this == DURATION;
    }
    
    /**
     * Check if this is a numeric entity.
     * 
     * @return true if numeric
     */
    public boolean isNumeric() {
        return this == MONEY || this == PERCENT || this == NUMBER || 
               this == ORDINAL || this == QUANTITY;
    }
    
    /**
     * Check if this is a technical entity (contact/web-related).
     * 
     * @return true if technical
     */
    public boolean isTechnical() {
        return this == EMAIL || this == URL || this == PHONE;
    }
    
    /**
     * Get entity type from string name (case-insensitive).
     * 
     * @param name Entity type name
     * @return EntityType or UNKNOWN if not found
     */
    public static EntityType fromString(String name) {
        if (name == null || name.trim().isEmpty()) {
            return UNKNOWN;
        }
        
        var normalized = name.trim().toUpperCase().replace(' ', '_').replace('-', '_');
        
        try {
            return valueOf(normalized);
        } catch (IllegalArgumentException e) {
            // Handle some common alternate names
            return switch (normalized) {
                case "ORG", "ORGANISATION" -> ORGANIZATION;
                case "LOC", "PLACE" -> LOCATION;
                case "PER" -> PERSON;
                case "DATETIME" -> DATE;
                case "MONETARY" -> MONEY;
                case "PERCENTAGE" -> PERCENT;
                case "NUMERIC" -> NUMBER;
                case "WEB", "WEBSITE" -> URL;
                case "TELEPHONE" -> PHONE;
                case "MAIL" -> EMAIL;
                default -> UNKNOWN;
            };
        }
    }
    
    /**
     * Get all person-related entity types.
     * 
     * @return Array of person-related types
     */
    public static EntityType[] getPersonTypes() {
        return new EntityType[]{PERSON, AUTHOR, NATIONALITY};
    }
    
    /**
     * Get all location-related entity types.
     * 
     * @return Array of location-related types
     */
    public static EntityType[] getLocationTypes() {
        return new EntityType[]{LOCATION, COUNTRY, STATE, CITY, ADDRESS};
    }
    
    /**
     * Get all temporal entity types.
     * 
     * @return Array of temporal types
     */
    public static EntityType[] getTemporalTypes() {
        return new EntityType[]{DATE, TIME, DURATION};
    }
    
    /**
     * Get all numeric entity types.
     * 
     * @return Array of numeric types
     */
    public static EntityType[] getNumericTypes() {
        return new EntityType[]{MONEY, PERCENT, NUMBER, ORDINAL, QUANTITY};
    }
    
    /**
     * Get display name for UI purposes.
     * 
     * @return Formatted display name
     */
    public String getDisplayName() {
        // Convert from ENUM_CASE to Title Case
        var words = name().toLowerCase().split("_");
        var result = new StringBuilder();
        
        for (int i = 0; i < words.length; i++) {
            if (i > 0) result.append(" ");
            var word = words[i];
            if (!word.isEmpty()) {
                result.append(Character.toUpperCase(word.charAt(0)));
                if (word.length() > 1) {
                    result.append(word.substring(1));
                }
            }
        }
        
        return result.toString();
    }
    
    @Override
    public String toString() {
        return getDisplayName() + " (" + description + ")";
    }
}