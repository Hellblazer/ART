package com.hellblazer.art.nlp.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive test suite for Entity class.
 * Tests immutability, validation, and BIO tagging functionality.
 */
@DisplayName("Entity Tests")
class EntityTest {

    @Test
    @DisplayName("Should create entity with valid parameters")
    void shouldCreateEntityWithValidParameters() {
        var entity = new Entity("John Doe", "PERSON", 0, 1, 0.95);
        
        assertThat(entity.getText()).isEqualTo("John Doe");
        assertThat(entity.getType()).isEqualTo("PERSON");
        assertThat(entity.getStartToken()).isZero();
        assertThat(entity.getEndToken()).isOne();
        assertThat(entity.getConfidence()).isEqualTo(0.95);
        assertThat(entity.getLength()).isEqualTo(2);
    }

    @Test
    @DisplayName("Should create entity with single token constructor")
    void shouldCreateEntityWithSingleTokenConstructor() {
        var entity = new Entity("Apple", "ORG", 5);
        
        assertThat(entity.getText()).isEqualTo("Apple");
        assertThat(entity.getType()).isEqualTo("ORG");
        assertThat(entity.getStartToken()).isEqualTo(5);
        assertThat(entity.getEndToken()).isEqualTo(5);
        assertThat(entity.getConfidence()).isOne();
        assertThat(entity.getLength()).isOne();
    }

    @Test
    @DisplayName("Should create entity without confidence parameter")
    void shouldCreateEntityWithoutConfidence() {
        var entity = new Entity("New York", "LOC", 0, 1);
        
        assertThat(entity.getText()).isEqualTo("New York");
        assertThat(entity.getType()).isEqualTo("LOC");
        assertThat(entity.getStartToken()).isZero();
        assertThat(entity.getEndToken()).isOne();
        assertThat(entity.getConfidence()).isOne();
    }

    @Test
    @DisplayName("Should validate null text")
    void shouldValidateNullText() {
        assertThatThrownBy(() -> new Entity(null, "PERSON", 0))
            .isInstanceOf(NullPointerException.class)
            .hasMessage("text cannot be null");
    }

    @Test
    @DisplayName("Should validate null type")
    void shouldValidateNullType() {
        assertThatThrownBy(() -> new Entity("John", null, 0))
            .isInstanceOf(NullPointerException.class)
            .hasMessage("type cannot be null");
    }

    @Test
    @DisplayName("Should validate negative start token")
    void shouldValidateNegativeStartToken() {
        assertThatThrownBy(() -> new Entity("John", "PERSON", -1))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("startToken must be non-negative");
    }

    @Test
    @DisplayName("Should validate end token less than start token")
    void shouldValidateEndTokenLessThanStartToken() {
        assertThatThrownBy(() -> new Entity("John", "PERSON", 5, 3))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("endToken must be >= startToken");
    }

    @Test
    @DisplayName("Should validate confidence below zero")
    void shouldValidateConfidenceBelowZero() {
        assertThatThrownBy(() -> new Entity("John", "PERSON", 0, 0, -0.1))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("confidence must be in [0.0, 1.0]");
    }

    @Test
    @DisplayName("Should validate confidence above one")
    void shouldValidateConfidenceAboveOne() {
        assertThatThrownBy(() -> new Entity("John", "PERSON", 0, 0, 1.1))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessage("confidence must be in [0.0, 1.0]");
    }

    @Test
    @DisplayName("Should accept boundary confidence values")
    void shouldAcceptBoundaryConfidenceValues() {
        var entityMin = new Entity("John", "PERSON", 0, 0, 0.0);
        var entityMax = new Entity("Jane", "PERSON", 0, 0, 1.0);
        
        assertThat(entityMin.getConfidence()).isZero();
        assertThat(entityMax.getConfidence()).isOne();
    }

    @Test
    @DisplayName("Should add token to entity (BIO tagging)")
    void shouldAddTokenToEntity() {
        var original = new Entity("New", "LOC", 10, 10, 0.9);
        var extended = original.addToken("York");
        
        // Original should be unchanged (immutable)
        assertThat(original.getText()).isEqualTo("New");
        assertThat(original.getEndToken()).isEqualTo(10);
        
        // Extended should have new text and token range
        assertThat(extended.getText()).isEqualTo("New York");
        assertThat(extended.getType()).isEqualTo("LOC");
        assertThat(extended.getStartToken()).isEqualTo(10);
        assertThat(extended.getEndToken()).isEqualTo(11);
        assertThat(extended.getConfidence()).isEqualTo(0.9);
        assertThat(extended.getLength()).isEqualTo(2);
    }

    @Test
    @DisplayName("Should update confidence preserving other fields")
    void shouldUpdateConfidencePreservingOtherFields() {
        var original = new Entity("Apple Inc", "ORG", 0, 1, 0.5);
        var updated = original.withConfidence(0.95);
        
        // Original should be unchanged (immutable)
        assertThat(original.getConfidence()).isEqualTo(0.5);
        
        // Updated should have new confidence, same other fields
        assertThat(updated.getText()).isEqualTo("Apple Inc");
        assertThat(updated.getType()).isEqualTo("ORG");
        assertThat(updated.getStartToken()).isZero();
        assertThat(updated.getEndToken()).isOne();
        assertThat(updated.getConfidence()).isEqualTo(0.95);
    }

    @Test
    @DisplayName("Should calculate correct length")
    void shouldCalculateCorrectLength() {
        var singleToken = new Entity("Apple", "ORG", 5);
        var multiToken = new Entity("New York City", "LOC", 0, 2);
        
        assertThat(singleToken.getLength()).isOne();
        assertThat(multiToken.getLength()).isEqualTo(3);
    }

    @Test
    @DisplayName("Should implement equals correctly")
    void shouldImplementEqualsCorrectly() {
        var entity1 = new Entity("John Doe", "PERSON", 0, 1, 0.95);
        var entity2 = new Entity("John Doe", "PERSON", 0, 1, 0.95);
        var entity3 = new Entity("Jane Doe", "PERSON", 0, 1, 0.95);
        var entity4 = new Entity("John Doe", "ORG", 0, 1, 0.95);
        var entity5 = new Entity("John Doe", "PERSON", 1, 2, 0.95);
        var entity6 = new Entity("John Doe", "PERSON", 0, 1, 0.90);
        
        // Same entity should equal itself
        assertThat(entity1).isEqualTo(entity1);
        
        // Identical entities should be equal
        assertThat(entity1).isEqualTo(entity2);
        assertThat(entity2).isEqualTo(entity1);
        
        // Different entities should not be equal
        assertThat(entity1).isNotEqualTo(entity3); // Different text
        assertThat(entity1).isNotEqualTo(entity4); // Different type
        assertThat(entity1).isNotEqualTo(entity5); // Different tokens
        assertThat(entity1).isNotEqualTo(entity6); // Different confidence
        
        // Null and different class
        assertThat(entity1).isNotEqualTo(null);
        assertThat(entity1).isNotEqualTo("not an entity");
    }

    @Test
    @DisplayName("Should implement hashCode correctly")
    void shouldImplementHashCodeCorrectly() {
        var entity1 = new Entity("John Doe", "PERSON", 0, 1, 0.95);
        var entity2 = new Entity("John Doe", "PERSON", 0, 1, 0.95);
        var entity3 = new Entity("Jane Doe", "PERSON", 0, 1, 0.95);
        
        // Equal entities should have equal hash codes
        assertThat(entity1.hashCode()).isEqualTo(entity2.hashCode());
        
        // Different entities should preferably have different hash codes
        assertThat(entity1.hashCode()).isNotEqualTo(entity3.hashCode());
    }

    @Test
    @DisplayName("Should provide meaningful toString")
    void shouldProvideMeaningfulToString() {
        var entity = new Entity("John Doe", "PERSON", 5, 6, 0.856);
        var str = entity.toString();
        
        assertThat(str).contains("John Doe");
        assertThat(str).contains("PERSON");
        assertThat(str).contains("[5-6]");
        assertThat(str).contains("0.856");
        assertThat(str).startsWith("Entity{");
    }

    @Test
    @DisplayName("Should handle edge cases for token ranges")
    void shouldHandleEdgeCasesForTokenRanges() {
        // Large token numbers
        var entity = new Entity("Token", "MISC", 1000, 1500, 0.5);
        assertThat(entity.getLength()).isEqualTo(501);
        
        // Equal start and end tokens
        var singleToken = new Entity("Word", "MISC", 42, 42, 1.0);
        assertThat(singleToken.getLength()).isOne();
    }

    @Test
    @DisplayName("Should maintain immutability through chained operations")
    void shouldMaintainImmutabilityThroughChainedOperations() {
        var original = new Entity("Apple", "ORG", 0);
        var step1 = original.addToken("Inc");
        var step2 = step1.withConfidence(0.8);
        var step3 = step2.addToken("Corporation");
        
        // Original should remain unchanged
        assertThat(original.getText()).isEqualTo("Apple");
        assertThat(original.getEndToken()).isZero();
        assertThat(original.getConfidence()).isOne();
        
        // Each step should be independent
        assertThat(step1.getText()).isEqualTo("Apple Inc");
        assertThat(step1.getEndToken()).isOne();
        assertThat(step1.getConfidence()).isOne();
        
        assertThat(step2.getText()).isEqualTo("Apple Inc");
        assertThat(step2.getConfidence()).isEqualTo(0.8);
        
        assertThat(step3.getText()).isEqualTo("Apple Inc Corporation");
        assertThat(step3.getEndToken()).isEqualTo(2);
        assertThat(step3.getConfidence()).isEqualTo(0.8);
    }
}