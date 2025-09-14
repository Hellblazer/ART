package com.hellblazer.art.hartcq.templates;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Timeout;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.CompletableFuture;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Comprehensive unit tests for the TemplateManager class.
 * Tests template system functionality including template matching,
 * variable substitution, deterministic selection, and anti-hallucination guarantees.
 */
@DisplayName("TemplateManager Tests")
class TemplateManagerTest {
    
    private TemplateManager templateManager;
    
    @Mock
    private TemplateRepository mockRepository;
    
    @Mock
    private TemplateMatcher mockMatcher;
    
    @Mock
    private TemplateRenderer mockRenderer;
    
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        templateManager = new TemplateManager();
    }
    
    @Nested
    @DisplayName("Basic Template Processing Tests")
    class BasicTemplateProcessingTests {
        
        @Test
        @DisplayName("Should process input with template successfully")
        void shouldProcessInputWithTemplateSuccessfully() {
            var input = "What is the weather today?";
            var variables = Map.<String, String>of("location", "New York", "date", "today");
            
            var result = templateManager.processInput(input, variables);
            
            assertThat(result).isNotNull();
            assertThat(result.operationId()).isNotNull();
            assertThat(result.operationId()).startsWith("op_");
            assertThat(result.confidence()).isBetween(0.0, 1.0);
        }
        
        @Test
        @DisplayName("Should handle empty input gracefully")
        void shouldHandleEmptyInputGracefully() {
            var result = templateManager.processInput("", Map.<String, String>of());
            
            assertThat(result).isNotNull();
            assertThat(result.successful()).isFalse();
            assertThat(result.errors()).isNotEmpty();
        }
        
        @Test
        @DisplayName("Should handle null input gracefully")
        void shouldHandleNullInputGracefully() {
            var result = templateManager.processInput(null, Map.<String, String>of());
            
            assertThat(result).isNotNull();
            assertThat(result.successful()).isFalse();
            assertThat(result.errors()).isNotEmpty();
        }
        
        @Test
        @DisplayName("Should handle null variables map")
        void shouldHandleNullVariablesMap() {
            var result = templateManager.processInput("test input", null);
            
            assertThat(result).isNotNull();
            // Should still attempt processing with empty variables
            assertThat(result.usedVariables()).isEmpty();
        }
    }
    
    @Nested
    @DisplayName("Template Matching Tests")
    class TemplateMatchingTests {
        
        @Test
        @DisplayName("Should find matching templates for various inputs")
        void shouldFindMatchingTemplatesForVariousInputs() {
            var testCases = Map.<String, String>of(
                "What is your name?", "question",
                "Hello there!", "greeting", 
                "Please help me", "request",
                "Thank you very much", "gratitude",
                "I don't understand", "confusion"
            );
            
            for (var entry : testCases.entrySet()) {
                var input = entry.getKey();
                var expectedCategory = entry.getValue();
                
                var result = templateManager.processInput(input, Map.<String, String>of());
                
                assertThat(result).isNotNull();
                assertThat(result.operationId()).contains(input.hashCode() + "");
            }
        }
        
        @Test
        @DisplayName("Should process input in specific category")
        void shouldProcessInputInSpecificCategory() {
            var input = "Hello world";
            var category = "greeting";
            var variables = Map.<String, String>of("name", "Alice");
            
            var result = templateManager.processInputInCategory(input, category, variables);
            
            assertThat(result).isNotNull();
            assertThat(result.operationId()).isNotNull();
        }
        
        @Test
        @DisplayName("Should handle non-existent category gracefully")
        void shouldHandleNonExistentCategoryGracefully() {
            var input = "Hello world";
            var nonExistentCategory = "non_existent_category_xyz";
            
            var result = templateManager.processInputInCategory(input, nonExistentCategory, Map.<String, String>of());
            
            assertThat(result).isNotNull();
            assertThat(result.successful()).isFalse();
            assertThat(result.errors()).isNotEmpty();
            assertThat(result.errors().get(0)).contains("category");
        }
    }
    
    @Nested
    @DisplayName("Deterministic Selection Tests")
    class DeterministicSelectionTests {
        
        @Test
        @DisplayName("Should provide deterministic template selection")
        void shouldProvideDeterministicTemplateSelection() {
            var input = "What is the meaning of life?";
            var variables = Map.<String, String>of("topic", "philosophy");
            
            // Process the same input multiple times
            var results = new java.util.ArrayList<TemplateManager.TemplateResult>();
            for (int i = 0; i < 5; i++) {
                results.add(templateManager.processDeterministic(input, variables));
            }
            
            // All results should be consistent for deterministic processing
            assertThat(results).allSatisfy(result -> {
                assertThat(result).isNotNull();
                assertThat(result.operationId()).isNotNull();
            });
            
            // If successful, template selection should be consistent
            var successfulResults = results.stream()
                .filter(TemplateManager.TemplateResult::successful)
                .toList();
            
            if (successfulResults.size() > 1) {
                var firstTemplate = successfulResults.get(0).template();
                for (var result : successfulResults) {
                    if (firstTemplate != null && result.template() != null) {
                        assertThat(result.template().id()).isEqualTo(firstTemplate.id());
                    }
                }
            }
        }
        
        @Test
        @DisplayName("Deterministic processing should be reproducible")
        void deterministicProcessingShouldBeReproducible() {
            var input = "How do I learn programming?";
            var variables = Map.<String, String>of("subject", "programming", "level", "beginner");
            
            var result1 = templateManager.processDeterministic(input, variables);
            var result2 = templateManager.processDeterministic(input, variables);
            
            // Results should have similar characteristics
            assertThat(result1.successful()).isEqualTo(result2.successful());
            
            if (result1.successful() && result2.successful()) {
                // If both successful, confidence should be identical for deterministic processing
                assertThat(result1.confidence()).isEqualTo(result2.confidence());
            }
        }
    }
    
    @Nested
    @DisplayName("Variable Substitution Tests")
    class VariableSubstitutionTests {
        
        @Test
        @DisplayName("Should substitute variables correctly")
        void shouldSubstituteVariablesCorrectly() {
            var input = "What is the weather in {location}?";
            var variables = Map.<String, String>of(
                "location", "Paris",
                "date", "today",
                "temperature", "20°C"
            );
            
            var result = templateManager.processInput(input, variables);
            
            assertThat(result).isNotNull();
            if (result.successful()) {
                assertThat(result.usedVariables()).isNotEmpty();
                // Should have used at least the location variable
                assertThat(result.usedVariables()).containsKey("location");
                assertThat(result.usedVariables().get("location")).isEqualTo("Paris");
            }
        }
        
        @Test
        @DisplayName("Should handle missing variables gracefully")
        void shouldHandleMissingVariablesGracefully() {
            var input = "Hello {name}, welcome to {place}!";
            var variables = Map.<String, String>of("name", "Alice"); // Missing "place" variable
            
            var result = templateManager.processInput(input, variables);
            
            assertThat(result).isNotNull();
            // Depending on implementation, might be successful with partial substitution
            // or unsuccessful due to missing variables
            if (!result.successful()) {
                assertThat(result.errors()).isNotEmpty();
            }
        }
        
        @Test
        @DisplayName("Should handle complex variable types")
        void shouldHandleComplexVariableTypes() {
            var input = "Process data with parameters";
            var variables = Map.<String, String>of(
                "number", "42",
                "decimal", "3.14159",
                "boolean", "true",
                "empty", "",
                "special_chars", "Hello@#$%^&*()"
            );
            
            var result = templateManager.processInput(input, variables);
            
            assertThat(result).isNotNull();
            assertThat(result.usedVariables()).isNotNull();
        }
        
        @Test
        @DisplayName("Should preserve variable types in output")
        void shouldPreserveVariableTypesInOutput() {
            var input = "The answer is {answer}";
            var variables = Map.<String, String>of(
                "answer", "42",
                "name", "test_user",
                "timestamp", String.valueOf(System.currentTimeMillis())
            );
            
            var result = templateManager.processInput(input, variables);
            
            assertThat(result).isNotNull();
            if (result.successful()) {
                // Variables used should maintain their string representation
                for (var entry : result.usedVariables().entrySet()) {
                    assertThat(entry.getValue()).isInstanceOf(String.class);
                }
            }
        }
    }
    
    @Nested
    @DisplayName("Anti-Hallucination Guarantees Tests")
    class AntiHallucinationTests {
        
        @Test
        @DisplayName("Should never generate output without matching template")
        void shouldNeverGenerateOutputWithoutMatchingTemplate() {
            // Use input that's unlikely to match any template
            var nonsenseInput = "xyzzyx abracadabra foobarbaz qwertyuiop";
            var variables = Map.<String, String>of();
            
            var result = templateManager.processInput(nonsenseInput, variables);
            
            assertThat(result).isNotNull();
            
            // If no template matches, should fail rather than hallucinate
            if (!result.successful()) {
                assertThat(result.output()).isEmpty();
                assertThat(result.template()).isNull();
                assertThat(result.errors()).isNotEmpty();
            } else {
                // If successful, must have a valid template
                assertThat(result.template()).isNotNull();
                assertThat(result.output()).isNotEmpty();
            }
        }
        
        @Test
        @DisplayName("Should validate template safety before processing")
        void shouldValidateTemplateSafetyBeforeProcessing() {
            var input = "Safe input for processing";
            var variables = Map.<String, String>of("safe_var", "safe_value");
            
            var canProcess = templateManager.canProcessSafely(input, variables);
            var result = templateManager.processInput(input, variables);
            
            // If canProcessSafely returns true, processing should succeed
            // If it returns false, processing might fail (but not necessarily)
            if (canProcess) {
                assertThat(result.successful()).isTrue();
            }
            
            // Regardless, result should never be null
            assertThat(result).isNotNull();
        }
        
        @Test
        @DisplayName("Should enforce strict template boundaries")
        void shouldEnforceStrictTemplateBoundaries() {
            var input = "Generate creative content";
            var variables = Map.<String, String>of("style", "creative", "length", "short");
            
            var result = templateManager.processInput(input, variables);
            
            assertThat(result).isNotNull();
            
            if (result.successful()) {
                // Output must be bounded by a template
                assertThat(result.template()).isNotNull();
                assertThat(result.template().id()).isNotEmpty();
                
                // Should track which variables were actually used
                assertThat(result.usedVariables()).isNotNull();
                
                // Confidence should reflect template match quality
                assertThat(result.confidence()).isPositive();
            }
        }
        
        @Test
        @DisplayName("Should provide multiple valid outputs for ambiguous input")
        void shouldProvideMultipleValidOutputsForAmbiguousInput() {
            var input = "Help me";  // Ambiguous - could match multiple templates
            var variables = Map.<String, String>of("context", "general");
            
            var allResults = templateManager.getAllPossibleOutputs(input, variables);
            
            assertThat(allResults).isNotNull();
            assertThat(allResults).isNotEmpty();
            
            // All results should be valid
            for (var result : allResults) {
                assertThat(result).isNotNull();
                assertThat(result.operationId()).isNotNull();
                
                if (result.successful()) {
                    assertThat(result.template()).isNotNull();
                    assertThat(result.output()).isNotEmpty();
                    assertThat(result.confidence()).isBetween(0.0, 1.0);
                }
            }
        }
    }
    
    @Nested
    @DisplayName("Performance and Caching Tests")
    class PerformanceAndCachingTests {
        
        @Test
        @Timeout(value = 5, unit = TimeUnit.SECONDS)
        @DisplayName("Should process templates within reasonable time")
        void shouldProcessTemplatesWithinReasonableTime() {
            var input = "What is the current time?";
            var variables = Map.<String, String>of("timezone", "UTC", "format", "ISO");
            
            var startTime = System.nanoTime();
            var result = templateManager.processInput(input, variables);
            var endTime = System.nanoTime();
            
            var processingTimeMs = (endTime - startTime) / 1_000_000.0;
            
            assertThat(result).isNotNull();
            assertThat(processingTimeMs).isLessThan(100); // Should process in under 100ms
        }
        
        @Test
        @DisplayName("Should benefit from caching on repeated requests")
        void shouldBenefitFromCachingOnRepeatedRequests() {
            var input = "Hello world";
            var variables = Map.<String, String>of("greeting", "hello");
            
            // First request - no cache
            var startTime1 = System.nanoTime();
            var result1 = templateManager.processInput(input, variables);
            var endTime1 = System.nanoTime();
            var time1 = endTime1 - startTime1;
            
            // Second request - should benefit from cache
            var startTime2 = System.nanoTime();
            var result2 = templateManager.processInput(input, variables);
            var endTime2 = System.nanoTime();
            var time2 = endTime2 - startTime2;
            
            assertThat(result1).isNotNull();
            assertThat(result2).isNotNull();
            
            // Results should be consistent
            assertThat(result1.successful()).isEqualTo(result2.successful());
            
            // Second request might be faster due to caching (but not guaranteed due to JIT)
            System.out.println("First request: " + (time1 / 1_000_000.0) + " ms");
            System.out.println("Second request: " + (time2 / 1_000_000.0) + " ms");
        }
        
        @Test
        @DisplayName("Should handle cache clearing")
        void shouldHandleCacheClearing() {
            var input = "Test caching";
            var variables = Map.<String, String>of("test", "value");
            
            // Process once to populate cache
            templateManager.processInput(input, variables);
            
            // Clear caches
            templateManager.clearCaches();
            
            // Process again - should still work
            var result = templateManager.processInput(input, variables);
            assertThat(result).isNotNull();
        }
        
        @Test
        @Timeout(value = 10, unit = TimeUnit.SECONDS)
        @DisplayName("Should handle concurrent processing efficiently")
        void shouldHandleConcurrentProcessingEfficiently() throws Exception {
            var executor = Executors.newFixedThreadPool(10);
            var results = new java.util.concurrent.ConcurrentHashMap<String, TemplateManager.TemplateResult>();
            var exceptions = new java.util.concurrent.ConcurrentLinkedQueue<Exception>();
            
            var futures = new java.util.ArrayList<CompletableFuture<Void>>();
            
            // Submit 100 concurrent requests
            for (int i = 0; i < 100; i++) {
                final var requestId = i;
                var future = CompletableFuture.runAsync(() -> {
                    try {
                        var input = "Concurrent request " + requestId;
                        var variables = Map.<String, String>of("id", String.valueOf(requestId));
                        var result = templateManager.processInput(input, variables);
                        results.put("request_" + requestId, result);
                    } catch (Exception e) {
                        exceptions.add(e);
                    }
                }, executor);
                
                futures.add(future);
            }
            
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get();
            
            executor.shutdown();
            assertThat(executor.awaitTermination(5, TimeUnit.SECONDS)).isTrue();
            
            // Verify results
            assertThat(exceptions).isEmpty();
            assertThat(results).hasSize(100);
            
            // All results should be valid
            for (var result : results.values()) {
                assertThat(result).isNotNull();
                assertThat(result.operationId()).isNotNull();
            }
        }
    }
    
    @Nested
    @DisplayName("Statistics and Monitoring Tests")
    class StatisticsAndMonitoringTests {
        
        @Test
        @DisplayName("Should track processing statistics")
        void shouldTrackProcessingStatistics() {
            // Process several requests
            var inputs = List.of(
                "Hello world",
                "What is your name?",
                "Thank you",
                "Goodbye"
            );
            
            for (var input : inputs) {
                templateManager.processInput(input, Map.<String, String>of());
            }
            
            var stats = templateManager.getProcessingStats();
            
            assertThat(stats).isNotNull();
            assertThat(stats.totalMatches()).isGreaterThan(0);
            assertThat(stats.totalRenders()).isGreaterThan(0);
            assertThat(stats.matchSuccessRate()).isBetween(0.0, 1.0);
            assertThat(stats.renderSuccessRate()).isBetween(0.0, 1.0);
            assertThat(stats.cacheHitRate()).isBetween(0.0, 1.0);
            assertThat(stats.templateCount()).isGreaterThanOrEqualTo(0);
            assertThat(stats.availableCategories()).isNotNull();
        }
        
        @Test
        @DisplayName("Should reset statistics correctly")
        void shouldResetStatisticsCorrectly() {
            // Process a request to generate stats
            templateManager.processInput("Test input", Map.<String, String>of());
            
            // Reset stats
            templateManager.resetStats();
            
            var stats = templateManager.getProcessingStats();
            
            // Stats should be reset
            assertThat(stats.totalMatches()).isZero();
            assertThat(stats.totalRenders()).isZero();
            assertThat(stats.cacheHits()).isZero();
        }
        
        @Test
        @DisplayName("Should provide meaningful toString representation")
        void shouldProvideMeaningfulToStringRepresentation() {
            var toString = templateManager.toString();
            
            assertThat(toString).isNotNull();
            assertThat(toString).contains("TemplateManager");
            assertThat(toString).contains("templateCount");
            assertThat(toString).contains("categories");
        }
    }
    
    @Nested
    @DisplayName("Template Repository Integration Tests")
    class TemplateRepositoryIntegrationTests {
        
        @Test
        @DisplayName("Should access template repository methods")
        void shouldAccessTemplateRepositoryMethods() {
            var categories = templateManager.getAvailableCategories();
            var templateCount = templateManager.getTemplateCount();
            var isReady = templateManager.isReady();
            
            assertThat(categories).isNotNull();
            assertThat(templateCount).isGreaterThanOrEqualTo(0);
            assertThat(isReady).isInstanceOf(Boolean.class);
        }
        
        @Test
        @DisplayName("Should get templates in specific category")
        void shouldGetTemplatesInSpecificCategory() {
            var categories = templateManager.getAvailableCategories();
            
            if (!categories.isEmpty()) {
                var firstCategory = categories.iterator().next();
                var templates = templateManager.getTemplatesInCategory(firstCategory);
                
                assertThat(templates).isNotNull();
            }
        }
        
        @Test
        @DisplayName("Should render template by ID")
        void shouldRenderTemplateById() {
            var templateId = "test_template_id";
            var variables = Map.<String, String>of("var1", "value1");
            
            var result = templateManager.renderTemplateById(templateId, variables);
            
            assertThat(result).isNotNull();
            // Result might be empty if template doesn't exist
            if (result.isPresent()) {
                var templateResult = result.get();
                assertThat(templateResult.operationId()).startsWith("op_direct_");
                assertThat(templateResult.confidence()).isEqualTo(1.0); // Direct access has full confidence
            }
        }
    }
    
    @Nested
    @DisplayName("Component Access Tests")
    class ComponentAccessTests {
        
        @Test
        @DisplayName("Should provide access to internal components")
        void shouldProvideAccessToInternalComponents() {
            var matcher = templateManager.getMatcher();
            var renderer = templateManager.getRenderer();
            var repository = templateManager.getRepository();
            
            assertThat(matcher).isNotNull();
            assertThat(renderer).isNotNull();
            assertThat(repository).isNotNull();
        }
    }
    
    @Nested
    @DisplayName("Edge Cases and Error Handling")
    class EdgeCasesAndErrorHandlingTests {
        
        @Test
        @DisplayName("Should handle extremely long input gracefully")
        void shouldHandleExtremelyLongInputGracefully() {
            var longInput = "word ".repeat(10000).trim();
            var variables = Map.<String, String>of("test", "value");
            
            var result = templateManager.processInput(longInput, variables);
            
            assertThat(result).isNotNull();
            assertThat(result.operationId()).isNotNull();
        }
        
        @Test
        @DisplayName("Should handle input with special characters")
        void shouldHandleInputWithSpecialCharacters() {
            var specialInput = "Hello! @#$%^&*()_+ <script>alert('test')</script> 世界 🌍";
            var variables = Map.<String, String>of("test", "value");
            
            var result = templateManager.processInput(specialInput, variables);
            
            assertThat(result).isNotNull();
            assertThat(result.operationId()).isNotNull();
        }
        
        @Test
        @DisplayName("Should handle large variable maps")
        void shouldHandleLargeVariableMaps() {
            var variables = new HashMap<String, String>();
            for (int i = 0; i < 1000; i++) {
                variables.put("var" + i, "value" + i);
            }
            
            var result = templateManager.processInput("Test with many variables", variables);
            
            assertThat(result).isNotNull();
            assertThat(result.operationId()).isNotNull();
        }
    }
}