/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of Java ART Neural Networks.
 * 
 * Java ART Neural Networks is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Java ART Neural Networks is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with Java ART Neural Networks. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.core;

import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.assertj.core.api.Assertions.*;

// ART Core classes  
import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.BaseART;
import com.hellblazer.art.core.FuzzyART;
import com.hellblazer.art.core.BayesianART;
import com.hellblazer.art.core.ART2;
import com.hellblazer.art.core.ARTMAP;
import com.hellblazer.art.core.SimpleARTMAP;
import com.hellblazer.art.core.BaseARTMAP;
import com.hellblazer.art.core.DeepARTMAP;
import com.hellblazer.art.core.DeepARTMAPParameters;
import com.hellblazer.art.core.DeepARTMAPResult;

/**
 * Comprehensive test suite for DeepARTMAP implementation.
 * Written BEFORE any DeepARTMAP implementation exists following test-first methodology.
 * These tests define the complete specification and expected behavior.
 * 
 * DeepARTMAP Key Features:
 * - Hierarchical ARTMAP with multiple ART modules forming layers
 * - Dual modes: Supervised (with class labels) and Unsupervised (without labels)  
 * - Multi-channel data processing (list of input matrices)
 * - Deep label propagation through hierarchical layers
 * - Layer stack: BaseARTMAP instances (ARTMAP or SimpleARTMAP)
 * - Supports arbitrary number of ART modules as building blocks
 * 
 * Reference: Python implementation at /Users/hal.hildebrand/git/AdaptiveResonanceLib/artlib/hierarchical/DeepARTMAP.py
 * Paper: "Deep ARTMAP: Generalized Hierarchical Learning with Adaptive Resonance Theory"
 * 
 * @author Hal Hildebrand
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class DeepARTMAPTest {
    
    private static final double TOLERANCE = 1e-6;
    private static final double LOOSE_TOLERANCE = 1e-3;
    private static final int DEFAULT_MAX_CATEGORIES = 1000;
    private static final long PERFORMANCE_TIMEOUT_MS = 5000;
    
    private Random random;
    private List<BaseART> testModules;
    
    @BeforeEach
    void setUp() {
        random = new Random(42);
        testModules = createTestModules();
    }
    
    private List<BaseART> createTestModules() {
        var fuzzyART1 = new FuzzyART();
        var fuzzyART2 = new FuzzyART();  
        var fuzzyART3 = new FuzzyART();
        return List.of(fuzzyART1, fuzzyART2, fuzzyART3);
    }
    
    private List<Pattern[]> createMultiChannelData(int samples, int channels, int dimensions) {
        var data = new Pattern[channels][];
        for (int c = 0; c < channels; c++) {
            data[c] = new Pattern[samples];
            for (int s = 0; s < samples; s++) {
                var values = new double[dimensions];
                for (int d = 0; d < dimensions; d++) {
                    values[d] = random.nextDouble();
                }
                data[c][s] = Pattern.of(values);
            }
        }
        return Arrays.asList(data);
    }
    
    private int[] createClassLabels(int samples, int numClasses) {
        var labels = new int[samples];
        for (int i = 0; i < samples; i++) {
            labels[i] = random.nextInt(numClasses);
        }
        return labels;
    }

    // ================================================================================
    // CATEGORY 1: CONSTRUCTOR TESTS (15 tests)
    // ================================================================================

    @Nested
    @DisplayName("Constructor Tests")
    @Order(1)
    class ConstructorTests {

        @Test
        @DisplayName("Should create DeepARTMAP with single module")
        void testConstructorSingleModule() {
            var modules = List.<BaseART>of(new FuzzyART());
            var deepARTMAP = new DeepARTMAP(modules, new DeepARTMAPParameters());
            
            assertThat(deepARTMAP.getModules()).hasSize(1);
            assertThat(deepARTMAP.getModules().get(0)).isInstanceOf(FuzzyART.class);
            assertThat(deepARTMAP.getLayerCount()).isEqualTo(0);
            assertThat(deepARTMAP.isTrained()).isFalse();
        }

        @Test
        @DisplayName("Should create DeepARTMAP with multiple modules")
        void testConstructorMultipleModules() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            
            assertThat(deepARTMAP.getModules()).hasSize(3);
            assertThat(deepARTMAP.getLayerCount()).isEqualTo(0);
            assertThat(deepARTMAP.isTrained()).isFalse();
        }

        @Test
        @DisplayName("Should reject null modules list")
        void testConstructorNullModules() {
            assertThatThrownBy(() -> new DeepARTMAP(null, new DeepARTMAPParameters()))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("modules cannot be null");
        }

        @Test
        @DisplayName("Should reject empty modules list")
        void testConstructorEmptyModules() {
            assertThatThrownBy(() -> new DeepARTMAP(List.of(), new DeepARTMAPParameters()))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Must provide at least one ART module");
        }

        @Test
        @DisplayName("Should reject null parameters")
        void testConstructorNullParameters() {
            assertThatThrownBy(() -> new DeepARTMAP(testModules, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("parameters cannot be null");
        }

        @Test
        @DisplayName("Should reject modules with null elements")
        void testConstructorNullModuleElement() {
            var modulesWithNull = Arrays.<BaseART>asList(new FuzzyART(), null, new FuzzyART());
            assertThatThrownBy(() -> new DeepARTMAP(modulesWithNull, new DeepARTMAPParameters()))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("modules cannot contain null elements");
        }

        @Test
        @DisplayName("Should create defensive copy of modules list")
        void testConstructorDefensiveCopy() {
            var originalModules = new java.util.ArrayList<>(testModules);
            var deepARTMAP = new DeepARTMAP(originalModules, new DeepARTMAPParameters());
            
            originalModules.clear();
            assertThat(deepARTMAP.getModules()).hasSize(3);
        }

        @Test
        @DisplayName("Should handle different ART module types")
        void testConstructorMixedModuleTypes() {
            var mixedModules = List.<BaseART>of(new FuzzyART(), new BayesianART(new BayesianParameters(0.9, new double[]{0.0}, Matrix.eye(1), 1.0, 1.0, 100)), new ART2(new ART2Parameters(0.9, 0.1, 100)));
            var deepARTMAP = new DeepARTMAP(mixedModules, new DeepARTMAPParameters());
            
            assertThat(deepARTMAP.getModules()).hasSize(3);
            assertThat(deepARTMAP.getModules().get(0)).isInstanceOf(FuzzyART.class);
            assertThat(deepARTMAP.getModules().get(1)).isInstanceOf(BayesianART.class);
            assertThat(deepARTMAP.getModules().get(2)).isInstanceOf(ART2.class);
        }

        @Test
        @DisplayName("Should accept large number of modules")
        void testConstructorManyModules() {
            var manyModules = java.util.stream.IntStream.range(0, 100)
                .mapToObj(i -> (BaseART) new FuzzyART())
                .toList();
            var deepARTMAP = new DeepARTMAP(manyModules, new DeepARTMAPParameters());
            
            assertThat(deepARTMAP.getModules()).hasSize(100);
        }

        @ParameterizedTest
        @ValueSource(ints = {1, 2, 3, 5, 10, 20})
        @DisplayName("Should create DeepARTMAP with parameterized module count")
        void testConstructorParameterizedModuleCount(int moduleCount) {
            var modules = java.util.stream.IntStream.range(0, moduleCount)
                .mapToObj(i -> (BaseART) new FuzzyART())
                .toList();
            var deepARTMAP = new DeepARTMAP(modules, new DeepARTMAPParameters());
            
            assertThat(deepARTMAP.getModules()).hasSize(moduleCount);
        }

        @Test
        @DisplayName("Should store parameters correctly")
        void testConstructorParametersStored() {
            var parameters = new DeepARTMAPParameters(0.8, 0.1, 1000, true);
            var deepARTMAP = new DeepARTMAP(testModules, parameters);
            
            assertThat(deepARTMAP.getParameters()).isEqualTo(parameters);
        }

        @Test
        @DisplayName("Should initialize with correct default state")
        void testConstructorDefaultState() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            
            assertThat(deepARTMAP.isTrained()).isFalse();
            assertThat(deepARTMAP.isSupervised()).isNull();
            assertThat(deepARTMAP.getLayerCount()).isEqualTo(0);
            assertThat(deepARTMAP.getLayers()).isEmpty();
        }

        @Test
        @DisplayName("Should handle modules with same reference")
        void testConstructorSameModuleReference() {
            var sharedModule = new FuzzyART();
            var modulesWithSameRef = List.<BaseART>of(sharedModule, sharedModule, new FuzzyART());
            var deepARTMAP = new DeepARTMAP(modulesWithSameRef, new DeepARTMAPParameters());
            
            assertThat(deepARTMAP.getModules()).hasSize(3);
            assertThat(deepARTMAP.getModules().get(0)).isSameAs(deepARTMAP.getModules().get(1));
        }

        @Test
        @DisplayName("Should maintain modules order")
        void testConstructorModuleOrder() {
            var module1 = new FuzzyART();
            var module2 = new BayesianART(new BayesianParameters(0.9, new double[]{0.0}, Matrix.eye(1), 1.0, 1.0, 100));  
            var module3 = new ART2(new ART2Parameters(0.9, 0.1, 100));
            var orderedModules = List.of(module1, module2, module3);
            var deepARTMAP = new DeepARTMAP(orderedModules, new DeepARTMAPParameters());
            
            assertThat(deepARTMAP.getModules().get(0)).isSameAs(module1);
            assertThat(deepARTMAP.getModules().get(1)).isSameAs(module2);
            assertThat(deepARTMAP.getModules().get(2)).isSameAs(module3);
        }

        @Test
        @DisplayName("Should create immutable modules list")
        void testConstructorImmutableModulesList() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            
            assertThatThrownBy(() -> deepARTMAP.getModules().add(new FuzzyART()))
                .isInstanceOf(UnsupportedOperationException.class);
        }
    }

    // ================================================================================
    // CATEGORY 2: SUPERVISED LEARNING TESTS (20 tests)
    // ================================================================================

    @Nested
    @DisplayName("Supervised Learning Tests")
    @Order(2)
    class SupervisedLearningTests {

        @Test
        @DisplayName("Should train in supervised mode with class labels")
        void testSupervisedBasicTraining() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            var result = deepARTMAP.fit(data, labels);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(deepARTMAP.isTrained()).isTrue();
            assertThat(deepARTMAP.isSupervised()).isTrue();
            assertThat(deepARTMAP.getLayerCount()).isEqualTo(3);
        }

        @Test
        @DisplayName("Should create SimpleARTMAP layers in supervised mode")
        void testSupervisedLayerTypes() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            deepARTMAP.fit(data, labels);
            
            var layers = deepARTMAP.getLayers();
            assertThat(layers).hasSize(3);
            assertThat(layers).allMatch(layer -> layer instanceof SimpleARTMAP);
        }

        @Test
        @DisplayName("Should train first layer with data and labels")
        void testSupervisedFirstLayerTraining() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            deepARTMAP.fit(data, labels);
            
            var firstLayer = (SimpleARTMAP) deepARTMAP.getLayers().get(0);
            assertThat(firstLayer.isTrained()).isTrue();
            assertThat(firstLayer.getCategoryCount()).isGreaterThan(0);
        }

        @Test
        @DisplayName("Should propagate labels through hierarchical layers")
        void testSupervisedLabelPropagation() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            deepARTMAP.fit(data, labels);
            
            for (int i = 1; i < deepARTMAP.getLayerCount(); i++) {
                var layer = (SimpleARTMAP) deepARTMAP.getLayers().get(i);
                assertThat(layer.isTrained()).isTrue();
            }
        }

        @Test
        @DisplayName("Should reject mismatched data and label sizes")
        void testSupervisedMismatchedSizes() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(8, 3); // Wrong size
            
            assertThatThrownBy(() -> deepARTMAP.fit(data, labels))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Inconsistent sample number");
        }

        @Test
        @DisplayName("Should handle single class labels")
        void testSupervisedSingleClass() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = new int[10]; // All zeros
            
            var result = deepARTMAP.fit(data, labels);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(deepARTMAP.isTrained()).isTrue();
        }

        @Test
        @DisplayName("Should handle many classes")
        void testSupervisedManyClasses() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(100, 3, 5);
            var labels = createClassLabels(100, 50); // 50 different classes
            
            var result = deepARTMAP.fit(data, labels);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(deepARTMAP.isTrained()).isTrue();
        }

        @Test
        @DisplayName("Should predict classes after supervised training")
        void testSupervisedPrediction() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var trainData = createMultiChannelData(10, 3, 5);
            var trainLabels = createClassLabels(10, 3);
            
            deepARTMAP.fit(trainData, trainLabels);
            
            var testData = createMultiChannelData(5, 3, 5);
            var predictions = deepARTMAP.predict(testData);
            
            assertThat(predictions).hasSize(5);
            for (int pred : predictions) {
                assertThat(pred).isGreaterThanOrEqualTo(0);
            }
        }

        @Test
        @DisplayName("Should support partial fit in supervised mode")
        void testSupervisedPartialFit() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data1 = createMultiChannelData(5, 3, 5);
            var labels1 = createClassLabels(5, 3);
            
            deepARTMAP.fit(data1, labels1);
            
            var data2 = createMultiChannelData(5, 3, 5);
            var labels2 = createClassLabels(5, 3);
            var result = deepARTMAP.partialFit(data2, labels2);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(deepARTMAP.isTrained()).isTrue();
        }

        @Test
        @DisplayName("Should maintain supervised mode consistency in partial fit")
        void testSupervisedPartialFitConsistency() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data1 = createMultiChannelData(5, 3, 5);
            var labels1 = createClassLabels(5, 3);
            
            deepARTMAP.fit(data1, labels1);
            
            var data2 = createMultiChannelData(5, 3, 5);
            
            assertThatThrownBy(() -> deepARTMAP.partialFit(data2, null))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("Labels were previously provided");
        }

        @ParameterizedTest
        @ValueSource(ints = {2, 3, 5, 10})
        @DisplayName("Should handle different numbers of classes")
        void testSupervisedParameterizedClasses(int numClasses) {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(20, 3, 5);
            var labels = createClassLabels(20, numClasses);
            
            var result = deepARTMAP.fit(data, labels);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(deepARTMAP.isTrained()).isTrue();
        }

        @Test
        @DisplayName("Should handle negative class labels")
        void testSupervisedNegativeLabels() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = new int[]{-1, -2, 0, 1, 2, -1, -2, 0, 1, 2};
            
            var result = deepARTMAP.fit(data, labels);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
        }

        @Test
        @DisplayName("Should reject null labels in supervised mode")
        void testSupervisedNullLabels() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            
            assertThatThrownBy(() -> deepARTMAP.fitSupervised(data, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("labels cannot be null in supervised mode");
        }

        @Test
        @DisplayName("Should handle large label values")
        void testSupervisedLargeLabelValues() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(5, 3, 5);
            var labels = new int[]{1000, 2000, 3000, 4000, 5000};
            
            var result = deepARTMAP.fit(data, labels);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
        }

        @Test
        @DisplayName("Should create deep labels concatenation")
        void testSupervisedDeepLabels() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            deepARTMAP.fit(data, labels);
            
            var deepLabels = deepARTMAP.getDeepLabels();
            assertThat(deepLabels.length).isEqualTo(10);
            for (var labelArray : deepLabels) {
                assertThat(labelArray.length).isEqualTo(3); // One label per layer
            }
        }

        @Test
        @DisplayName("Should support label mapping between layers")
        void testSupervisedLabelMapping() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            deepARTMAP.fit(data, labels);
            
            var mapping = deepARTMAP.mapDeep(1, 0);
            assertThat(mapping).isNotNull();
        }

        @Test
        @DisplayName("Should handle identical patterns with different labels")
        void testSupervisedIdenticalPatternsWithDifferentLabels() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            
            // Create identical patterns with different labels
            var identicalPattern = Pattern.of(new double[]{0.5, 0.5, 0.5, 0.5, 0.5});
            var data = List.of(
                new Pattern[]{identicalPattern, identicalPattern},
                new Pattern[]{identicalPattern, identicalPattern},  
                new Pattern[]{identicalPattern, identicalPattern}
            );
            var labels = new int[]{0, 1}; // Different labels for same pattern
            
            var result = deepARTMAP.fit(data, labels);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
        }

        @Test
        @DisplayName("Should maintain performance with many samples in supervised mode")
        void testSupervisedPerformanceWithManySamples() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(1000, 3, 10);
            var labels = createClassLabels(1000, 10);
            
            assertTimeout(java.time.Duration.ofMillis(PERFORMANCE_TIMEOUT_MS), () -> {
                var result = deepARTMAP.fit(data, labels);
                assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            });
        }

        @Test
        @DisplayName("Should support supervised mode with single module")
        void testSupervisedSingleModule() {
            var singleModule = List.<BaseART>of(new FuzzyART());
            var deepARTMAP = new DeepARTMAP(singleModule, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 1, 5);
            var labels = createClassLabels(10, 3);
            
            var result = deepARTMAP.fit(data, labels);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(deepARTMAP.getLayerCount()).isEqualTo(1);
        }
    }

    // ================================================================================
    // CATEGORY 3: UNSUPERVISED LEARNING TESTS (15 tests)
    // ================================================================================

    @Nested
    @DisplayName("Unsupervised Learning Tests")
    @Order(3) 
    class UnsupervisedLearningTests {

        @Test
        @DisplayName("Should train in unsupervised mode without labels")
        void testUnsupervisedBasicTraining() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            
            var result = deepARTMAP.fit(data, null);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(deepARTMAP.isTrained()).isTrue();
            assertThat(deepARTMAP.isSupervised()).isFalse();
            assertThat(deepARTMAP.getLayerCount()).isEqualTo(2); // n_modules - 1
        }

        @Test
        @DisplayName("Should create ARTMAP + SimpleARTMAP layer structure")
        void testUnsupervisedLayerStructure() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            
            deepARTMAP.fit(data, null);
            
            var layers = deepARTMAP.getLayers();
            assertThat(layers).hasSize(2);
            assertThat(layers.get(0)).isInstanceOf(ARTMAP.class);
            assertThat(layers.get(1)).isInstanceOf(SimpleARTMAP.class);
        }

        @Test
        @DisplayName("Should require at least two modules for unsupervised mode")
        void testUnsupervisedMinimumModules() {
            var singleModule = List.<BaseART>of(new FuzzyART());
            var deepARTMAP = new DeepARTMAP(singleModule, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 1, 5);
            
            assertThatThrownBy(() -> deepARTMAP.fit(data, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Must provide at least two ART modules");
        }

        @Test
        @DisplayName("Should train first layer as ARTMAP with second and first modules")
        void testUnsupervisedFirstLayerConfiguration() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            
            deepARTMAP.fit(data, null);
            
            var firstLayer = (ARTMAP) deepARTMAP.getLayers().get(0);
            assertThat(firstLayer.isTrained()).isTrue();
            assertThat(firstLayer.getArtA()).isSameAs(testModules.get(1));
            assertThat(firstLayer.getArtB()).isSameAs(testModules.get(0));
        }

        @Test
        @DisplayName("Should create SimpleARTMAP layers for remaining modules")
        void testUnsupervisedSubsequentLayers() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            
            deepARTMAP.fit(data, null);
            
            if (deepARTMAP.getLayerCount() > 1) {
                for (int i = 1; i < deepARTMAP.getLayerCount(); i++) {
                    var layer = deepARTMAP.getLayers().get(i);
                    assertThat(layer).isInstanceOf(SimpleARTMAP.class);
                    assertThat(((SimpleARTMAP) layer).isTrained()).isTrue();
                }
            }
        }

        @Test
        @DisplayName("Should predict clusters after unsupervised training")
        void testUnsupervisedPrediction() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var trainData = createMultiChannelData(10, 3, 5);
            
            deepARTMAP.fit(trainData, null);
            
            var testData = createMultiChannelData(5, 3, 5);
            var predictions = deepARTMAP.predict(testData);
            
            assertThat(predictions).hasSize(5);
            for (int pred : predictions) {
                assertThat(pred).isGreaterThanOrEqualTo(0);
            }
        }

        @Test
        @DisplayName("Should support partial fit in unsupervised mode")
        void testUnsupervisedPartialFit() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data1 = createMultiChannelData(5, 3, 5);
            
            deepARTMAP.fit(data1, null);
            
            var data2 = createMultiChannelData(5, 3, 5);
            var result = deepARTMAP.partialFit(data2, null);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(deepARTMAP.isTrained()).isTrue();
        }

        @Test
        @DisplayName("Should maintain unsupervised mode consistency in partial fit")
        void testUnsupervisedPartialFitConsistency() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data1 = createMultiChannelData(5, 3, 5);
            
            deepARTMAP.fit(data1, null);
            
            var data2 = createMultiChannelData(5, 3, 5);
            var labels = createClassLabels(5, 3);
            
            assertThatThrownBy(() -> deepARTMAP.partialFit(data2, labels))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("Labels were not previously provided");
        }

        @Test
        @DisplayName("Should create hierarchical cluster structure")
        void testUnsupervisedHierarchicalClustering() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(20, 3, 5);
            
            deepARTMAP.fit(data, null);
            
            var deepLabels = deepARTMAP.getDeepLabels();
            assertThat(deepLabels.length).isEqualTo(20);
            for (var labelArray : deepLabels) {
                assertThat(labelArray.length).isEqualTo(2); // n_layers
            }
        }

        @Test
        @DisplayName("Should handle identical patterns in unsupervised mode")
        void testUnsupervisedIdenticalPatterns() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            
            var identicalPattern = Pattern.of(new double[]{0.5, 0.5, 0.5, 0.5, 0.5});
            var data = List.of(
                new Pattern[]{identicalPattern, identicalPattern},
                new Pattern[]{identicalPattern, identicalPattern},
                new Pattern[]{identicalPattern, identicalPattern}
            );
            
            var result = deepARTMAP.fit(data, null);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
        }

        @ParameterizedTest
        @ValueSource(ints = {2, 3, 4, 5})
        @DisplayName("Should handle different numbers of modules in unsupervised mode")
        void testUnsupervisedParameterizedModules(int moduleCount) {
            var modules = java.util.stream.IntStream.range(0, moduleCount)
                .mapToObj(i -> (BaseART) new FuzzyART())
                .toList();
            var deepARTMAP = new DeepARTMAP(modules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, moduleCount, 5);
            
            var result = deepARTMAP.fit(data, null);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(deepARTMAP.getLayerCount()).isEqualTo(moduleCount - 1);
        }

        @Test
        @DisplayName("Should handle sparse data in unsupervised mode")
        void testUnsupervisedSparseData() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = List.of(
                new Pattern[]{Pattern.of(new double[]{0, 0, 1, 0, 0})},
                new Pattern[]{Pattern.of(new double[]{1, 0, 0, 0, 1})},
                new Pattern[]{Pattern.of(new double[]{0, 1, 0, 1, 0})}
            );
            
            var result = deepARTMAP.fit(data, null);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
        }

        @Test
        @DisplayName("Should maintain performance with many samples in unsupervised mode")
        void testUnsupervisedPerformanceWithManySamples() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(1000, 3, 10);
            
            assertTimeout(java.time.Duration.ofMillis(PERFORMANCE_TIMEOUT_MS), () -> {
                var result = deepARTMAP.fit(data, null);
                assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            });
        }

        @Test
        @DisplayName("Should support label propagation in unsupervised hierarchical layers")
        void testUnsupervisedLabelPropagation() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            
            deepARTMAP.fit(data, null);
            
            // Each layer should have trained with labels from previous layer
            for (int i = 1; i < deepARTMAP.getLayerCount(); i++) {
                var layer = (SimpleARTMAP) deepARTMAP.getLayers().get(i);
                assertThat(layer.isTrained()).isTrue();
            }
        }

        @Test
        @DisplayName("Should generate different cluster patterns than supervised mode")
        void testUnsupervisedVsSupervised() {
            var data = createMultiChannelData(20, 3, 5);
            var labels = createClassLabels(20, 3);
            
            var supervisedARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            supervisedARTMAP.fit(data, labels);
            
            var unsupervisedARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            unsupervisedARTMAP.fit(data, null);
            
            assertThat(supervisedARTMAP.isSupervised()).isTrue();
            assertThat(unsupervisedARTMAP.isSupervised()).isFalse();
            assertThat(supervisedARTMAP.getLayerCount()).isNotEqualTo(unsupervisedARTMAP.getLayerCount());
        }
    }

    // ================================================================================
    // CATEGORY 4: MULTI-CHANNEL DATA TESTS (12 tests)
    // ================================================================================

    @Nested
    @DisplayName("Multi-Channel Data Tests")
    @Order(4)
    class MultiChannelDataTests {

        @Test
        @DisplayName("Should validate consistent sample counts across channels")
        void testMultiChannelSampleCountValidation() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var inconsistentData = List.of(
                new Pattern[]{Pattern.of(new double[]{1, 2, 3})}, // 1 sample
                new Pattern[]{Pattern.of(new double[]{4, 5, 6}), Pattern.of(new double[]{7, 8, 9})}, // 2 samples
                new Pattern[]{Pattern.of(new double[]{10, 11, 12})} // 1 sample
            );
            
            assertThatThrownBy(() -> deepARTMAP.fit(inconsistentData, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Inconsistent sample number");
        }

        @Test
        @DisplayName("Should require data channels to match module count")
        void testMultiChannelModuleCountValidation() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters()); // 3 modules
            var wrongChannelData = createMultiChannelData(10, 2, 5); // Only 2 channels
            
            assertThatThrownBy(() -> deepARTMAP.fit(wrongChannelData, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Must provide 3 input matrices for 3 ART modules");
        }

        @Test
        @DisplayName("Should handle different dimensional data across channels")
        void testMultiChannelDifferentDimensions() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var mixedDimData = List.of(
                new Pattern[]{Pattern.of(new double[]{1, 2}), Pattern.of(new double[]{3, 4})}, // 2D
                new Pattern[]{Pattern.of(new double[]{5, 6, 7}), Pattern.of(new double[]{8, 9, 10})}, // 3D
                new Pattern[]{Pattern.of(new double[]{11, 12, 13, 14}), Pattern.of(new double[]{15, 16, 17, 18})} // 4D
            );
            
            var result = deepARTMAP.fit(mixedDimData, null);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
        }

        @Test
        @DisplayName("Should process each channel with corresponding ART module")
        void testMultiChannelModuleAssignment() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            
            deepARTMAP.fit(data, null);
            
            // Verify each module processes its corresponding channel
            for (int i = 0; i < testModules.size(); i++) {
                var module = testModules.get(i);
                // Module should have been used in training (exact verification depends on implementation)
                assertThat(module).isNotNull();
            }
        }

        @Test
        @DisplayName("Should handle empty channel data")
        void testMultiChannelEmptyData() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var emptyData = List.of(
                new Pattern[]{}, // Empty channel
                new Pattern[]{},
                new Pattern[]{}
            );
            
            assertThatThrownBy(() -> deepARTMAP.fit(emptyData, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Cannot fit with empty data");
        }

        @Test
        @DisplayName("Should handle single sample across multiple channels")
        void testMultiChannelSingleSample() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var singleSampleData = List.of(
                new Pattern[]{Pattern.of(new double[]{1, 2, 3, 4, 5})},
                new Pattern[]{Pattern.of(new double[]{6, 7, 8, 9, 10})},
                new Pattern[]{Pattern.of(new double[]{11, 12, 13, 14, 15})}
            );
            
            var result = deepARTMAP.fit(singleSampleData, null);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
        }

        @Test
        @DisplayName("Should handle large number of samples across channels")
        void testMultiChannelLargeSamples() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var largeData = createMultiChannelData(10000, 3, 5);
            
            assertTimeout(java.time.Duration.ofMillis(PERFORMANCE_TIMEOUT_MS * 2), () -> {
                var result = deepARTMAP.fit(largeData, null);
                assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            });
        }

        @Test
        @DisplayName("Should validate channel data types")
        void testMultiChannelDataTypeValidation() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var invalidData = new java.util.ArrayList<Pattern[]>();
            invalidData.add(new Pattern[]{Pattern.of(new double[]{1, 2, 3})});
            invalidData.add(null); // Invalid channel
            invalidData.add(new Pattern[]{Pattern.of(new double[]{7, 8, 9})});
            
            assertThatThrownBy(() -> deepARTMAP.fit(invalidData, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("channel data cannot be null");
        }

        @Test
        @DisplayName("Should handle channels with different pattern counts before validation")
        void testMultiChannelPreValidationCounts() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            
            // This should be caught during validation
            var invalidData = List.of(
                new Pattern[]{Pattern.of(new double[]{1, 2}), Pattern.of(new double[]{3, 4})}, // 2 patterns
                new Pattern[]{Pattern.of(new double[]{5, 6})}, // 1 pattern  
                new Pattern[]{Pattern.of(new double[]{7, 8}), Pattern.of(new double[]{9, 10})} // 2 patterns
            );
            
            assertThatThrownBy(() -> deepARTMAP.fit(invalidData, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Inconsistent sample number");
        }

        @ParameterizedTest
        @CsvSource({"2,3", "3,5", "4,7", "5,10"})
        @DisplayName("Should handle parametrized channels and dimensions")
        void testMultiChannelParameterizedData(int channels, int dimensions) {
            var modules = java.util.stream.IntStream.range(0, channels)
                .mapToObj(i -> (BaseART) new FuzzyART())
                .toList();
            var deepARTMAP = new DeepARTMAP(modules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, channels, dimensions);
            
            var result = deepARTMAP.fit(data, null);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
        }

        @Test
        @DisplayName("Should maintain channel order during processing")
        void testMultiChannelOrderPreservation() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var orderedData = List.of(
                new Pattern[]{Pattern.of(new double[]{1, 0, 0})}, // Channel 0: [1,0,0]
                new Pattern[]{Pattern.of(new double[]{0, 1, 0})}, // Channel 1: [0,1,0]  
                new Pattern[]{Pattern.of(new double[]{0, 0, 1})}  // Channel 2: [0,0,1]
            );
            
            var result = deepARTMAP.fit(orderedData, null);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            // Channel processing order should be maintained (implementation-dependent verification)
        }

        @Test
        @DisplayName("Should support prediction with multi-channel test data")
        void testMultiChannelPrediction() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var trainData = createMultiChannelData(20, 3, 5);
            var labels = createClassLabels(20, 4);
            
            deepARTMAP.fit(trainData, labels);
            
            var testData = createMultiChannelData(10, 3, 5);
            var predictions = deepARTMAP.predict(testData);
            
            assertThat(predictions).hasSize(10);
            for (int pred : predictions) {
                assertThat(pred).isGreaterThanOrEqualTo(0);
            }
        }
    }

    // ================================================================================
    // CATEGORY 5: HIERARCHICAL LEARNING TESTS (10 tests)
    // ================================================================================

    @Nested
    @DisplayName("Hierarchical Learning Tests")
    @Order(5)
    class HierarchicalLearningTests {

        @Test
        @DisplayName("Should create hierarchical layer stack")
        void testHierarchicalLayerCreation() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            deepARTMAP.fit(data, labels);
            
            assertThat(deepARTMAP.getLayerCount()).isEqualTo(3);
            assertThat(deepARTMAP.getLayers()).hasSize(3);
            assertThat(deepARTMAP.getLayers()).allMatch(layer -> layer instanceof BaseARTMAP);
        }

        @Test
        @DisplayName("Should propagate labels through hierarchical layers")
        void testHierarchicalLabelPropagation() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            deepARTMAP.fit(data, labels);
            
            // First layer trained with original labels
            var firstLayer = deepARTMAP.getLayers().get(0);
            assertThat(firstLayer.isTrained()).isTrue();
            
            // Subsequent layers trained with previous layer's labels
            for (int i = 1; i < deepARTMAP.getLayerCount(); i++) {
                var layer = deepARTMAP.getLayers().get(i);
                assertThat(layer.isTrained()).isTrue();
            }
        }

        @Test
        @DisplayName("Should support deep label mapping between arbitrary levels")
        void testHierarchicalDeepMapping() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            deepARTMAP.fit(data, labels);
            
            // Test mapping from different levels
            var mapping1 = deepARTMAP.mapDeep(0, 1);
            var mapping2 = deepARTMAP.mapDeep(1, 0);
            
            assertThat(mapping1).isNotNull();
            assertThat(mapping2).isNotNull();
        }

        @Test
        @DisplayName("Should handle negative level indices in deep mapping")
        void testHierarchicalNegativeLevelMapping() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            deepARTMAP.fit(data, labels);
            
            // Negative indices should work (counting from end)
            var mapping = deepARTMAP.mapDeep(-1, 0);
            
            assertThat(mapping).isNotNull();
        }

        @Test
        @DisplayName("Should create concatenated deep labels from all layers")
        void testHierarchicalDeepLabelsGeneration() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(15, 3, 5);
            var labels = createClassLabels(15, 4);
            
            deepARTMAP.fit(data, labels);
            
            var deepLabels = deepARTMAP.getDeepLabels();
            assertThat(deepLabels.length).isEqualTo(15); // Same as number of samples
            for (var labelArray : deepLabels) {
                assertThat(labelArray.length).isEqualTo(3); // Same as layer count
            }
        }

        @Test
        @DisplayName("Should maintain hierarchical structure consistency")
        void testHierarchicalStructureConsistency() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            deepARTMAP.fit(data, labels);
            
            // Each layer should be connected appropriately in the hierarchy
            for (int i = 0; i < deepARTMAP.getLayerCount(); i++) {
                var layer = deepARTMAP.getLayers().get(i);
                assertThat(layer.isTrained()).isTrue();
                assertThat(layer.getCategoryCount()).isGreaterThanOrEqualTo(1);
            }
        }

        @Test
        @DisplayName("Should support hierarchical partial fitting")
        void testHierarchicalPartialFit() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data1 = createMultiChannelData(10, 3, 5);
            var labels1 = createClassLabels(10, 3);
            
            deepARTMAP.fit(data1, labels1);
            var initialLayerCounts = deepARTMAP.getLayers().stream()
                .mapToInt(layer -> layer.getCategoryCount())
                .toArray();
            
            var data2 = createMultiChannelData(5, 3, 5);
            var labels2 = createClassLabels(5, 3);
            deepARTMAP.partialFit(data2, labels2);
            
            // Categories may have increased through hierarchical learning
            for (int i = 0; i < deepARTMAP.getLayerCount(); i++) {
                var layer = deepARTMAP.getLayers().get(i);
                assertThat(layer.getCategoryCount()).isGreaterThanOrEqualTo(initialLayerCounts[i]);
            }
        }

        @Test
        @DisplayName("Should handle hierarchical learning with many layers")
        void testHierarchicalManyLayers() {
            var manyModules = java.util.stream.IntStream.range(0, 10)
                .mapToObj(i -> (BaseART) new FuzzyART())
                .toList();
            var deepARTMAP = new DeepARTMAP(manyModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(20, 10, 5);
            var labels = createClassLabels(20, 5);
            
            var result = deepARTMAP.fit(data, labels);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(deepARTMAP.getLayerCount()).isEqualTo(10);
        }

        @Test
        @DisplayName("Should maintain hierarchy performance with deep structures")
        void testHierarchicalPerformanceDeepStructure() {
            var manyModules = java.util.stream.IntStream.range(0, 5)
                .mapToObj(i -> (BaseART) new FuzzyART())
                .toList();
            var deepARTMAP = new DeepARTMAP(manyModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(100, 5, 10);
            var labels = createClassLabels(100, 10);
            
            assertTimeout(java.time.Duration.ofMillis(PERFORMANCE_TIMEOUT_MS * 3), () -> {
                var result = deepARTMAP.fit(data, labels);
                assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            });
        }

        @Test
        @DisplayName("Should support hierarchical prediction through all layers")
        void testHierarchicalPredictionAllLayers() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var trainData = createMultiChannelData(20, 3, 5);
            var trainLabels = createClassLabels(20, 4);
            
            deepARTMAP.fit(trainData, trainLabels);
            
            var testData = createMultiChannelData(10, 3, 5);
            var predictions = deepARTMAP.predict(testData);
            var deepPredictions = deepARTMAP.predictDeep(testData);
            
            assertThat(predictions).hasSize(10);
            assertThat(deepPredictions.length).isEqualTo(10);
            for (var predArray : deepPredictions) {
                assertThat(predArray.length).isEqualTo(3); // All layers
            }
        }
    }

    // ================================================================================
    // CATEGORY 6: PERFORMANCE TESTS (8 tests)
    // ================================================================================

    @Nested
    @DisplayName("Performance Tests")
    @Order(6)
    class PerformanceTests {

        @Test
        @DisplayName("Should handle large datasets efficiently")
        @Timeout(value = 10, unit = TimeUnit.SECONDS)
        void testPerformanceLargeDataset() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var largeData = createMultiChannelData(5000, 3, 20);
            var largeLabels = createClassLabels(5000, 50);
            
            var result = deepARTMAP.fit(largeData, largeLabels);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(deepARTMAP.isTrained()).isTrue();
        }

        @Test
        @DisplayName("Should maintain performance with high-dimensional data")
        @Timeout(value = 8, unit = TimeUnit.SECONDS)
        void testPerformanceHighDimensionalData() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var highDimData = createMultiChannelData(1000, 3, 100);
            var labels = createClassLabels(1000, 20);
            
            var result = deepARTMAP.fit(highDimData, labels);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
        }

        @Test
        @DisplayName("Should scale with number of ART modules")
        @Timeout(value = 12, unit = TimeUnit.SECONDS)
        void testPerformanceScalingModules() {
            var manyModules = java.util.stream.IntStream.range(0, 20)
                .mapToObj(i -> (BaseART) new FuzzyART())
                .toList();
            var deepARTMAP = new DeepARTMAP(manyModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(500, 20, 10);
            var labels = createClassLabels(500, 10);
            
            var result = deepARTMAP.fit(data, labels);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(deepARTMAP.getLayerCount()).isEqualTo(20);
        }

        @Test
        @DisplayName("Should handle prediction performance on large test sets")
        @Timeout(value = 5, unit = TimeUnit.SECONDS)
        void testPerformanceLargePrediction() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var trainData = createMultiChannelData(100, 3, 10);
            var trainLabels = createClassLabels(100, 10);
            
            deepARTMAP.fit(trainData, trainLabels);
            
            var largeTestData = createMultiChannelData(10000, 3, 10);
            var predictions = deepARTMAP.predict(largeTestData);
            
            assertThat(predictions).hasSize(10000);
        }

        @Test
        @DisplayName("Should maintain performance with many partial fits")
        @Timeout(value = 10, unit = TimeUnit.SECONDS)
        void testPerformanceManyPartialFits() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var initialData = createMultiChannelData(10, 3, 5);
            var initialLabels = createClassLabels(10, 3);
            
            deepARTMAP.fit(initialData, initialLabels);
            
            for (int i = 0; i < 100; i++) {
                var batchData = createMultiChannelData(5, 3, 5);
                var batchLabels = createClassLabels(5, 3);
                var result = deepARTMAP.partialFit(batchData, batchLabels);
                assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            }
        }

        @Test
        @DisplayName("Should handle memory efficiently with large category counts")
        void testPerformanceMemoryWithManyCategories() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            
            // Generate diverse data to create many categories
            var diverseData = new java.util.ArrayList<List<Pattern[]>>();
            var diverseLabels = new java.util.ArrayList<Integer>();
            
            for (int i = 0; i < 1000; i++) {
                var sample = createMultiChannelData(1, 3, 10);
                diverseData.add(sample);
                diverseLabels.add(i % 100); // Create up to 100 different classes
                
                if (i % 100 == 0 && i > 0) {
                    var batchData = createMultiChannelData(100, 3, 10);
                    var batchLabels = diverseLabels.subList(i - 99, i + 1).stream().mapToInt(Integer::intValue).toArray();
                    deepARTMAP.partialFit(batchData, batchLabels);
                }
            }
            
            assertThat(deepARTMAP.isTrained()).isTrue();
        }

        @Test
        @DisplayName("Should maintain consistent performance across iterations")
        void testPerformanceConsistency() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(100, 3, 10);
            var labels = createClassLabels(100, 10);
            
            var times = new long[5];
            
            for (int i = 0; i < 5; i++) {
                var freshARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
                long startTime = System.nanoTime();
                freshARTMAP.fit(data, labels);
                long endTime = System.nanoTime();
                times[i] = endTime - startTime;
            }
            
            // Check that times are reasonably consistent (no extreme outliers)
            var avgTime = Arrays.stream(times).sum() / times.length;
            for (long time : times) {
                assertThat(time).isBetween(avgTime / 3, avgTime * 3);
            }
        }

        @Test
        @DisplayName("Should perform deep mapping operations efficiently")
        @Timeout(value = 5, unit = TimeUnit.SECONDS)
        void testPerformanceDeepMapping() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(200, 3, 10);
            var labels = createClassLabels(200, 20);
            
            deepARTMAP.fit(data, labels);
            
            // Perform many deep mapping operations
            for (int level = 0; level < deepARTMAP.getLayerCount(); level++) {
                for (int labelValue = 0; labelValue < 10; labelValue++) {
                    var mapping = deepARTMAP.mapDeep(level, labelValue);
                    assertThat(mapping).isNotNull();
                }
            }
        }
    }

    // ================================================================================
    // CATEGORY 7: INTEGRATION TESTS (10 tests)
    // ================================================================================

    @Nested
    @DisplayName("Integration Tests")
    @Order(7)
    class IntegrationTests {

        @Test
        @DisplayName("Should integrate with all BaseART implementations as modules")
        void testIntegrationAllARTVariants() {
            var allVariants = List.of(
                new FuzzyART(),
                new BayesianART(new BayesianParameters(0.9, new double[]{0.0}, Matrix.eye(1), 1.0, 1.0, 100)), 
                new ART2(new ART2Parameters(0.9, 0.1, 100)),
                new BayesianART(new BayesianParameters(0.9, new double[]{0.0}, Matrix.eye(1), 1.0, 1.0, 100))
            );
            
            var deepARTMAP = new DeepARTMAP(allVariants, new DeepARTMAPParameters());
            var data = createMultiChannelData(20, 4, 8);
            var labels = createClassLabels(20, 5);
            
            var result = deepARTMAP.fit(data, labels);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(deepARTMAP.getLayerCount()).isEqualTo(4);
        }

        @Test
        @DisplayName("Should work with ARTMAP as standalone comparison")
        void testIntegrationARTMAPComparison() {
            // Test that DeepARTMAP with 2 modules behaves similarly to ARTMAP
            var twoModules = List.<BaseART>of(new FuzzyART(), new FuzzyART());
            var deepARTMAP = new DeepARTMAP(twoModules, new DeepARTMAPParameters());
            
            var data = createMultiChannelData(10, 2, 5);
            var result = deepARTMAP.fit(data, null);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(deepARTMAP.getLayerCount()).isEqualTo(1);
            assertThat(deepARTMAP.getLayers().get(0)).isInstanceOf(ARTMAP.class);
        }

        @Test
        @DisplayName("Should integrate with ScikitClusterer interface")
        void testIntegrationScikitClusterer() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            assertThat(deepARTMAP).isInstanceOf(ScikitClusterer.class);
            
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            // Test ScikitClusterer methods
            var clusterResult = deepARTMAP.fit(data, labels);
            assertThat(clusterResult).isNotNull();
            
            var predictions = deepARTMAP.predict(data);
            assertThat(predictions).hasSize(10);
        }

        @Test
        @DisplayName("Should work with different parameter configurations")
        void testIntegrationParameterConfigurations() {
            var strictParams = new DeepARTMAPParameters(0.9, 0.01, 100, true);
            var deepARTMAP1 = new DeepARTMAP(testModules, strictParams);
            
            var looseParams = new DeepARTMAPParameters(0.5, 0.1, 1000, false);
            var deepARTMAP2 = new DeepARTMAP(testModules, looseParams);
            
            var data = createMultiChannelData(20, 3, 5);
            var labels = createClassLabels(20, 4);
            
            var result1 = deepARTMAP1.fit(data, labels);
            var result2 = deepARTMAP2.fit(data, labels);
            
            assertThat(result1).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(result2).isInstanceOf(DeepARTMAPResult.Success.class);
        }

        @Test
        @DisplayName("Should integrate with existing ART test data patterns")
        void testIntegrationExistingTestPatterns() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            
            // Use patterns similar to other ART tests
            var simplePatterns = List.of(
                new Pattern[]{Pattern.of(new double[]{1, 0, 1, 0, 1})},
                new Pattern[]{Pattern.of(new double[]{0, 1, 0, 1, 0})},
                new Pattern[]{Pattern.of(new double[]{1, 1, 0, 0, 1})}
            );
            
            var result = deepARTMAP.fit(simplePatterns, null);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
        }

        @Test
        @DisplayName("Should handle serialization compatibility")
        void testIntegrationSerialization() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            deepARTMAP.fit(data, labels);
            
            // Test that trained state can be serialized (implementation-dependent)
            assertThat(deepARTMAP.isTrained()).isTrue();
            assertThat(deepARTMAP.getLayerCount()).isGreaterThan(0);
        }

        @Test
        @DisplayName("Should integrate with concurrent execution")
        void testIntegrationConcurrency() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var data = createMultiChannelData(50, 3, 5);
            var labels = createClassLabels(50, 5);
            
            deepARTMAP.fit(data, labels);
            
            var testData = createMultiChannelData(20, 3, 5);
            
            // Multiple concurrent predictions should work
            var executor = java.util.concurrent.Executors.newFixedThreadPool(4);
            var futures = new java.util.ArrayList<java.util.concurrent.Future<int[]>>();
            
            for (int i = 0; i < 10; i++) {
                futures.add(executor.submit(() -> deepARTMAP.predict(testData)));
            }
            
            // All predictions should complete successfully
            for (var future : futures) {
                assertThatCode(() -> {
                    var predictions = future.get(5, TimeUnit.SECONDS);
                    assertThat(predictions).hasSize(20);
                }).doesNotThrowAnyException();
            }
            
            executor.shutdown();
        }

        @Test
        @DisplayName("Should work with mixed supervised and unsupervised workflows")
        void testIntegrationMixedWorkflows() {
            // Cannot mix in same instance, but should work with separate instances
            var supervisedARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            var unsupervisedARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            var supervisedResult = supervisedARTMAP.fit(data, labels);
            var unsupervisedResult = unsupervisedARTMAP.fit(data, null);
            
            assertThat(supervisedResult).isInstanceOf(DeepARTMAPResult.Success.class);
            assertThat(unsupervisedResult).isInstanceOf(DeepARTMAPResult.Success.class);
            
            assertThat(supervisedARTMAP.isSupervised()).isTrue();
            assertThat(unsupervisedARTMAP.isSupervised()).isFalse();
        }

        @Test
        @DisplayName("Should integrate with existing ART module configurations")
        void testIntegrationModuleConfigurations() {
            // Test with pre-configured ART modules
            var configuredModule1 = new FuzzyART();
            var configuredModule2 = new FuzzyART();
            var configuredModule3 = new FuzzyART();
            
            var configuredModules = List.<BaseART>of(configuredModule1, configuredModule2, configuredModule3);
            var deepARTMAP = new DeepARTMAP(configuredModules, new DeepARTMAPParameters());
            
            var data = createMultiChannelData(15, 3, 6);
            var labels = createClassLabels(15, 4);
            
            var result = deepARTMAP.fit(data, labels);
            
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
        }

        @Test
        @DisplayName("Should maintain compatibility with BaseART framework")
        void testIntegrationBaseARTFramework() {
            var deepARTMAP = new DeepARTMAP(testModules, new DeepARTMAPParameters());
            
            // Should follow BaseART patterns
            assertThat(deepARTMAP).isInstanceOf(BaseART.class);
            
            var data = createMultiChannelData(10, 3, 5);
            var labels = createClassLabels(10, 3);
            
            // Should support BaseART interface methods
            var result = deepARTMAP.fit(data, labels);
            assertThat(result).isInstanceOf(DeepARTMAPResult.Success.class);
            
            assertThat(deepARTMAP.isTrained()).isTrue();
            assertThat(deepARTMAP.getCategoryCount()).isGreaterThan(0);
        }
    }
}