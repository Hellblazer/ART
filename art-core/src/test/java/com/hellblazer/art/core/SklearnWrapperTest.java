package com.hellblazer.art.core;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SklearnWrapperTest {

    @Test
    void testFuzzyARTFactory() {
        var wrapper = SklearnWrapper.fuzzyART(0.7, 0.001, 1.0);
        assertNotNull(wrapper);
        
        // Test basic functionality
        var data = Arrays.asList(
            new double[]{0.1, 0.2, 0.3},
            new double[]{0.4, 0.5, 0.6},
            new double[]{0.7, 0.8, 0.9}
        );
        
        wrapper.fit(data);
        var predictions = wrapper.predict(data);
        
        assertNotNull(predictions);
        assertEquals(data.size(), predictions.length);
    }

    @Test
    void testBayesianARTFactory() {
        var wrapper = SklearnWrapper.bayesianART(0.7, 3);
        assertNotNull(wrapper);
        
        // Test basic functionality
        var data = Arrays.asList(
            new double[]{0.1, 0.2, 0.3},
            new double[]{0.4, 0.5, 0.6},
            new double[]{0.7, 0.8, 0.9}
        );
        
        wrapper.fit(data);
        var predictions = wrapper.predict(data);
        
        assertNotNull(predictions);
        assertEquals(data.size(), predictions.length);
    }

    @Test
    void testGaussianARTFactory() {
        var wrapper = SklearnWrapper.gaussianART(0.7, 0.01, 2.0);
        assertNotNull(wrapper);
        
        // Test basic functionality
        var data = Arrays.asList(
            new double[]{0.1, 0.2, 0.3},
            new double[]{0.4, 0.5, 0.6},
            new double[]{0.7, 0.8, 0.9}
        );
        
        wrapper.fit(data);
        var predictions = wrapper.predict(data);
        
        assertNotNull(predictions);
        assertEquals(data.size(), predictions.length);
    }

    @Test
    void testHypersphereARTFactory() {
        var wrapper = SklearnWrapper.hypersphereART(0.7, 0.1, 10.0);
        assertNotNull(wrapper);
        
        // Test basic functionality
        var data = Arrays.asList(
            new double[]{0.1, 0.2, 0.3},
            new double[]{0.4, 0.5, 0.6},
            new double[]{0.7, 0.8, 0.9}
        );
        
        wrapper.fit(data);
        var predictions = wrapper.predict(data);
        
        assertNotNull(predictions);
        assertEquals(data.size(), predictions.length);
    }

    @Test
    void testEllipsoidARTFactory() {
        var wrapper = SklearnWrapper.ellipsoidART(0.7, 0.1, 2.0, 1.0);
        assertNotNull(wrapper);
        
        // Test basic functionality
        var data = Arrays.asList(
            new double[]{0.1, 0.2, 0.3},
            new double[]{0.4, 0.5, 0.6},
            new double[]{0.7, 0.8, 0.9}
        );
        
        wrapper.fit(data);
        var predictions = wrapper.predict(data);
        
        assertNotNull(predictions);
        assertEquals(data.size(), predictions.length);
    }

    @Test
    void testFitAndPredict() {
        var wrapper = SklearnWrapper.fuzzyART(0.7, 0.001, 1.0);
        
        var trainData = Arrays.asList(
            new double[]{0.1, 0.2},
            new double[]{0.8, 0.9},
            new double[]{0.2, 0.1},
            new double[]{0.9, 0.8}
        );
        
        wrapper.fit(trainData);
        
        var testData = Arrays.asList(
            new double[]{0.15, 0.25},
            new double[]{0.85, 0.95}
        );
        
        var predictions = wrapper.predict(testData);
        
        assertNotNull(predictions);
        assertEquals(testData.size(), predictions.length);
        
        // Similar patterns should get same category
        assertTrue(predictions[0] >= 0);
        assertTrue(predictions[1] >= 0);
    }

    @Test
    void testFitPredict() {
        var wrapper = SklearnWrapper.fuzzyART(0.7, 0.001, 1.0);
        
        var data = Arrays.asList(
            new double[]{0.1, 0.2},
            new double[]{0.8, 0.9},
            new double[]{0.2, 0.1},
            new double[]{0.9, 0.8}
        );
        
        var labels = wrapper.fitPredict(data);
        
        assertNotNull(labels);
        assertEquals(data.size(), labels.length);
        
        // All labels should be valid category indices
        for (int label : labels) {
            assertTrue(label >= 0);
        }
    }

    @Test
    void testPartialFit() {
        var wrapper = SklearnWrapper.fuzzyART(0.7, 0.001, 1.0);
        
        // First batch
        var batch1 = Arrays.asList(
            new double[]{0.1, 0.2},
            new double[]{0.8, 0.9}
        );
        
        wrapper.partialFit(batch1);
        var categoriesAfterBatch1 = wrapper.getCategoryCount();
        assertTrue(categoriesAfterBatch1 > 0);
        
        // Second batch
        var batch2 = Arrays.asList(
            new double[]{0.5, 0.5},
            new double[]{0.3, 0.7}
        );
        
        wrapper.partialFit(batch2);
        var categoriesAfterBatch2 = wrapper.getCategoryCount();
        assertTrue(categoriesAfterBatch2 >= categoriesAfterBatch1);
    }

    @Test
    void testGetSetParams() {
        var wrapper = SklearnWrapper.fuzzyART(0.7, 0.001, 1.0);
        
        var params = wrapper.getParams();
        assertNotNull(params);
        assertTrue(params.containsKey("vigilance"));
        assertEquals(0.7, params.get("vigilance"));
        
        // Modify vigilance
        Map<String, Object> newParams = Map.of("vigilance", 0.8);
        wrapper.setParams(newParams);
        
        var updatedParams = wrapper.getParams();
        assertEquals(0.8, updatedParams.get("vigilance"));
    }

    @Test
    void testScore() {
        var wrapper = SklearnWrapper.fuzzyART(0.7, 0.001, 1.0);
        
        var data = Arrays.asList(
            new double[]{0.1, 0.2},
            new double[]{0.8, 0.9},
            new double[]{0.2, 0.1},
            new double[]{0.9, 0.8}
        );
        
        wrapper.fit(data);
        
        // Check that we have categories after fitting
        assertTrue(wrapper.getCategoryCount() > 0, "No categories created");
        
        var score = wrapper.score(data);
        // Score should be non-negative (silhouette-like metric)
        assertFalse(Double.isNaN(score), "Score is NaN");
        assertTrue(score >= 0.0, "Score was: " + score);
    }

    @Test
    void testTransform() {
        var wrapper = SklearnWrapper.fuzzyART(0.7, 0.001, 1.0);
        
        var data = Arrays.asList(
            new double[]{0.1, 0.2},
            new double[]{0.8, 0.9},
            new double[]{0.2, 0.1},
            new double[]{0.9, 0.8}
        );
        
        wrapper.fit(data);
        
        var testData = Arrays.asList(
            new double[]{0.15, 0.25},
            new double[]{0.85, 0.95}
        );
        
        var transformed = wrapper.transform(testData);
        
        assertNotNull(transformed);
        assertEquals(testData.size(), transformed.length);
        assertEquals(wrapper.getCategoryCount(), transformed[0].length);
        
        // Each row should sum to 1 (one-hot encoding)
        for (var row : transformed) {
            var sum = Arrays.stream(row).sum();
            assertEquals(1.0, sum, 0.001);
        }
    }

    @Test
    void testFitTransform() {
        var wrapper = SklearnWrapper.fuzzyART(0.7, 0.001, 1.0);
        
        var data = Arrays.asList(
            new double[]{0.1, 0.2},
            new double[]{0.8, 0.9},
            new double[]{0.2, 0.1},
            new double[]{0.9, 0.8}
        );
        
        var transformed = wrapper.fitTransform(data);
        
        assertNotNull(transformed);
        assertEquals(data.size(), transformed.length);
        assertTrue(wrapper.getCategoryCount() > 0);
        assertEquals(wrapper.getCategoryCount(), transformed[0].length);
    }

    @Test
    void testEmptyDataHandling() {
        var wrapper = SklearnWrapper.fuzzyART(0.7, 0.001, 1.0);
        
        List<double[]> emptyData = Arrays.asList();
        
        wrapper.fit(emptyData);
        assertEquals(0, wrapper.getCategoryCount());
        
        var predictions = wrapper.predict(emptyData);
        assertNotNull(predictions);
        assertEquals(0, predictions.length);
    }

    @Test
    void testSingleSampleHandling() {
        var wrapper = SklearnWrapper.fuzzyART(0.7, 0.001, 1.0);
        
        var singleSample = Arrays.asList(
            new double[]{0.5, 0.5}
        );
        
        wrapper.fit(singleSample);
        assertEquals(1, wrapper.getCategoryCount());
        
        var prediction = wrapper.predict(singleSample);
        assertEquals(1, prediction.length);
        assertEquals(0, prediction[0]);
    }

    @Test
    void testLargeDataset() {
        var wrapper = SklearnWrapper.fuzzyART(0.9, 0.001, 1.0);
        
        // Generate larger dataset
        var data = new java.util.ArrayList<double[]>();
        for (int i = 0; i < 100; i++) {
            data.add(new double[]{
                Math.random(),
                Math.random(),
                Math.random()
            });
        }
        
        wrapper.fit(data);
        assertTrue(wrapper.getCategoryCount() > 0);
        assertTrue(wrapper.getCategoryCount() <= 100);
        
        var predictions = wrapper.predict(data);
        assertEquals(data.size(), predictions.length);
    }

    @Test
    void testDifferentVigilanceLevels() {
        var data = Arrays.asList(
            new double[]{0.1, 0.2},
            new double[]{0.2, 0.3},
            new double[]{0.8, 0.9},
            new double[]{0.9, 0.8}
        );
        
        // Low vigilance - fewer categories
        var lowVigilance = SklearnWrapper.fuzzyART(0.5, 0.001, 1.0);
        lowVigilance.fit(data);
        var lowCategories = lowVigilance.getCategoryCount();
        
        // High vigilance - more categories
        var highVigilance = SklearnWrapper.fuzzyART(0.95, 0.001, 1.0);
        highVigilance.fit(data);
        var highCategories = highVigilance.getCategoryCount();
        
        assertTrue(highCategories >= lowCategories);
    }
}