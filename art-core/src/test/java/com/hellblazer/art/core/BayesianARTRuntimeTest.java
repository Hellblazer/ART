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

import com.hellblazer.art.core.algorithms.BayesianART;
import com.hellblazer.art.core.parameters.BayesianParameters;
import com.hellblazer.art.core.utils.Matrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Simple runtime test for BayesianART to verify implementation works
 * 
 * @author Hal Hildebrand
 */
public class BayesianARTRuntimeTest {
    
    @Test
    public void testBasicCreation() {
        // Test basic creation
        var priorMean = new double[]{0.0, 0.0};
        var priorCovMatrix = new Matrix(2, 2);
        priorCovMatrix.set(0, 0, 1.0);
        priorCovMatrix.set(1, 1, 1.0);
        
        var params = new BayesianParameters(
            0.7,           // vigilance
            priorMean,     // prior mean
            priorCovMatrix, // prior covariance  
            0.1,           // noise variance
            1.0,           // prior precision
            100            // max categories
        );
        
        var bayesianART = new BayesianART(params);
        assertNotNull(bayesianART);
        
        // Test basic properties
        assertEquals(0, bayesianART.getCategoryCount());
        assertFalse(bayesianART.is_fitted());
    }
    
    @Test 
    public void testPatternCreation() {
        var pattern = Pattern.of(1.0, 2.0, 3.0);
        assertNotNull(pattern);
        assertEquals(3, pattern.dimension());
        assertEquals(1.0, pattern.get(0));
        assertEquals(2.0, pattern.get(1));
        assertEquals(3.0, pattern.get(2));
    }
    
    @Test
    public void testMatrixBasics() {
        var matrix = new Matrix(2, 2);
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);
        matrix.set(1, 1, 4.0);
        
        assertEquals(1.0, matrix.get(0, 0));
        assertEquals(2.0, matrix.get(0, 1));
        assertEquals(3.0, matrix.get(1, 0));
        assertEquals(4.0, matrix.get(1, 1));
    }
}