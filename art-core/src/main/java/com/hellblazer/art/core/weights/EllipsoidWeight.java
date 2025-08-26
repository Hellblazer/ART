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
package com.hellblazer.art.core.weights;

import com.hellblazer.art.core.Pattern;
import com.hellblazer.art.core.DenseVector;
import com.hellblazer.art.core.WeightVector;
import com.hellblazer.art.core.utils.Matrix;
/**
 * Weight vector for EllipsoidART representing an ellipsoidal category region.
 * 
 * @param center Center point of the ellipsoid
 * @param covariance Covariance matrix defining ellipsoid shape and orientation
 * @param sampleCount Number of samples used to train this ellipsoid
 * 
 * @author Hal Hildebrand
 */
public record EllipsoidWeight(
    DenseVector center,
    Matrix covariance, 
    long sampleCount
) implements WeightVector {
    
    public EllipsoidWeight {
        if (center == null) {
            throw new IllegalArgumentException("center cannot be null");
        }
        if (covariance == null) {
            throw new IllegalArgumentException("covariance cannot be null");
        }
        if (center.dimension() != covariance.getRowCount() || 
            center.dimension() != covariance.getColumnCount()) {
            throw new IllegalArgumentException("center dimension must match covariance matrix dimensions");
        }
        if (sampleCount < 0) {
            throw new IllegalArgumentException("sampleCount cannot be negative");
        }
    }
    
    @Override
    public int dimension() {
        return center.dimension();
    }
    
    @Override
    public double get(int index) {
        return center.get(index);
    }
    
    @Override
    public double l1Norm() {
        double sum = 0.0;
        for (int i = 0; i < center.dimension(); i++) {
            sum += Math.abs(center.get(i));
        }
        return sum;
    }
    
    @Override
    public WeightVector update(Pattern input, Object parameters) {
        // EllipsoidWeight updates are handled by EllipsoidART.updateWeights() method
        // Individual weights cannot update themselves without the full ellipsoidal clustering context
        // This method returns this weight unchanged as per WeightVector interface contract
        return this;
    }
    
    @Override
    public String toString() {
        return "EllipsoidWeight{center=" + center + 
               ", covariance=" + covariance + 
               ", samples=" + sampleCount + "}";
    }
}