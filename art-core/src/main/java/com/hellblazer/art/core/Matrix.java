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

/**
 * Matrix implementation for BayesianART - MINIMAL STUB FOR TEST COMPILATION
 * This is a minimal implementation to allow tests to compile.
 * 
 * @author Hal Hildebrand
 */
public class Matrix {
    private final double[][] data;
    private final int rows;
    private final int cols;
    
    public Matrix(double[][] data) {
        this.data = data;
        this.rows = data.length;
        this.cols = data.length > 0 ? data[0].length : 0;
    }
    
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }
    
    public double get(int row, int col) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw new IndexOutOfBoundsException("Index (" + row + ", " + col + ") out of bounds for " + rows + "x" + cols + " matrix");
        }
        return data[row][col];
    }
    
    public void set(int row, int col, double value) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw new IndexOutOfBoundsException("Index (" + row + ", " + col + ") out of bounds for " + rows + "x" + cols + " matrix");
        }
        data[row][col] = value;
    }
    
    public int getRowCount() {
        return rows;
    }
    
    public int getColumnCount() {
        return cols;
    }
    
    public Matrix multiply(double scalar) {
        var result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }
    
    public Matrix add(Matrix other) {
        if (other.rows != rows || other.cols != cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for addition");
        }
        var result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }
    
    public Matrix inverse() {
        if (rows != cols) {
            throw new IllegalArgumentException("Matrix must be square to compute inverse");
        }
        
        // Use Gaussian elimination with partial pivoting
        int n = rows;
        var augmented = new Matrix(n, 2 * n);
        
        // Create augmented matrix [A|I]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented.set(i, j, get(i, j));
                augmented.set(i, j + n, i == j ? 1.0 : 0.0);
            }
        }
        
        // Forward elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented.get(k, i)) > Math.abs(augmented.get(maxRow, i))) {
                    maxRow = k;
                }
            }
            
            // Swap rows
            if (maxRow != i) {
                for (int j = 0; j < 2 * n; j++) {
                    double temp = augmented.get(i, j);
                    augmented.set(i, j, augmented.get(maxRow, j));
                    augmented.set(maxRow, j, temp);
                }
            }
            
            // Check for singular matrix
            if (Math.abs(augmented.get(i, i)) < 1e-12) {
                throw new ArithmeticException("Matrix is singular and cannot be inverted");
            }
            
            // Make diagonal element 1
            double pivot = augmented.get(i, i);
            for (int j = 0; j < 2 * n; j++) {
                augmented.set(i, j, augmented.get(i, j) / pivot);
            }
            
            // Eliminate column
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented.get(k, i);
                    for (int j = 0; j < 2 * n; j++) {
                        augmented.set(k, j, augmented.get(k, j) - factor * augmented.get(i, j));
                    }
                }
            }
        }
        
        // Extract inverse matrix
        var inverse = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse.set(i, j, augmented.get(i, j + n));
            }
        }
        
        return inverse;
    }
    
    public double determinant() {
        if (rows != cols) {
            throw new IllegalArgumentException("Matrix must be square to compute determinant");
        }
        
        if (rows == 1) {
            return get(0, 0);
        }
        if (rows == 2) {
            return get(0, 0) * get(1, 1) - get(0, 1) * get(1, 0);
        }
        
        // Use LU decomposition for larger matrices
        int n = rows;
        var copy = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                copy.set(i, j, get(i, j));
            }
        }
        
        double det = 1.0;
        int swapCount = 0;
        
        // Gaussian elimination with partial pivoting
        for (int i = 0; i < n; i++) {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(copy.get(k, i)) > Math.abs(copy.get(maxRow, i))) {
                    maxRow = k;
                }
            }
            
            // Swap rows if needed
            if (maxRow != i) {
                for (int j = 0; j < n; j++) {
                    double temp = copy.get(i, j);
                    copy.set(i, j, copy.get(maxRow, j));
                    copy.set(maxRow, j, temp);
                }
                swapCount++;
            }
            
            // Check for zero pivot
            if (Math.abs(copy.get(i, i)) < 1e-12) {
                return 0.0; // Singular matrix
            }
            
            det *= copy.get(i, i);
            
            // Eliminate below pivot
            for (int k = i + 1; k < n; k++) {
                double factor = copy.get(k, i) / copy.get(i, i);
                for (int j = i; j < n; j++) {
                    copy.set(k, j, copy.get(k, j) - factor * copy.get(i, j));
                }
            }
        }
        
        // Adjust sign for row swaps
        return swapCount % 2 == 0 ? det : -det;
    }
    
    public double[][] toArray() {
        var result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, result[i], 0, cols);
        }
        return result;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        Matrix matrix = (Matrix) obj;
        if (rows != matrix.rows || cols != matrix.cols) return false;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Double.compare(data[i][j], matrix.data[i][j]) != 0) {
                    return false;
                }
            }
        }
        return true;
    }
    
    @Override
    public int hashCode() {
        int result = Integer.hashCode(rows);
        result = 31 * result + Integer.hashCode(cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result = 31 * result + Double.hashCode(data[i][j]);
            }
        }
        return result;
    }
    
    @Override
    public String toString() {
        var sb = new StringBuilder();
        sb.append("Matrix").append(rows).append("x").append(cols).append("[");
        for (int i = 0; i < rows; i++) {
            if (i > 0) sb.append(", ");
            sb.append("[");
            for (int j = 0; j < cols; j++) {
                if (j > 0) sb.append(", ");
                sb.append(String.format("%.6f", data[i][j]));
            }
            sb.append("]");
        }
        sb.append("]");
        return sb.toString();
    }
}