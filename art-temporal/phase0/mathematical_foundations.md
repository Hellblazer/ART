# Phase 0: Mathematical Foundations
## Temporal ART Research Project

---

## Week 1-2: Equation Extraction and Analysis

### Core Equations from Kazerounian & Grossberg (2014)

#### 1. Working Memory Dynamics (STORE 2 Model)
```
Equation 1: Item Storage with Primacy Gradient
dX_i/dt = -A_i * X_i + (B - X_i) * S_i - X_i * Σ(j≠i) I_ij

Where:
- X_i = activation of item i in working memory
- A_i = decay rate (position-dependent)
- B = upper bound (1.0)
- S_i = self-excitation signal
- I_ij = lateral inhibition from position j to i
```

#### 2. Masking Field Dynamics
```
Equation 5: Multi-Scale Competitive Dynamics
dY_jk/dt = -α * Y_jk + (β - Y_jk) * [f(Y_jk) + I_jk] - Y_jk * Σ g(Y_lm)

Where:
- Y_jk = activity of cell at position j, scale k
- α = passive decay (0.1)
- β = upper bound (1.0)
- f(Y_jk) = self-excitation function
- I_jk = bottom-up input
- g(Y_lm) = lateral inhibition from cell (l,m)
```

#### 3. Transmitter Dynamics
```
Equation 7: Habituative Transmitter Gates
dZ_i/dt = ε(1 - Z_i) - Z_i(λ * S_i + μ * S_i²)

Where:
- Z_i = transmitter level at synapse i
- ε = recovery rate (0.005 - very slow)
- λ = linear depletion rate (0.1)
- μ = quadratic depletion rate (0.05)
- S_i = presynaptic signal strength
```

#### 4. Adaptive Filter Learning
```
Equation 12: Competitive Instar Learning
dW_ij/dt = L * Y_j * (X_i * Z_i - W_ij)

Where:
- W_ij = weight from working memory i to masking field j
- L = learning rate (0.1)
- Y_j = masking field activation (winner)
- X_i = working memory activation
- Z_i = transmitter gate value
```

---

## Week 3-4: Convergence Analysis

### Stability Conditions

#### Shunting Dynamics Stability
For the shunting equation to converge, we need:

```python
# Python/NumPy analysis
import numpy as np

def analyze_shunting_stability(A, B, C):
    """
    Analyze stability of shunting dynamics
    dx/dt = -Ax + (B-x)E - (x+C)I
    """
    # Jacobian matrix at equilibrium
    def jacobian(x_eq, E, I):
        return -(A + E + I)

    # Equilibrium point
    x_eq = (B*E - C*I) / (A + E + I)

    # Eigenvalue (1D case)
    eigenvalue = jacobian(x_eq, E, I)

    # Stability condition: eigenvalue < 0
    is_stable = eigenvalue < 0

    return {
        'equilibrium': x_eq,
        'eigenvalue': eigenvalue,
        'stable': is_stable,
        'convergence_rate': abs(eigenvalue)
    }

# Test parameter ranges
A_range = [0.05, 0.1, 0.2]
for A in A_range:
    result = analyze_shunting_stability(A=A, B=1.0, C=0.0)
    print(f"A={A}: Stable={result['stable']}, Rate={result['convergence_rate']:.3f}")
```

#### Multi-Scale Interaction Stability
For asymmetric inhibition between scales:

```
Stability Criterion:
Σ_k |W_jk| < 1  (spectral radius < 1)

Where W_jk is the inhibition strength from scale k to j
```

---

## Week 5-6: Numerical Methods Selection

### Integration Methods Comparison

#### 1. Euler Method (Current Implementation)
```java
// Simple but potentially unstable
x[t+1] = x[t] + dt * f(x[t])

Pros: Simple, fast
Cons: Can be unstable for stiff equations
Stability: Requires dt < 2/|λ_max|
```

#### 2. Runge-Kutta 4 (Recommended)
```java
// More stable and accurate
k1 = dt * f(x[t])
k2 = dt * f(x[t] + k1/2)
k3 = dt * f(x[t] + k2/2)
k4 = dt * f(x[t] + k3)
x[t+1] = x[t] + (k1 + 2*k2 + 2*k3 + k4) / 6

Pros: Much more stable, 4th order accurate
Cons: 4x computation per step
Stability: Larger stable dt range
```

#### 3. Adaptive Step Size
```java
// Automatically adjust dt for stability
public class AdaptiveIntegrator {
    private static final double TOL = 1e-6;

    public double[] integrate(double[] state, double t, double dt) {
        double[] k1 = computeDerivative(state, t);
        double error = estimateError(k1, dt);

        if (error > TOL) {
            // Reduce step size
            dt = dt * 0.9 * Math.pow(TOL/error, 0.2);
            return integrate(state, t, dt);
        }

        // Accept step
        return updateState(state, k1, dt);
    }
}
```

---

## Week 7-8: Parameter Characterization

### Critical Parameter Relationships

#### 1. Time Scale Hierarchy
```
τ_working_memory < τ_masking_field < τ_transmitter < τ_weights

Typical values:
- Working memory: 10-100 ms
- Masking field: 50-500 ms
- Transmitter: 500-5000 ms
- Weight adaptation: 1000-10000 ms
```

#### 2. Stability Boundaries
```java
public class ParameterValidator {
    public boolean isStable(Parameters p) {
        // Condition 1: Decay rates positive
        if (p.decayRate <= 0) return false;

        // Condition 2: Time step constraint
        double maxEigenvalue = computeMaxEigenvalue(p);
        if (p.dt >= 2.0 / maxEigenvalue) return false;

        // Condition 3: Learning rate constraint
        if (p.learningRate >= 1.0) return false;

        // Condition 4: Inhibition/excitation balance
        double ratio = p.inhibitionStrength / p.excitationStrength;
        if (ratio < 0.5 || ratio > 2.0) return false;

        return true;
    }
}
```

#### 3. Performance vs Accuracy Tradeoffs
| Parameter | Accuracy Impact | Performance Impact | Recommended Range |
|-----------|----------------|-------------------|------------------|
| dt | High (stability) | High (iterations) | 0.001-0.01 |
| numScales | Medium (capacity) | High (memory) | 3-5 |
| numCells | High (resolution) | High (computation) | 50-200 |
| vigilance | High (selectivity) | Low | 0.5-0.9 |

---

## Validation Test Suite

### 1. Convergence Tests
```java
@Test
public void testShuntingConvergence() {
    ShuntingDynamics dynamics = new ShuntingDynamics(A=0.1, B=1.0, C=0.0);
    double x = 0.5;
    double E = 0.2, I = 0.1;

    // Run to equilibrium
    for (int t = 0; t < 1000; t++) {
        x = dynamics.update(x, E, I, dt=0.01);
    }

    // Check convergence
    double expected = dynamics.computeEquilibrium(E, I);
    assertEquals(expected, x, 1e-3);
}
```

### 2. Stability Tests
```java
@Test
public void testNumericalStability() {
    // Test with extreme inputs
    double[] extremeInputs = {0.0, 1e-10, 1.0, 1e10};

    for (double input : extremeInputs) {
        double result = dynamics.update(0.5, input, 0.1, 0.01);
        assertTrue("Output must be bounded", result >= 0 && result <= 1.0);
        assertFalse("Output must not be NaN", Double.isNaN(result));
    }
}
```

### 3. Conservation Tests
```java
@Test
public void testWeightNormalization() {
    CompetitiveInstar learning = new CompetitiveInstar();
    double[][] weights = learning.getWeights();

    // Check normalization
    for (int j = 0; j < weights.length; j++) {
        double sum = 0;
        for (int i = 0; i < weights[j].length; i++) {
            sum += weights[j][i];
        }
        assertEquals("Weights must sum to 1", 1.0, sum, 1e-6);
    }
}
```

---

## Risk Assessment

### High-Risk Areas

#### 1. Numerical Instability (Probability: 70%)
**Symptoms**: Explosions, oscillations, NaN values
**Mitigation**:
- Use RK4 instead of Euler
- Adaptive time stepping
- Bounds checking

#### 2. Parameter Sensitivity (Probability: 60%)
**Symptoms**: Small changes cause large effects
**Mitigation**:
- Parameter sweep analysis
- Automatic tuning
- Robust defaults

#### 3. Scaling Issues (Probability: 80%)
**Symptoms**: Single scale works, multi-scale fails
**Mitigation**:
- Incremental scaling
- Careful inhibition balance
- Scale-specific testing

---

## Next Steps

### Immediate Actions (This Week)
1. [ ] Implement convergence analyzer
2. [ ] Create parameter sweep tool
3. [ ] Build stability test suite
4. [ ] Document equation derivations
5. [ ] Set up experiment tracking

### Code to Write First
```java
// Priority 1: Mathematical validator
public class MathematicalValidator {
    // All validation methods here
}

// Priority 2: Convergence analyzer
public class ConvergenceAnalyzer {
    // Eigenvalue analysis
    // Lyapunov functions
    // Phase portraits
}

// Priority 3: Parameter optimizer
public class ParameterOptimizer {
    // Grid search
    // Gradient-based optimization
    // Stability constraints
}
```

---

*This foundation work is critical for project success. Do not skip!*