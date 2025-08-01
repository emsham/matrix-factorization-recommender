# Matrix Factorization for Movie Recommendations using Coordinate Ascent MAP Estimation

A complete implementation of probabilistic matrix factorization using Coordinate Ascent Maximum A Posteriori (MAP) estimation from scratch in Python. This project demonstrates advanced probabilistic machine learning techniques for collaborative filtering and recommendation systems.

## Project Overview

This project implements a Gaussian-based matrix factorization model for movie rating prediction, using coordinate ascent optimization with Bayesian regularization. The implementation includes:

- **Custom Coordinate Ascent MAP**: Full implementation of the optimization algorithm with gradient computation
- **Latent Factor Discovery**: Automatic learning of user preferences and item attributes
- **Hyperparameter Optimization**: Grid search for optimal model configuration
- **Comprehensive Visualization**: Multiple visualization tools for model analysis and convergence monitoring

## Dataset

The project analyzes the **MovieLens dataset**, processing user-movie ratings to learn latent representations for personalized movie recommendations.

## Key Features

### Core Implementation

- **CoordinateAscentMAPEstimate Class**: Complete implementation of matrix factorization with MAP estimation
- **Log-Likelihood Optimization**: Convergence monitoring and Bayesian objective function
- **Model Parameters**: Automatic updates of θ (user factors) and β (item factors) matrices
- **Regularization**: Configurable Gaussian priors (λ*θ, λ*β) for preventing overfitting

### Advanced Functionality

- **Optimal Parameter Selection**: Automated selection of K (latent factors) and γ (learning rate)
- **Train-Test Split**: Random hold-out validation for model evaluation
- **Convergence Detection**: Windowed relative tolerance checking with divergence handling
- **Missing Data Handling**: Robust handling of sparse rating matrices

### Visualization Suite

- **Hyperparameter Analysis**: 2D plots showing log-likelihood vs. gamma for different K values
- **3D Surface Plots**: Comprehensive visualization of K, γ, and log-likelihood relationships
- **Convergence Plots**: Training progress monitoring with iteration-wise log-likelihood
- **Model Comparison**: Grid search results visualization

## Technical Stack

- **Python 3.x**
- **NumPy**: Efficient matrix operations and numerical computations
- **Pandas**: Data manipulation and pivot operations
- **SciPy**: Statistical functions for Gaussian distributions
- **Matplotlib**: Comprehensive data visualization
- **Scikit-learn**: Train-test splitting utilities

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

### Usage Example

```python
from CoordinateAscentMAPEstimate import CoordinateAscentMAPEstimate

# Initialize the model
model = CoordinateAscentMAPEstimate(
    dataframe=movie_lens,
    lambda_theta=1,
    lambda_beta=1,
    sigma=1
)

# Find optimal hyperparameters
best_K, best_gamma, best_log_likelihood, results = model.optimal_fit(
    Ks=[10, 50, 100],
    gammas=[0.0001, 0.001, 0.01, 0.1]
)

# Fit with optimal parameters
thetas, betas, converged = model.basic_fit(K=100, gamma=0.001)
```

## Results & Analysis

### Model Performance

- **Convergence**: Achieved within reasonable iterations for most hyperparameter combinations
- **Optimal Configuration**: K=100 latent factors with γ=0.001 learning rate
- **Scalability**: Efficient matrix operations enabling analysis of large rating matrices

### Key Findings

- **Latent Factors**: 100 dimensions capture user preferences and movie characteristics effectively
- **Learning Rate**: γ=0.001 provides optimal balance between convergence speed and stability
- **Regularization**: Gaussian priors effectively prevent overfitting while maintaining predictive accuracy
- **Model Robustness**: Handles sparse data and missing ratings gracefully

## Mathematical Foundation

The implementation is based on the MAP estimation framework:

**Objective Function**:

```
log p(θ, β | X) ∝ log p(X | θ, β) + log p(θ) + log p(β)
```

**Gradient Updates**:

- **θ update**: User preference factors with L2 regularization
- **β update**: Item attribute factors with L2 regularization
- **Prediction**: R̂ = θβᵀ

## Project Structure

```
├── notebook.ipynb                # Main implementation and analysis
└── README.md                     # Project documentation
```

## Experimental Design

1. **Data Preprocessing**: Pivot user-item-rating matrix construction
2. **Model Training**: Coordinate ascent with gradient-based updates
3. **Hyperparameter Search**: Systematic grid search over K and γ
4. **Evaluation**: Log-likelihood on held-out test data
5. **Visualization**: Multi-dimensional analysis of results

## Educational Value

This project demonstrates:

- **Probabilistic Modeling**: Implementation of Bayesian matrix factorization
- **Optimization Theory**: Coordinate ascent and gradient computation
- **Recommendation Systems**: Practical application of collaborative filtering
- **Software Engineering**: Clean, modular code with comprehensive documentation

## Performance Metrics

- **Log-Likelihood**: Monitoring of optimization objective
- **Convergence Rate**: Relative tolerance-based convergence checking
- **Prediction Accuracy**: Evaluation on held-out test ratings
- **Computational Efficiency**: Optimized matrix operations using NumPy

## Implementation Highlights

### Code Quality

- **Object-Oriented Design**: Clean class structure with separation of concerns
- **Documentation**: Comprehensive docstrings explaining each method
- **Error Handling**: Robust handling of edge cases and numerical issues
- **Modularity**: Reusable components for different factorization tasks

### Technical Features

- **Numerical Stability**: Careful handling of log computations and matrix operations
- **Memory Efficiency**: Sparse matrix support for large-scale datasets
- **Convergence Monitoring**: Advanced windowed checking with divergence detection
- **Hyperparameter Tuning**: Automated grid search with result tracking

## Future Enhancements

- Implementation of stochastic gradient descent for improved scalability
- Extension to implicit feedback and side information
- Comparison with other factorization methods (SVD++, NMF, ALS)
- Online learning capabilities for real-time recommendations
- GPU acceleration for large-scale deployments

## References

- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems
- Salakhutdinov, R., & Mnih, A. (2008). Bayesian Probabilistic Matrix Factorization using MCMC
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning

---

_This project was developed as part of graduate coursework in Probabilistic Models and Machine Learning at Columbia University._
