# COMP0171-Bayesian-Deep-Learning-CW1

- Final Grades: 91/100
  - Comments:
  - [Part 1: 15/14] Good implementation, good short answer overall but note that updating one at a time is often *worse* for correlated variables — would get stuck!
  - [Part 2: 17/21] Your SGD with a very low learning rate seems to be getting “stuck”, or stopping early somehow — you’re actually still quite a bit off the optimal parameters. (You can see this even by taking your estimates and adding a little big of Gaussian noise as jitter — this often even leads to higher values of log_joint from what you have, on the quadratic features…). Minor bug in evidence computation. Otherwise looks good. Your features don’t include sine/cosine, although in the short answer you say they do? The features you have actually perform worse than the quadratic features with interactions, both in terms of test accuracy and model evidence… For last part of short answer — note that the MAP estimate already has “regularization”, so careful when describing what is “different” about using the Laplace posterior predictive distribution rather than just the MAP predictions.

---

## Overview

This project demonstrates key Bayesian techniques applied in deep learning, with a focus on:
- **Probabilistic Modeling:** Understanding Bayesian approaches for parameter estimation.
- **Bayesian Classifiers:** Applying Bayesian inference for classification tasks.
- **Posterior Estimation:** Implementing methods to approximate distributions of parameters given data.

The repository includes Jupyter notebooks that guide you through theoretical concepts, implementations, and experiments in Bayesian deep learning.

---

## Project Structure.

Below is an outline of the notebooks included in this repository:

- **(Part 1) Seven Scientists.ipynb**  
  *Description:* This notebook explores Bayesian inference in a simplified setting where multiple scientists contribute measurements. It demonstrates how Bayesian updates refine posterior distributions given multiple observations. The code includes defining prior distributions, updating beliefs with observed data, and visualizing the posterior distributions using Bayesian inference principles.

- **(Part 2) Bayesian Classifiers.ipynb**  
  *Description:* This notebook covers Bayesian classification techniques, comparing Bayesian methods with standard classifiers. The code includes the implementation of Gaussian Naive Bayes, Bayesian decision rules, and posterior computation. Different datasets are used to analyze classification performance under uncertainty.

---

## Usage

- **Running the Notebooks:**  
  Open any of the provided `.ipynb` files with Jupyter Notebook or Jupyter Lab to explore the experiments and implementations.

- **Code Highlights:**  
  - **Probabilistic Modeling:** The implementation includes defining prior distributions and Bayesian updates.
  - **Inference and Classification:** The Bayesian Classifier notebook applies probabilistic models to classification tasks, showing step-by-step computations of posterior probabilities.
  - **Data Processing:** The notebooks demonstrate handling real and synthetic datasets, preprocessing steps, and visualizing uncertainty in predictions.
  
- **Customization:**  
  Users can modify hyperparameters of prior distributions, experiment with different likelihood functions, and extend the analysis to other datasets.

---

## Part 1 Task 1

### Function Overview

The function `log_joint` computes the **unnormalized log joint probability** of the parameters $$\mu$$ (a scalar) and $$\sigma$$ (a vector of length 7), given the prior distributions and observed data.

### Function Breakdown

```python
def log_joint(mu, sigma, alpha=50, beta=0.5):
```

- $$\mu$$ : A scalar parameter (mean of a normal distribution).
- $$\sigma$$ : A tensor (vector of length 7) representing standard deviations.
- $$\alpha$$ : Standard deviation of the Gaussian prior on `mu`, defaulted to **50**.
- $$\beta$$ : Rate of the Exponential prior on each `sigma_i`, defaulted to **0.5**.

### Input Validation

```python
assert mu.ndim == 0
assert sigma.shape == (7,)
if torch.any(sigma <= 0):
    return float('-inf')
```

- Ensures `mu` is a scalar and `sigma` has the expected shape.
- Returns **negative infinity (-inf)** if any `sigma_i` is **≤ 0**, ensuring all elements in `sigma` are strictly positive.

### Prior Distribution on `mu`

```python
log_p_mu = dist.Normal(0, alpha).log_prob(mu)
```

- Assumes $$(\mu \sim 𝒩(0, α²))$$, a normal distribution with mean 0 and variance $$\alpha^2$$.
- Computes the **log probability** of `mu` under this prior.

### Prior Distribution on `sigma`

```python
log_p_sigma = dist.Exponential(beta).log_prob(sigma).sum(0)
```

- Each  follows an **Exponential distribution**:\
 $$\(\sigma_i \sim ℰ(β)\)$$, meaning $$\(p(σᵢ) = β e^(-βσᵢ)\)$$.
- Computes the **log probability** for all `sigma_i`.
- `.sum(0)` sums over all 7 elements, assuming independence.

### Likelihood Calculation

```python
log_likelihood = dist.Normal(mu, sigma).log_prob(measurements).sum(0)
```

- Observations follow $$\(z_i \sim 𝒩(μ, σᵢ²)\)$$.
- Computes log probability of `measurements` given normal likelihood $$\(p(zᵢ | μ, σᵢ)\)$$.
- `.sum(0)` aggregates across observations.

### Combining Log Probabilities

```python
log_joint = log_p_mu + log_p_sigma + log_likelihood
return log_joint
```

- The total log-joint probability is the **sum** of:
  - `log_p_mu` → Log prior for `mu`
  - `log_p_sigma` → Log prior for `sigma`
  - `log_likelihood` → Log likelihood of observed data
- Returns the **log joint probability**.

### Key Takeaways

- Ensures valid inputs (`sigma > 0`).
- Uses **log probabilities** to avoid numerical instability.
- Applicable for **Bayesian inference**, e.g., **posterior estimation** or **MCMC sampling**.

---
## Task 2 Part 1

### Function Overview
The goal is to implement an **MCMC sampler** using the **Metropolis-Hastings algorithm** to sample from the posterior distribution:

$$ p(\mu, \sigma \mid x, \alpha, \beta) $$

This requires implementing two key functions:

1. **Proposal Distribution:** `get_mcmc_proposal(mu, sigma)` 
   - Defines proposal distributions for both $$\mu$$ and $$\sigma$$.
   
2. **MCMC Step:** `mcmc_step(mu, sigma, alpha, beta)`
   - Proposes new values for $$\mu$$ and $$\sigma$$, computes acceptance probability, and decides whether to accept the new values.

### Proposal Distribution
Defines normal proposal distributions centered at the current values:

$$q(\mu' | \mu) = \mathcal{N}(\mu, 0.5^2)$$
$$q(\sigma' | \sigma) = \mathcal{N}(\sigma, 0.1^2)$$

These ensure small, controlled updates to the parameters.

### MCMC Step
- Samples new candidates $$\mu'$$ and $$\sigma'$$ from the proposal distributions.
- Computes the log-joint probability ratio:
  
  $$r = \log p(\mu', \sigma' | x) - \log p(\mu, \sigma | x)$$
  
- Converts it into an acceptance probability:
  
  $$a = \min(1, e^r)$$
  
- Accepts $$\mu'$$, $$\sigma'$$ with probability $$a$$, otherwise retains the current values.

### `run_mcmc`
Given by professor. Runs the Metropolis-Hastings algorithm for a fixed number of iterations, storing accepted samples and computing the acceptance rate.

### `alg_parameters`
Defines:
- **Total MCMC iterations:** `N_samples = 50000`
- **Burn-in period:** `N_burnin = 3000`


Changing **`N_samples`** and **`N_burnin`** affects the accuracy and efficiency of MCMC sampling:
* **`N_samples` (Total Iterations)**
  * **Higher**: Better posterior approximation, but slower.
  * **Lower**: Faster but risks poor estimation.
* **`N_burnin` (Burn-in Period)**
  * **Higher**: Removes initial biased samples, improving accuracy.
  * **Lower**: May retain non-converged samples, leading to bias.

### **Key Trade-offs**
* Ensure **`N_samples` is large enough** for reliable posterior estimates.
* **`N_burnin` ~10-20%** of `N_samples` to discard early non-converged samples.
* Use **trace plots** or **convergence tests** to fine-tune values.

These control the convergence and stabilization of the Markov chain.

## Experiments and Results

### Acceptance Rate
- The acceptance rate of **~22.5%** is within the optimal range (20-50%), indicating a well-tuned proposal distribution.
- A lower rate suggests too small steps, while a higher rate may indicate inefficient exploration.

### Trace Plot & Histogram for $$\mu$$
- The **trace plot** shows convergence after an initial burn-in, meaning the MCMC chain stabilizes.
- The **histogram** suggests a unimodal posterior distribution centered around **$$\mu \approx 10$$**.

### Trace Plot for $$\sigma$$ Values
- Different **scientists' standard deviations** show distinct trends.
- Some chains appear more stable, while others fluctuate, indicating varied uncertainty levels across different sources.

### Boxplot of $$\sigma$$ Estimates
- Shows the distribution of estimated standard deviations across different scientists.
- Some measurements exhibit **higher uncertainty (larger spread)** while others remain tightly concentrated.

### Key Takeaways
- The MCMC chain **converges properly**, making posterior inference reliable.
- Variability in $$\sigma$$ suggests that some scientists' measurements have **higher uncertainty** than others.
- The **proposal distribution** is well-tuned, leading to an optimal acceptance rate.

---
## Task 3 Part 1

### Posterior Expectations
We estimate the expectation and probability of $$\mu$$ under the posterior distribution:

1. **Expected Value of $$\mu$$**
   - The estimated expectation is **$$E[\mu] = 9.8172$$**.
   - This represents the mean of the posterior samples after the burn-in phase.

2. **Posterior Probability $$Pr(\mu < 9)$$**
   - The estimated probability is **$$Pr(\mu < 9) = 0.0141$$**.
   - This indicates a very low probability that $$\mu$$ is less than 9 under the posterior distribution.

### Key Takeaways
- The MCMC chain **converges properly**, making posterior inference reliable.
- Variability in $$\sigma$$ suggests that some scientists' measurements have **higher uncertainty** than others.
- The **proposal distribution** is well-tuned, leading to an optimal acceptance rate.
- The expectation estimation confirms that the inferred $$\mu$$ aligns with the posterior distribution, while the low probability of $$\mu < 9$$ indicates high confidence in the estimate.

---
## Task 1 Part 2

This section covers Bayesian logistic regression using two sets of feature transformations applied to data $$\mathbf{x} \in \mathbb{R}^2$$.

### Feature Transformations
We define two types of feature maps:

1. **Simple Feature Map**
   $$\phi_{simple}(\mathbf{x})=[1, x_1, x_2]$$

2. **Quadratic Feature Map**
   $$\phi_{quadratic}(\mathbf{x})=[1, x_1, x_2, x_1x_2, x_1^2, x_2^2]$$

The quadratic map expands the feature space to capture nonlinear relationships, whereas the simple map keeps it linear.

### Model Fitting Approaches
We implement Bayesian logistic regression using:
- **MAP Estimation** (Penalized Maximum Likelihood)
- **Laplace Approximation** (Gaussian Approximate Posterior)

### Dataset
A synthetic dataset is used for classification, visualized as follows:

```python
X_train, y_train, X_test, y_test = torch.load("data.pt")
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, edgecolors='k')
```

This generates a scatter plot of the training data, displaying class separability.

### Feature Space Dimensions
The dimensionality of the feature mappings:
```python
print("Dimension of Phi, 'features_simple':", features_simple(X_train).shape)
print("Dimension of Phi, 'features_quadratic':", features_quadratic(X_train).shape)
```

Output:
```plaintext
Dimension of Phi, 'features_simple': torch.Size([100, 3])
Dimension of Phi, 'features_quadratic': torch.Size([100, 6])
```
This confirms that the simple feature map produces a **3D** feature space, while the quadratic transformation expands it to **6D**.

---

## Bayesian Logistic Regression Model

We define a Bayesian logistic regression model with the following structure:

$$\mathbf{w} \sim \mathcal{N}(0, \sigma^2 I)$$

$$\hat{y}_i = \text{Logistic}(\mathbf{w}^T \phi(\mathbf{x}_i))$$

$$y_i \sim \text{Bernoulli}(\hat{y}_i)$$, where $$i = 1, \dots, N$$, and the **Logistic function** is defined as:

$$\text{Logistic}(z) = \frac{1}{1 + \exp\{-z\}}$$

In PyTorch, this is implemented using `torch.sigmoid`.

### Prediction Function

The function `predict_probs_MAP(Phi, w)` computes the probability estimates using:

$$\hat{y} = \text{sigmoid}(\Phi \mathbf{w})$$

- This takes a **design matrix** $$(\Phi)$$ and a **weight vector** $$(\mathbf{w})$$.
- Uses matrix multiplication (`@` in PyTorch) to compute logits.
- Applies `torch.sigmoid` to output probabilities between $$0$$ and $$1$$.

### Log-Joint Probability Function

The function `log_joint(Phi, y, w, sigma)` computes the log probability:

$$\log p(y, \mathbf{w} | \Phi, \sigma)$$

Breaking it down:

1. **Prior log probability**:
   - Assumes $$\mathbf{w}$$ follows a Gaussian prior $$\mathcal{N}(0, \sigma^2 I)$$.
   - Computed using the squared sum of $$\mathbf{w}$$ divided by $$\sigma^2$$.
   - Includes a normalization term using $$\log(2 \pi \sigma^2)$$.

2. **Likelihood computation**:
   - Uses Bernoulli likelihood since $$y_i$$ is binary.
   - Computes predicted probabilities $$\hat{y}$$ using logistic regression.
   - Applies $$\log$$ to prevent numerical underflow.

3. **Combining the two**:
   - Returns the sum of prior and likelihood log-probabilities.
   - Ensures stability by adding a small constant $$1e-6$$ in $$\log$$ terms.

### Summary

- **`predict_probs_MAP`** computes probability estimates using logistic regression.
- **`log_joint`** combines prior and likelihood terms to compute the posterior log-probability.
- The model enables **Bayesian inference** by incorporating uncertainty in $$\mathbf{w}$$ through its prior.

---
## Task 2 Part 2
### MAP Estimation

Maximum a Posteriori (MAP) estimation is used to find the most probable value of $$\mathbf{w}$$ given the data, by maximizing the log joint probability:

$$w_{MAP} =arg_w max logp(y,w∣Φ,σ)$$


### Implementation Details
- **Optimization Approach:** Uses gradient-based optimization with PyTorch's `torch.optim`.
- **Loss Function:** Defined as the **negative log joint probability** (since optimizers minimize by default).
- **Prior on $$\mathbf{w}$$:** Assumes a Gaussian prior $$\mathcal{N}(0,1)$$.
- **Likelihood:** Uses the Bernoulli likelihood with sigmoid activation for classification.
- **Gradient Updates:** Uses **Stochastic Gradient Descent (SGD)** or **Adagrad** optimizer.

### Steps in MAP Estimation
1. **Initialize Weights:**
   - Set $$\mathbf{w}$$ to zero and require gradient computation.

2. **Define Optimizer:**
   - Use `torch.optim.SGD` with a learning rate (e.g., 0.005) and weight decay for regularization.

3. **Iterate for Optimization:**
   - Compute the **log prior** assuming a Gaussian distribution.
   - Compute the **log-likelihood** using Bernoulli distribution and sigmoid activation.
   - Compute **negative log joint probability** (to minimize loss).
   - Perform **backpropagation** and **gradient updates**.
   - Store **loss values** for convergence analysis.

### Summary
- MAP estimation finds optimal weights $$\mathbf{w}_{MAP}$$ by maximizing the posterior log joint.
- Uses gradient descent to iteratively refine $$\mathbf{w}$$.
- Works with **any feature representation**, including both **simple** and **quadratic** mappings.

## Visualization of Classifier Results

We visualize the decision boundaries of the Bayesian logistic regression model using **contour plots** for both feature mappings.

### Simple Features
- The **decision boundary** (black dashed line) is approximately **linear**, which aligns with the simplicity of the feature mapping.
- **Training accuracy: 0.68**, **Test accuracy: 0.68**.
- This suggests the model captures some separation but lacks flexibility to model complex decision boundaries.

### Quadratic Features
- The **decision boundary** is **curved**, demonstrating the impact of polynomial feature expansion.
- **Training accuracy: 0.86**, **Test accuracy: 0.90**.
- Higher accuracy indicates that the model captures **nonlinear patterns** in the data more effectively.

### Key Takeaways
- The quadratic feature mapping significantly improves classification accuracy by introducing **nonlinear boundaries**.
- The **simple feature mapping** is limited to **linear separability**, leading to suboptimal performance on complex data.
- Proper feature engineering plays a **crucial role** in Bayesian logistic regression.

---
## Task 3 Part 2

## Laplace Approximation

Laplace approximation is used to approximate the posterior distribution over the weights by computing a **Gaussian approximation** around the MAP estimate. This involves:

1. **Computing the covariance matrix** as the **inverse Hessian** of the log-joint density.
2. **Sampling from the approximate posterior** to make **Bayesian predictions**.

### Key Functions

#### **`compute_laplace_cov(Phi, y, w_MAP)`**
- Computes an **approximate posterior covariance** for the model weights using the **Hessian matrix**.
- The Hessian matrix is derived by combining:
  - **Prior contribution**: This enforces a Gaussian prior over weights.
  - **Likelihood contribution**: Based on the second derivative of the log-likelihood function, capturing how the model's predictions change with weight updates.
- The **inverse** of the Hessian matrix gives the covariance estimate, representing **uncertainty in the weights**.
- This helps quantify **how confidently the model estimates each weight** given the observed data.

#### **`predict_bayes(Phi, w_MAP, Cov)`**
- This function **makes probabilistic predictions** using the **posterior distribution of the weights**.
- Instead of making a single prediction using the MAP estimate $$\mathbf{w}_{MAP}$$, it:
  1. **Samples multiple weight vectors** from a **Multivariate Gaussian distribution**, centered at $$\mathbf{w}_{MAP}$$ with covariance $$\text{Cov}$$.
  2. **Computes predictions** using each sampled weight.
  3. **Averages the predictions**, integrating over the weight uncertainty.
- This **Monte Carlo sampling approach** leads to **smoother, less overconfident predictions**, especially in regions with limited data.

## Visualization of Laplace Approximation

### Simple vs Quadratic Features
We compare **MAP estimation** and **Laplace approximation** for both feature mappings:

- **MAP Estimation**: Uses point estimates of $$\mathbf{w}$$.
- **Laplace Approximation**: Incorporates uncertainty by marginalizing over sampled $$\mathbf{w}$$ values.

#### **Simple Features**
- The **decision boundary remains linear**.
- **Laplace approximation smooths predictions**, reducing sharp transitions between classes.

#### **Quadratic Features**
- **MAP-based decision boundaries are sharper**.
- **Laplace-based boundaries are softer**, capturing more **uncertainty in class probabilities**.

### **Key Takeaways**
- **Laplace approximation provides uncertainty estimates** by integrating over weight distributions.
- **Predictions become smoother**, reducing overconfidence in decisions.
- **Quadratic features improve accuracy** by enabling non-linear separation of data.

---
## Task 4 Part 2
## Model Comparison

To compare different models, we compute the **marginal likelihood approximation** using the **Laplace approximation**. This **evidence estimate** provides a way to evaluate which feature mapping (simple or quadratic) better fits the data.

### **Key Concept: Model Evidence**
- The **log-evidence** (marginal likelihood) is computed as:
  
  $$\log p(y | \Phi) = \log p(y | w_{MAP}, \Phi) + \log p(w_{MAP}) - \frac{1}{2} \log | H | - \frac{D}{2} \log(2 \pi)$$
  
  where:
  - **Likelihood Term**: $$\log p(y | w_{MAP}, \Phi)$$ is the model's fit to data.
  - **Prior Term**: $$\log p(w_{MAP})$$ accounts for weight regularization.
  - **Determinant Term**: $$-\frac{1}{2} \log | H |$$ adjusts for model complexity (from the Hessian matrix $$H$$).
  - **Constant Term**: $$-\frac{D}{2} \log(2 \pi)$$ is a normalization factor.

### **Implementation Details**
- **`compute_laplace_log_evidence(Phi, y, w_MAP, Cov)`**
  - Computes the **log-evidence** by summing contributions from likelihood, prior, and complexity penalty.
  - Uses **determinant of Hessian** to penalize overfitting.
  
### **Results & Interpretation**
After computing the **model evidence estimates**:
- **Simple Features Evidence**: $$-67.88$$
- **Quadratic Features Evidence**: $$-57.96$$

Since the **quadratic feature model** has a **higher log-evidence**, it suggests that the quadratic transformation better captures patterns in the data **without excessive overfitting**.

### **Key Takeaways**
- **Higher log-evidence** indicates a **better trade-off** between model fit and complexity.
- **Quadratic features perform better** than simple ones, suggesting that adding polynomial terms improves generalization.
- **Laplace approximation helps avoid overfitting** by penalizing overly complex models.

---





