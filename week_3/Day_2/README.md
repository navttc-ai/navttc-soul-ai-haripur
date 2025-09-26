# 📘 Probability Tutorial

This tutorial covers essential topics in **Probability and Statistics**, ideal for beginners and students. You’ll learn the basics of probability, types, and important concepts like joint, marginal, and conditional probability, probability distributions, and Bayesian probability.

---

## 🎯 1. Probability

### 🔹 Definition

Probability measures how likely an event is to occur. It is defined as:

$P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of outcomes}}$

### 🔹 Example (Python)

```python
# Example: Probability of rolling a 4 on a fair 6-sided die
favorable_outcomes = 1
total_outcomes = 6
P_A = favorable_outcomes / total_outcomes
print("P(A) =", P_A)
```

### 🧠 Assignment

1. Calculate the probability of drawing a red card from a standard deck.
2. What is the probability of getting heads when flipping two coins?

---

## 🎯 2. Joint, Marginal, and Conditional Probability

### 🔹 Joint Probability

Probability of two events happening together.

$P(A \cap B) = P(A) \times P(B)$

```python
P_A = 0.5
P_B = 0.4
P_A_and_B = P_A * P_B
print("Joint Probability P(A and B) =", P_A_and_B)
```

### 🔹 Marginal Probability

Probability of a single event regardless of others.

$P(A) = \sum_B P(A, B)$

```python
# Example data
joint_probs = {'A_and_B1': 0.2, 'A_and_B2': 0.3}
P_A = sum(joint_probs.values())
print("Marginal Probability P(A) =", P_A)
```

### 🔹 Conditional Probability

Probability of one event given another has occurred.

$P(A|B) = \frac{P(A \cap B)}{P(B)}$

```python
P_A_and_B = 0.2
P_B = 0.4
P_A_given_B = P_A_and_B / P_B
print("Conditional Probability P(A|B) =", P_A_given_B)
```

### 🧠 Assignment

1. Given $P(A) = 0.6$, $P(B) = 0.5$, $P(A \cap B) = 0.3$, find $P(A|B)$.
2. Create Python code to visualize joint and conditional probability using `matplotlib`.

---

## 🎯 3. Probability Distributions

Probability distributions show how probabilities are distributed over values.

### Types:

* **Discrete**: Probability Mass Function (PMF)
* **Continuous**: Probability Density Function (PDF)

---

## 🎯 4. Discrete Probability Distribution

Examples include:

* **Bernoulli Distribution** (Two outcomes)
* **Binomial Distribution** (Number of successes in trials)

```python
import numpy as np
from scipy.stats import binom

n = 10  # trials
p = 0.5  # probability of success
x = np.arange(0, n+1)

pmf = binom.pmf(x, n, p)
print(pmf)
```

### 🧠 Assignment

* Plot the **Binomial PMF** using `matplotlib`.
* Compute mean and variance.

---

## 🎯 5. Continuous Probability Distribution

Examples include:

* **Normal Distribution** (bell-shaped)
* **Exponential Distribution** (waiting times)

```python
from scipy.stats import norm
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y = norm.pdf(x, 0, 1)

plt.plot(x, y)
plt.title('Normal Distribution')
plt.show()
```

### 🧠 Assignment

* Plot the PDF for Normal Distribution.
* Change mean and standard deviation and observe the effect.

---

## 🎯 6. Bayesian Probability

### 🔹 Definition

Bayes’ Theorem updates probability based on new evidence.

$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$

### 🔹 Example (Python)

```python
P_A = 0.3
P_B_given_A = 0.7
P_B = 0.5

P_A_given_B = (P_B_given_A * P_A) / P_B
print("P(A|B) =", P_A_given_B)
```

### 🧠 Assignment

1. A disease test is 99% accurate. If 1% of the population has the disease, compute probability a person who tests positive actually has it.
2. Implement Bayes’ Theorem with different prior probabilities.

---

## ✅ Summary

* **Probability** measures likelihood.
* **Joint Probability**: Two events together.
* **Marginal Probability**: Single event.
* **Conditional Probability**: Given another event.
* **Distributions**: Show how probabilities spread.
* **Bayesian Probability**: Updates beliefs with evidence.

---

💡 **Next Steps:**

* Practice with real datasets.
* Visualize distributions.
* Explore `scipy.stats` for more distributions.

