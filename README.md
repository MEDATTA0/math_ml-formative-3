# Math for Machine Learning — Formative Assignment 3

Group submission covering four topics in probability and optimization for ML: fitting a
Gaussian Mixture Model with Expectation-Maximization, applying Bayes' theorem to sentiment
analysis, and deriving/verifying gradient descent for linear regression by hand and in code.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Part 1 - Probability Distributions (Gaussian Mixture Model)](#part-1--probability-distributions-gaussian-mixture-model)
- [Part 2 - Bayesian Sentiment Analysis](#part-2--bayesian-sentiment-analysis)
- [Part 3 -- Gradient Descent Derivation (Handwritten)](#part-3--gradient-descent-derivation-handwritten)
- [Part 4 - Gradient Descent Implementation](#part-4--gradient-descent-implementation)
- [How Part 3 and Part 4 Relate](#how-part-3-and-part-4-relate)
- [Setup / How to Run](#setup--how-to-run)
- [Contributors](#contributors)

## Repository Structure

| Part   | File                                                | Type             |
|--------|-----------------------------------------------------|------------------|
| Part 1 | `Formative_3_Part1_Probability_Distributions.ipynb` | Jupyter notebook |
| Part 2 | `Part_2_Bayesian_sentiment_implementation.ipynb`    | Jupyter notebook |
| Part 3 | `iterations.pdf`                                    | Handwritten PDF  |
| Part 4 | `part4.ipynb`                                       | Jupyter notebook |

## Part1 - Probability Distributions (Gaussian Mixture Model)

**File:** `Formative_3_Part1_Probability_Distributions.ipynb`

Fits a 2-component Gaussian Mixture Model from scratch (Expectation-Maximization implemented
with plain `numpy`) to the classic **Galton parent-child heights** dataset.

- Combines deduplicated father heights (n=205) with all child heights (n=934) into one
  dataset (n=1139, range 56–79 inches).
- Runs EM for 15 iterations, tracking `μ`, `σ²`, and mixture weight `π` for each component.
- **Converged parameters:**
  - Children component: μ = 66.402, σ² = 11.528, π = 0.814
  - Fathers component: μ = 70.750, σ² = 2.574, π = 0.186
  - Final log-likelihood: −3050.601
- Plots the observed height histogram against the two fitted Gaussian components and their mixture.
- Finishes with a Bayesian posterior classification example: for a height of 72 inches,
  P(Child | 72) = 0.4192 and P(Father | 72) = 0.5808 → predicted class **Father**.

**Requires:** `GaltonFamilies.csv` (not included in this repo - place it in the same directory
as the notebook before running, and you'll see the result).

## Part 2 - Bayesian Sentiment Analysis

**File:** `Part_2_Bayesian_sentiment_implementation.ipynb`

Applies Bayes' theorem **by hand** to score sentiment keywords against the **IMDB 50k movie reviews** dataset.

- Formula used: `P(Positive | keyword) = P(keyword | Positive) × P(Positive) / P(keyword)`
- Positive keywords tested: *excellent, brilliant, masterpiece, wonderful*
- Negative keywords tested: *boring, awful, worst, waste*
- **Results: P(Positive | keyword):**

  | Keyword     | Posterior  | Keyword  | Posterior  |
  |-------------|------------|----------|------------|
  | wonderful   | 0.812      | boring   | 0.194      |
  | excellent   | 0.807      | awful    | 0.098      |
  | brilliant   | 0.761      | worst    | 0.092      |
  | masterpiece | 0.730      | waste    | 0.069      |

- With a balanced 50/50 prior, positive keywords push the posterior well above 0.5 and
  negative keywords well below it. Posteriors land around 0.73–0.81 rather than near 1.0
  because these words still appear in some opposite-sentiment reviews (negation, sarcasm).

**Requires:** `IMDB Dataset.csv`. The notebook was originally built for Google Colab and
mounts Google Drive at `/content/drive/MyDrive/Math_ML_Formative_3/IMDB Dataset.csv`; adjust
the path if running locally.

## Part 3 - Gradient Descent Derivation (Handwritten)

**File:** `iterations.pdf` (4 pages)

Handwritten derivation of gradient descent for linear regression: manual predictions, the
mean-squared-error cost, and the analytical gradient, worked out step-by-step across 3
iterations.

## Part 4 - The Gradient Descent Implementation

**File:** `part4.ipynb`

Python implementation that reproduces the iterations derived by hand in Part 3, using the same dataset (`X = [[1,3],[4,10]]`, `y = [5,6]`).

- Implements the analytical gradient (`dJ/dm = (2/n) * X^T (y_hat − y)`, `dJ/db = (2/n) (y_hat − y)`) **and**
  a numerical gradient via SciPy's `scipy.optimize.approx_fprime`, as specified the assignment's
  requirement.
- Runs 3 iterations of gradient descent (learning rate 0.01), printing both the numerical and
  analytical gradient at each step to confirm they agree (small differences come from
  floating-point step size in the numerical approximation).
- **Final result after 3 iterations:** m = [-1.369562, 1.095182], b = [0.93624], giving
  predictions y_hat = [2.852224, 6.409812] against actual y = [5, 6].
- Plots parameter trajectories (`m`, `b`) and the MSE cost across iterations.

## How Part 3 and Part 4 Relate

Part 4 is the coded verification of Part 3: the same gradient descent iterations that we worked out by hand in `iterations.pdf` are reproduced in `part4.ipynb`,
with the analytical gradient checked against a numerical gradient computed via SciPy.

## Setup / How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

## Team Members

| Name              | Parts                               |
|-------------------|-------------------------------------|
| Blessing Ingabire | Part 1, Part 3 (iteration 2)        |
| Joshua Agonzibwa  | Part 2, Part 3 (iteration 1)        |
| Ibrahim Maâzou    | Part 3 (setup, iteration 3), Part 4 |
