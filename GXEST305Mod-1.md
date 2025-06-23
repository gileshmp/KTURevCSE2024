# Module 1: Introduction to AI and Machine Learning

Welcome to the foundational module on Artificial Intelligence and Machine Learning. In this module, we will journey from the basic definition of ML to building our first predictive models and understanding the structure of a neural network.

---

### 1. What is Machine Learning?

At its core, **Machine Learning (ML)** is a subfield of Artificial Intelligence (AI) that gives computers the ability to learn from data without being explicitly programmed.

Think about a traditional program to filter spam emails. You would have to write explicit rules:
*   `IF` the email contains the words "viagra" or "free money", `THEN` mark as spam.
*   `IF` the email is from a known contact, `THEN` mark as not spam.

This approach is brittle. Spammers can easily change words ("v1agra", "fr3e m0ney") to bypass these rules.

The **Machine Learning approach** is different. Instead of writing rules, you show the computer thousands of examples of emails that have already been labeled as "Spam" or "Not Spam". The ML algorithm "learns" the patterns associated with spam on its own.



The output of this learning process is a **model**, which is essentially a smart, data-driven program that can now make predictions on new, unseen emails.

**Key Terminology:**
*   **Training Data:** The example data used to "teach" the algorithm (e.g., labeled emails).
*   **Model:** The output of the training process; a set of learned patterns.
*   **Inference (or Prediction):** Using the trained model to make a prediction on new data.

---

### 2. Types of Machine Learning Systems

Machine Learning systems are broadly categorized into three types based on the nature of the data and the learning task.

#### a) Supervised Learning

This is the most common type of ML. The "supervision" comes from the fact that the training data is **labeled**. The algorithm learns to map an input (`X`) to an output (`y`) based on example input-output pairs.

> **Analogy:** Learning with a teacher or a flashcard. The card has a question (input) on one side and the answer (label) on the other.

There are two main types of supervised learning problems:
1.  **Regression:** The output variable (`y`) is a continuous, numerical value.
    *   **Goal:** Predict a quantity.
    *   **Branch-Specific Examples:**
        *   **Mechanical/Civil Engineering:** Predicting the load-bearing capacity of a beam based on its material properties and dimensions.
        *   **Electrical Engineering:** Predicting the power demand for a city based on time of day, weather, and historical data.
        *   **Computer Science:** Predicting the time it will take for a program to execute based on its complexity.

2.  **Classification:** The output variable (`y`) is a category or a class.
    *   **Goal:** Predict a label.
    *   **Branch-Specific Examples:**
        *   **Mechanical Engineering:** Classifying a machine part as 'functional', 'needs maintenance', or 'defective' based on sensor readings (vibration, temperature).
        *   **Civil Engineering:** Classifying a soil sample into types (e.g., 'clay', 'silt', 'sand') based on its properties.
        *   **Computer Science:** Classifying an email as 'Spam' or 'Not Spam'.

#### b) Unsupervised Learning

In unsupervised learning, the training data is **unlabeled**. The algorithm's goal is to explore the data and find some inherent structure or pattern within it without any pre-defined outcomes.

> **Analogy:** Being given a box of mixed fruits and asked to sort them into groups. You don't know the names of the fruits, but you can group them based on color, size, and shape.

Common unsupervised learning tasks include:
1.  **Clustering:** Grouping similar data points together.
    *   **Branch-Specific Examples:**
        *   **Computer Science:** Grouping customers with similar purchasing behaviors for targeted marketing.
        *   **Civil Engineering:** Identifying zones within a city that have similar traffic patterns.
        *   **Electrical Engineering:** Detecting anomalies or faulty sensors in a power grid by finding data points that don't belong to any cluster.

2.  **Dimensionality Reduction:** Simplifying the data by reducing the number of input variables (features).

#### c) Reinforcement Learning

This type of learning involves an **agent** that interacts with an **environment**. The agent learns to perform actions that maximize a cumulative **reward**. It learns from the consequences of its actions through trial and error, rather than from labeled data.

> **Analogy:** Training a pet. You give it a treat (reward) for good behavior (correct action) and a scolding (penalty) for bad behavior.

**Branch-Specific Examples:**
*   **Mechanical/Robotics Engineering:** Training a robotic arm to pick and place objects efficiently. The reward is given for successful placements.
*   **Civil Engineering:** Developing an intelligent traffic light control system that adapts to real-time traffic to minimize congestion. The reward is a reduction in average wait time.
*   **Electrical Engineering:** Optimizing the energy distribution in a smart grid. The agent decides where to route power to maximize efficiency and minimize cost.

---

### 3. Challenges in Machine Learning

While powerful, ML is not magic. Common challenges include:
*   **Insufficient Data:** ML algorithms often need a large amount of data to learn effectively.
*   **Poor Quality Data:** Garbage in, garbage out. Errors, outliers, and noise in the data can lead to a poor model.
*   **Overfitting:** The model learns the training data *too well*, including its noise and quirks. It performs great on the data it was trained on but fails to generalize to new, unseen data. (Like a student who memorizes the textbook but can't answer a slightly different question).
*   **Underfitting:** The model is too simple to capture the underlying structure of the data. It performs poorly on both the training data and new data. (Like using a straight line to describe a wave pattern).
*   **Feature Engineering:** Selecting and transforming the most relevant input variables (features) for the model can be a difficult and time-consuming task.

---

### 4. Supervised Learning Example: Linear Regression

Let's explore the simplest regression model: **Linear Regression**. Its goal is to model a linear relationship between a single input feature (independent variable, `x`) and a continuous target (dependent variable, `y`).

The model is defined by the equation of a straight line:
$y = \beta_0 + \beta_1 x$

Where:
*   $y$ is the value we want to predict.
*   $x$ is the input feature.
*   $\beta_1$ is the **slope** of the line. It represents the change in $y$ for a one-unit change in $x$.
*   $\beta_0$ is the **y-intercept**. It's the value of $y$ when $x=0$.

The "learning" part of linear regression is finding the best values for $\beta_0$ and $\beta_1$ that make the line fit the data as closely as possible.

#### Finding Coefficients with the Least Squares Method

How do we define the "best" line? The most common method is the **Ordinary Least Squares (OLS)**. The idea is to minimize the **sum of the squared errors (SSE)**.

1.  For each data point $(x_i, y_i)$, the model makes a prediction,

$\hat{y}_i=\beta_0 +\beta_1 x_i$


4.  The error (or residual) for that point is the difference: $e_i = y_i - \hat{y}_i$.
5.  We square these errors to avoid negative and positive errors canceling each other out.
6.  The goal is to find the $\beta_0$ and $\beta_1$ that minimize the sum of all these squared errors:
   
    $$\min \sum_{i=1}^{n} (y_i -\hat{y}_i)^2 =\sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2$$

While this can be solved using calculus, we can easily find these values using a tool like Microsoft Excel.

#### Demonstration using Microsoft Excel

Let's take a simple dataset: Hours Studied vs. Exam Score.

| Hours Studied (x) | Exam Score (y) |
| ----------------- | -------------- |
| 1                 | 55             |
| 2                 | 62             |
| 3                 | 78             |
| 4                 | 75             |
| 5                 | 85             |
| 6                 | 92             |

**Steps in Excel:**
1.  Enter the data into two columns.
2.  Select the data and go to `Insert > Chart > Scatter`. This will create a scatter plot.
3.  Right-click on any data point in the chart and select `Add Trendline...`.
4.  In the `Format Trendline` pane that appears, make sure `Linear` is selected.
5.  Check the boxes for `Display Equation on chart` and `Display R-squared value on chart`.

You will see an equation on your chart, something like `y = 6.8x + 48.6`. This means our model is:
*   $\beta_1 = 6.8$ (For each extra hour studied, the score is predicted to increase by 6.8 points).
*   $\beta_0 = 48.6$ (A student who studies for 0 hours is predicted to get a score of 48.6).



---

### 5. Classification Example: Logistic Regression

What if our output isn't a number, but a category like "Yes/No" or "Pass/Fail"? Let's represent this as `1` (Pass) and `0` (Fail).

Consider this data:

| Hours Studied (x) | Result (y) |
| ----------------- | ---------- |
| 1                 | 0 (Fail)   |
| 2                 | 0 (Fail)   |
| 3                 | 0 (Fail)   |
| 4                 | 1 (Pass)   |
| 5                 | 1 (Pass)   |
| 6                 | 1 (Pass)   |

#### Why Linear Regression Fails for Classification

If we try to fit a linear regression line to this data, we run into problems.


1.  **Nonsensical Predictions:** The line extends infinitely in both directions. It can predict values like `1.5` or `-0.2`. What does a "pass probability" of 150% or -20% mean? Nothing. We need our output to be constrained between 0 and 1.
2.  **Poor Fit:** A straight line is not a good way to model a sharp jump from 0 to 1.

#### The Solution: The Sigmoid Function

We need a function that takes any real number (the output of our linear equation, $z = \beta_0 + \beta_1 x$) and "squashes" it into the range (0, 1). The perfect candidate is the **Sigmoid (or Logistic) function**:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$



In **Logistic Regression**, we don't predict `y` directly. We predict the *probability* that `y=1`.

$$ P(y=1 | x) = \sigma(\beta_0 + \beta_1 x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}} $$

#### The Logit Function and Log-Odds

How do we link the linear part ($z$) to the probability ($p$)? This is done via the **logit function**, which is the inverse of the sigmoid.

The "odds" of an event is the ratio of the probability of it happening to the probability of it not happening: $\text{Odds} = \frac{p}{1-p}$.

The **logit function** is the natural logarithm of the odds (the "log-odds"):
$$ \text{logit}(p) = \ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x $$

This transformation allows us to use a linear model ($\beta_0 + \beta_1 x$) to predict a value (log-odds) that can then be converted into a meaningful probability between 0 and 1.

For a hands-on implementation and visualization of these concepts, please refer to the Colab notebook:
> [Link to Colab: Computational Part of Logistic Regression](https://colab.research.google.com/drive/1M-VamLi6RXknEuW8BN-vELdoqmbenwgu?usp=sharing)

---

### 6. Unsupervised Example: K-Means Clustering

K-Means is a popular algorithm for **clustering**. The goal is to partition `n` observations into `k` clusters in which each observation belongs to the cluster with the nearest mean (cluster centroid).

Let's use the famous `iris` dataset. It contains measurements for 150 iris flowers from 3 different species (Setosa, Versicolor, Virginica). We will use two features, `Sepal Length` and `Sepal Width`, for easy visualization.

**The K-Means Algorithm:**
1.  **Choose K:** Decide on the number of clusters you want to find (e.g., `K=3`).
2.  **Initialize Centroids:** Randomly select `K` data points to be the initial cluster centers.
3.  **Assign Points:** Assign each data point to its closest centroid.
4.  **Update Centroids:** Recalculate the position of each centroid by taking the mean of all data points assigned to it.
5.  **Repeat:** Repeat steps 3 and 4 until the centroids no longer move significantly.

#### Visualization and Analysis

When we run K-Means on the iris dataset (with K=3), we get clusters. We can visualize this by plotting the data and coloring each point according to its assigned cluster.



**Comparison with Actual Labels:**
The iris dataset has actual labels (the true species). We can compare the clusters found by the algorithm to the true species. Often, the algorithm does a good job, but it's not always perfect. The algorithm is simply finding mathematical structure. What if we had set `K=4`? The algorithm would have happily partitioned the data into 4 clusters, even though we know there are only 3 species. This highlights a key aspect of unsupervised learning: the results require human interpretation.

For a hands-on demonstration of clustering the iris dataset, see the notebook below:
> [Link to Colab: Clustering of Iris Dataset](https://colab.research.google.com/drive/1qtzkT4VfcmK5J4uUH8ACUYWhGwIJCMn9?usp=sharing)

---

### 7. Artificial Neural Networks (ANN)

ANNs are a powerful class of ML models inspired by the structure of the human brain.

#### The Biological Inspiration vs. The Artificial Model

| Biological Neuron         | Artificial Neuron (Perceptron) |
| ------------------------- | ------------------------------ |
| **Dendrites** (Receives signals) | **Inputs (`x`)** (Receives numerical data) |
| **Soma** (Processes signals) | **Processing Unit** (Sums weighted inputs + bias) |
| **Axon** (Transmits signal) | **Output (`y`)** (Sends a numerical signal) |
| **Synapse Strength** (Controls signal strength) | **Weights (`w`)** (Controls input importance) |

#### The Perceptron: The Simplest ANN

A perceptron is a single neuron. It computes a weighted sum of its inputs and applies a **step function** to decide whether to fire (output 1) or not (output 0).

$z = \sum w_i x_i + b$

`output = 1 if z > threshold else 0`



#### Unravelling the "Black Box": From Perceptron to MLP

A single perceptron can only learn linearly separable patterns. To solve complex problems, we stack them in layers to create a **Multi-Layer Perceptron (MLP)**, which has an Input Layer, one or more Hidden Layers, and an Output Layer.

This architecture is not a "black box"; it's a beautiful mathematical construction.

*   **Linear Combination:** The connection from one layer to the next is a linear operation. Each neuron calculates its own weighted sum: $z = \mathbf{W} \cdot \mathbf{x} + \mathbf{b}$. This is like running multiple linear regressions in parallel.

*   **Activation Function (Non-linearity):** If we only stacked linear operations, the entire network would be just one big linear function. To learn complex patterns, each neuron applies a non-linear **activation function** (like Sigmoid or ReLU) to its sum `z`. **This is the key to an ANN's power.**

*   **Backpropagation (Error Minimization):** This is the learning algorithm for ANNs. It's an application of calculus (the chain rule) for iterative optimization.
    1.  **Forward Pass:** Feed an input through the network to get a prediction.
    2.  **Calculate Error:** Compare the prediction with the true label to calculate a loss.
    3.  **Backward Pass (Backpropagation):** The error is propagated backward, calculating how much each weight and bias contributed to the error.
    4.  **Update Weights:** Adjust the weights to reduce the error.
    5.  **Repeat:** This process is repeated thousands of times, causing the network to slowly converge to a set of weights that minimizes the error.

A **Deep Neural Network (DNN)** is simply an ANN with many hidden layers.

### 8. The Universal Approximation Theorem

This theorem provides the theoretical guarantee for why ANNs are so powerful.

> **Statement:** A feed-forward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$, under mild assumptions on the activation function.

**In simple terms:** It means that no matter how complicated a continuous function is, there is a neural network that can approximate it to any desired degree of accuracy. It doesn't tell us *how* to find the right weights, but it guarantees that a solution *exists*.

### 9. Demonstration of MLP for Regression and Classification

An MLP can be adapted for both regression and classification tasks.

*   **For Regression:** The output layer has a single neuron with no activation function (or a linear activation). The model is trained to minimize a loss like Mean Squared Error (MSE).
    *   *Example:* Predicting house prices based on features like area, number of bedrooms, and location.

*   **For Classification:** The output layer has as many neurons as there are classes. A `softmax` activation function is typically used, which converts the outputs into a probability distribution. The model is trained to minimize a loss like Categorical Cross-Entropy.
    *   *Example: MNIST Dataset:* This is a classic dataset of handwritten digits (0-9). An MLP can be trained to classify these images.
        *   **Input:** A flattened 28x28 pixel image (784 input neurons).
        *   **Hidden Layers:** One or more hidden layers with ReLU activation.
        *   **Output Layer:** 10 neurons (one for each digit, 0-9) with `softmax` activation. The neuron with the highest probability is the model's prediction.

---

> **Note:** From this module, theory questions like the types of machine learning, steps in linear and logistic regression, simple problems to cluster a tabular data with at most two features and 10 samples and basics of ANN will be used for both internal and end semester assessments.


# Module 2: Mathematical Foundations of AI and Data Science

In this module, we will explore the mathematical backbone of modern AI and Data Science: Linear Algebra. We will see how data is represented as matrices and how decomposing these matrices helps us uncover hidden patterns and simplify complex data.

---

### 1. Data and its Representation

Before we can analyze data, we need to understand what it is and how it's stored. In machine learning, data is almost always organized into a structured format.

#### Interactive Activity: Data Collection

To understand this, let's create our own dataset. Please fill out this Google Form with some basic, anonymous details.

> **Instructor's Note:** Create a simple Google Form with the following fields:
> *   Gender (Options: Male, Female, Other)
> *   Age (Numeric)
> *   Height (in cm, Numeric)
> *   Weight (in kg, Numeric)
> *   Briefly state your expectation for this course (Text)

Once the data is collected, it can be viewed in a Google Sheet. It will look something like this:

| Timestamp           | Gender | Age | Height | Weight | Expectation                                |
| ------------------- | ------ | --- | ------ | ------ | ------------------------------------------ |
| 2023/10/27 10:01 AM | Male   | 20  | 175    | 70     | To learn how to build AI models.           |
| 2023/10/27 10:01 AM | Female | 19  | 162    | 55     | Understand the math behind AI.             |
| 2023/10/27 10:02 AM | Male   | 21  | 180    | 85     | To get a good job in the data science field. |
| ...                 | ...    | ... | ...    | ...    | ...                                        |

This spreadsheet is, for all practical purposes, a **matrix**.

*   **Samples (Rows):** Each row in the sheet represents a single observation or data point. In this case, each row is a student. These are also called *instances* or *records*.
*   **Features (Columns):** Each column represents a specific attribute or property of our samples. 'Gender', 'Age', and 'Height' are all features. These are also called *variables* or *attributes*.

So, our data is an `m x n` matrix, where `m` is the number of students (samples) and `n` is the number of attributes (features).

#### Data Types

Notice that the features are of different types:
*   **Numerical (Continuous):** Can take any value within a range (e.g., `Height`, `Weight`).
*   **Numerical (Discrete):** Can only take integer values (e.g., `Age`).
*   **Categorical:** Represents distinct categories (e.g., `Gender`).
*   **Text:** Free-form text (e.g., `Expectation`).

Computers, and especially linear algebra, work with numbers. Therefore, a crucial step in any ML pipeline is to convert non-numeric data (like 'Gender' and 'Expectation') into a numerical format. This process is called **vectorization**. For example, 'Gender' could be mapped as `Male=0`, `Female=1`, `Other=2`. Text requires more complex techniques.

---

### 2. Matrix Decomposition: The "Why"

**Matrix Decomposition** (or factorization) is the process of breaking down a matrix into its constituent parts—a product of simpler, more fundamental matrices.

> **Analogy:** Think of prime factorization. The number `12` doesn't tell us much on its own. But its factorization, `2 × 2 × 3`, reveals its fundamental building blocks. Similarly, decomposing a matrix reveals its fundamental properties, such as its dominant directions, its rank, and how it transforms space.

**Why is this relevant?**
In data science, our data matrix often contains redundant information, noise, and complex relationships. Decomposition helps us:
*   **Simplify the data:** By identifying the most important patterns.
*   **Reduce noise:** By isolating and removing the less important components.
*   **Extract meaningful features:** By finding the "latent" or hidden structure in the data.
*   **Solve systems of equations more efficiently.**

We will focus on two key decomposition techniques: **Singular Value Decomposition (SVD)** and its application in **Principal Component Analysis (PCA)**.

---

### 3. Singular Value Decomposition (SVD)

SVD is arguably the most powerful and general matrix decomposition method. It states that *any* `m x n` matrix `A` can be factored into three other matrices:

$$ A = U \Sigma V^T $$



#### Visual Example: Image Compression

The most intuitive way to understand SVD is through image compression.
1.  A grayscale image is just a matrix of pixel intensities (e.g., a 512x512 matrix).
2.  We perform SVD on this image matrix `A` to get `U`, `Σ`, and `Vᵀ`.
3.  The matrix `Σ` contains the **singular values** ($\sigma_1, \sigma_2, ...$) along its diagonal, sorted in descending order of importance. The first few singular values capture the most significant, large-scale features of the image, while later ones represent finer details and noise.
4.  We can reconstruct an approximation of the image by using only the top `k` singular values (and the first `k` columns of `U` and `V`).

$$ A_k = U_k \Sigma_k V_k^T $$

As we increase `k`, the reconstructed image gets closer to the original, but we need to store more data. SVD allows us to find a sweet spot between image quality and storage size.


*With just a few singular values (k=20), we get a good approximation of the original image.*

#### The Mathematical Formulation and Terms

For an `m x n` matrix `A`:
*   **U:** An `m x m` **orthogonal matrix**. Its columns are the **left-singular vectors**. They form an orthonormal basis for the column space of A.
*   **Σ (Sigma):** An `m x n` **diagonal matrix**. The diagonal entries $\sigma_1, \sigma_2, ...$ are the **singular values** of A. They are always non-negative and are sorted in descending order ($\sigma_1 \ge \sigma_2 \ge ... \ge 0$).
*   **Vᵀ:** An `n x n` **orthogonal matrix** (transpose of V). The columns of V (rows of Vᵀ) are the **right-singular vectors**. They form an orthonormal basis for the row space of A.

#### Solved Example (Simple 2x2 Matrix)

Let's find the SVD for 

```math
A = \begin{pmatrix} 3 & 0 \\ 4 & 5 \end{pmatrix} 
```

**Step 1: Find V by working with $A^T A$.**

```math
 A^T A = \begin{pmatrix} 3 & 4 \\ 0 & 5 \end{pmatrix} \begin{pmatrix} 3 & 0 \\ 4 & 5 \end{pmatrix} = \begin{pmatrix} 25 & 20 \\ 20 & 25 \end{pmatrix}
```
Now, find the eigenvalues ($\lambda$) and eigenvectors ($v$) of $A^T A$.
The characteristic equation is $(25-\lambda)^2 - 20^2 = 0 \implies (25-\lambda-20)(25-\lambda+20) = 0 \implies (5-\lambda)(45-\lambda) = 0$.
So, the eigenvalues are $\lambda_1 = 45, \lambda_2 = 5$.

The corresponding (normalized) eigenvectors are 
```math
$ v_1 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix} $ and $ v_2 = \frac{1}{\sqrt{2}} \begin{pmatrix} -1 \\ 1 \end{pmatrix}
```
These form the columns of V: 
```math
V = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}.
```

**Step 2: Find Σ.**
The singular values are the square roots of the eigenvalues:
$\sigma_1 = \sqrt{45} = 3\sqrt{5}$
$\sigma_2 = \sqrt{5}$
So, $ \Sigma = \begin{pmatrix} 3\sqrt{5} & 0 \\ 0 & \sqrt{5} \end{pmatrix} $.

**Step 3: Find U.**
The columns of U are given by $u_i = \frac{1}{\sigma_i} A v_i$.
$ u_1 = \frac{1}{3\sqrt{5}} \begin{pmatrix} 3 & 0 \\ 4 & 5 \end{pmatrix} \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{3\sqrt{10}} \begin{pmatrix} 3 \\ 9 \end{pmatrix} = \frac{1}{\sqrt{10}} \begin{pmatrix} 1 \\ 3 \end{pmatrix} $.
$ u_2 = \frac{1}{\sqrt{5}} \begin{pmatrix} 3 & 0 \\ 4 & 5 \end{pmatrix} \frac{1}{\sqrt{2}} \begin{pmatrix} -1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{10}} \begin{pmatrix} -3 \\ 1 \end{pmatrix} $.
So, $ U = \frac{1}{\sqrt{10}} \begin{pmatrix} 1 & -3 \\ 3 & 1 \end{pmatrix} $.

**Final Result:** $ A = U \Sigma V^T = \left(\frac{1}{\sqrt{10}} \begin{pmatrix} 1 & -3 \\ 3 & 1 \end{pmatrix}\right) \begin{pmatrix} 3\sqrt{5} & 0 \\ 0 & \sqrt{5} \end{pmatrix} \left(\frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}\right) $.

#### The Geometry of SVD

SVD tells us that any linear transformation `A` can be broken down into a sequence of three fundamental geometric operations:
1.  **A Rotation (Vᵀ):** It takes the standard input basis vectors and rotates them to align with the principal directions (eigenvectors of $A^TA$).
2.  **A Scaling (Σ):** It stretches or shrinks the space along these new rotated axes, with the scaling factors being the singular values.
3.  **Another Rotation (U):** It rotates the scaled result into the final output coordinate system.



---

### 4. Principal Component Analysis (PCA)

#### The Curse of Dimensionality

Imagine you have a dataset with 2 features. You can plot it on a 2D plane. If you have 3 features, you can visualize it in 3D space. What about 100 features? Or 10,000?

The **Curse of Dimensionality** refers to various problems that arise when working with high-dimensional data:
*   **Data becomes sparse:** The volume of the space increases exponentially with the number of dimensions. To maintain the same data density, you would need an exponentially larger number of samples.
*   **Computational cost:** More dimensions mean more calculations, making algorithms slower.
*   **Overfitting:** Models can easily find spurious patterns in high-dimensional noise, leading to poor generalization.

We need a way to reduce the number of dimensions while preserving as much of the useful information as possible. This is where PCA comes in.

#### PCA: The Solution

**Principal Component Analysis (PCA)** is a dimensionality reduction technique that transforms a dataset into a new coordinate system. The axes of this new system are the **Principal Components (PCs)**.

*   **Principal Component 1 (PC1):** The direction in the data with the **maximum variance**. It captures the most information.
*   **Principal Component 2 (PC2):** The direction with the second-most variance, which is also **orthogonal (perpendicular)** to PC1.
*   ...and so on. Each subsequent PC captures the maximum remaining variance while being orthogonal to all previous PCs.

These PCs are **latent vectors**—they are not any of the original features but are linear combinations of them. By keeping only the first few PCs, we can reduce the dimensionality of our data while retaining most of its original variance (information).



#### The PCA Algorithm (with a Simple Example)

Let's find the principal components for the following data matrix:
$ X = \begin{pmatrix} 1 & 1 \\ 2 & 3 \\ 3 & 2 \end{pmatrix} $

**Step 1: Standardize the Data**
First, find the mean of each column (feature):
Mean of X1 = (1+2+3)/3 = 2
Mean of X2 = (1+3+2)/3 = 2
Subtract the mean from each data point:
$$ X_{centered} = \begin{pmatrix} 1-2 & 1-2 \\ 2-2 & 3-2 \\ 3-2 & 2-2 \end{pmatrix} = \begin{pmatrix} -1 & -1 \\ 0 & 1 \\ 1 & 0 \end{pmatrix} $$
*(For simplicity, we'll skip division by standard deviation here, but it's crucial in real applications).*

**Step 2: Calculate the Covariance Matrix**
The covariance matrix `C` is given by $C = \frac{1}{N-1} X_{centered}^T X_{centered}$.
$ C = \frac{1}{2} \begin{pmatrix} -1 & 0 & 1 \\ -1 & 1 & 0 \end{pmatrix} \begin{pmatrix} -1 & -1 \\ 0 & 1 \\ 1 & 0 \end{pmatrix} = \frac{1}{2} \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix} = \begin{pmatrix} 1 & 0.5 \\ 0.5 & 1 \end{pmatrix} $

**Step 3: Find Eigenvalues and Eigenvectors of the Covariance Matrix**
We solve the characteristic equation for `C`: $(1-\lambda)^2 - 0.5^2 = 0$.
This gives eigenvalues $\lambda_1 = 1.5$ and $\lambda_2 = 0.5$.

The corresponding (normalized) eigenvectors are:
*   For $\lambda_1 = 1.5$: $ v_1 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix} $ (This is **PC1**)
*   For $\lambda_2 = 0.5$: $ v_2 = \frac{1}{\sqrt{2}} \begin{pmatrix} -1 \\ 1 \end{pmatrix} $ (This is **PC2**)

**Step 4: Interpret the Results**
The principal components are the eigenvectors of the covariance matrix.
*   **PC1** points in the direction `(1, 1)`, and it explains a proportion of the variance equal to $\frac{\lambda_1}{\lambda_1+\lambda_2} = \frac{1.5}{2.0} = 75\%$.
*   **PC2** points in the direction `(-1, 1)`, and it explains the remaining $25\%$ of the variance.

To reduce our 2D data to 1D, we would project the centered data onto PC1, as it captures the most information.

---

> **Note:** From this module, theory questions like the SVD and PCA algorithm and problems to find the U, Σ, and Vᵀ matrices (from SVD) and to find the principal components of matrices (from PCA) will be used for both internal and end semester assessments.
