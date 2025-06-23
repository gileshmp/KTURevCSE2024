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
