# IITGN_ASSIGNMENTS

TASK-1


 1. **What is the HAR Dataset?**
The **Human Activity Recognition (HAR)** dataset contains data collected from smartphones with built-in sensors like accelerometers and gyroscopes. This data represents different activities like walking, sitting, standing, etc.

There are two types of data in this dataset:
**Featurized data**: This data contains features (important information) extracted from raw sensor data. For example, it might include statistics like the average or standard deviation of sensor readings during an activity.
**Raw sensor data**: This is the original data from the sensors, such as accelerometer and gyroscope readings.

 2. **Bias-Variance Tradeoff**
The **Bias-Variance Tradeoff** is a key concept in machine learning that helps us understand how the complexity of a model affects its performance:
- **Bias**: If a model is too simple (underfitting), it won't capture all the patterns in the data, leading to poor performance (high bias).
- **Variance**: If a model is too complex (overfitting), it will fit the training data very well but won't generalize well to new data (high variance).

The goal is to find a good balance between bias and variance by avoiding both underfitting and overfitting.

### 3. **Using a Decision Tree Classifier**
A **Decision Tree** is a model that makes decisions by asking a series of questions based on features in the dataset. The **depth** of the tree determines how many levels of questions the model will ask:
- **Shallow depth**: The tree is simpler and may not capture all the patterns in the data (underfitting).
- **Deep depth**: The tree becomes more complex and may start memorizing the data, leading to overfitting.

### 4. **Steps to Complete the Task**
- **Use featurized data**: Work with the pre-processed data where important features have been extracted.
- **Vary the tree's depth**: Train a Decision Tree with different depths to observe how its performance changes.
- **Demonstrate the Bias-Variance Tradeoff**: By adjusting the tree depth, observe how the model performs on both training and testing data. A shallow tree may underperform (high bias), while a very deep tree may perform well on training data but poorly on test data (high variance).
- **Visualize the results**: Plot the performance (e.g., accuracy) for various tree depths to illustrate the relationship between model complexity and performance.

### 5. **Objectives to Focus On**
- **Underfitting**: Occurs when the tree is too shallow (low depth), failing to capture enough information from the data.
- **Overfitting**: Occurs when the tree is too deep (high depth), learning too much from the training data and including noise, leading to poor generalization to new data.
- **Finding the optimal depth**: The objective is to find the tree depth that balances model complexity, where the model performs well on both training and test data.

### 6. **Visualizing the Bias-Variance Tradeoff**
By plotting **training accuracy** and **testing accuracy** for various tree depths, you will observe:
- As the depth increases, training accuracy will generally increase.
- After a certain depth, test accuracy will start to decline, indicating that the model is overfitting.



TASK -2

### Task 2: Train and Compare Classic ML Models

In **Task 2**, you are tasked with training and comparing multiple classic machine learning models using the **featurized dataset** (the dataset with pre-extracted features). The models you need to work with are:

1. **Random Forest Classifier**
2. **Decision Tree Classifier**
3. **Logistic Regression**
4. **AdaBoost Classifier**

These models will be trained and evaluated on the same dataset, but using different evaluation techniques to compare their performance.

### 1. **Models to be Trained**

- **Random Forest Classifier**: This is an ensemble method that uses multiple decision trees to improve classification accuracy. It reduces overfitting by averaging multiple trees.
  
- **Decision Tree Classifier**: This model creates a tree-like structure to make decisions based on features. It is simple and interpretable but can overfit if not properly tuned.

- **Logistic Regression**: A linear model used for binary and multi-class classification problems. It estimates probabilities and classifies data based on linear decision boundaries.

- **AdaBoost Classifier**: An ensemble method that focuses on improving the performance of weak classifiers (e.g., decision trees) by giving more weight to incorrectly classified instances during training.

### 2. **Evaluation Techniques**

To evaluate these models, two cross-validation methods will be used:

#### a. **K-Fold Cross-Validation (K-Fold CV)**

- **K-Fold CV** is a method where the data is split into **K subsets** (folds). The model is trained on K-1 subsets and tested on the remaining subset. This process is repeated K times, with each subset used as a test set once.
- The results are averaged over all folds to get a more reliable estimate of the model's performance.

#### b. **Leave-One-Subject-Out Cross-Validation (LOSO-CV)**

- **LOSO-CV** is specifically useful when you have subject-based data. In this case, one subject’s data is left out as the test set, while the remaining data (from other subjects) is used for training.
- This method simulates how the model would perform on unseen subjects, providing a more realistic evaluation in scenarios where you might encounter new users or subjects.

### 3. **Performance Metrics to Compare the Models**

You will compare the performance of the models based on the following metrics:

- **Accuracy**: The percentage of correct predictions made by the model out of all predictions.
  
- **Precision**: The proportion of true positive predictions among all positive predictions made by the model. It helps to measure how many of the positive predictions were actually correct.
  
- **Recall**: The proportion of actual positives correctly identified by the model. It shows how well the model identifies positive instances.
  
- **F1-Score**: The harmonic mean of precision and recall. It is a more balanced metric when you need to consider both precision and recall, especially in cases where there is an imbalance between classes.

### 4. **Steps to Complete the Task**

- **Train the Models**: For each model (Random Forest, Decision Tree, Logistic Regression, AdaBoost), you'll train them using the featurized dataset.
  
- **Evaluate Using K-Fold CV**: Perform **K-Fold Cross-Validation** on each model. This will give you an estimate of the model's performance on different splits of the data.
  
- **Evaluate Using LOSO-CV**: Perform **Leave-One-Subject-Out Cross-Validation**, where each subject’s data is excluded one at a time and used as the test set, while the rest is used for training.
  
- **Calculate the Metrics**: For each model, calculate **Accuracy**, **Precision**, **Recall**, and **F1-Score** using the results from K-Fold and LOSO cross-validation.

- **Compare the Models**: Once all models are evaluated using the two cross-validation techniques, compare their performance based on the calculated metrics to determine which model performs best overall.

### 5. **Purpose of This Task**

- Comparing the performance of multiple classic machine learning models.
- Evaluating the models in a more robust way using K-Fold and LOSO Cross-Validation, which helps understand how they generalize to new data (especially important when working with subject-based datasets like HAR).
- Selecting the best-performing model based on different performance metrics such as accuracy, precision, recall, and F1-Score.

