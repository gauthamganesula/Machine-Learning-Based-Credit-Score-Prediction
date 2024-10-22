# Machine-Learning-Based-Credit-Score-Prediction


This project focuses on predicting credit scores using machine learning techniques. The primary dataset consists of various financial and personal data , such as annual income, loan details, credit history, and more. A Random Forest Classification model was implemented to classify credit scores into three categories: Low, Medium, and High.

**Project Overview**

The goal of this project was to build a robust model that can predict the credit score of a person based on the provided financial and personal data. This project follows the entire process from data cleaning, building an initial model, optimizing the model using hyperparameter tuning, and finally, making predictions on a separate test dataset.

**Key Steps Involved**

1) **Data Cleaning:** The dataset had missing values and some irrelevant features. Initially, the missing values were handled by either filling them with the mean for numerical columns or 'Unknown' for categorical columns. Unnecessary features like SSN and other identifiers were dropped to avoid overfitting and irrelevant contributions to the model.

2) **Label Encoding:** Categorical features were converted into numerical format using Label Encoding to ensure the machine learning algorithm could process them effectively.

3) **Initial Model Development:** A basic Random Forest Classifier model was built using the cleaned training data. This model was evaluated to understand its initial performance, providing the following results:

Accuracy on Validation Set: 0.78055 (approximately 78%)

**Classification Report:**
    | Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
|   0   |   0.73    |  0.71  |   0.72   |  3527   |
|   1   |   0.78    |  0.78  |   0.78   |  5874   |
|   2   |   0.80    |  0.81  |   0.80   | 10599   |

Overall Accuracy: 78%

Macro Average F1-Score: 0.77

Weighted Average F1-Score: 0.78

While the initial accuracy and weighted F1-scores were solid, i have identified opportunities to improve the model’s performance, particularly in handling different credit score categories (low, medium, and high). To achieve this, we used GridSearchCV to fine-tune the hyperparameters.

4)  **Model Optimization with GridSearchCV:** To enhance the model's performance, GridSearchCV was used to fine-tune the hyperparameters of the Random Forest Classifier. By experimenting with different combinations of parameters such as max_depth, n_estimators, min_samples_leaf, and min_samples_split, the following optimal set of hyperparameters was identified:

**Best Parameters:**

A) max_depth: 20

B) max_features: 'sqrt'

C) min_samples_leaf: 1

D) min_samples_split: 2

E) n_estimators: 200

Upon re-evaluating the model with these optimized hyperparameters, the performance slightly decreased:

Accuracy on Validation Set (After Tuning): 0.77195 (approximately 77%) 

**Classification Report:**
   | Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
|   0   |   0.71    |  0.71  |   0.71   |  3527   |
|   1   |   0.78    |  0.75  |   0.76   |  5874   |
|   2   |   0.79    |  0.80  |   0.80   | 10599   |

5) **Predicting Credit Scores**
   Once the optimal model was identified, i used it to predict the credit score for individuals in the test dataset. The model outputs a prediction in one of the three categories:

**0 (Low Credit Score)**

**1 (Medium Credit Score)**

**2 (High Credit Score)**

These predictions were then saved to a CSV file for further analysis or usage.

6)  **Data Visualizations**
To gain deeper insights, I have  generated visualizations that:

Displayed the distribution of credit scores across the dataset.

Showed the relationship between various features and the predicted credit scores.

These visualizations helped us better understand the patterns in the data and how different factors contribute to the final credit score prediction.

**Conclusion**

Through this project, I have built a reliable machine learning model for credit score prediction. The model was optimized using GridSearchCV, and visualizations provided key insights into the dataset and predictions. The final model predicts credit scores into three categories: Low (0), Medium (1), and High (2).


