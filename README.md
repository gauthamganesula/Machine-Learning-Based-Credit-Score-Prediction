# Machine-Learning-Based-Credit-Score-Prediction


This project focuses on predicting credit scores using machine learning techniques. The primary dataset consists of various financial and personal attributes, such as annual income, loan details, credit history, and more. A Random Forest Classification model was implemented to classify credit scores into three categories: Low, Medium, and High.

Project Overview

The goal of this project was to build a robust model that can predict the credit score of a person based on the provided financial and personal data. This project follows the entire process from data cleaning, building an initial model, optimizing the model using hyperparameter tuning, and finally, making predictions on a separate test dataset.

Key Steps Involved

Data Cleaning: The dataset had missing values and some irrelevant features. Initially, the missing values were handled by either filling them with the mean for numerical columns or 'Unknown' for categorical columns. Unnecessary features like SSN and other identifiers were dropped to avoid overfitting and irrelevant contributions to the model.

Label Encoding: Categorical features were converted into numerical format using Label Encoding to ensure the machine learning algorithm could process them effectively.

Building the Initial Model: A basic Random Forest Classifier was first used to train the model. We evaluated its performance on the validation set, and the accuracy was around 0.78. While the model performed reasonably well, there was room for improvement, especially in precision and recall for the different classes.


Hyperparameter Tuning with GridSearchCV: To enhance the model’s performance, we used GridSearchCV to fine-tune the model’s hyperparameters. Various combinations of parameters, such as the number of estimators, maximum depth, and minimum samples required for a split, were tested to identify the best-performing configuration.


Best Parameters Identified: After GridSearchCV was applied, the best parameters were:
arduino

Post-Tuning Accuracy: After hyperparameter tuning, the accuracy slightly improved to 0.77. While this was not a drastic improvement in accuracy, the model’s precision, recall, and F1-score improved for various credit score classes, particularly for distinguishing between Medium and High scores.


Visualization: To further understand the dataset and model, some visualizations were created:
A Credit Score Distribution plot was generated to see the distribution of credit scores across the dataset.
A Feature Importance plot from the Random Forest model was created to highlight which features were most impactful in predicting the credit score.
A Correlation Heatmap was used to visualize the relationships between different features in the dataset.

Model Performance

Initial Evaluation:
The first evaluation of the Random Forest model gave an accuracy of 0.78. This performance was decent but indicated room for improvement, especially for precision and recall in predicting 'Medium' and 'High' credit scores.

Hyperparameter Tuning:
To improve the model, GridSearchCV was used to tune the hyperparameters. The final optimized model performed slightly better, with an accuracy of 0.77 on the validation set. The benefit of this process was not only the accuracy but also more balanced precision, recall, and F1-scores across all the classes.

Final Thoughts

This project demonstrates the full pipeline of a machine learning model—starting from data cleaning and feature encoding, to building, tuning, and evaluating a Random Forest model. The final model can predict credit scores based on the provided attributes with reasonable accuracy.

Visualizations

Several visualizations were also created to better understand the data and the model's performance, such as the Credit Score Distribution, Feature Importance, and a Correlation Heatmap.
