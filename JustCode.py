import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

train_df =pd.read_csv('train.csv', low_memory=False)
test_df =pd.read_csv('test.csv',low_memory=False)
#Cleaning The Data 
train_df.drop('Customer_ID', axis=1, inplace=True)  
test_df.drop('Customer_ID', axis=1, inplace=True)
train_df['Name'].fillna('Unknown', inplace=True)
train_df['Type_of_Loan'].fillna('Unknown', inplace=True)
train_df['Credit_History_Age'].fillna('Unknown', inplace=True)

test_df['Name'].fillna('Unknown', inplace=True)
test_df['Type_of_Loan'].fillna('Unknown', inplace=True)
test_df['Credit_History_Age'].fillna('Unknown', inplace=True)

numerical_columns_train = ['Age', 'Monthly_Inhand_Salary', 'Num_of_Delayed_Payment', 
                           'Num_Credit_Inquiries', 'Amount_invested_monthly', 'Monthly_Balance']

numerical_columns_test = ['Age', 'Monthly_Inhand_Salary', 'Num_of_Delayed_Payment', 
                           'Num_Credit_Inquiries', 'Amount_invested_monthly', 'Monthly_Balance']

for col in numerical_columns_train:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')  
    train_df[col].fillna(train_df[col].mean(), inplace=True)  
for col in numerical_columns_test:
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')  
    test_df[col].fillna(test_df[col].mean(), inplace=True)  


categorical_columns = train_df.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))  
    label_encoders[col] = le  


for col in categorical_columns:
    if col in test_df.columns:  
        test_df[col] = test_df[col].apply(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

train_df.replace('______', 'Unknown', inplace=True)
test_df.replace('______', 'Unknown', inplace=True)

print("Missing values in training data after cleaning:")
print(train_df.isnull().sum())

print("Missing values in test data after cleaning:")
print(test_df.isnull().sum())

train_df.to_csv('cleaned_train.csv', index=False)
test_df.to_csv('cleaned_test.csv', index=False)

print("Data cleaning completed and cleaned files saved.")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('cleaned_train.csv')
test_df = pd.read_csv('cleaned_test.csv')

train_df.drop(['Customer_ID', 'ID'], axis=1, errors='ignore', inplace=True)
test_df.drop(['Customer_ID', 'ID'], axis=1, errors='ignore', inplace=True)

non_numeric_columns = train_df.select_dtypes(include=['object']).columns
print(f"Non-numeric columns in training data: {list(non_numeric_columns)}")
# Applying Label Encoding to all non numerical data 
label_encoders = {}

for col in non_numeric_columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))  
    label_encoders[col] = le  
for col in non_numeric_columns:
    if col in test_df.columns:
        test_df[col] = test_df[col].apply(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

#Feature and Traget Selection 
X = train_df.drop('Credit_Score', axis=1)  
y = train_df['Credit_Score'] 
# Train and Split 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#The RandomForest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy on Validation Set: {accuracy}")
print("Classification Report:\n", classification_report(y_val, y_pred))
#GridSearchCV to fine-tune the hyperparameters for optimal performance
from sklearn.model_selection import GridSearchCV

# Smaller parameter grid to reduce time
param_grid = {
    'n_estimators': [100, 200],  
    'max_depth': [10, 20],       
    'min_samples_split': [2, 5], 
    'min_samples_leaf': [1, 2],  
    'max_features': ['sqrt']     
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy on Validation Set after tuning: {accuracy}")
print("Classification Report:\n", classification_report(y_val, y_pred))

#Testing the model

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

test_df = pd.read_csv('cleaned_test.csv')

categorical_columns = ['Month', 'Name', 'SSN', 'Occupation', 'Type_of_Loan', 'Credit_History_Age', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Mix']
for col in categorical_columns:
    test_df[col].fillna('Unknown', inplace=True)

for col in categorical_columns:
    if col in test_df.columns:
        if col in label_encoders:
            le_classes = list(label_encoders[col].classes_)
            le_classes.append('Unknown')  
            label_encoders[col].classes_ = np.array(le_classes)
            test_df[col] = test_df[col].apply(lambda x: x if x in le_classes else 'Unknown')
            test_df[col] = label_encoders[col].transform(test_df[col].astype(str))
        else:
            le = LabelEncoder()
            test_df[col] = le.fit_transform(test_df[col].astype(str))
            label_encoders[col] = le  

for col in test_df.columns:
    if col not in categorical_columns:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        test_df[col].fillna(test_df[col].mean(), inplace=True)

print("Missing values in the cleaned test data after handling:")
print(test_df.isnull().sum())

test_predictions = best_model.predict(test_df)

output = pd.DataFrame({'ID': test_df.index, 'Predicted_Credit_Score': test_predictions})
output.to_csv('test_predictions.csv', index=False)

print("Predictions on test set saved to test_predictions.csv")
#To display the results 
predections_df = pd.read_csv('test_predictions.csv')
predections_df.head()

#Visulazations 
#1) Distrubution Of Credit Score 
train_df = pd.read_csv('cleaned_train.csv')

plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='Credit_Score', palette='Set2')
plt.title('Distribution of Credit Score')
plt.xlabel('Credit Score')
plt.ylabel('Count')
plt.show()

#2)Corelation Matrix For Numerical Coloums 

plt.figure(figsize=(12, 8))
corr_matrix = train_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

#3) Scatterplot of Annual_Income VS Credit_Score 

plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x='Annual_Income', y='Credit_Score', hue='Credit_Score', palette='Set1')
plt.title('Annual Income vs Credit Score Category')
plt.xlabel('Annual Income')
plt.ylabel('Credit Score Category')
plt.show()

#4) Box Plot to show Credit Score across different Occupations
plt.figure(figsize=(12, 6))
sns.boxplot(data=train_df, x='Occupation', y='Credit_Score', palette='Set3')
plt.title('Credit Score Distribution by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Credit Score Category')
plt.xticks(rotation=90)
plt.show()
