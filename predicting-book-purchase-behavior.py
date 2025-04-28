import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


# Load the dataset
charlesBookClubDataFrame = pd.read_csv('/Users/imadulislamchowdhury/Downloads/ml2/final_exam/CharlesBookClub.csv')

# Inspect the dataset
print("Dataset Head:")
print(charlesBookClubDataFrame.head())

print("\nDataset Description:")
print(charlesBookClubDataFrame.describe())

print("\nDataset Info:")
print(charlesBookClubDataFrame.info())

# Summarize patterns in book purchases
charlesBookClubColumns = charlesBookClubDataFrame.columns[7:-4] # Columns representing book genres
charlesBookClubPurchase = charlesBookClubDataFrame[charlesBookClubColumns].sum()

print("\nSummary of Book Purchases:")
print(charlesBookClubPurchase)

#Transform
# Remove irrelevant columns
charlesBookClubDataFrame = charlesBookClubDataFrame.drop(columns=['Seq#', 'ID#'])

# Check for missing values
missing_values = charlesBookClubDataFrame.isnull().sum()

# Print the columns with missing values
print("Columns with missing values:")
print(missing_values[missing_values > 0])

# Create a binary target column from Yes_Florence and No_Florence
charlesBookClubDataFrame['FlorenceBuyer'] = charlesBookClubDataFrame['Yes_Florence'].apply(lambda x: 1 if x == 1 else 0)

# Drop the original Yes_Florence and No_Florence columns
charlesBookClubDataFrame.drop(columns=['Yes_Florence', 'No_Florence'], inplace=True)
print(charlesBookClubDataFrame.info())

print()
print("Technical Checkpoint 1: ")
print()
print("Which features appear to influence specialty travel purchases?")
print("The primary factor which is influencing the customer behavior in the purchase of the Florence travel book include previous purchases of books relating to Florence (with .8635), \nthe total expenditure by the customer, and the timing of their initial purchase.")
print()

# Split data into features and target
X = charlesBookClubDataFrame.drop(columns=['FlorenceBuyer'])
y = charlesBookClubDataFrame['FlorenceBuyer']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Random Forest classification model
charlesBookClubModel = RandomForestClassifier(n_estimators=100, random_state=42)
charlesBookClubModel.fit(X_train, y_train)

# Predict on the test set
y_pred = charlesBookClubModel.predict(X_test)

# Evaluate the model using accuracy, precision, and recall
charlesBookClubAccuracy = accuracy_score(y_test, y_pred)
charlesBookClubPrecision = precision_score(y_test, y_pred)
charlesBookClubRecall = recall_score(y_test, y_pred)

print(f"Model Accuracy: {charlesBookClubAccuracy}")
print(f"Model Precision: {charlesBookClubPrecision}")
print(f"Model Recall: {charlesBookClubRecall}")

# Plot feature importance
charlesBookClubFeatureImportances = charlesBookClubModel.feature_importances_
charlesBookClubFeatures = X.columns

plt.figure(figsize=(10, 6))
plt.barh(charlesBookClubFeatures, charlesBookClubFeatureImportances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance Plot for Charles Book Club')
plt.show()

# Interpret key drivers of customer behavior
charlesBookClubImportantFeatures = pd.Series(charlesBookClubFeatureImportances, index=charlesBookClubFeatures).sort_values(ascending=False)
print("\nKey Drivers of Customer Behavior:")
print(charlesBookClubImportantFeatures.head(10))

# Convert genre purchase data into binary format
charlesBookClubGenreColumns = charlesBookClubDataFrame.columns[7:-4]
charlesBookClubDataFrame_binary = charlesBookClubDataFrame[charlesBookClubGenreColumns].applymap(lambda x: 1 if x > 0 else 0)

# Apply Apriori algorithm to extract rules with support, confidence, and lift
charlesBookClubFrequentItemsets = apriori(charlesBookClubDataFrame_binary, min_support=0.01, use_colnames=True)
charlesBookClubRules = association_rules(charlesBookClubFrequentItemsets, metric="lift", min_threshold=1.0)

# Sort rules by lift and get the top 3 rules
charlesBookClubTopRules = charlesBookClubRules.sort_values(by='lift', ascending=False).head(3)

print("Top 3 Association Rules:")
print(charlesBookClubTopRules)

#Recommendations
for index, rule in charlesBookClubTopRules.iterrows():
    charlesBookClubAntecedents = ', '.join(list(rule['antecedents']))
    charlesBookClubConsequents = ', '.join(list(rule['consequents']))

    print(f"\nRule {index + 1}:")
    print(f"Antecedents: {charlesBookClubAntecedents}")
    print(f"Consequents: {charlesBookClubConsequents}")
    print(f"Support: {round(rule['support'], 3)}")
    print(f"Confidence: {round(rule['confidence'], 3)}")
    print(f"Lift: {round(rule['lift'], 3)}")
    print("Actionable Insights:")
    print(f"Customers who buy {charlesBookClubAntecedents} are likely to also buy {charlesBookClubConsequents}.")
    print(f"Consider bundling these genres or offering combo discounts on {charlesBookClubAntecedents} + {charlesBookClubConsequents} in marketing campaigns.")

print()
print("Technical Checkpoint 2: ")
print()
print("Why might an ensemble method outperform a simple logistic regression here?")
print("An ensemble approach can surpass a basic logistic regression model by integrating multiple models to better capture complex feature relationships and interactions. \nThis method mitigates overfitting and enhances generalization. \nTechniques such as Random Forest and Gradient Boosting are particularly effective in managing non-linearities and feature interactions, resulting in improved predictive accuracy across various datasets.")
print()