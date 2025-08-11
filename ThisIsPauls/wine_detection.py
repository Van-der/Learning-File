import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB,GaussianNB

import shap
#loading the wine dataset
wine=load_wine()
print(dir(wine))
df=pd.DataFrame(wine.data,columns=wine.feature_names)
df['target'] = wine.target
print(df.head())
#creating objects for MultinomialNB and GaussianNB
mnb=MultinomialNB()
gnb=GaussianNB()

#feature segregation
x=df.drop('target', axis=1)
y=df['target']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=25) #splitting the dataset into training and testing sets 
#the indepecdent variables are x_scaled and the dependent variable is just y

#fitting the models
mnb.fit(x_train, y_train)
print("MultinomialNB Score:", mnb.score(x_test, y_test))
gnb.fit(x_train, y_train)
print("GaussianNB Score:", gnb.score(x_test, y_test))


# Predicting the class of a new sample
new_sample_data = [[13.0, 2.5, 2.0, 18.0, 100.0, 2.5, 2.0, 0.3, 1.0, 3.0, 1.0, 2.5, 800.0]]
new_sample = pd.DataFrame(new_sample_data, columns=x.columns)


#shap explanation for MultinomialNB and GaussianNB
# Get the predictions for the new sample
mnb_prediction = mnb.predict(new_sample)[0]
gnb_prediction = gnb.predict(new_sample)[0]

print("\n--- Explaining MultinomialNB Prediction ---")
# Create a SHAP explainer for the MultinomialNB model
explainer_mnb = shap.KernelExplainer(mnb.predict_proba, x_train)
# Get the SHAP values for the new sample
shap_values_mnb = explainer_mnb.shap_values(new_sample)

print("SHAP values for MultinomialNB prediction (Class {}):".format(mnb_prediction))
# Slice and reshape the SHAP values to a 2D array for the predicted class
shap_values_for_predicted_class_mnb = shap_values_mnb[0, :, mnb_prediction]
shap_df_mnb = pd.DataFrame(shap_values_for_predicted_class_mnb.reshape(1, -1), columns=x.columns, index=['SHAP Value'])
print(shap_df_mnb)

# --- Top 6 Features for MultinomialNB ---
print("\n--- Top 6 Most Important Features for MultinomialNB ---")
# Calculate the absolute SHAP values to find the magnitude of contribution
abs_shap_scores_mnb = shap_df_mnb.iloc[0].abs()
# Sort the absolute scores in descending order and get the top 6 features
top_6_features_mnb = abs_shap_scores_mnb.nlargest(6)
# Get the original SHAP values for these top 6 features
top_6_shap_df_mnb = shap_df_mnb[top_6_features_mnb.index]
print(top_6_shap_df_mnb.T.rename(columns={'SHAP Value': 'SHAP Score'}))




print("\n--- Explaining GaussianNB Prediction ---")
# Create a SHAP explainer for the GaussianNB model
explainer_gnb = shap.KernelExplainer(gnb.predict_proba, x_train)
# Get the SHAP values for the new sample
shap_values_gnb = explainer_gnb.shap_values(new_sample)

print("SHAP values for GaussianNB prediction (Class {}):".format(gnb_prediction))
# Apply the same slicing and reshaping logic to GaussianNB's output
shap_values_for_predicted_class_gnb = shap_values_gnb[0, :, gnb_prediction]
shap_df_gnb = pd.DataFrame(shap_values_for_predicted_class_gnb.reshape(1, -1), columns=x.columns, index=['SHAP Value'])
print(shap_df_gnb)

# --- Top 6 Features for GaussianNB ---
print("\n--- Top 6 Most Important Features for GaussianNB ---")
# Calculate the absolute SHAP values for GaussianNB
abs_shap_scores_gnb = shap_df_gnb.iloc[0].abs()
# Sort the absolute scores and get the top 6 features
top_6_features_gnb = abs_shap_scores_gnb.nlargest(6)
# Get the original SHAP values for these top 6 features
top_6_shap_df_gnb = shap_df_gnb[top_6_features_gnb.index]
print(top_6_shap_df_gnb.T.rename(columns={'SHAP Value': 'SHAP Score'}))



print("MultinomialNB Prediction:", mnb.predict(new_sample))
print("GaussianNB Prediction:", gnb.predict(new_sample))
# Saving the model using joblib

import joblib
joblib.dump(mnb, 'mnb_wine_model.pkl')
joblib.dump(gnb, 'gnb_wine_model.pkl')
