"""
Analyzing Decomposition in Sanitation Systems (Python Pipeline)

Hey there! This script is the main workhorse for my science fair project on how hygiene products break down over time. Here you'll find all the core data wrangling, stats, and machine learning bits. If you're looking for extra stats or fancier plots, check out the R script!
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

# --- Load the data ---
csv_file = 'degradation_data.csv'
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Couldn't find '{csv_file}'. Please make sure your CSV is in the same folder as this script!")

# Read in the measurements (each row = one observation from a trial)
df = pd.read_csv(csv_file)

# --- Clean up and prep the data ---
# Make sure mass and day are numbers (sometimes Excel/CSV can be weird)
for col in ['mass_g', 'day']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Drop any rows where we couldn't get a mass or day
# (If you see a lot of rows dropped, check your CSV for typos!)
df = df.dropna(subset=['mass_g', 'day'])
df['trial_id'] = df['trial_id'].astype(str)
df['product_type'] = df['product_type'].astype(str)

# --- Calculate percent mass remaining ---
# For each trial, figure out what % of the original mass is left each day
# (This makes it easier to compare different products and trials)
df['initial_mass'] = df.groupby('trial_id')['mass_g'].transform('first')
df['percent_remaining'] = df['mass_g'] / df['initial_mass'] * 100

# --- Linear regression: does product type or day matter? ---
# Let's see how much product type and time affect decomposition
model = ols('percent_remaining ~ day * product_type', data=df).fit()
with open('linear_regression_summary.txt', 'w') as f:
    f.write(model.summary().as_text())

# --- Repeated Measures ANOVA ---
# This checks if the differences we see are statistically significant, accounting for repeated trials
try:
    aovrm = AnovaRM(df, 'percent_remaining', 'trial_id', within=['day', 'product_type'])
    res = aovrm.fit()
    with open('anova_results.txt', 'w') as f:
        f.write(str(res))
except Exception as e:
    with open('anova_results.txt', 'w') as f:
        f.write(f"ANOVA could not be performed: {e}\n")

# --- Neural network: can we predict decomposition? ---
# Let's see if a simple neural net can learn the pattern of mass loss over time
X = pd.get_dummies(df[['day', 'product_type']])
y = df['percent_remaining']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Convert to float32 to make TensorFlow happy
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

model_nn = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])
model_nn.compile(optimizer='adam', loss='mse')
model_nn.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# Save the neural net's predictions for later comparison
y_pred = model_nn.predict(X_test).flatten()
pred_df = pd.DataFrame({'actual': y_test.values, 'predicted': y_pred})
pred_df.to_csv('nn_predictions.csv', index=False)

# --- Visualization: how does decomposition look over time? ---
agg = df.groupby(['product_type', 'day'])['percent_remaining'].mean().reset_index()
plt.figure(figsize=(10,6))
for product in agg['product_type'].unique():
    subset = agg[agg['product_type'] == product]
    plt.plot(subset['day'], subset['percent_remaining'], marker='o', label=product)
plt.xlabel('Day')
plt.ylabel('Average % Mass Remaining')
plt.title('Decomposition of Hygiene Products Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('decomposition_report.png')
plt.close()

# --- Save a summary table for easy reference ---
agg.to_excel('decomposition_summary.xlsx', index=False)

print("All done! Check out the output files for results, stats, and plots. If you want to dig deeper, try running the R script for extra analysis.") 