import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Step 1: Load training data and ideal functions into Pandas DataFrames
training_data_file = "train.csv"
ideal_functions_file = "ideal.csv"
test_data_file = "test.csv"

training_data = pd.read_csv(training_data_file)
ideal_functions = pd.read_csv(ideal_functions_file)
test_data = pd.read_csv(test_data_file)

# Step 1.2: Create SQLite database and load data into separate tables
db_path = "your_database.db"

engine = create_engine(f"sqlite:///{db_path}")

# Load training data into a table named 'training_data'
training_data.to_sql("training_data", engine, if_exists="replace", index=False)

# Load ideal functions into a table named 'ideal_functions'
ideal_functions.to_sql("ideal_functions", engine, if_exists="replace", index=False)

# Step 1.3: Load test data
# Since we are not storing test data permanently in the database, we'll skip loading it into a table.

# Step 2: Implement Least-Squares to find ideal functions
def ideal_function(x, params):
    # Define the ideal function to fit the training data
    # For example, a simple quadratic function: y = ax^2 + bx + c
    num_params = len(params)
    a, b, c = params[:3]  # Extract the first three parameters for the quadratic function
    return a * x**2 + b * x + c

# Fit the ideal function to each training dataset using least_squares
fit_params = []
for i in range(1, 5):  # Assuming there are four training datasets
    x_train = training_data["x"]
    y_train = training_data[f"y{i}"]

    # Get the initial guess for the parameters
    initial_guess = np.ones(3)

    # Perform least squares fitting with bounds for the parameters
    result = least_squares(ideal_function, initial_guess, args=(y_train,), bounds=(0, np.inf))

    # Get the optimized parameters
    params = result.x
    fit_params.append(params)

# Calculate the chosen ideal functions with the least deviation
max_deviations = []
for i, params in enumerate(fit_params):
    x_train = training_data["x"]
    y_train = training_data[f"y{i+1}"]
    y_fit = ideal_function(x_train, params)
    deviations = y_train - y_fit
    max_deviation = np.max(np.abs(deviations))
    max_deviations.append(max_deviation)

chosen_indices = np.argsort(max_deviations)[:4]
chosen_ideal_functions = [fit_params[i] for i in chosen_indices]

# Step 2.2: Store the four chosen ideal functions in a new table in the database
chosen_ideal_df = pd.DataFrame(chosen_ideal_functions, columns=["a", "b", "c"])
chosen_ideal_df.to_sql("chosen_ideal_functions", engine, if_exists="replace", index=False)

# Step 2.3: Visualize the training data, ideal functions, and chosen ideal functions using matplotlib
fig, ax = plt.subplots(figsize=(10, 6))

# Plot training data
for i in range(1, 5):
    ax.plot(training_data["x"], training_data[f"y{i}"], label=f"Training Data {i}")

# Plot ideal functions
x_values = training_data["x"]
for i in range(1, 51):
    params = ideal_functions.iloc[:, i]
    y_ideal = ideal_function(x_values, params)
    ax.plot(x_values, y_ideal, "--", label=f"Ideal Function {i}")

# Plot chosen ideal functions
for i, params in enumerate(chosen_ideal_functions):
    y_chosen = ideal_function(x_values, params)
    ax.plot(x_values, y_chosen, label=f"Chosen Ideal Function {i+1}", linewidth=2)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Training Data, Ideal Functions, and Chosen Ideal Functions")
ax.legend()
plt.grid(True)
plt.show()

# Step 3.1: Create a DataFrame to store the test data mapping
test_mapping_df = pd.DataFrame(columns=["X", "Y", "Deviation", "Chosen_Ideal_Function"])

# Step 3.2: Visualize the test data mapping using matplotlib
fig, ax = plt.subplots(figsize=(10, 6))

# Plot training data
for i in range(1, 5):
    ax.plot(training_data["x"], training_data[f"y{i}"], label=f"Training Data {i}")

# Plot ideal functions
x_values = training_data["x"]
for i in range(1, 51):
    params = ideal_functions.iloc[:, i]
    y_ideal = ideal_function(x_values, params)
    ax.plot(x_values, y_ideal, "--", label=f"Ideal Function {i}")

# Plot chosen ideal functions
for i, params in enumerate(chosen_ideal_functions):
    y_chosen = ideal_function(x_values, params)
    ax.plot(x_values, y_chosen, label=f"Chosen Ideal Function {i+1}", linewidth=2)

# Define a list of colors to be used for the scatter plot
colors = ['red', 'blue', 'green', 'purple']

# Plot test data mapping
for index, row in test_mapping_df.iterrows():
    x_test = row["X"]
    y_test = row["Y"]
    chosen_ideal_func = row["Chosen_Ideal_Function"]
    deviation = row["Deviation"]
    
    ax.scatter(x_test, y_test, c=colors[chosen_ideal_func], marker="x", s=50, label=f"Test Data {index+1}")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Training Data, Ideal Functions, Chosen Ideal Functions, and Test Data Mapping")
ax.legend()
plt.grid(True)
plt.show()

# Close the database connection
engine.dispose()
