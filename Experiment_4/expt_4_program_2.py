# -*- coding: utf-8 -*-
"""Expt_4_Program_2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sTJPxFZ3nHh5i-YOkG8ihxrmi1r9dbFB
"""

import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset (Assuming you have a 'titanic.csv' or similar dataset)
# Replace this with your actual dataset and adjust column names accordingly
data = pd.read_csv('titanic.csv')

# Prepare the dataset (Example using Titanic dataset)
X = data[['pclass', 'sex']]  # Use appropriate feature columns
X['sex'] = X['sex'].map({'male': 0, 'female': 1})  # Convert 'sex' to numerical values
y = data['survived']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Create the main tkinter window
root = tk.Tk()
root.title("Logistic Regression Results By muskan 211P067")

# Generate the classification report
report = classification_report(y_test, model.predict(X_test), output_dict=True)
report_text = pd.DataFrame(report).transpose().round(2).to_string()

# Display the classification report
text = tk.Text(root, height=10, width=60)
text.insert(tk.END, report_text)
text.pack()

# Gender selection
gender_label = tk.Label(root, text="Select Gender:")
gender_label.pack(pady=5)
gender = ttk.Combobox(root, values=["male", "female"])
gender.pack(pady=5)

# Passenger class selection
pclass_label = tk.Label(root, text="Select Pclass:")
pclass_label.pack(pady=5)
pclass = ttk.Combobox(root, values=[1, 2, 3])
pclass.pack(pady=5)

# Function to filter survivors based on gender and class
def show_survivors():
    filtered = data[
        (data['sex'] == gender.get()) &
        (data['pclass'] == int(pclass.get())) &
        (data['survived'] == 1)
    ]
    result_text = f"Survivors: {len(filtered)}"
    result_label.config(text=result_text)

# Button to trigger the survivor filter function
button = ttk.Button(root, text="Show Survivors", command=show_survivors)
button.pack(pady=10)

# Label to display the number of survivors
result_label = tk.Label(root, text="")
result_label.pack(pady=5)

# Run the tkinter application
root.mainloop()

