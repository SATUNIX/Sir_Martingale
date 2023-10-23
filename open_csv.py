'''
IMPORTING CSV FINANCIAL DATA INTO MATHPLOT LIB FOR COMPARISONS



'''


# Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt

# Function to read data from a CSV file and plot it using Matplotlib
def plot_csv_data(file_name):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    
    # Extract the true and predicted values
    true_values = df['True_Values']
    predicted_values = df['Predicted_Values']
    
    # Plot the true and predicted values
    plt.plot(true_values, label='True Values')
    plt.plot(predicted_values, label='Predicted Values')
    plt.legend()
    plt.show()

# Example usage: Plotting the data from the saved CSV file
plot_csv_data('/mnt/data/Saved_Data.csv')
