import numpy as np
import pandas as pd

def load_auto():

	# import data
	Auto = pd.read_csv('/Users/falk/Documents/python_projects/Deep_learning_for_image_analysis/Assignment_1/Auto.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()

	# Extract relevant data features
	
	columns_to_normalize = ['cylinders','displacement','horsepower','weight', 'acceleration','year','origin']

	# Normalize the selected columns using min-max normalization
	normalized_data = (Auto[columns_to_normalize] - Auto[columns_to_normalize].min()) / (Auto[columns_to_normalize].max() - Auto[columns_to_normalize].min())

	# Replace the original columns with the normalized data
	Auto[columns_to_normalize] = normalized_data
	
	X_train_1 = Auto[['horsepower']].values
	X_train_7 = Auto[['cylinders','displacement','horsepower','weight', 'acceleration','year','origin']].values
	Y_train = Auto[['mpg']].values




	return X_train_1, X_train_7, Y_train

