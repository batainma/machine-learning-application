import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as mlp
from datetime import datetime
from datetime import timedelta

# CSV modifier to create usable test and training data for an RNN
# Mohammed Bataineh
def main():

	# Read in the training and test data
	TrainCSV = "./train.csv"
	TestCSV = "./test.csv"
	df = pd.read_csv(TrainCSV)
	df2 = pd.read_csv(TestCSV)
	
	print("\nOriginal dataset:")
	print(df.head())
	print(".\n.\n.")
	
	# Isolate the cell tower that I want to use
	df = isolateCells('Cell_000111', df)
	df2 = isolateCells('Cell_000111', df2)

	print(df.shape)
	print(df2.shape)
	
	df = formatIndex(df)
	df2 = formatIndex(df2)
	print("\nFormatted dataset:")
	print(df.head())

	# Put the new data into my folder
	df.to_csv("./modifiedTrain.csv")
	df2.to_csv("./modifiedTest.csv")


# Isolates the cell that I want to predict
def isolateCells(cell_name, df):
	df = df[df['CellName'] == cell_name]
	#df.reset_index(drop=True, inplace=True)
	print("\nFiltered dataset:")
	print(df.head())
	return df


# Makes the data index-able by adding the hour to the date column
def formatIndex(df):
	# Increment through the dataframe
	for i in df.index:
		# Grab the date and hour values
		date = df.at[i, 'Date']
		hour = df.at[i, 'Hour']
		
		# Convert to a datetime object and add the hour
		dateObject = datetime.strptime(date, '%m/%d/%Y %H:%M')
		dateObject = dateObject + timedelta(hours=hour.item())
		dateObject = datetime.strftime(dateObject, '%Y/%m/%d %H:%M')

		# Replace date with our datetime objects
		df.at[i, 'Date'] = dateObject

	# Set and sort the indices
	df = df.set_index("Date")
	df = df.sort_index()
	return df




if __name__ == "__main__":
	main()
