import tensorflow as tf
import numpy as np
from tensorflow import keras
import csv
import pandas as pd
import seaborn as sns
import os
import glob

root_dir = os.getcwd()

# root directionary for the training data
training_data_dir = root_dir+"/artists/"

# training data files
training_data_files = glob.glob(training_data_dir+"*.csv")
li = []

#get the training data using panda
for filename in training_data_files:
	dataframe = pd.read_csv(filename, header = 0, index_col = None)
	li.append(dataframe)

training_dataframes = pd.concat(li, axis = 0, ignore_index=True)

#extract only relevant data columns
df_condensed = training_dataframes[['artist_name', 'artist_death_year', 'artist_nationality', 'auction_house', 
'auction_department', 'auction_location', 'auction_date', 'auction_currency', 'exchange_rate_to_usd', 'auction_lot_count', 
'lot_place_in_auction', 'work_medium', 'work_execution_year', 'work_width', 'work_height', 'work_measurement_unit', 'hammer_price']]

#filter out rows with missing data
training_dataframes_filtered = df_condensed[(df_condensed.auction_department != -1) & (df_condensed.auction_lot_count != -1) 
& (df_condensed.work_execution_year != -1) & (df_condensed.work_height != -1) & (df_condensed.work_width != -1) 
& (df_condensed.work_measurement_unit != -1) & (df_condensed.hammer_price != -1)]

print(training_dataframes_filtered.sample(20))

# convert panda dataframe to numpy array
np_data = training_dataframes_filtered.values

print(len(np_data))
