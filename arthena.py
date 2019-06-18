import tensorflow as tf
import numpy as np
from tensorflow import keras
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

root_dir = os.getcwd()

# root directionary for the training data
training_data_dir = root_dir+"/artists/"

# training data files
training_data_files = glob.glob(training_data_dir+"*.csv")
li = []
#inch to cm converstion
in_to_cm = 2.54
#mm to cm converstion
mm_to_cm = 0.1

#get the training data using panda
for filename in training_data_files:
	dataframe = pd.read_csv(filename, header = 0, index_col = None)
	li.append(dataframe)

training_dataframes = pd.concat(li, axis = 0, ignore_index=True)

#extract only relevant data columns
#exluded features - 
#artist nationalities: 2 of the 3 is training sets are american artist, this feature will more likely create bias and it is already correlated with artist's name feature
#artist birth year: It is strongly correclate with artist's death year and death year is a a feature that is already included
#auction_sale_id: each sale id is unique, not a great training feature
#lot_id: The data point is highly correlated to lot_place_in_auction, which is already captured
#lot_description: skip for now, but could potentially use an nlp clustering algorithms to explore key word/hammer price correlation
#lot_link: not relevant to the training model
#work_title: same as description, no discernable insight for the corrent model
#work dimentions: features being captured in work_width and work_height
#buyer_premium: ignored per instruction
#lot_place_in_auction and auction_lot_count:  Was included in V1, model performed much better without them.  They attributed to overfitting.

df_condensed = training_dataframes[['artist_name', 'artist_death_year', 'auction_house', 
'auction_department', 'auction_location', 'auction_date', 'auction_currency', 'exchange_rate_to_usd',  
 'work_medium', 'work_execution_year', 'work_width', 'work_height', 'work_measurement_unit', 'hammer_price', 'estimate_low', 'estimate_high']]


#filter out rows with missing data
training_dataframes_filtered = df_condensed[(df_condensed.auction_department != -1) & (df_condensed.auction_department != "-1") 
 & (df_condensed.work_execution_year > 0) & (df_condensed.work_height > 0) & (df_condensed.work_width > 0) 
 & (df_condensed.work_measurement_unit != -1) & (df_condensed.hammer_price > 0) & (df_condensed.estimate_low > 0) & (df_condensed.estimate_high > 0)]


#determine what is the final training data size after filtering
print (len(training_dataframes_filtered))
#visualize the training data using seaborn
#sns.set()
#sns.pairplot(training_dataframes_filtered[["hammer_price", "estimate_low", "estimate_high"]], diag_kind="kde")
#plt.show()

#get unique values for non-numerical data
#['Andy Warhol' 'Pablo Picasso' 'Sol Lewitt']
artist_names = training_dataframes_filtered.artist_name.unique()
#['Post-War & Contemporary' 'Prints & Multiples' 'Photographs' 'Decorative Arts' 'Regional Specific' 'Impressionist & Modern']
auction_departments = training_dataframes_filtered.auction_department.unique()
#['Sothebys' 'Christies']
auction_houses = training_dataframes_filtered.auction_house.unique()
#['London' 'New York' 'Paris' 'Online' 'Amsterdam' 'Milan']
auction_locations = training_dataframes_filtered.auction_location.unique()
#['GBP' 'USD' 'EUR']
auction_currencies = training_dataframes_filtered.auction_currency.unique()
#['print' 'drawing' 'painting' 'watercolor' 'photograph' 'sculpture' 'color drawing' 'poster' 'pastel' 'decorative arts']
work_mediums = training_dataframes_filtered.work_medium.unique()


#new headers to be added to the dataframe to allow one-hot encoding
one_hot_headers = np.concatenate((artist_names, auction_departments, auction_houses, auction_locations, auction_currencies,
	work_mediums))

#default all the one-hot fields to 0
for x in one_hot_headers:
	training_dataframes_filtered[x] = 0

#create one-hot encoding training data, normalize currency and work measurement denominations and auction dates
for index, row in training_dataframes_filtered.iterrows():
	#only take the year and month of the auction date, normalize month as it has a fixed range.
	auction_year = int(row.auction_date[0:4])
	#auction_month = float(row.auction_date[5:7])/12
	training_dataframes_filtered.at[index, "auction_year"] = auction_year
	#training_dataframes_filtered.at[index, "auction_month"] = auction_month

	#derive a new feature as auction year minus artist death year
	gap_year = auction_year - row.artist_death_year
	training_dataframes_filtered.at[index, "gap_year"] = gap_year

	for name in artist_names:
		if row.artist_name == name:
			training_dataframes_filtered.at[index, name] = 1

	for department in auction_departments:
		if row.auction_department == department:
			training_dataframes_filtered.at[index, department] = 1

	for house in auction_houses:
		if row.auction_house == house:
			training_dataframes_filtered.at[index, house] = 1

	for location in auction_locations:
		if row.auction_location == location:
			training_dataframes_filtered.at[index, location] = 1

	for currency in auction_currencies:
		if row.auction_currency == currency:
			training_dataframes_filtered.at[index, currency] = 1

	for medium in work_mediums:
		if row.work_medium == medium:
			training_dataframes_filtered.at[index, medium] = 1

	if row.auction_currency != 'USD':
		hammer_price_USD = row.hammer_price/row.exchange_rate_to_usd
		training_dataframes_filtered.at[index, 'hammer_price'] = hammer_price_USD

	if row.work_measurement_unit == 'in':
		training_dataframes_filtered.at[index, 'work_width'] = row.work_width*in_to_cm
		training_dataframes_filtered.at[index, 'work_height'] = row.work_height*in_to_cm

	if row.work_measurement_unit == 'mm':
		training_dataframes_filtered.at[index, 'work_width'] = row.work_width*mm_to_cm
		training_dataframes_filtered.at[index, 'work_height'] = row.work_height*mm_to_cm



#print (training_dataframes_filtered.sample(10))

df_label = training_dataframes_filtered[['hammer_price']]
df_training = training_dataframes_filtered.drop(['hammer_price', 'artist_name','auction_department', 'exchange_rate_to_usd',
	'auction_date', 'auction_house', 'auction_location', 'auction_currency', 'work_measurement_unit', 'work_medium'], axis = 1)

df_estimate_lows = training_dataframes_filtered[['estimate_low']]

# convert panda dataframe to numpy array
training_data = df_training.values
training_label = df_label.values
estimate_lows = df_estimate_lows.values

print(training_data.shape)
print(training_label.shape)

#batch size
batch_size = 3630
time_step = 1

#reshape traning data into 3D for RNN
#adding time step to training data and label
#training_data = training_data.reshape(batch_size, time_step, training_data.shape[1])
#training_label = training_label.reshape(batch_size, time_step)

#print(training_data.shape)
#print(training_label.shape)
#create training model, using Relu activation for hidden layer and linear for output layer
#add dropout layer to reduce over fitting
model = tf.keras.Sequential([
		tf.keras.layers.Dense(1024, activation = 'relu'),
        tf.keras.layers.Dense(512, activation = 'relu'),
        #tf.keras.layers.SimpleRNN(units=1000, input_shape=(training_data.shape[1], training_data.shape[2]), activation = "relu", return_sequences = True),
        #tf.keras.layers.SimpleRNN(units=500, activation = "relu"),
        tf.keras.layers.Dense(1)
])

#compile the model 
#optimizer: ADAM
#Metrics: MAPE, MSE, MAE and Cosine
model.compile(loss= 'MAPE',
                optimizer='adam',
                metrics=[ 'mape', 'mae'])

#train the data, split bdetween 90/10 between training and validation data
model.fit(training_data, training_label, epochs=2000, batch_size=batch_size, validation_split = 0.05)
model.summary()

#get predictions based on trained model
predicted_results = model.predict(training_data)

#get normalized residuals of the results.
norm_residual = (training_label - predicted_results) / estimate_lows

residuals = np.concatenate(norm_residual, axis= 0)

#exclude outliers from the histogram
first_edge = -1
last_edge = 3

median = np.round(np.median(residuals),3)
mean = np.round(np.mean(residuals),3)

print(median)
print(mean)

n, bins, patches = plt.hist(x=residuals, bins="auto", range=(first_edge,last_edge))
plt.text(1, 250, r'mean='+str(mean)+', median='+str(median))
plt.title('(hammer_price - predicted_hammer_price) / estimate_low')
plt.show()



