We would like to train a model to predict the hammer_price of lots at upcoming auctions (some upcoming lots are included in your dataset). Note that this means that you can't use future data to predict the past. You may use as many or as few features as you like except for buyers_premium (it's a function of the final sale price. See SCHEMA.md for more details). This model should be optimized to minimize the relative error of each lot. Good solutions will have a MAPE below 30%.

- Did you perform any data cleaning, filtering or transformations to improve the model fit?
  Yes.  I filtered out the following features:
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

I also performed data cleaning to ignore rows with missing or incorrect training features.
I converted the categorical features using one-hot encoding
I parsed the auction dates into feature, year and month
I derived a feature gap_year, defined as auction_year - artist_death_year


- Why did you choose this model?
  I chose the set of the relevant features based on my personal experience of art prices and auction.  Given the limited traning data, I probably wouldn't use this model to predict the hammer price of the art works outside of the three artist in the training data.  I used ReLu activation function for its simplicity, computational efficiency and its non-linearity, which enables back-propagation.  I use 2 hidden layers because that seemed to work best for this model based on trail and error.  Additional hidden layers started to yield overfitting behavior.  I have also tried leakyrelu to compensate for Relu's asymetrical shortcomings but it didn't yield any substantially better result.



- What loss function did you use? Why did you pick this loss function?
 For the first version, I used MAPE loss function to hoping to minimize the relative error of each auction.  I also used MAE with additional hidden layers and dropouts.  It yielded similar results.  I would want to try RMSE, but it is not a off the shelf tensorflow loss function at the moment.


- Describe how your model is validated.
  I split the training data into 95/5. 95% for model training and 5% for validation to test for overfitting.  I used 95% of data for training instead of traditional 80% due to limited traning data I have after data cleaning. 



- What error metrics did you evaluate your model against?
  The metric is also MAPE so we care about minizing the relative error of each lot



- Generate and report a histogram of (hammer_price - predicted_hammer_price) / estimate_low (normalized model residuals). What are the mean and median values of this distribution?