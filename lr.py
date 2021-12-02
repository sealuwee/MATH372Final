import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn.model_selection import train_test_split
from statsmodels.regression.linear_model import OLS
import seaborn as sns
import matplotlib.pyplot as plt

class LR:
	global df 
	# starts as none until df is committed on object instantiation
	df = None

	def __init__(self, dataset, response, eval_metric, verbose=0):
		# string that represents the column in the dataset that you would like to specifty as the response.
		self.response = response
		# default eval metric = , otherwise unless specified as something else
		self.eval_metric = eval_metric
		# string that represents a path to the dataset
		self.dataset = dataset
		# verbosity
		self.verbose = verbose


	def preprocessing(self):
		PATH = self.dataset

	def show_output():
		pass

	def __processSubset(self, feature_set):
   		# Fit model on feature_set and calculate RSS
	    model = sm.OLS(y,X[list(feature_set)])
	    regr = model.fit()
	    
	    if self.eval_metric == "RSS":
	    	RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()

	    #TODO: write more conditional statements based on what eval metric is stated by the user

	    return {"model":regr, "evaluation metric {}".format(self.eval_metric):RSS}


	def __getBest(k):
    
		tic = time.time()

		results = []

		for combo in itertools.combinations(X.columns, k):
			results.append(processSubset(combo))

		# Wrap everything up in a nice dataframe
		models = pd.DataFrame(results)

		# Choose the model with the highest RSS
		best_model = models.loc[models['RSS'].argmin()]

		toc = time.time()
		print("Processed", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")

		# Return the best model, along with some other useful information about the model
		return best_model




