import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn.model_selection import train_test_split
from statsmodels.regression.linear_model import OLS
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.regressionplots import *
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
import itertools
import time

#TODO: replace all values for y & X with response and design matrix
#TODO: test cross validation and different metrics provided in object instantiation
#TODO: add plot functions from scratch paper2
#TODO/Stretch: write preprocessing for ranks

class LR:
	# global variables to represent the dataframe
	# X represents the design matrix
	# y represents the repsonse

	global df
	global X
	global y 
	df = None
	X = None
	y = None

	def __init__(self, dataset, response, eval_metric, verbose=0):
		# string that represents the column in the dataset that you would like to specifty as the response.
		self.response = response
		# default eval metric = , otherwise unless specified as something else
		# preferable eval metric == 
		# ridge+lasso increase RSS because ew're worried about overfitting
		#TODO: need argument that specifies inference or prediction 
		# if inference == different way for model selection lasso for variability
		# Mallows CP AIC BIC
		self.eval_metric = eval_metric
		# string that represents a path to the dataset
		self.dataset = dataset
		# verbosity
		self.verbose = verbose


		df,X,y = self.__preprocessing()
		models = self.__crossvalidation()

		print(models_best.loc[2, "model"].summary())

	def __preprocessing(self):
		PATH = self.dataset
		df = pd.read_csv(PATH)
		for col in df.columns.tolist():
		    if df[col].isna().sum() > 0:
		        print('NaN values found in column : ', col)
		        #drop NA values
		        df = df.dropna()
		        #reset the index
		        df = df[np.isfinite(california).all(1)].reset_index(drop="True")
		    else:
		        print('No NaN values found in column : ', col)

		for col,dtype in dict(df.dtypes).items():
		    if dtype == np.dtype('O'):
		        print('Handling object dtype column: \"{}" in design matrix with One Hot Encoding'.format(col))
		        # variable to represent the collection of one hot encoded columns
		        ohe = pd.get_dummies(df, drop_first=True)
		        df = df.drop(col,axis=1)
		        df = pd.concat([df,ohe],axis=1)

   		#TODO: specify the response column for y
   		#TODO: specify the design matrix for X
   		y = df[str(self.response)]
		y = y.loc[:,~y.columns.duplicated()]
		X = df.drop(str(self.response),axis=1)

		return df, X, y

	def __processSubset(self, feature_set):
   		# Fit model on feature_set and calculate RSS

	    model = sm.OLS(y,X[list(feature_set)])
	    regr = model.fit()
	    #TODO: 
	    #within a certain # of predictors 
	    #using best subset selection
	    #all the metrics will give you the same answers
	    #as long as you specify the amount of predictors in getBest(k)

	    #inputs being the RSS and # pof predictors
	    #in this case of this project we want 

	    #more preds = higher AIC, etc.
	    #can't tell unless you compare the models by the amount of predictors
	    #take the best N predictor model vs. M predictor model
	    #plot a chart with X being the # preds Y being AIC value
	    
	    if self.eval_metric == "RSS":
	    	RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()

	    #TODO: write more conditional statements based on what eval metric is stated by the user

	    return {"model":regr, "evaluation metric {}".format(self.eval_metric):RSS}

	def __getBest(self,k):
		# get best subset from collection of models
		tic = time.time()

		results = []

		for combo in itertools.combinations(X.columns, k):
			results.append(self.__processSubset(combo))

		# Wrap everything up in a nice dataframe
		models = pd.DataFrame(results)

		# Choose the model with the highest RSS
		best_model = models.loc[models['RSS'].argmin()]

		toc = time.time()
		print("Processed", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")

		# Return the best model, along with some other useful information about the model
		return best_model

	def __crossvalidation(self):
		
		models_best = pd.DataFrame(columns=["{}".format(self.eval_metric), "model"])

		tic = time.time()
		for i in range(1,8):
		    models_best.loc[i] = self.__getBest(i)

		toc = time.time()
		print("Total elapsed time:", (toc-tic), "seconds.")

		return models_best

	def plot(self):
		#TODO: add arguments 
		# ideally we want to write one function to plot things based on what we specify in the arguments

		pass


if __name__ == "__main__":
	x = LR(dataset="loan_data.csv",response="int.rate",eval_metric="RSS")
	x