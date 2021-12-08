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
from sklearn.metrics import mean_squared_error
import itertools
import time

# going to try interfacing R to use regsubsets()
import rpy2
# or maybe we dont have to

#TODO: replace all values for y & X with response and design matrix
#TODO: test cross validation and different metrics provided in object instantiation
#TODO: add plot functions from scratch paper2
#TODO/Stretch: write preprocessing for ranks

class LR:
	# global variables to represent the dataframe
	# X represents the design matrix
	# y represents the repsonse
	# k represents the number of features after preprocessing
	global k
	global df
	global X
	global y
	global _df
	global fw_df 
	global models_df
	df = None
	X = None
	y = None
	k = 0
	_df = None
	fw_df = None
	models_df = None

	def __init__(self, dataset, response, selection='forward', prediction=True, eval_metric='RSS', verbose=0):
		# string that represents a path to the dataset
		self.dataset = dataset
		# string that represents the column in the dataset that you would like to specifty as the response.
		self.response = response
		# default eval metric = , otherwise unless specified as something else
		# preferable eval metric == 
		# ridge+lasso increase RSS because we're worried about overfitting
		#TODO: need argument that specifies inference or prediction 
		# if inference == different way for model selection lasso for variability
		# Mallows CP AIC BIC
		# default selection is forward because exhaustive selection is of 2^n complexity and we need reproducability.
		self.selection = selection
		self.prediction = prediction
		#unless otherwise stated, prediction will be done by default.
		# either RSS or R_squared
		self.eval_metric = eval_metric
		# verbosity, in case you do not want to see that many graphs 0 for bare minimum, any number greater than 0 for verbose
		self.verbose = verbose

		# Object instantiation begins
		self.__preprocessing()
		# the amount of models to be created would be n choose r
		# this would require that this function has 2^n complexity
		# models = self.__exhaustiveSubsetSelection(4)
		# best_model = self.__getBestModel()

		if self.selection == 'forward':
			self.__forwardSelection()
			# print(len(models))
			# print(fw_df.columns)
			# print(len(models_df))
			# print(models_df.columns)
			# print(len(fw_df))
			# print(fw_df.head())
			if verbose > 0:
				self.plot_RSS_and_R_squared(fw_df)
				self.__displayDataFrame(fw_df,5)

			else:
				print('Skipping plotting RSS and R_squared values for {} selection'.format(self.selection))
				print('Performing Model comparisons with the chosen metric: {}'.format(self.eval_metric))

		else:
			print('\n Exhaustive selection is 2^n in complexity. \
				\n that would require that we would have some sort of threading implemented to do Best Subset Selection \
				\n Hopefully we can do this in the future, but for now please use : selection=\'forward\' \n')


		# the verbose option allows the user to determine how much information will be presented and accessible on object instantiation.
		# global _df

		# The verbose option provides a dataframe with all scores
		# if self.eval_metric != None:
			# Instantiate new dataframe to add columnar values for the best scores for all eval metrics for plotting purposes.
			# donot plot RSS/RSquared
			# _df = models[models.groupby('num_features')['RSS'].transform(min) == models['RSS']]
			# _df['min_RSS'] = _df.groupby('num_features')['RSS'].transform(min)

			# _df = models[models.groupby('num_features')['R_squared'].transform(max) == models['R_squared']]
			# _df['max_R_squared'] = _df.groupby('num_features')['R_squared'].transform(max)

			# _df = models[models.groupby('num_features')['AIC'].transform(min) == models['AIC']]
			# _df['min_AIC'] = _df.groupby('num_features')['AIC'].transform(min)

			# _df = models[models.groupby('num_features')['BIC'].transform(min) == models['BIC']]
			# _df['min_BIC'] = _df.groupby('num_features')['BIC'].transform(min)

			# _df = models[models.groupby('num_features')['Adjusted_R_squared'].transform(max) == models['Adjusted R_squared']]
			# _df['max_Adjusted_R_squared'] = _df.groupby('num_features')['Adjusted_R_squared'].transform(max)

		# else:
		# 	print("\n The following eval metrics are required to be specified during object instantiation: \
		# 		   \n eval_metric params : \
		# 		   \n RSS , R_squared, AIC, BIC, adjR_squared \
		# 		   \n Default = RSS ")

		# maybe only make a plot that shows all the metrics

		# else: 
		# 	# In the non verbose option we only populate a dataframe with the user's chosen eval metric for plotting purposes.
		# 	if self.eval_metric != None:
		# 		if self.eval_metric=='RSS':
		# 			_df = models[models.groupby('num_features')['RSS'].transform(min) == models['RSS']]
		# 			_df['min_RSS'] = _df.groupby('num_features')['RSS'].transform(min)
					
		# 		if self.eval_metric=='R_squared':
		# 			_df = models[models.groupby('num_features')['R_squared'].transform(max) == models['R_squared']]
		# 			_df['max_R_squared'] = _df.groupby('num_features')['R_squared'].transform(max)

		# 		if self.eval_metric=='AIC':
		# 			_df = models[models.groupby('num_features')['AIC'].transform(min) == models['AIC']]
		# 			_df['min_AIC'] = _df.groupby('num_features')['AIC'].transform(min)


		# 		if self.eval_metric=='BIC':
		# 			_df = models[models.groupby('num_features')['BIC'].transform(min) == models['BIC']]
		# 			_df['min_BIC'] = _df.groupby('num_features')['BIC'].transform(min)

		# 		if self.eval_metric=='adjR_squared':
		# 			_df = models[models.groupby('num_features')['Adjusted_R_squared'].transform(max) == models['Adjusted R_squared']]
		# 			_df['max_Adjusted_R_squared'] = _df.groupby('num_features')['Adjusted_R_squared'].transform(max)
	
		# 	else:
		# 		print("\n The following eval metrics are required to be specified during object instantiation: \
		# 			   \n eval_metric params : \
		# 			   \n RSS , R_squared, AIC, BIC, adjR_squared \
		# 			   \n Default = RSS ")

		# if self.verbose > 0:
		# 	self.__displayDataFrame(df,10)
		# 	#TODO: PLOT FUNCTION HERE
		# else:
		# 	self.__displayDataFrame(df,3)
		# 	#TODO: PLOT FUNCTION HERE

		# self.__displaySummary(best_model)
		# this should show a plot of the models


	# helper functions for display purposes
	def __displayDataFrame(self,df,n):
		#displays n results from a resulting dataframe
		print(df.head(n))

	def __displaySummary(self,model):
		model.summary()

	# private functions in order of operations
	def __preprocessing(self):
		global df
		global y
		global X
		global k

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
				ohe =pd.get_dummies(df, drop_first=True)
				df = df.drop(col,axis=1)
				df = pd.concat([df,ohe],axis=1)
			else:
				pass

		y = df[str(self.response)]
		if y.shape[1]>1:
			y = y.loc[:,~y.columns.duplicated()]
		X = df.drop(str(self.response),axis=1)
		k = len(X.columns.tolist())

		return df, X, y, k

	def __processSubset(self,combination):
		# Fit model on feature_set and calculate RSS

		model = sm.OLS(y,X[list(combination)])
		model = model.fit()
		RSS = mean_squared_error(y,model.predict(X[list(combination)])) * len(y)
		R_squared = model.rsquared
		# AIC = model.aic
		# BIC = model.bic
		# adj_R_squared = model.rsquared_adj

		return model,RSS,R_squared


	def __exhaustiveSubsetSelection(self,i):
		# get best subset from collection of models
		# This currently does not work
		
		models, feature_list, num_features = [],[],[]
		RSS_list, R_squared_list, AIC_list, BIC_list, adj_R_squared_list = [],[],[],[],[]

		#Looping over all possible combinations: from 1 to k
		for i in range(1, i+1):
			for combo in itertools.combinations(X.columns,i):
				result = self.__processSubset(combo) 
				models.append(result[0])
				RSS_list.append(result[1])                 
				R_squared_list.append(result[2])
				# AIC_list.append(result[3])
				# BIC_list.append(results[4])
				# adj_R_squared_list.append(results[5])
				feature_list.append(combo)
				num_features.append(len(combo))   

			# return a dataframe of best subset selection

		return pd.DataFrame({'model': models,
							 'num_features': num_features,
							 'RSS': RSS_list, 
							 'R_squared':R_squared_list,
							 # 'AIC': AIC_list,
							 # 'BIC': BIC_list,
							 # 'Adjusted_R_squared': adj_R_squared_list,
							 'features':feature_list})

	def __forwardSelection(self):
		
		remaining_features = list(X.columns.values)
		features = []
		models = []
		RSS_list, R_squared_list = [np.inf], [np.inf] #Due to 1 indexing of the loop...
		features_list = dict()

		for i in range(1,k+1):
			
			best_RSS = np.inf

			for combo in itertools.combinations(remaining_features,1):
				
				RSS = self.__processSubset(combo)   #Store temp result 
				
				if RSS[1] < best_RSS:
					best_model = RSS[0]
					best_RSS = RSS[1]
					best_R_squared = RSS[2]
					best_feature = combo[0]

		    #Updating variables for next loop
			models.append(best_model)
			features.append(best_feature)
			remaining_features.remove(best_feature)

			#Saving values for plotting
			RSS_list.append(best_RSS)
			R_squared_list.append(best_R_squared)
			features_list[i] = features.copy()

		global fw_df
		global models_df

		fw_df = pd.concat([pd.DataFrame({'features':features_list}),pd.DataFrame({'RSS':RSS_list, 'R_squared': R_squared_list})], axis=1, join='inner')
		models_df = pd.DataFrame({'models':models})
		# models_df = pd.concat([pd.DataFrame({'features': features_list}),pd.DataFrame({'models':models})], axis=1,join='outer')
		
		fw_df['num_features'] = fw_df.index
		# models_df['num_features'] = models.index
		# remove first row because there's a NaN feature value for the first model
		# models_df = models_df.iloc[1:,:]
		fw_df = pd.concat([fw_df,models_df],axis=1)
		fw_df = fw_df.iloc[1:,:]

		return fw_df

	def __computeModelComparison(self, df):

		m = len(y)
		p = len(X.columns)
		hat_sigma_squared = (1/(m - p -1)) * min(_df['RSS'])
		# math for these model metrics
		AIC = (1/(m*hat_sigma_squared)) * (df['RSS'] + 2 * df['num_features'] * hat_sigma_squared )
		BIC = (1/(m*hat_sigma_squared)) * (df['RSS'] +  np.log(m) * df['num_features'] * hat_sigma_squared )
		C_p = (1/m) * (df['RSS'] * df['num_features'] * hat_sigma_squared)
		Adjusted_R_squared = 1 - ( (1 - df['R_squared'])*(m-1)/(m-df['num_features'] -1))

		return df

	def __getBestModel(self,df):
			
		# if self.eval_metric == 'RSS':
		# if self.eval_metric == 'R_squared':		
		# if self.eval_metric == 'AIC':
		# if self.eval_metric == 'BIC':
		# if self.eval_metric == 'C_p':
		# if self.eval_metric == 'Adjusted_R_squared':
		return 

	def __plotFittedValuesAndResiduals(self):
		#TODO: add plotting here once we're ready to start working with outliers

		f, ax = plt.subplots(figsize=(13, 13))
		sns.despine(f, left=True, bottom=True)
		sns.scatterplot(x=model_1.fittedvalues, y=model_1.resid,
						palette="ch:r=-.2,d=.3_r",
						sizes=(1, 8), linewidth=0, ax=ax)
		plt.axhline(y=0,color='red')
		plt.xlabel("fitted values", fontsize=20)
		plt.ylabel("residuals",fontsize=20)
		plt.show()

	def model(self):
		# returns the best model chosen by the program

		return

	def plot_RSS_and_R_squared(self, df):
		# if you wanted to plot RSS and R squared you can.
		fig = plt.figure(figsize = (16,6))
		ax = fig.add_subplot(1,2,1)
		ax.scatter(df['num_features'],df['RSS'], alpha = .2, color = 'darkblue' )
		# ax.plot(df['num_features'],df['min_RSS'],color = 'r', label = 'Best subset')
		ax.set_xlabel('# Features')
		ax.set_ylabel('RSS')
		ax.set_title('RSS - {} subset selection'.format(self.selection))
		ax.legend()

		ax = fig.add_subplot(1, 2, 2)
		ax.scatter(df['num_features'],df['R_squared'], alpha = .2, color = 'darkblue' )
		# ax.plot(df['num_features'],df['max_R_squared'],color = 'r', label = 'Best subset')
		ax.set_xlabel('# Features')
		ax.set_ylabel('R squared')
		ax.set_title('R_squared - {} subset selection'.format(self.selection))
		ax.legend()

		plt.show()

	def plot_best_subset_selection(self,df):
		#TODO: add arguments 
		# ideally we want to write one function to plot things based on what we specify in the arguments

		if self.verbose > 0:

			fig = plt.figure(figsize = (16,6))
			# ax = fig.add_subplot(2, 3, 1)
			
			# plots should give you new information / meaningful information
			# incresed complexity versus lower error.

			# ax.scatter(df['num_features'],df['RSS'], alpha = .2, color = 'darkblue' )
			# ax.plot(df['num_features'],df['min_RSS'],color = 'r', label = 'Best subset')
			# ax.set_xlabel('# Features')
			# ax.set_ylabel('RSS')
			# ax.set_title('RSS - Best subset selection')
			# ax.legend()

			# ax = fig.add_subplot(2, 3, 2)
			# ax.scatter(df.['num_features'],df.['R_squared'], alpha = .2, color = 'darkblue' )
			# ax.plot(df['num_features'],df['max_R_squared'],color = 'r', label = 'Best subset')
			# ax.set_xlabel('# Features')
			# ax.set_ylabel('R squared')
			# ax.set_title('R_squared - Best subset selection')
			# ax.legend()

			ax = fig.add_subplot(2, 2, 1)
			ax.scatter(df['num_features'],df['AIC'], alpha = .2, color = 'darkblue' )
			ax.plot(df['num_features'],df['min_AIC'],color = 'r', label = 'Best subset')
			ax.set_xlabel('# Features')
			ax.set_ylabel('AIC')
			ax.set_title('AIC - Best subset selection')
			ax.legend()

			ax = fig.add_subplot(2, 2, 2)
			ax.scatter(df['num_features'],df['BIC'], alpha = .2, color = 'darkblue' )
			ax.plot(df['num_features'],df['min_BIC'],color = 'r', label = 'Best subset')
			ax.set_xlabel('# Features')
			ax.set_ylabel('BIC')
			ax.set_title('BIC - Best subset selection')
			ax.legend()

			ax = fig.add_subplot(2, 2, 3)
			ax.scatter(df['num_features'],df['Adjusted_R_squared'], alpha = .2, color = 'darkblue' )
			ax.plot(df['num_features'],df['max_Adjusted_R_squared'],color = 'r', label = 'Best subset')
			ax.set_xlabel('# Features')
			ax.set_ylabel('Adjusted R squared')
			ax.set_title('Adjusted R squared - Best subset selection')
			ax.legend()

		else: 

			fig = plt.figure(figsize = (16,6))

			if self.eval_metric == "RSS":
				ax = fig.add_subplot(1, 1, 1)
				ax.scatter(df['num_features'],df['RSS'], alpha = .2, color = 'darkblue' )
				ax.plot(df['num_features'],df['min_RSS'],color = 'r', label = 'Best subset')
				ax.set_xlabel('# Features')
				ax.set_ylabel('RSS')
				ax.set_title('RSS - Best subset selection')
				ax.legend()

			if self.eval_metric == "R_squared":
				ax = fig.add_subplot(1, 1, 1)
				ax.scatter(df['num_features'],df['R_squared'], alpha = .2, color = 'darkblue' )
				ax.plot(df['num_features'],df['max_R_squared'],color = 'r', label = 'Best subset')
				ax.set_xlabel('# Features')
				ax.set_ylabel('R squared')
				ax.set_title('R_squared - Best subset selection')
				ax.legend()


			if self.eval_metric == "AIC":
				ax = fig.add_subplot(1, 1, 1)
				ax.scatter(df['num_features'],df['AIC'], alpha = .2, color = 'darkblue' )
				ax.plot(df['num_features'],df['min_AIC'],color = 'r', label = 'Best subset')
				ax.set_xlabel('# Features')
				ax.set_ylabel('AIC')
				ax.set_title('AIC - Best subset selection')
				ax.legend()

			if self.eval_metric == "BIC":
				ax = fig.add_subplot(1, 1, 1)
				ax.scatter(df['num_features'],df['BIC'], alpha = .2, color = 'darkblue' )
				ax.plot(df['num_features'],df['min_BIC'],color = 'r', label = 'Best subset')
				ax.set_xlabel('# Features')
				ax.set_ylabel('BIC')
				ax.set_title('BIC - Best subset selection')
				ax.legend()

			if self.eval_metric == "Adjusted_R_squared":
				ax = fig.add_subplot(1, 1, 1)
				ax.scatter(df['num_features'],df['Adjusted_R_squared'], alpha = .2, color = 'darkblue' )
				ax.plot(df['num_features'],df['max_Adjusted_R_squared'],color = 'r', label = 'Best subset')
				ax.set_xlabel('# Features')
				ax.set_ylabel('Adjusted R squared')
				ax.set_title('Adjusted R squared - Best subset selection')
				ax.legend()
		
		plt.show()


if __name__ == "__main__":
	x = LR(dataset="loans_small.csv",response="int.rate",selection='forward', prediction=True, eval_metric="RSS", verbose=0)
	x