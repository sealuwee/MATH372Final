import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn.model_selection import train_test_split
from statsmodels.regression.linear_model import OLS
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.regressionplots import *
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
import pylab as pyl
from glmnet import ElasticNet


class LR(object):

	# global variables 

	global k
	global df
	global X
	global y
	global _df
	global fw_df 
	global models_df
	global df1
	global best_model
	global ridge_model
	global lasso_model
	df = None
	X = None
	y = None
	k = 0
	_df = None
	fw_df = None
	models_df = None
	df1 = None
	best_model = None
	ridge_model = None
	lasso_model = None

	# constructor for object instantiation.
	# this is where the bulk of the processes will be completed.

	def __init__(self, dataset, response, selection='forward', prediction=True, eval_metric='RSS', verbose=0):
		# string that represents a path to the dataset
		self.dataset = dataset
		# string that represents the column in the dataset that you would like to specifty as the response.
		self.response = response
		# default eval metric = , otherwise unless specified as something else
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
		global df1
		global best_model
		global ridge_model
		global lasso_model

		if self.selection == 'forward':
			
			self.__forwardSelection()

			if self.verbose > 0:
				self.__plot_RSS_and_R_squared(fw_df)
				self.__displayDataFrame(fw_df,5)
				print('\nPerforming Model comparisons with the chosen metric: {}\n'.format(self.eval_metric))
				print('\nCalling the method .model_selection() will provide an accessible dataframe\n')
				df1 = self.__computeModelComparison(fw_df)
				print('\nSince verbosity is set to {} the dataframe can be seen below before it has been sorted'.format(self.verbose))
				self.__displayDataFrame(df1,k)
				print('\nNow we will plot these model comparisons \n \
					Since verbose = {},\n \
					We are going to show the plots for all the eval metrics'.format(self.verbose))
				self.__plot_model_selection(df1)

			else:
				print('Skipping plotting RSS and R_squared values for {} selection'.format(self.selection))
				print('Performing Model comparisons with the chosen metric: {}'.format(self.eval_metric))
				print('\nCalling the method .model_selection() will provide an accessible dataframe\n')
				df1 = self.__computeModelComparison(fw_df)
				print('\nSince verbosity is set to {} the dataframe can be seen below before it has been sorted'.format(self.verbose))
				self.__displayDataFrame(df1,3)
				print('\nNow we will plot these model comparisons \n \
					Since verbose = {},\n \
					We are only going to show the plot for the chosen eval_metric = {}'.format(self.verbose,self.eval_metric))
				# self.__plot_model_selection(df1)

			# pull out the best model from the dataframe we created

			best_model = self.__getBestModel(df1)
			# print(best_model.columns)

			print('\nHere is the best model given the chosen eval_metric = {}\n'.format(self.eval_metric))

			print(best_model)

			print('\nThis model is accessible from the following line of code : \n \
				')

			print('\n Continuining we can conduct tests to discover the following : \n \
				- Outliers \n \
				- Influential Points \n \
				- Points of High Leverage \n \
				- Including plots with hypothesis tests\n')

			print('\nAgain we will check for the verbose flag = {} \n \
				to determine what information to show next'.format(self.verbose))

			model = self.get_best_model()

			self.__displaySummary(model)

			self.__plotFittedValuesAndResiduals(model)

			print('Checking correlation between fitted values and residuals . . .')

			self.__assertCorrelatedValues(self.__getCorrelatedValues(model))

			print('\nNow begins the normality assumption . . . ')

			print('\nOutputting histogram of residuals . . . ')

			self.__plot_histogram_residuals(model.resid)

			print('\nOutputting QQ Plot . . . ')

			sm.qqplot(model.resid, line='45')
	
			pyl.show()

			outliers = self.__checkOutliers()

			print('Here are the calculated points of high leverage')

			print(outliers)

			print('\n Now we will discover influential points. . . ')

			KSscore = self.__KSTest(model.resid)

			self.__assertKSTest(KSscore,model)

			self.__nonNormalityTest(model.resid)

			BPresults = self.__breuschPaganResults(model)

			transformations = self.__performTransformations(BPresults)

			if transformations == True:
				print('\nFrom this point, we would recommend that there be transformations made with some form of penalized regression.')

				ridge_model = __performRidgeRegresion(self)
				lasso_model = __performLasso(self)

				if self.verbose > 0:
					print('Plotting Ridge model.')
					self.__plotGLMNet(ridge_model,2)
					print('Plotting Lasso model.')
					self.__plotGLMNet(lasso_model,1)	

				print('\n The Ridge and Lasso models have been saved.\
					\n They may be accessed with the method .ridge() & .lasso() respectively.')

			else: 
				print('\nNo penalized regression model will be created.')
				print('\nFrom here our analysis is complete. \
					\n If you would like to access the best model it can be accessed with the method .get_best_model() ')

			print('\n \n \n \n Thank you ! \n \n \n')

		else:
			print('\n Exhaustive selection is 2^n in complexity. \
					\n that would require that we would have some sort of threading implemented to do Best Subset Selection \
					\n Hopefully we can do this in the future, but for now please use : selection=\'forward\' \n')

	# helper functions for display purposes
	def __displayDataFrame(self,df,n=5):
		#displays n results from a resulting dataframe
		print(df.head(n))

	def __displaySummary(self,model):
		print(model.summary())

	# private functions in order of operations
	def __preprocessing(self):
		global df
		global y
		global X
		global k
		print('\n Beginning preprocessing . . .\n')

		PATH = self.dataset
		df = pd.read_csv(PATH)
		for col in df.columns.tolist():
		    if df[col].isna().sum() > 0:
		        print('NaN values found in column : ', col)
		        #drop NA values
		        df = df.dropna()
		        #reset the index
		        df = df[np.isfinite().all(1)].reset_index(drop="True")
		    else:
		        print('No NaN values found in column : ', col)

		for col,dtype in dict(df.dtypes).items():
			if dtype == np.dtype('O'):
				print('Handling object dtype column: \"{}" in design matrix with One Hot Encoding'.format(col))
				# variable to represent the collection of one hot encoded columns
				df = pd.get_dummies(df, drop_first=True)
				# print(ohe.columns)
				# df = df.drop(col,axis=1)
				# df = pd.concat([df,ohe],axis=1,join='inner')
			else:
				pass

		y = df[str(self.response)]
		# if y.shape[1]>1:
		# 	y = y.loc[:,~y.columns.duplicated()]
		X = df.drop(str(self.response),axis=1)
		# have to purposely add a column of 1s
		# need to remove the intercept here because it's proving to be the best single feature lol
		# X = sm.add_constant(X)
		k = len(X.columns.tolist())

		print('\n Preprocessing has concluded . . . \n')

		return df, X, y, k

	def __performTransformations(self, BPresult):

		if BPresult > 0:

			print('\nGiven the following Breusch-Pagan result that we found we will be performing the following transformations')

			print('\nFirst we will do a transformation that takes the log of the dependent variable')

			global y
			
			y = np.log(y)

			print('\nTransformation complete.')

			return True

		else: 

			print('\n We will not be doing any transformations because of the results from the Breusch-Pagan test.')

			return False


	def __performRidgeRegresion(self):

		# fit the transformed value of y
		# alpha = 0 for ridge
		ridge = glmnet(X,y,alpha=0)

		return ridge

	def __performLasso(self):
		
		# fit the transformed value of y
		# alpha = 1 for lasso
		lasso = glmnet(X,y,alpha=1)

		return lasso

	def ridge(self):
		#returns the ridge_model
		return ridge_model

	def lasso(self):
		# returns the lasso_model
		return lasso_model

	def __plotGLMNet(self,model,L):

		_x = model.lambda_path_
		_y = model.coef_

		f,ax = plt.subplots(figsize=(12,6))
		sns.despine(f,left=True,bottom=True)
		sns.scatterplot(x=_x, y=_y, 
						palette="ch:r=-.2,d=.3_r",
						sizes=(1, 8), linewidth=0, ax=ax)

		plt.xlabel("Lambdas".format(n))
		if L == 2:
			plt.ylabel("L2 Norm")
		if L == 1:
			plt.ylabel("L1 Norm")

		plt.xlim([10,10000])
		plt.show()

	def __processSubset(self,combination):
		# Fit model on feature_set and calculate RSS

		model = sm.OLS(y,X[list(combination)])
		model = model.fit()
		# RSS = mean_squared_error(y,model.predict(X[list(combination)])) * len(y)
		RSS = ((model.predict(X[list(combination)])-y)**2).sum()
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

	def __computeModelComparison(self, df1):

		m = len(y)
		p = len(models_df)
		# computing sigmahatsquared
		hat_sigma_squared = (1/(m - p -1)) * min(df1['RSS'])
		# math for these model metrics
		df1['AIC'] = (1/(m*hat_sigma_squared)) * (df1['RSS'] + 2 * df1['num_features'] * hat_sigma_squared )
		df1['BIC'] = (1/(m*hat_sigma_squared)) * (df1['RSS'] +  np.log(m) * df1['num_features'] * hat_sigma_squared )
		df1['C_p'] = (1/m) * (df1['RSS'] * df1['num_features'] * hat_sigma_squared)
		df1['Adjusted_R_squared'] = 1 - ( (1 - df1['R_squared'])*(m-1)/(m-df1['num_features'] -1))

		return df1

	def __getBestModel(self,df):
			
		if self.eval_metric == 'RSS':
			tmp_df = df.set_index('RSS')
			# print(tmp_df.index)
			best_model = tmp_df.loc[df['RSS'].min()]

		if self.eval_metric == 'R_squared':
			tmp_df = df.set_index('R_squared')
			best_model = tmp_df.loc[df['R_squared'].max()]

		if self.eval_metric == 'AIC':
			tmp_df = df.set_index('AIC')
			best_model = tmp_df.loc[df['AIC'].min()]

		if self.eval_metric == 'BIC':
			tmp_df = df.set_index('BIC')
			best_model = tmp_df.loc[df['BIC'].min()]

		if self.eval_metric == 'C_p':
			tmp_df = df.set_index('C_p')
			best_model = tmp_df.loc[df['C_p'].min()]

		if self.eval_metric == 'Adjusted_R_squared':
			tmp_df = df.set_index('Adjusted_R_squared')
			best_model = tmp_df.loc[df['Adjusted_R_squared'].max()]

		return best_model

	def __checkOutliers(self):
		print('Checking for outliers . . . \n')
		# have to purposely add a column of 1s for the intercept here
		print('Points of high levereage are determined by the diagonals from the Hat Matrix')
		_X = sm.add_constant(X)
		_X = _X.to_numpy()
		hat = _X.dot(np.linalg.inv(_X.T.dot(_X)).dot(_X.T))
		# print(hat)
		hat_diag = np.diagonal(hat)
		# print(hat_diag)
		n = len(hat_diag)
		print("There are : {} diagonal entries".format(len(hat_diag)))
		# n diagonal entries
		trace = round(hat_diag.sum())
		print("Sum of the diagonal entries: {} ".format(trace))
		# diag = (_X.T * np.linalg.inv(_X.T.dot(_X)).dot(_X.T)).sum(0)
		# print(diag)
		print("Now we can plot to find points of high leverage . . . \n")

		f,ax = plt.subplots(figsize=(12,6))
		sns.despine(f,left=True,bottom=True)
		sns.scatterplot(x=range(0,n), y=hat_diag, 
						palette="ch:r=-.2,d=.3_r",
						sizes=(1, 8), linewidth=0, ax=ax)

		plt.axhline(y=2*trace/n,color='green')
		plt.axhline(y=3*trace/n,color='red')
		plt.xlabel("Range from 0,{}".format(n))
		plt.ylabel("Diagonals")
		plt.show()

		# these are our points of high leverage

		return np.where(hat_diag > 3*trace/n)

	def __nonNormalityTest(self, residuals):
		print(stats.shapiro(residuals))

		if stats.shapiro(residuals)[1] < 0.05:
			print("\n Shapiro p-value is {} which is less than 0.05.\
				\n This means that the non-normality test fails for the residuals of this model.".format(stats.shapiro(residuals)[1]))
		
		elif stats.shapiro(residuals)[1] < 0.1 and stats.shapiro(residuals)[1] > 0.05:
			print("\n Shapiro p-value is {} which is greater than 0.05 and less than 0.1.\
				\n This means that the non-normality test is questionable for the residuals of this model.".format(stats.shapiro(residuals)[1]))
		
		elif stats.shapiro(residuals)[1] > 0.1:
			print("\n Shapiro p-value is {} which is greater than 0.1.\
				\n This means that the non-normality test passes for the residuals of this model.".format(stats.shapiro(residuals)[1]))

	def __breuschPaganResults(self, model):
		names = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
		test = sm.het_breuschpagan(model.resid, model.model.exog)
		
		if test[1] < 0.05:
			print("\n Breusch-Pagan p-value is {} which is less than 0.05.\
				\n This means heteroscedasticity is present in this model".format(stats.shapiro(residuals)[1]))

			print('\n We suggest transformations.')
			
			return 1

		elif test[1] < 0.1 and test[1] > 0.05:
			print("\n Breusch-Pagan p-value is {} which is greater than 0.05 and less than 0.1.\
				\n This means that the heteroscedasticity of this model is questionable.".format(stats.shapiro(residuals)[1]))
			
			print('\nWe suggest that we do not do transformations')
			
			return 0

		elif test[1] > 0.1:
			print("\n Breusch-Pagan p-value is {} which is greater than 0.1.\
				\n This means that we do not have sufficient evidence to say that heteroscedasticity is present in the regression model.".format(stats.shapiro(residuals)[1]))
			
			print('\nWe suggest that we do not do transformations')
			
			return -1

	def __studentizedResiduals(self, model):
		# plot studentized residuals
		_X = sm.add_constant(X)
		_X = _X.to_numpy()
		hat = _X.dot(np.linalg.inv(_X.T.dot(_X)).dot(_X.T))
		# print(hat)
		hat_diag = np.diagonal(hat)
		# print(hat_diag)
		n = len(hat_diag)
		p = round(hat_diag.sum()) - 1
		alpha = 0.5


		f,ax = plt.subplots(figsize=(12,6))
		sns.despine(f,left=True,bottom=True)
		sns.scatterplot(x=range(0,n), y=model.outlier_test()['student_resid'], 
						palette="ch:r=-.2,d=.3_r",
						sizes=(1, 8), linewidth=0, ax=ax)

		qt1 = stats.t.ppf(1 - alpha / 2, n - p - 2)
		qt2 = stats.t.ppf(1 - alpha / 2, n - p - 2)

		plt.axhline(y=qt1,color='red')
		plt.axhline(y=qt2,color='red')
		
		plt.xlabel("Range from 0,{}".format(n))
		plt.ylabel("Studentized Residuals")

		plt.show()

	def __getCorrelatedValues(self,model):

		corrResidFittedValues = np.correlate(model.fittedvalues,model.resid)

		return corrResidFittedValues


	def __assertCorrelatedValues(self,corr):
		# :params:
		# corr : correlated value
		# return : print statement stating the significance of the correlation

		if corr < 0.00001:
			print('\n The correlation = {} between these two data is incredibly close to 0, therefore we can conclude that the linearity assumption is met.\n'.format(corr))

	def __plotFittedValuesAndResiduals(self,model):

		print('Plotting Fitted Values and Residuals of model : {}'.format(str(model)))

		f, ax = plt.subplots(figsize=(12, 6))
		sns.despine(f, left=True, bottom=True)
		sns.scatterplot(x=model.fittedvalues, y=model.resid,
						palette="ch:r=-.2,d=.3_r",
						sizes=(1, 8), linewidth=0, ax=ax)
		plt.axhline(y=0,color='red')
		plt.xlabel("fitted values", fontsize=20)
		plt.ylabel("residuals",fontsize=20)
		plt.show()

	def model_selection(self):

		return fw_df

	def __plot_RSS_and_R_squared(self, df):
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

	def get_best_model(self):

		print('This object is a fitted model with methods found in the statsmodels.OLS.fit() documentation.\n')

		return best_model['models']

	def __plot_histogram_residuals(self, residuals):

		num_bins = 10
		n, bins, patches = plt.hist(residuals, num_bins, color='red', alpha=0.5)
		plt.xlabel('Residuals')
		plt.ylabel('Distribution')
		plt.title('Histogram of Residuals')

		plt.show()

	def __KSTest(self, residuals):

		print('\n Now it is time for the Kolmogorov-Smirnov test. \n')

		print(stats.kstest(residuals, 'norm', N=len(residuals)))

		if stats.kstest(residuals, 'norm',N=len(residuals))[1] > 0.9:
			print('\nThere is strong evidence to suggest that we should not reject the null hypothesis, and that the residuals are normally distributed.\n \
				Therefore we can perform inference with this information with confidence.')

			return 1

		if stats.kstest(residuals, 'norm',N=len(residuals))[1] < 0.1:
		 	print('\nThere is strong evidence to suggest that we should reject the null hypothesis and that the residuals are NOT normally distributed.\n \
		 		Therefore, we should not perform inference (t-tests, F-tests, Confidence intervals, etc.)')

		 	return 0

	def __assertKSTest(self,KSscore,model):
		# determine whether or not we can do confidence intervals, t-tests, F-tests etc. 
		if KSscore > 0:
			# if we should not reject the null hypothesis then we do plot the studenized residuals
			self.__studentizedResiduals(model)

		else:
			print('\n We will not be plotting the studenized resiuals at this level because the residuals are not normally distributed according to the Kolmogorov-Smirnov test. \n')

	def __plot_model_selection(self,df):
		# ideally we want to write one function to plot things based on what we specify in the arguments

		if self.verbose > 0:

			fig = plt.figure(figsize = (12,12))

			ax = fig.add_subplot(2, 2, 1)
			ax.scatter(df['num_features'],df['AIC'], alpha = .2, color = 'darkblue' )
			# ax.plot(df['num_features'],df['min_AIC'],color = 'r', label = 'Best subset')
			ax.set_xlabel('# Features')
			ax.set_ylabel('AIC')
			ax.set_title('AIC - {} subset selection'.format(self.selection))
			ax.legend()

			ax = fig.add_subplot(2, 2, 2)
			ax.scatter(df['num_features'],df['BIC'], alpha = .2, color = 'darkblue' )
			# ax.plot(df['num_features'],df['min_BIC'],color = 'r', label = 'Best subset')
			ax.set_xlabel('# Features')
			ax.set_ylabel('BIC')
			ax.set_title('BIC - {} subset selection'.format(self.selection))
			ax.legend()

			ax = fig.add_subplot(2, 2, 3)
			ax.scatter(df['num_features'],df['Adjusted_R_squared'], alpha = .2, color = 'darkblue' )
			# ax.plot(df['num_features'],df['max_Adjusted_R_squared'],color = 'r', label = 'Best subset')
			ax.set_xlabel('# Features')
			ax.set_ylabel('Adjusted R squared')
			ax.set_title('Adjusted R squared - {} subset selection'.format(self.selection))
			ax.legend()

			ax = fig.add_subplot(2, 2, 4)
			ax.scatter(df['num_features'],df['C_p'], alpha = .2, color = 'darkblue' )
			# ax.plot(df['num_features'],df['C_p'],color = 'r', label = 'Best subset')
			ax.set_xlabel('# Features')
			ax.set_ylabel('C_p')
			ax.set_title('C_p - {} subset selection'.format(self.selection))
			ax.legend()

		else: 

			fig = plt.figure(figsize = (12,6))

			if self.eval_metric == "RSS":
				ax = fig.add_subplot(1, 1, 1)
				ax.scatter(df['num_features'],df['RSS'], alpha = .2, color = 'darkblue' )
				# ax.plot(df['num_features'],df['min_RSS'],color = 'r', label = 'Best subset')
				ax.set_xlabel('# Features')
				ax.set_ylabel('RSS')
				ax.set_title('RSS - {} subset selection'.format(self.selection))
				ax.legend()

			if self.eval_metric == "R_squared":
				ax = fig.add_subplot(1, 1, 1)
				ax.scatter(df['num_features'],df['R_squared'], alpha = .2, color = 'darkblue' )
				# ax.plot(df['num_features'],df['max_R_squared'],color = 'r', label = 'Best subset')
				ax.set_xlabel('# Features')
				ax.set_ylabel('R squared')
				ax.set_title('R_squared - {} subset selection'.format(self.selection))
				ax.legend()


			if self.eval_metric == "AIC":
				ax = fig.add_subplot(1, 1, 1)
				ax.scatter(df['num_features'],df['AIC'], alpha = .2, color = 'darkblue' )
				# ax.plot(df['num_features'],df['min_AIC'],color = 'r', label = 'Best subset')
				ax.set_xlabel('# Features')
				ax.set_ylabel('AIC')
				ax.set_title('AIC - {} subset selection'.format(self.selection))
				ax.legend()

			if self.eval_metric == "BIC":
				ax = fig.add_subplot(1, 1, 1)
				ax.scatter(df['num_features'],df['BIC'], alpha = .2, color = 'darkblue' )
				# ax.plot(df['num_features'],df['min_BIC'],color = 'r', label = 'Best subset')
				ax.set_xlabel('# Features')
				ax.set_ylabel('BIC')
				ax.set_title('BIC - {} subset selection'.format(self.selection))
				ax.legend()

			if self.eval_metric == "Adjusted_R_squared":
				ax = fig.add_subplot(1, 1, 1)
				ax.scatter(df['num_features'],df['Adjusted_R_squared'], alpha = .2, color = 'darkblue' )
				# ax.plot(df['num_features'],df['max_Adjusted_R_squared'],color = 'r', label = 'Best subset')
				ax.set_xlabel('# Features')
				ax.set_ylabel('Adjusted R squared')
				ax.set_title('Adjusted R squared - {} subset selection'.format(self.selection))
				ax.legend()
			if self.eval_metric == "C_p":
				ax = fig.add_subplot(1, 1, 1)
				ax.scatter(df['num_features'],df['C_p'], alpha = .2, color = 'darkblue' )
				# ax.plot(df['num_features'],df['C_p'],color = 'r', label = 'Best subset')
				ax.set_xlabel('# Features')
				ax.set_ylabel('C_p')
				ax.set_title('C_p - {} subset selection'.format(self.selection))
				ax.legend()
		
		plt.show()


if __name__ == "__main__":
	x = LR(dataset="loan_data.csv",response="log.annual.inc",selection='forward', prediction=True, eval_metric="BIC", verbose=0)
	x