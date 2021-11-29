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

	def __init__(self, dataset, response, eval_metric):
		# string that represents the column in the dataset that you would like to specifty as the response.
		self.response = response
		# default eval metric = , otherwise unless specified as something else
		self.eval_metric = eval_metric
		# string that represents a path to the dataset
		self.dataset = dataset


	def preprocessing(self):
		PATH = self.dataset

	def show_output():
		pass



		