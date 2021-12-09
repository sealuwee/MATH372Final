# MATH372Final

Math 372 final project, hope I get a good grade thanks in advance professor

## Introduction

The purpose of this project was to create a function that produces the following information:

- Data pre-processing (removal of missing observations)

- Model selection that incorporates cross-validation, using various metrics
(adjusted R^2, Mallows’ C_p, AIC, BIC)

- Where appropriate, shrinkage methods (Lasso and ridge regression)

- Checking for and accounting for outliers, influential points, and points of
high leverage, by providing visual plots and by applying hypothesis tests

- Visual and numerical display of diagnostics for normality, homoscedasticity and linearity

- Determination of appropriate transformations on y and/or inclusion of
higher-order or interaction terms

The way I approached this project was to do all of these steps in Python with the handiness of Object Oriented Programming. In the functionality section, you can find explanations for the code and reasoning behind producing the information during object instantiation. 


## Requirements:

** MUST HAVE Python version >= 3.7 **

## Installation:

```
git clone https://github.com/sealuwee/MATH372Final.git
cd MATH372Final/
pip install -r requirements.txt 
```

## Sample functionality in the CLI

Step 1)

```
ls
```

Step 2)

Be sure lr.py is in your working directory.

Be sure that you have followed the Installation of requirements.txt to the environment of your choice.

Step 3) 

In your CLI

```
python3 lr.py
```

From here you will be presented with a vast amount of information that occurs on object instantiation.

This information is presented as summaries and statistical explanations of the dataset ```loans_data.csv``` that describes loan data from Lending Club. 

## Explanations of function parameters

```
dataset: Path to dataset file in CSV format
response: Specified column in said dataset that the user would define as the response vector
selection: Subset selection method, currently only forward is implemented because exhaustive subset selection is a 2^n operation, and for the purposes of this project I believed forward selection would suffice.
prediction: Boolean to determine if prediction is true, if not do inference
eval_metric: Specified evaluation metric for model selection, choices are : ['RSS','R_squared','AIC','BIC','C_p','Adjusted_R_squared']
verbose: 1 for verbose 0 for non-verbose. Verbose is a flag for how much information there would be to be displayed.
```

## Further explanation of methods

Most methods in the class are private methods because they are necessary for the information presented on object instantiation. 

Accessible methods include :

```
.model_selection() : returns a dataframe of all models and their corresponding evaluation metrics that were produced from feature selection
.get_best_model() : returns the best model selected from the provided parameter eval_metric
```

The following methods are only accessible if transformations were applied and necessary for further analysis. 

```
.ridge() : returns the fitted Ridge regression model
.lasso() : returns the fitted Lasso model
```

## Reproducability for new datasets

One would be able to access the methods listed above if you used the LR() object in practice.

```python
from lr import LR

linreg = LR(dataset='path/to/dataset',
			response='response_variable',
			selection='forward',
			prediction=True,
			eval_metric='Adjusted_R_squared',
			verbose=1)

# On object instantiation, there would be a summary output and information about statistical tests that compute the fitted response and the residuals of the best model along with plots and diagnostics to describe outliers, points of high leverage etc., combined with suggestions as to what transformations would have to be done in order to deal with heteroskedacity.

best_model = linreg.get_best_model()
ridge = linreg.ridge()

```
## Sources

- http://www.science.smith.edu/~jcrouser/SDS293/labs/lab8-py.html
- https://xavierbourretsicotte.github.io/subset_selection.html