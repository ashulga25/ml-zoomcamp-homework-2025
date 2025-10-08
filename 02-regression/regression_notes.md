## Part 2. Linear Regression
----------------------------
## 2.1 Car Price Prediction Project

data source www.kaggle.com/CooperUnion/cardataset

Project Plan:
- Prepare data and perform EDA (Exploratory Data analysis)
- Use linear regression for predicting price
- Understand the internals of linear regression
- Evaluating the model with RMSE (Root Mean Squarred Error) method and use RMSA on validation data
- Feature engineering (adding new features based on data in dataset)
- Regularization
- Using the model


code here

https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/chapter-02-car-price
(02-carprice.ipynb and data.csv)

---
import pandas as pd
import numpy as np

data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/tree/master/chapter-02-car-price/data.csv'

!wget $data

df = pd.read_csv('data.csv')
df.head()

df.['Transmission type']


------------------------------------------------------
# 2.2 Data Preparation - normalization of table columns:

df.columns = df.columns.str.lower().str.replace(' ', '_')
df.head()


normalization of table contents (all to lowcase):
df.dtypes		# shows info about type of column. We are interested in `objects` (which are strings)

df.dtypes	 == 'object'	(not int64, not float)
df.dtypes[df.dtypes	== 'object']				# we get the list of column names
df.dtypes[df.dtypes	== 'object'].index			# this would be an index of strings

strings = list(df.types[df.dtypes	== 'object'].index)		# we will make it a list of indexes and store it in a variable

for col in strings:											# now we want to get through the list and make changes
  df[col] = df[col].str.lower().str.replace(' ', '_')		# and put all strings to lower and replace ` ` with `_`
															# this way we would normalize all data to the same lower case and remove spaces


--------------------------------
# 2.3 - Exploratory Data Analysis
								
								
df.head()										# now all data normalized
df.dtypes										# checking the columns to see what values are there


df										# just tprint the data frame

for col in df.columns:					
	print(col)
	print(df[col].head())					# print some statistics	and first 5 rows with values
	print(df[col].unique())					# print some statistics	and unique values
	print(df[col].unique()[:5])				# print some statistics	and FIRST 5 unique values
	print(df[col].nunique())				# print the number of unique values
	print()



Now visualize the MSRP column:

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

sns.histplot(df.msrp, bins=50)			# bins is how wide are the bars.


										# 1e6 = 10 in 6th = 1.000.000
										
# `Long tail` distribution - when almost all data is in left part of the graph, leaving a tail with a few or single records far-right (like a tail)
# The `long tail` distribution is typical for prices, because most of the things are within some range (i.e. public things).
# And there are `special`, unique, non-public items with much higher price
# `long tail` distribution IS NOT  good for ML, it owuld actually screw the learning

sns.histplot(df.msrp[df.msrp < 100000], bins=50)		# would create a plot with better distribution


------------------
Theory:
In order to get rid of `long tail` we apply `logarithmic distribution` for the price and we would get 
more compact values

np.log([1, 10, 1000, 100000])		# would generate logarithmic values and they would be not that large in difference

# the issue with logarythm scale that it would generate error, so it SHOULD NOT start from 0,
# so in order to protect from the error, it is pretty common to add 1 to each value, like

np.log([0 + 1, 1+1, 10+1, 1000+1, 100000+1])

in order not to add 1 manually everytime, we would use function log1p which add 1 automatically to each element in array

np.log1p([0, 1, 10, 1000, 100000])

---------------------

So we would use:

price_logs = np.log1p(df.msrp)
price_logs								# all values are smaller now

sns.histplot(price_logs, bins=50)		# now we have a proper plot

Now plot resembles the normal distribution with bell-peak form, this is ideal for ML-models and ML-models 
work great with them. So normalize the data in a way to get the bell-like normal distribution over 
the criteria you want models to learn upon


Checking the missing values:

df.isnull()						# would show if there is a NULL value.	
df.isnull().sum()				# would summarize per each column

in some columns there would be thousand missing records
we would take it into account when training the model


---------------------------------------------
# 2.4 Setting up the validation framework

Splitting the dataset into 3 parts: TRAINing (60%) - VALidate (20%) - TESTing (20%)
									Xtr Ytr				Xv Yv			Xt Yt

n = len(df)				# size of our dataframe

len(df) * 0.2 = 2382	# number of rows to leave in VAL and TEST data-subsets


n_val = int(n * 0.2)			# convert the number to integer for the size of validation data subset
n_test = int(n * 0.2)			# convert the number to integer for the size of validation data subset
n_train = n - n_val - n_tes

---
df.iloc[[0, 1, 2]]				# would create 3 data sets

d.iloc[:10]						# returns first 10 records, for 0 till 9th
----
so we run:
df_val = df.iloc[:n_val]					# records for validation
df_test = df.iloc[n_val:n_val+test]			# records for testing
df_train = df.iloc[n_val+test:]				# records for training


OR ANOTHER ORDER:
df_train = df.iloc[:n_train]					# records for training
df_val = df.iloc[n_train:n_train+n_val]			# records for validation
df_test = df.iloc[n_train+n_val:]				# records for testing

however, the initial dataset is arranged by car maker, so by dividing the dataset by index would put only specific
brands/records into each subset. So we need to SHUFFLE!
in order for the shuffle to be reproducsible - we need to use a custom seed:

idx = np.arrange(n)
np.random.seed(2)
np.random.shuffle(idx)										# we shuffle
df_train = df.iloc[idx[:n_train]]							# records for training
df_val = df.iloc[idx[n_train:n_train+n_val]]				# records for validation
df_test = df.iloc[idx[n_train+n_val:]]						# records for testing


df_train
df_val
df_test

len(df_train), len(df_val), len)df_test)
(we should have 7150, 2382, 2382)


Next, we do index reset, as it changed after the shuffle:

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)



Next we need to do log1p transformation to get y variable for the plot:

y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

len(y_train), len(y_val), len(y_test)
>>	7150, 2382, 2382

Once we got `y`, we need to remove it from dataset in order not to use it accidentally (becasue then  model would 
learn over it occasionally and it would be like a perfect model).

del df_train['msrp']
del df_test['msrp']
del df_val['msrp']


----------
# 2.5 - Linear Regression

---
Theory:
g(X) ~ Y		# Y is price in current task

g - model (linear regression)
X - feature matrix
Y - target

g(Xi) ~ Yi			# for a single record, one car
					# Where Xi is a vector Xi = [Xi1, Xi2, ..., Xin]


so our function is taking all these items and produces Y:
g(Xi1, Xi2, ..., Xin) ~ Yi


so will check the training dataset now: df_train.iloc[10]

we'll take 3 params:
- engine_hp  	453
- city_mpg		11
- popularity	86

so Xi = [453, 11, 86]

so our function should do smth like:

def g(xi):
		do something
	return 10000	# price Y
	
	
Linear regression formula:		g(Xi) = W0 + W1	* Xi1 + W2 * Xi2 + W3 * Xi3

so we need to find  such values of W0, W1, W2, W3	that would fullfill the equation:
g(Xi) = W0 + SUM ( Wj * Xij) for j from 1 to 3:		or for j from 1 to n in general case:

W0 = 0
W = [1, 1, 1]

now we implement the formula:

def linear_regression(xi):
	n = len(xi)
	
	pred = w0
	
	for j in range(n):
		pred = pred + w[j] * xi[j]
		
	return pred						# price
	
	
If we run it - we would get not the result we want:
linear_regression(xi)
>> 550		

NOTA BENE:
this is still logarithmic value which we need to revert back!


if we change weights to:
W0=7.17
W=[0.01, 0.04, 0.002]

then the output of our function would be: 12.312 - this is log(y-1).
And we did log(y+1) 	# log1p
so we need to undo the logarithm, meaning we need to do the exponent: np.exp(12.312) - 1
So it would generate the predicted price: 222347.222
we can make use of shortcut function:	np.expm1(12.312) = 222347.222

if we run np.log1p(22347.222) 		# we would get 12.213


----------------------------------
# 2.6 Linear regression vector form

general formula would be:
g(Xi) = W0 + SUM ( Wj * Xij) for j from 1 to n	

the SUM ( Wj * Xij) for j from 1 to n	is a .dot -product (vector-vector multiplication):

g(Xi) = W0 + SUM ( Wj * Xij) for j from 1 to n	= W0 + Xi(Transposed) * W
so Xi vector transposed (into vertical) multiplied to W vector (also vertical):

a formula would be like:

def dot(Xi, W):
	n = len(Xi)
	
	res = 0.0
	
	for j in range(n):
		res = res + Xi[j] * W[j]
	
	return res
	
def linear_regression(Xi):
	return W0 + dot(Xi, W)
	

We can make the formula even simpler: - we add a Xi0 element, which always equal 1, so W0 * Xi0 == W0 * 1 = W0
- would also be a part of vector-vector multiplication:
W = [W0, W1, W2, ... Wn]				# Wn+1 element vector

Xi = [Xi0, Xi1, Xi2, ..., Xin]]			#  also Xn+1 element vector, where Xi0 = 1

and multiplication looks like:

WTr * Xi == Xi Tr * W == W0 + W1 * X1 + W2 * X2 + ... Wn * Xn

WTr == W transposed


so finally our function would look like:

W_new = [W0] + W
W_new


def linear_regression(Xi):
	xi = [1] + xi
	return dot(Xi, W_new)


linear_regression(Xi) 
>> 12.312


In more general case:
Xm * (n+1)	* W
X						W
[1 X11 X12 ... X1n]	 	[W0]	= X1Tr * W
[1 X21 X22 ... X2n]		[W1]	= X2Tr * W
[1 X31 X32 ... X3n]	*	[W2]	= X3Tr * W
....					...
[1 Xm1 Xm2 ... Xmn]		[Wn]	= XmTr * W

so we multiply each row X and vector W, and in fact it is matrix-matrix multiplication

So it is X * W	matrix-matrix multiplication

W0 - is `bias term` - how much a car should cost if we don't know 
anything about that car (basic car cost). The base line.

taking another exaple:

x1  = [1, 148, 24, 1385 ]
x2  = [1, 132, 25, 2031 ]
...
x10 = [1, 453, 11, 86]


X = [x1, x2, x10]		# list of lists
X = np.array(X)
X

w0 = 7.17
w = [0.01, 0.04, 0.002]
w_new = [w0] + w

and now we just do multiplication of the matrix and vector:
def linear_regression(X):
	return X.dot(w_new)

linear_regression(X)
>> array([12.38, 13.552, 12.312)	# array of price predictions


-------------
# 2.7 Trainining a linear regression model

g(x) = X * W ~ Y	OR	X * W = Y	But it is ideally, so as close as possible

So we need to find a W, so: X-1 (inverse) * X * W = X-1 (inverse) * Y ==> I * W = X-1 * Y  ==> I * W = X-1(inverse) * Y

But we can't find the X-1(inverse) - it does not exist.

So we can do different:
Xtr * X * W = Xtr * Y

Xtr == X transpose
Xtr * X = so called Gram Matrix  (n+1) X (n+1).
For this case X-1(inverse) does exist. 

So we could get:

(Xtr * X)-1 (inverse) * Xtr * X * W = (Xtr * X)-1 (inverse) * Xtr * Y

(Xtr * X)-1 (inverse) * Xtr * X = I (identity matrix) and evaporates, 

so we are left with: I * W = (Xtr * X)-1 (inverse) * Xtr * Y  ==> 		W = (Xtr * X)-1 (inverse) * Xtr * Y
becasue I * W = W.
So W is the closest solution to this system of equation.


implementation:
# 1 - append 1s into array:

ones = np.ones(X.shape[0])	# look at number of rows and create the arrays of ones
ones

# 2) - matrix of params:

X  = [
		[148, 24, 1385 ],
        [132, 25, 2031 ],
        [453, 11, 86   ],
		[158, 24, 185  ],
        [172, 25, 201  ],
        [413, 11, 86   ],
		[38,  54, 185  ],
        [142, 25, 431  ],
        [453, 31, 86   ],

X = np.array(X)
X

# 3) - adding row of ones:


np.column_stack([ones, X])					# takes vectors and stacks them together

np.column_stack([ones, ones, X])			# takes vectors and stacks them together	(can stack multiple vectors)

list(np.column_stack([ones, X]))			# verification


so we run to add row of ones:

X = np.column_stack([ones, X])


# 4) transposing matrix:

XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)
>> array of values


# verification:

XTX.dot(XTX_inv)
>> array([])		# - this is not exactly Identity matrix, but very close to it, with very small values out of diagonal.
					
					
				
# so we could rount it up and it would be Idenity matrix

XTX.dot(XTX_inv).round(1)
>> array( [[1.,  0., 0.],
		   [-0., 1., 0.],
           [0.,  0., 1.]]  )

y = [10000, 20000, 15000, 20050, 10000, 20000, 15000, 25000, 12000 ]

w_full = XTX_inv.dot(X.T).dot(y)
w_full		# print the result

w0 = w_full[0]		- this is the bias
w = w_full[1:]		- these are all other weights

w0, w

>> (25844.75, array([ -16.089, -199.47, -1.2280])

- so actually it means we havea base price of the car 25844 and then for each year of car - we extract some amount out of its' price


Let's put it all in a function:
		   
def train_linear_regression(X, y):
	ones = np.ones(X.shape[0])
	X = np.column_stack([ones, X])
	XTX = X.T.dot(X)
	XTX_inv = np.linalg.inv(XTX)
	w_full = XTX_inv.dot(X.T).dot(y)
	
	return w_full[0], f_full[1:]		# we return result as tupple, the bias term and the weights in array
	


-------------
# 2.8 Car price baseline model


df_train.dtypes			# listing the details about dataset columns 

base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']	# - taking the relevant columns


df_train[base].isnull().sum					# checking for the NULL values and their number

df_train[base].fillna(0).isnull().sum()		# filling the NULL with 0 and checking again

X_train = df_train[base].fillna(0).values				# we extract values

w0, w = train_linear_regression(X_train, y_train)		# we trained the model, and got some weights
>> 7.9272, array ([])


y_pred = w0 + X_train.dot(w)		# this would be a predicted price


# building plot:

sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
sns.histplot(y_pred, color='blue', alpha=0.5, bins=50)

we got a blue graph with original mode, and red graph with predicted prices.
However the model is shafted left (meaning prices of cars would be cheaper). Which mean the model is not ideal!

However using only graph to check the model is not correct, and we will check it via RMSE-method (root-mean squarred error method)

-------------------
# 2.9 RMSE-method


RMSE = Square root from [ 1/m * SUM (from 1 to M) of (g(Xi) - Yi)) in power 2 ]
where m is number of observations.

So actually RMSE is: square root from [the sum of all ((differences between the predicted price minus actual price) 
														multiplied by itself) 
										and then averaged for all the differences (divided by the number of entries)]
										
y_pred = g(Xi) = [10, 9, 11, ...., 10]

y_train	= [9, 9, 10.5, ....., 11,5]


difference would be:
y_error_diff = [1, 0, 0.5, ...., -1.5]

next we square it:
y_error_diff_squarred = [1, 0, 0.25, ...., 2.25]

mean_squared_error = (1 + 0 + 0,25 + 2,25 ) / 4 = 0.875

Next we take root out of it	0.875 = 0.93

def rmse (y, y_pred):
	se = (y - y_pred) ** 2
	mse = se.mean()				# we can directly invoke mean method and get mean
	return np.sqrt(mse)			# return square root of mean
	

rmse(y_train, y_pred)
			

We apply RMSE to the same TRAIN dataset!


			
--------------------------
# 2.10 Validating the model 

- we'll apply RMSE to the VAL data-subset.



Code to train model:

base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']	# - taking the relevant columns

X_train = df_train[base].fillna(0).values				# we extract values

	
w0, w = train_linear_regression(X_train, y_train)		# we trained the model, and got some weights
>> 7.9272, array ([])


y_pred = w0 + X_train.dot(w)		# this would be a predicted price


function to prepare dataset (by removing empty values):

def prepare_X(df):
	df_num = df[base]					# data-frame with numbers	
	df_num = df_num.fillna(0)			# filling with 0 the NULL records 
	X = df_num.values					# filling the result dataset with values
	return X
	
The steps would be:	

# Training part	

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)


# Validation part - we prepare the validation dataset and check the error

X_val = prepare_X(df_val)

y_pred = w0 + X_val.dot(w)				# run predictions

rmse(y_val, y_pred)
						# calculate rmse of the predicted dataset
>> 0.76165

the RMSE of train dataset was 
0.75549226 - so it is pretty similar

------------------------------
# 2.11 Feature Engineering


We have a field year, which specifies the age of car production. 
Based on it we would calculate the age of the car. 

df_train.year.max()				# the largest number in the field `year`. We would accept it as current year.

2017 - df_train.year.max()		# some cars would be 0 years old, some - 10+ years old


So we would add a new field into dataset:

def prepare_X(df):
	df = df.copy()				# we make a copy, so original df won't be changed.
    
	df['age'] = 2017 - df.year
	features = base + ['age']
	
	df_num = df[features]					# data-frame with numbers	
	df_num = df_num.fillna(0)			# filling with 0 the NULL records 
	X = df_num.values					# filling the result dataset with values
	
	return X


run it:

X_train = prepare_X(df_train)

df_train.columns				# no new column		
X_train							# we added a new column `age`, we can see it in 6th column.

now we run the same code, and our model would be better - the MSRE would be smaller:

X_train = prepare+X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)
>> 0.5172				# smaller than 0.7616!!!


# Building a new plot:

sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
sns.histplot(y_val, color='blue', alpha=0.5, bins=50)		# we predict on validation dataset!

now the graph is better and closed to original one (still not ideal!)


----------------------------------
# 2.12 - Categorical Variables (new feature)

Categorical Variables - strings

Exclusion - door is not a numerical variable, because number of doors is about car type, not the benefit of more doors


A typical way of encoding such categoriacal values - we represent it with binary columns:

number_of_doors:
[2, 3, 4, 2]
transforms into:
2_doors = [1, 0, 0, 1]
3_doors = [0, 1, 0, 0]
4_doors = [0, 0, 1, 0]

there is a feature `astype` for it in numpy:

for v in [2, 3, 4]:
	df_train['num_doors_%s' % v ] = (df_train.number_of_doors == v).astype('int')


we would add it to our function:

def prepare_X(df):
	df = df.copy()						# we would create a new list, in order no to modify the original one
	features = base.copy()				# same here
	
	df['age'] = 2017 - df.year	
	features.append('age')			 	# otherwise, the ['age'] column would be appended every time 
	
	for v in [2, 3, 4]:
		df['num_doors_%s' % v ] = (df.number_of_doors == v).astype('int')
		features.append('num_doors_%s' % v)
		
	df_num = df[features]
	df_num = df_num.fillna(0)
	X = df_num.values
	
	return X
	
# testing how it works:

prepare_X(df_train)				# now we have a new set of columns
	

# Training model:

X_train = prepare_X(df_train)


now we run the same code, and our model would be better - the MSRE would be smaller:

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)
>> 0.515799					# previos was 0.5172055461, so improvement is very limited


Doing same categorization with the `make` field:

df.make.nunique()
>> 48						# quite a lot to add into dataset.
							# so instead we would add only the most popular ones
							
car_makers = list(df.make.value_counts().head().index)		# this way we would get only the popular ones
>>  chevrolet   1123
	ford		881
	volkswagen	809
	toyota		746
	dodge		626
	nissan		558
	gmc			515
	honda		449
	...

we would take the top 5 and include them same way as doors:

def prepare_X(df):
	df = df.copy()						# we would create a new list, in order no to modify the original one
	features = base.copy()				# same here
	
	df['age'] = 2017 - df.year	
	features.append('age')			 	# otherwise, the ['age'] column would be appended every time 
	
	for v in [2, 3, 4]:
		df['num_doors_%s' % v ] = (df.number_of_doors == v).astype('int')
		features.append('num_doors_%s' % v)
		
	for v in car_makers:
		df['maker_%s' % v ] = (df.maker == v).astype('int')
		features.append('maker_%s' % v)
		
	df_num = df[features]
	df_num = df_num.fillna(0)
	X = df_num.values
	
	return X

	
# re-running the training and our model would be better - the MSRE would be smaller, but only for a fraction:

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)
>> 0.507603884955					# previous was 0.5157995641, so improvement is very limited, less than 1%



# next we can categorize 'engine_fuel_type', 'transmission_type', 'driven_wheels', 
# 'market_category', 'vehicle_size', 'vehicle_style'

df_train.dtypes

# we'll put all of them into a list:

categorical_variables = ['maker', 'engine_fuel_type', 'transmission_type', 'driven_wheels', 'market_category', 'vehicle_size', 'vehicle_style' ]


# and then create a dictionary to contain top 5 most popular items:
categories = {}

for c in categorical_variables:
	categories[c] = list(df[c].value_counts.head().index)


	
now we would update our function:

def prepare_X(df):
	df = df.copy()						# we would create a new list, in order no to modify the original one
	features = base.copy()				# same here
	
	df['age'] = 2017 - df.year	
	features.append('age')			 	# otherwise, the ['age'] column would be appended every time 
	
	for v in [2, 3, 4]:
		df['num_doors_%s' % v ] = (df.number_of_doors == v).astype('int')
		features.append('num_doors_%s' % v)

    for c, values in categories.items():
		for v in values:
			df['%s_%s' % (c, v)] = (df[c] == v).astype('int')
			features.append('%s_%s' % (c, v) )
		
#	for v in car_makers:
#		df['maker_%s' % v ] = (df.maker == v).astype('int')
#		features.append('maker_%s' % v)
		
	df_num = df[features]
	df_num = df_num.fillna(0)
	X = df_num.values
	
	return X

# Doing same thing - train model one more time:
# re-running the training and our model would be better - the MSRE would be smaller, but only for a fraction:

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)
>> 41.4541 					# previous was 0.5157995641, so smth went wrong!

w0
>> -1.37 (it a scientific notation) - it is a huge negative number (-13706224874057856)
							so we made a mistake somewhere in the model.
							
							
-------------------------------
# 2.13 Regularization

							
def prepare_X(df):
	df = df.copy()						# we would create a new list, in order no to modify the original one
	features = base.copy()				# same here
	
	df['age'] = 2017 - df.year	
	features.append('age')			 	# otherwise, the ['age'] column would be appended every time 
	
	for v in [2, 3, 4]:
		df['num_doors_%d' % v ] = (df.number_of_doors == v).astype('int')
		features.append('num_doors_%d' % v)

    for name, values in categories.items():
		for value in values:
			df['%s_%s' % (name, value)] = (df[name] == value).astype('int')
			features.append('%s_%s' % (name, value) )
		
#	for v in car_makers:
#		df['maker_%s' % v ] = (df.maker == v).astype('int')
#		features.append('maker_%s' % v)
		
	df_num = df[features]
	df_num = df_num.fillna(0)
	X = df_num.values
	
	return X
	

Train model:

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)
>> 41.4541 					# previous was 0.5157995641, so smth went wrong!

w0
>> -1.37 (it a scientific notation) - it is a huge negative number (-13706224874057856)
	

The issue	why we got an incorrect value is W = (Xtr * X)-1 inverse * Xtr * Y

gram matrix `(Xtr * X)-1 inverse`  sometimes this matrix may not exist!

sometimes in the matrix some columns may be duplicates and it leads to situation when `(Xtr * X)-1 inverse` won't exist.
columns 2 and 3 are same!

X = [ [4, 4, 4],
	  [3, 5, 5],
	  [5, 1, 1],
	  [5, 4, 4],
	  [7, 5, 5],
	  [4, 5, 5],
	]
	
X = np.array(X)
X

XTX = X.T.dot(X)
- would show us that transposed matrix would also have the duplicate rows.
In such situation the inverse simply does not exist!
In Linear Algebra it is said `we can express one column with another column` - 
which means one column is a duplicate of another.

np.linalg.inv(XTX) 		# numpy would generate error that matrix is singular and inverse can not be found.



It was not our case exactly, but probably in our data there was noise, when one column was very 
similar to another!

TO solve this issue we can add a small number to the diagonal of the transposed matrix:
XTX = X.T.dot(X)
XTX
>> ([[ 140+0.0001,      111,         111.0000004 ],
     [ 111,        108+0.0001,       108.0000005 ], 
	 [ 111.000004, 108.0000005,      108.000001+0.0001 ]]
	 
we can do it automaticall via adding numpy function np.eye(), which is diagonal matrix of size 3: [ [1, 0, 0]
																									[0, 1, 1]
																									[0, 1, 1] ]

XTX = XTX + 0.01 * np.eye(3)  == XTX + [ [0.01, 0, 0],
										 [0, 0.01, 0],
										 [0, 0, 0.01]]

actually 0.01 becomes a parameter. The larger we make it - the smaller numbers are generated in the matrix np.linalg.inv(XTX)

Using this approach we would re-implement the linear_regression function:
---
def train_linear_regression_reg (X, y, r=0.001):
	ones = np.ones(X.shape[0])
	X = np.column_stack([ones, X])
	
	XTX = X.T.dot(X)
	XTX = XTX + r * np.eye(XTX.shape[0])
	
	XTX_inv = np.linalg.inv(XTX)
	w_full = XTX_inv.dot(X.T).dot(y)
	
	return w_full[0], w_full[1:]
---
	
Now we train the model one mode time:	
	

X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X_train, y_train, r=0.01)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)
>> 0.46082082 					# previous was 0.5157995641, so we got improvement!


So by adding a number to diagonal we were able to control weights
to regularize parameter.

if we set too high:       r = 1000 		- there would be a huge rmse error, about  0.9664
if we make it very small: r = 0.0 		-  we got even a larger rmse error, like 266.599


------------------------
# 2.14 - Tuning the model

we would try a bunch of different values for r from 0 to 10 and see what happens if there would be improvement in rmse:

for r in [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]:
		X_train = prepare_X(df_train)
		w0, w = train_linear_regression_reg(X_train, y_train, r=r)

		X_val = prepare_X(df_val)
		y_pred = w0 + X_val.dot(w)
		
		score = rmse(y_val, y_pred)
		
		print(r, w0, score)
		
>>	0.0		3066
    1e-05	6.63		266.599
	0.0001	7.129		0.46081		# good one!
	0.001	7.130		0.46081		# another good one!
	0.01	7.05		0.46087		# also good one!
	0.1		7.00		0.46087
	1		6.25		0.46158
	10		4.72		0.4726

so with this score it becomes obvious with r = 0.001 we get close to optimal results


r = 0.001
X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X_train, y_train, r=r)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
		
score = rmse(y_val, y_pred)
score		

>> 0.46081	
		
-----------------------
# 2.15 Using the model and testing it over the TEST data-subset

So far we trained model at TRAIN data-subset, applied it to the validation data-subset and got some RMSE.
Now we would trait it over TRAIN+VAL = FULL_TRAIN and test over TEST data-subset and check RMSE there.


df_full_train = pd.concat([df_train, df_val])			# pd.concat would combine 2 datasets together

df_full_train = df_full_train.reset_index(drop=True)	# we rebuild index

df_full_train

X_full_train = prepare_X(df_full_train)
X_full_train


same with prices column - we combine it:

y_full_train = np.concatenate([y_train, y_val])

w0, W = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)
w0
>> 7.1
score = rmse(y_test, y_pred)
score
>> 0.460000

Now preparing the testing data-subset:
X_test = prepare_X(df_test)
y_pred = w0 +X_test.dot(W)
score = rmse(y_test, y_pred)
score
>> 0.460007		# it is almost same as previous



Finally, we would like to predict the price of a car:

we would take any car from our dataset and pretend it is a new car:
# we would create a python dictionary with all the values about the car:		

a_car = df_test.iloc[20].to_dict()		# toyota sienna 2015
		
# Also we need to prepare it:

df_small = pd.DataFrame([a_car])
df_small

X_small = prepare_X(df_small)

y_pred = w0 + X_small.dow(W)
y_pred = y_pred[0]
y_pred
>> 10.63

this is still logarythm of the price, so we take exponent:
np.expm1(y_pred)
>> 41459 USD (predicted price)

the actual cost of the car:
y_test[20]
>> 10.4631

np.expm1(y_test)
>> 35000 USD, so our prediction for 61459 USD higher!!!

-----------------------------
# 2.16 Summary


1 - we downloaded a dataset
2 - cleaned the dataset and prepared it tp be uniform
3 - did exporatory data analysis and identified we have a long tail so we have 
	applied logarythmic distribution to get a bell-curve
4 - then we did a split of data set into TRAIN, VAL, TEST sub-dataset
5 - checked how linear regression works for a single example 1 row and then expanded it into a matrix
	The result of linear regression is the weights-vector.
6 - we checked it at the graph versus the original distribution and saw the model is not doing well (distorted)
7 - we learned what is RMSE method and applied it to calculate the error.
8 - build the validation function
9 - added extra features/fields to the dataset (added column `age`)
10 - added categorical variables with a bunch of binary columns
11 - step 10 resulted into huge error, so we had to add the regularization by adding a small number to diagonal matrix
12 - fine-tuned the model with different values of regularization
13 - used the model by combining the TRAIN and VAL dataset into one FULL_TRAIN dataset 
14 - applied model to one of a records from dataset to check the predicted price... prediction was 20% higher than actual price.


Next steps:
- we included only top 5 features. What will happen if we include 10, 15, 20 ?

Other projects to learn the topic better:
- predict the price of a house - e.g. boston dataset
- https://archive.ics.uci.edu/ml/datasets.php?task=reg
- https://archive.ics.uci.edu/ml/datasets/Student+Performance
