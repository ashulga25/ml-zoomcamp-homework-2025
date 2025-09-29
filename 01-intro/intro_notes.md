# General Theory

Feature matrix - X (2 dimensional array, a table)
Output	Y	(1 dimensional array, list)

X -> training model -> get Y

so we apply function g (our model) at matix X to get Y
g(X)  ~= Y	

model `g` applied to matrix `X` to get target `Y`

this is formal definition of "supervised ML"

based on the model - for spam filter the prediction is spam probality
                   - for car price modeling - the price 



# There are different types of supervised ML:

1) regression - returns a number for car price from 0 to infinity

house price - also regression.
Any number where out is a number (from "minus infinity to plus infinity"), the problem is
regression problem, so - the model is called regression model.


2) if we need to identify the object on a picture - the model is solving classification problem,
so it is a classification model.

we don't output a number, but a category.
same with spam - it is also a classification, we predict if smth is spam or not.
This is actually `binary classification`.


3) there are different subclasses of classification - like milticlass:
i.e. we need to classify images, if there are cars, dogs, cats at the image.

There may be tens, hundreds, thousands of categories.


4) another type of supervised learning, called `ranking`, it is used in e-cmmerce to 
rank exisitng products bassed if customer would potentially like them or not and give them a score.

so for example all products are ranked between 0 and 1 and top 6 products are shown as recommended
to the customer.
Google is doing something similar. So they show not only what is relevant based on user 
search criteria typed in, but also they score the results based on what they know about user
to show him the most relevant resutls.


So the learniing is called superwised because we have a matrix of inputs and expected resutls,
and the goal of supervised to learning to generate a function `g` that would provide the 
result `Y` when applied to the input `X` with acceptable precision.


So `g` tries to extract some patterns and put them into a form of model.



## CRISP-DM - methodology for organizing ML projects

CRISP-DM - cross-industry standard process for data mining

steps:
1 understand the problem (business understanding -> data understanding)

2 collect data	(data preparation)

3 Train the model (modeling -> evaluation)

4 use it ( -> Deployment)


process:
business understanding -> data understanding -> data preparation -> modeling -> evaluation -> Deployment

Business understanding:

		But first think if machine learning is right for this task
		Analyze to what extent it's a problem.
		Will ML solve it?

		We need to work on he problem with some goal/KPI/metric to measure success.

		Also we need to understand what data we have to solve the porblem
		(maybe we need to buy dataset or collect/accumulate dataset)

Data Understanding:

	During data understanding we need to answer ourselves:
	We need to identify the data sources:
		- do we have a report span button?
		- Is the data behind this button good enough?
		- Is it reliable?
		- Do we track it correctly?
		- Is the dataset large enough?
		- Do we need to get more data?

	Sometimes data sources may influence the goal.
	
	
Data Preparation:
	- transform the data so it can be put into ML algorithm
	- clean the data
	- build the pipelines
	- convert into tabular form
	

Modeling:
		- training the models: - the actual Machine Learning
		- we may try different models and select a best one.
		Which model to choose:
			- Logistic regression
			- Decision tree
			- Neural network
			- many others...

	Sometimes we may go back to data preparation in order to `Add new features` or `Fix data issues`


Evaluation:
	- measure how well the model solves the business problem.
	- evaluate the model trained on the test group
	
	Do a retrospective:
	- was the goal achievable?
	- did we solve/measure the right thing
	
	After that we may decide to:
	- go back and adjust the goal
	- rollout the model to more users/all users
	- stop working on the project
	
Deployment:
		If everything goes well, we may decide to run into online evaluation on live users.
		Meaning deploy the model and evaluate its' usage results
		
		Typically the model is deployed to 5% of users and if everything goes well -
		then it is deployed to all other users:
		- Roll the mode to all users
		- monitor the resutls
		- ensure the quality and maintainability
		
Finally: Iterate!

			ML projects require many iterations!
			Can we improve the model? Is it good enough?
			
Good approach:
	Start simple
	Learn from feedback
	Improve			


Summary:

1 - Business understanding: defina a measurable goal. Ask: do we need ML?

2 - Data understanding: do we have the date? Is it good?

3 - Data preparation: transform data into a table, so we can put it into ML

4 - Modelling: to select the best model, use the VALIDATION SET

5 - Evaluation: validate that the goal is reached

6 - Deployment: roll out to production to all users

7 - Iterate: start simple, learn from the feedback, improve



Model Selection
		- process to select the best model
		
		
we divide our data set into Train data  (Tran) (80% of total data amount) 
and Validation data (VAL) (20%).
Then train model on Train Data. Apply it on Val and get the result of for example 66% accuracy for Logistic Regression

The we decide to use a different model (Decision Tree) and get 60% accuracy

The we decide to use a different model (Random Forest) and get 67% accuracy

The we decide to use a different model (Gradient Boosting) and get 56% accuracy

The we decide to use a different model (Neural Network) and get 80% accuracy


Yet, there may be a problem, where we take the Val and get totally different accuracy, compared to the Tran data accuracy!
Some models would generate just 20% accuracy, some 40% accuracy, some 100% accuracy of the Val dataset!

It was just a lucky coincidence! And if we take another Val dataset - we would get totally different accuracy results!

This is all probalisitic result.

So we could take 20%  of data for VALIDation
And we could take 20%  of data for TESTing
And we could take 60%  of data for TRAINing

And then based on TESTing we select the best suitable model.

(but this proportion 60-20-20 could be totally different).


So the process is:
1 - SPLIT
2 - TRAIN
3 - VALIDATE
3a - compare results on different models
4 - select a best one
5 - TEST (to make sure model did not get particularly lucky)
6 - check

To get better results - we can first train model on TRAIN data, then check on VALidation data.
Then combine TRAIN + VAL data into a new dataset and train model on it one more time.
And then test it on TEST dataset.



## Introduction to  NumPy

Create a new repo in github and then in Code select CodeSpaces - it would launch 
a new instance of VS Code in browser

authorize it
run  PS1="> "	- make a link so there is no long-path printed before the command

ctrl + ~  - opens a new terminal

pip install jupiter numpy pandas scikit-learn seaborn

Homework answers:
import pandas as pd
pd.__version__

Q1. run `pd.__version__` in jupyter notebook

OR 
$ pip list
$
and get the version: 2.3.1


# Inside jupyter notebook

import numpy as np
np.zeros(5) 		# creates array of size 5 filled with `0`
np.ones(10) 		# creates array of size 10 filled with `1`
np.full(10, 2.5)	# fills array with specified number

create array `a` with values from input:
a = np.array([1, 2, 3, 5, 7, 12])
a[2] - access to the 3rd element in array. In

a[2] = 19	- assigns value `19` to the 3rd element of array
a			# print the array

np.arrange(10) 		# creates as array from 0 to 9 (default starts from 0) 
np.arrange(3, 10)	# create array with increase by 1 from 3 till 10 (so only 7 elements!)

np.linspace(0, 1, 11)	# creates array of size 11 with even values between records
						# (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0).
						# if we use np.linspace(0, 1, 10) - there would be not nice increments
						
# Multi-diensional arrays

np.zeros((5, 2))	# array of 5 rows with 2 columns


# Same with np.array to manually create it:

n = np.array([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ])

to access the value of 2 - we use n[0, 1] = 2
also we can update it:
n[0, 1] = 20
we can access the whole row via n[1]
update the whole row via:
n[2] = [1, 1, 1]


n[:, 1]		- accessing the whole 2nd column:

n[:]	- returns whole array

n[0,1] = 20		# update single element of 2-dim array

n[:, 2] = [0, 1, 2]		- reassign item in 2nd column

n[:, 2]





# Random generated arrays with np.random.rand()

np.random.rand(5, 2)	- standard UNIFORM distribution

np.random.rand(5, 2)	- different result upon second run

np.random.seed(10)		- to get reproducible results via fixing the seed
np.random.rand(5, 2)

np.random.seed(10)		- to get reproducible results via fixing the seed
np.random.rand(5, 2)	- standard NORMAL distribution


# For getting integers we can simply multiply by 100:

100 * np.random.rand(5, 2)

OR use another command:

np.random.randint(low=0, high=100, size=(5, 2))		# generate 5 rows, 2 columns, from 0 to 99 (100 excluded)


# fill array with zeros
a = np.zeros(5)


# Element wise-operations:

b = np.arange(5)

b + 1		# would add `1` to each item
b * 2 		# would multiply by `2` the each item
b / 100		# would divide by `100` the each item

10 + (b / 100)

c = (10 + (b * 3)) ** 2	/ 100	# multiply by 3 add 10 and then take to the power of 2 and divided by 100

# Combining operations:
d = b + c

e = c / b + 10


# Comparison operations:

a >= 2		# print which elements of array are higher than 2

a > b		# compare elements of array a and b


a[a > b]	# the inside statmnet returns an array of Tru/False. And the outer statement 
			# returns elements of array for which the condition is true.
			
a = [5, 3, 2, 1, 0]
b = [1, 2, 3, 0, 5]			

a[a > b]	- would return ([5, 3, 1])


# Summarizing Operations
a.min()		# returns smallest number
a.max()		# returns largest number

a.sum()		# sum of all elements
a.mean()	# mean value
a.std()		# standard deviation


# Linear Algebra (LA) refresh

In LA vectors are usually columns, while in numpy - vectors are usually arrays
Multiply or add oprations are typically applied to the each element of vector.

[u] + [v] = [u1 + v1, u2 + v2, ...]

so new matrix is a combination of 2 added vectors/matrices

in numpy

u = np.array([2, 4, 5, 6])
v = np.array([1, 0, 0, 2])

u + v
>> array ([3, 4, 5, 8])

2 * u
>> array ([4, 8, 10, 12])


vector-vector multiplication (dot product)

u * v = [u1 * v1 + u2 * v2 + u3 * v3 + u4 * v4] = 14

# Formula is SUM(Ui*Vi) for i from 1 till n

in order to turn rows into columns - we use transpose operations = Vrow * U column = Vtr * U

def vector_vector_multiplication(u, v):
  assert u.shape[0] = v.shape[0]
  
  u.shape[0] - is the size of array, the [0] element is array size.
  That's why we need to make sure vestors have the same size.
  
  n = u.shape[o]
  results = 0.0
  for i in range(n):
	result = result + u[i] * v[i]
	
  return result
  
vector_vector_multiplication(u, v)
>> 14

there is alreaady a function for it called .dot()

u.dov(v)
>> 14



# Matrix-vector multiplication

U = [[(2,4,5,6], [(1,2,1,2)], [(3, 1, 2, 1)]]

V = [1, 0.5, 2, 1]

U*v is take first row (first vector and multiply it at V) 

Utr0 * V = W0
Utr1 * V = W1
Utr2 * V = W2

the dimensionality should be the same!


def matrix_vector_multiplication(U, v):
	# we need to check the numnber of columns
	# so we check that size of 2nd dimension of matrix matches the size of vector
  	assert U.shape[1] == v.shape[0]
  
	num_rows = U.shape[0]  
	result = np.zeros(num_rows)
	
	for i in range(num_rows):
	# so for each row of matrix we perform vector_vector_multiplication
		result[i] = vector_vector_multiplication(U[i], v)
		
	return result



# In numpy we already have a function for it `.dot()`
# dot function knows how to implement it
 U.dot(v)  


# Matrix-matrix multipliication

U = [[(2,4,5,6], [(1,2,1,2)], [(3, 1, 2, 1)]]

V = [[(1, 1, 2], [(0, 0.5, 1)], [(0, 2, 1)], [(2, 1, 0)]]


# We take matrix V and brake in into columns and multiply it for matrix U to a vector V0, then U*v1, U*v2
Result is U*V = UV == [Uv0, Uv1, Uv2]



def matrix_matrix_multipliication(U, V):
	# we need to make sure that dimensions match
		assert U.shape[1] == V.shape[0]
			
		num_rows = U.shape[0]
		num_cols = V.shape[1]
		
		result = np.zeros((num_rows, num_cols))
		
		for i in range(num_cols):
			vi = V[:, i]
			Uvi = matrix_vector_multiplication(U, vi)
			result[;, i] = Uvi
		
		return result
	

# The same is implemented in numpy via .dot() function
$ U.dot(V)	


$ Identity matrix I - is a diagonal of matrix
identity of matrix is like 1: x * 1 = x, 1 * x = x

U * I = U  <==> I*U = U


np.eye(10) - generates matrix size of 10, with 1 on diagonal and all other items are 0.

I = np.eye(3)
V.dot(I) = returns V


# Matrix Inverse (only square matrix have inverse)

A-1 * A = I (square matrix with 1 diagonal)


Vs = V[[0, 1, 2]]
Vs
>>
array([[1. , 1. , 2. ],
       [0. , 0.5, 1. ],
	   [0. , 2. , 1. ]])
	   
Vs_inv = np.linalg.inv(Vs)
Vs_inv.dot(Vs)
>>
array([[1. , 0. ,  0. ],
       [0. , 1. ,  0. ],
	   [0. , 0. ,  1. ]])
	   
This is useful for liner regression


# 1.9 Pandas

data = [
	['Nissan', 'Stanza', 1991, 138, 4, 'MANUAL', 'sedan', 2000],
	['Hyundai', 'Sonata', 2017, None, 4, 'AUTOMATIC', 'sedan, 27150],
	['Lotus', 'Elise', 2010, 218, 4, 'MANUAL', 'convertible', 54990],
	['GMC', 'Acadia', 2017, 194, 4, 'AUTOMATIC', '4dr SUV', 34450],
	['Nissan', 'Frontier', 2017, 261, 6, 'MANUAL', 'Pickup', 32340]
]

columns = [
	'Make', 'Model', 'Year', "Engine HP', "Engine Cylinders', 
	'Transmission Type', 'Vehicle_Style', "MSRP'
]

df = pd.DataFrame(data, columns=columns)
df

Also, we can input it as list of dictionaries in JSON format
data1 = [
	{
	"Make": "Nissan",
	...
	...
	...
	"MSRP": 32340	
	}
	....
]

df2 = pd.DataFrame(data1)

df.head() 		# shows first 5 rows
df.head(n=3) 		# shows first 3 rows


each column is data series, to access a particular column run (make)

df.Make		OR 		df['Make]


df['Enfine HP']		# if colum name contians space, -, +, etc.


Getting a sub-set of only 3 columns:

df[['Make', 'Model', 'MSRP']]

df['id'] = [1, 2, 3, 4, 5 ]			# adding colulmn id to existing ones
df.id								# now extracts data
df['id'] = [10, 20, 30, 40, 50] 	# updating the column values from [1, 2, 3, 4, 5] to [10, 20, 30, 40, 50]

del df['id']						# remove column from dataframe


df.index			# Built-in index of rows, starts from 0

df.loc[1]			# get element (row) by location
df.loc[[1, 2]]		# get multiple rows

df.index = ['a', 'b', 'c', 'd', 'e' ]		# redefininig index

df.loc[1] 	# won' work now


Also there is `Positioanl Index`, usually used in lists / numpy arrays
Once we redefine regular inddex, then we can use .iloc() for positional index

df.iloc[1]
df.iloc[1, 2, 4]


resetting index - keeps previous re-asigned index and creates a new column for numerical indexes

df.reset_index()

df.reset_index(drop=True)		# drops previously re-assigned index


Element-wise operations

df['Engine HP'] / 100

df['Engine HP'] * 2

we operate arrays, but under-the-hood pandas use numpy


df['Year'] >= 2015	# returns records after 2015


df[df['Year'] >= 2015]	-	filters out and returns rows where year column is larger than 2015


df[
	df['Make'] == 'Nissan'		# return only rows where car makes is Nissan
]


Combined query

df [
	(df['Make'] == 'Nissan') & (df['Year'] >= 2015)
]


# String operations 


## NOTA BENE:
By default df is not overwritten!


'STRr'.lower()										# set all strings to lower case
'machine learning zoomcamp'.replace(' ', '_')		# replace ' ', with '_'


df['Vehicle_Stype']			# if records are not standartized, we can fix it

df['Vehicle_Stype'].str.lower()		# invokes string methods

df['Vehicle_Stype'].str.replace(' ', '_')		# still not overwrites the records


df['Vehicle_Stype'] = df['Vehicle_Stype'].str.replace(' ', '_')		# NOW overwrites the records

df['Vehicle_Stype']



# Summarizing operators

df.MSRP.min()
df.MSRP.max()
df.MSRP.mean()
df.MSRP.describe()				# provides statistic info for the specified column, but only for numerical values!

df.MSRP.describe().round(2)		# provides statistic info for the specified column, but only for numerical values, 
									rounded to 2 digits after decimal point


df.describe()					# describes the whole array

values with numbers called `numerical`
values with string called  `categorical`


df.Make.nunique()				# shows the number of unique values

df.nunique						# shows unique infor for all columns

for ML we don't want to have NaN or NULL values. So we can check it via

df.isnull()						# returns true if value is missing

df.isnull().sum()				# tells now many missing values in each column


# Grouping (group by)

group by transmission type and then calculate the average price:

```
SELECT transmission_type, AVF(MSRP)
FROM cars
GROUP By
	transmission_type
```

df.groupby('Transmission Type').MSRP.mean()			# group by transmission type and fine mean MSRP (price)
df.groupby('Transmission Type').MSRP.min()			# group by transmission type and fine smallest MSRP (price)
df.groupby('Transmission Type').MSRP.max()			# group by transmission type and fine largest MSRP (price)


df.MSRP.value		# returns a list



# Converting data-frame back to json:

df.to_dict(orient='records')		# returns a json, which we can save to a file

Completed.


# Homework notes

pip list  

and get the version: 2.3.1

Q1: 2.3.1
Q2. Records count: 9704
Q3. types of fuel: 2 (gasoline and diesel)
Q4. How many columns in the dataset have missing values? 4
Q5: Fuel efficiency: 23.75
Q6 - After filling the NaN fields of the horsepower column with the most frequent value
	of that column - the Median value of horsepower column has increased. It was obvious, as
	the mode (most frequest value) was larger than the median (mean) value. By updating about
	700+ records with the value (152) larger than median value (149) in the dataset of 9704 
	records - > we definitely move the distribution closer to the most frequent value.
	
	Answer is `Yes, it increased`

Q7 - Sum of weights: result close to 0.51