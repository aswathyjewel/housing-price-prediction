#!/usr/bin/env python
# coding: utf-8

# In[1]:



#---------------------------------------------------------------
# @author Bhargav Vemula C0818081
# @author Govind Vijayan C0819805
# @author Jaison Kayamkattil Jacob C0814631
# @author Prasanth Moothedath Padmakumar C0796752

# Dataset File downloaded from https://www.kaggle.com/aariyan101/usa-housingcsv

# The inputs are as follows
# ‘Avg. Area Income’ – Avg. The income of the householder of the city house is located.
# ‘Avg. Area House Age’ – Avg. Age of Houses in the same city.
# ‘Avg. Area Number of Rooms’ – Avg. Number of Rooms for Houses in the same city.
# ‘Avg. Area Number of Bedrooms’ – Avg. Number of Bedrooms for Houses in the same city.
# ‘Area Population’ – Population of the city.
# ‘Address’ – Address of the houses.

# The output is
# Price = Price of the house in USD 
#---------------------------------------------------------------


# ### Import Libraries

# In[2]:


# Pandas library for data manipulation and analysis
# Numpy library for some standard mathematical functions
# Matplotlib library to visualize the data in the form of different plot
# Seaborn library for visualizing statistical graphics and work on top of Matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#To display plot within the document
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read the dataset from csv file

# In[3]:


HousingData = pd.read_csv('USA_Housing.csv')


# In[4]:


# Display first 5 rows of the dataset using head function
HousingData.head()


# ### Checking data set for null values and duplicate entries

# In[5]:


# Print summary of the dataframe 
HousingData.info( )


# In[6]:


# Checking count of duplicate entries in the data set
HousingData.duplicated().sum()


# In[7]:


# Describing the data set 
HousingData.describe( )


# In[8]:


# Display columns in the data set
HousingData.columns


# ### Visualizing the dataset

# In[9]:


# Pair plot
sns.pairplot(HousingData)


# In[10]:


# Distribution plot
sns.displot(HousingData['Price'])


# In[11]:


# From the pairplot we can see there are correlations to some extend
# In Distribution plot, the bell shape indicates the data is normalised


# In[12]:


# Heat map
sns.heatmap(HousingData.corr(), annot=True)


# In[13]:


# From heat map it visible that variables Income, House age, Population, Rooms are showing good correlation
# With the attribute number of bedrooms showing least correlation


# ### Training our Linear Regression Model
# 

# In[14]:


# X represents the independent variables (predictors)
# Address field is removed since it contains text data and not relevant for the linear regression
X = HousingData[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]

# Y represents the dependent variable (predictand)
y = HousingData['Price']


# In[15]:


# Import train_test_split from sklearn library to split the data set to train and test data set
from sklearn.model_selection import train_test_split

# two third of the whole data set is used for training and rest is used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[16]:


# first 5 rows of predictors from the training data
X_train.head()


# In[17]:


# first 5 rows of predictand from the training data
y_train.head()


# In[18]:


# first 5 rows of predictors from the test data
X_test.head()


# In[19]:


# first 5 rows of predictand from the test data
y_test.head()


# In[20]:


#importing the algorithm to be used 
from sklearn.linear_model import LinearRegression

model = LinearRegression()


# In[21]:


# Fitting the model to the training data to find coefficients
model.fit(X_train,y_train)


# In[22]:


# Predict Price for test data and storing the values in a variable
predictions = model.predict(X_test)


# In[23]:


# Slope or intercept
model.intercept_


# In[24]:


# Display Coefficient of each variables in tabular form
coefficient = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
coefficient.head() 


# In[25]:


# For unit change in area income price varies by 21.61635$
# For unit change in house age price varies by 165221.119872$
# For unit change in Rooms price varies by 121405.376596$
# For unit change in Bedrooms price varies by 1318.718783$
# For unit change in Population price varies by 15.225196$


# ### Comparing Test values against Prediction

# In[26]:


# scatter plot
plt.scatter(y_test,predictions)
plt.xlabel('Test')
plt.ylabel("Predicted")
plt.show()


# In[27]:


# The scatter plot obtained is in line shape


# In[33]:


# Distribution plot
sns.displot((y_test-predictions),bins=50);


# In[29]:


# The distribution plot has almost formed a bell shape


# In[30]:


# Depicting the difference in Predicted and actual value for first 5 predictions
difference = predictions - y_test
diffTable = pd.DataFrame({'Actual-Value': y_test, 'Predicted-Value':predictions, 'Difference':difference})
diffTable.head()


# In[34]:


# Plotting a bargraph for showing difference in the first 10 actual and predicted results
comparisonDf = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}).head(10)
comparisonDf.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5')
plt.grid(which='minor', linestyle=':', linewidth='0.5')
plt.ylabel("House Price")
plt.xlabel("Row id in dataset")
plt.show()


# ### Metrics

# In[32]:


# Three Different methods from metrics library is used to check the accuracy of the prediction
from sklearn import metrics

print('Coefficient of determination, R2 Score = {:f}'.format(metrics.r2_score(y_test, predictions)))
print('Mean Absoulute Error = {:f}'.format(metrics.mean_absolute_error(y_test, predictions)))
print('Root Mean Squared Error = {:f}'.format(np.sqrt(metrics.mean_squared_error(y_test, predictions))))

