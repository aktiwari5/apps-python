#!/usr/bin/env python
# coding: utf-8

# NumPy stands for Numerical Python and is the core library for numeric and scientific computing

# It basically consist of multi-dimensional array objects and a collection of routines for processing those arrays

# In[9]:


import numpy as np
n1 = np.array([10,20,30,40])
n1


# In[61]:


n2 = np.array([[10,20,30,40],[60,70,80,90]])
n2


# In[62]:


type(n1)


# In[63]:


n1.ndim


# In[64]:


n2.ndim


# Initializing NumPy array with zeros

# In[65]:


z1 = np.zeros((1,2))
z1


# In[66]:


z2 = np.zeros((5,5))
z2


# In[67]:


o1 = np.ones((4,4))
o1


# Initializing NumPy array with same Number

# In[68]:


a1 = np.full((2,2),10)
a1


# Initializing NumPy array withing a range

# In[69]:


r1 = np.arange(10,20)
r1


# In[70]:


r2 = np.arange(10,60,5)


# In[71]:


r2


# **********************************************************************************

# Pandas stands for Panel Data and is the core library for data manipulation and data analysis

# It consists of single and muti-dimensional data structures for data manipulation
# a. Single Dimensional data structure is called as "Series Object"
# b. Multi-dimensional data structure is called as "Data Frames"

# Series Object is one-dimensional labeled array

# In[10]:


import pandas as pd
s1  = pd.Series([1,2,3,4,5])
s1


# In[73]:


type(s1)


# In[74]:


s2 = pd.Series([1,2,3,4,5,6],index=['a','b','c','d','e','f'])
s2


# In[75]:


s1[3]


# In[76]:


s1[-3:]


# In[77]:


s1[:3]


# Dataframe is a 2-dimesional labelled data-structure. Dataframe comprises of rows and columns

# In[78]:


pd.DataFrame({"Name":['Bob','Sam','Anne'],"Marks":[76,25,92]})


# In[79]:


d2={"Name":['Bob','Sam','Anne'],"Marks":[76,25,92]}
d2


# In[80]:


d2.keys()


# In[81]:


courses = pd.read_csv('HRApps-Data-Courses.csv')


# In[82]:


courses


# In[83]:


courses.head(2)


# In[84]:


courses.tail(3)


# In[85]:


courses.describe()


# In[86]:


courses.shape


# In[87]:


courses.sort_index(ascending=False)


# In[88]:


courses_sorted= courses.sort_values('Title')
courses_sorted.head(17)


# Let’s sort our dataframe by lowest Hours and Department.

# In[89]:


columns = ['Department', 'Hours']
order = [False, True]
courses.sort_values(by=columns, ascending=order)


# In[99]:


us_babies = pd.read_csv("us_baby_names.csv")


# In[100]:


us_babies


# #### Question : what were the 5 most popular baby names in 2014 in the united states?

# Answer:
#     1. slice out rows for 2014
#     2. Sort rows in decending order by count
#     3. Get first five rows

# In[102]:


us_babies['Year']


# In[103]:


us_babies['Year'] == 2014


# In[105]:


us_babies.loc[us_babies['Year'] == 2014, :]


# In[106]:


us_babies_2014 = us_babies.loc[us_babies['Year'] == 2014, :]


# In[107]:


us_babies_2014


# In[109]:


sorted_us_2014 = us_babies_2014.sort_values('Count', ascending=False)


# In[110]:


sorted_us_2014


# In[111]:


sorted_us_2014.iloc[0:5]


# In[113]:


#states_babies = pd.read_csv('state_baby_names.csv')


# In[114]:


#states_babies


# #### Questions : what were the 5 most popular baby names in California

# Answer:
#     1. slice out the rows for the year 2014
#     2. Slice out the rows for the state California
#     3. Sort the rows in decending order by Count
#     4. Get the first five rows

# In[118]:


#states_babies_2014 = states_babies.loc[states_babies['Year'] == 2014,:]


# In[129]:


#ca_babies_2014 = states_babies_2014.loc[states_babies_2014['State'] == 'CA',:]


# In[131]:


#sorted_ca_babies_2014 = ca_babies_2014.sort_values('Count', ascending = False)


# In[132]:


#sorted_ca_babies_2014.iloc[0:5]


# #### Question : what were the most popular Male and Female baby names in California

# Answer:
#     1. Slice out the rows for California
#     2. Group resulting DataFrame by 'Year' and 'Gender'
#     3. Compute the most popular name for each Group

# In[135]:


#ca_states_babies = states_babies.loc[states_babies['State'] == 'CA',:]


# In[137]:


#ca_states_babies


# In[139]:


def popular(s):
    """Receives s, a Panda series, contaning baby names in order of highest count to lowest count.
    Retuns the most popular name in s."""
    return s.iloc[0]


# In[141]:


#ca_grouped = ca_states_babies.sort_values('Count', ascending = False).groupby(['Year', 'Gender']).agg(popular)


# In[143]:


#ca_grouped.sort_values('Year', ascending = False)


# ##### Question: How frequent your baby name occurs across the year in US among baby names

# Answer:
#     1. slice out the rows containing my first name
#     2. Create a Horizontal bar plot for occurence of my name over the years

# In[163]:


us_nisha = us_babies.loc[us_babies['Name'] == 'Ajit', :]


# In[164]:


us_nisha


# In[165]:


us_nisha.plot.barh(x = 'Year', y = 'Count')


# In[167]:


#crime = pd.read_csv('crime_boston.csv')


# In[168]:


#crime


# In[170]:


#crime.isnull() # to return the True for null values in Data Frame


# In[171]:


#crime.isnull().any(axis = 1)  # Output is Panda series that contains boolean. It represents each row with boolean indicating does that row has missing value. True means yes it had and False mean no missing data


# In[173]:


#rows_with_missing_vals = crime.isnull().any(axis = 1)
#crime[rows_with_missing_vals]


# In[175]:


#crime_cleaned =  crime.drop(columns = ['YEAR','MONTH','HOUR']) # Drop the irrelavent columns from table


# In[177]:


#crime_cleaned['OFFENSE_CODE_GROUP'].unique()


# In[182]:


#crime_cleaned['DAY_OF_WEEK'].unique()


# In[183]:


#crime_cleaned = crime_cleaned.drop(columns = 'Location')


# #### Qualitative Data Virtualization


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


listings = pd.read_csv('AB_NYC_2019.csv')


# In[187]:


listings


# In[188]:


sns.countplot(x = 'neighbourhood_group' , data = listings)
plt.show()


# In[189]:


sns.barplot(x = 'neighbourhood_group', y = 'price', data = listings)
plt.show()


# In[191]:


sns.barplot(x = 'neighbourhood_group', y = 'price', data = listings, ci = False)
plt.show()


# #### Quantitative Data Virtualization

# In[192]:


plt.hist(listings['price'])
plt.xlabel('price (in US dollars)')
plt.show()


# In[198]:


plt.hist(listings['price'], bins = np.arange(0, 1100, 40))
plt.xlabel('price (in US dollars)')
plt.show()


# In[201]:


plt.scatter(x = listings['price'], y = listings['number_of_reviews'])
plt.xlabel('price')
plt.ylabel('number of reviews')
plt.show()


# In[202]:


plt.scatter(x = listings['price'], y = listings['number_of_reviews'])
plt.xlabel('price')
plt.ylabel('number of reviews')
plt.xlim(0, 1100)
plt.show()


# In[205]:


plt.scatter(x = listings['price'], y = listings['number_of_reviews'], s = 4)
plt.xlabel('price')
plt.ylabel('number of reviews')
plt.xlim(0, 1100)
plt.show()


# In[ ]:





# ***********************************************************************************************

# An introduction to seaborn¶
# Seaborn is a library for making statistical graphics in Python. It is built on top of matplotlib and closely integrated with pandas data structures.
# 
# Here is some of the functionality that seaborn offers:
# 
# A dataset-oriented API for examining relationships between multiple variables
# 
# Specialized support for using categorical variables to show observations or aggregate statistics
# 
# Options for visualizing univariate or bivariate distributions and for comparing them between subsets of data
# 
# Automatic estimation and plotting of linear regression models for different kinds dependent variables
# 
# Convenient views onto the overall structure of complex datasets
# 
# High-level abstractions for structuring multi-plot grids that let you easily build complex visualizations
# 
# Concise control over matplotlib figure styling with several built-in themes
# 
# Tools for choosing color palettes that faithfully reveal patterns in your data
# 
# Seaborn aims to make visualization a central part of exploring and understanding data. Its dataset-oriented plotting functions operate on dataframes and arrays containing whole datasets and internally perform the necessary semantic mapping and statistical aggregation to produce informative plots.

# In[97]:


fmri = sns.load_dataset("fmri")
fmri.head()


# In[98]:


sns.set()
tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", col="time",
            hue="smoker", style="smoker", size="size",
            data=tips);


# In[14]:


country_population = pd.read_csv("world-population-data.csv")


# In[15]:


country_population.head()


# In[16]:


country_population.head(235)


# In[17]:


plt.hist(country_population['Population'])
plt.xlabel('population (in million)')
plt.show()


# In[18]:


sns.barplot(x = 'Country', y = 'Population', data = country_population, ci = False)
plt.show()


# In[20]:


plt.scatter(x = country_population['Country'], y = country_population['Population'], s = 4)
plt.xlabel('Country Names')
plt.ylabel('Population')
plt.show()


# In[ ]:




