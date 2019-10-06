### Data processing
### Author: Klaudia Łubian - StatsEye
### Date: 06/10/2019

# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Settings 
sns.set()
%matplotlib inline

pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Import dataset. It was created with records of absenteeism at work from July 2007 to July 2010 at a courier company in Brazil.
raw_data = pd.read_csv('data/Absenteeism_at_work.csv', delimiter=";")
raw_data.head()

# Make a copy of raw_data
df = raw_data.copy()
df.info()

# Remove unnecessary columns
df.drop(['ID'], axis=1, inplace=True)

# Plot bar charts for all vars
plt.rcParams['figure.figsize']=25,10              
df.hist()                                 
plt.show() 

### Dependent variable

df['Absenteeism time in hours'].value_counts()
df['Absenteeism time in hours'].hist(bins=100)
# Clearly count data. The general advice is to analyze these with some variety of a Poisson model.

### Turn categorical variables into dummies 

# Reason for absence 
sorted(df['Reason for absence'].unique())

# To treat is as categorical variable, turn all options into dummies. Drop first column to avoid mullticollinearity (reason 0 represents no reason given), and then combine reasons for absence as they are related (https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work)
reasons = pd.get_dummies(df['Reason for absence'], drop_first=True)
reasons.info()
reason_disease = reasons.iloc[:, 0:14].max(axis=1)
reason_pregnancy = reasons.iloc[:, 15:17].max(axis=1)
reason_external = reasons.iloc[:, 18:21].max(axis=1)
reason_visit = reasons.iloc[:, 22:28].max(axis=1)

df = pd.concat([df, reason_disease, reason_pregnancy, reason_external, reason_visit], axis=1)
df.drop('Reason for absence', axis=1, inplace=True)
df.info()

# Change column names
print(df.columns.values)

column_names = ['Month of absence', 'Day of the week', 'Seasons',
       'Transportation expense', 'Distance from Residence to Work',
       'Service time', 'Age', 'Work load Average/day ', 'Hit target',
       'Disciplinary failure', 'Education', 'Son', 'Social drinker',
       'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index',
       'Absenteeism time in hours', 'Reason_disease',
       'Reason_pregnancy', 'Reason_external', 'Reason_visit']

df.columns = column_names
df.info()

# Education

df['Education'].value_counts()
# These values repressent high school, graduate, post-graduate and master-Phd. Majority of employees are high school graduate and around 100 of them have college or higher degree. I will recode high school to 0 and rest to 1.

df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
df['Education'].value_counts()

# Two variables seem to have very low prevalence
df['Social smoker'].value_counts()
df['Disciplinary failure'].value_counts()
# Disciplinary failure - too few cases with 1s, delete from the dataset
df.drop('Disciplinary failure', axis=1, inplace=True)

### Investigate correlation of variables

colormap = plt.cm.RdBu
plt.figure(figsize=(15,15))
plt.title('Pearson Correlation of Features', y=1.0, size=10)
sns.heatmap(df.corr(),linewidths=0.2,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
# BMI highly correlated with Weight (0.9), with Height not, but it does not seem to be relevant for prediction - keep BMI only
# Service time and Age are also strongly correlated
# other correlations acceptable

df.drop(['Height'], axis=1, inplace=True)
df.drop(['Weight'], axis=1, inplace=True)

sns.jointplot("Age", "Service time", df, kind='kde');
# Age is is more strongly related with the dependent variable than Service time, drop Service time
df.drop(['Service time'], axis=1, inplace=True)
df.info()

# Checkpoint
df_preprocessed = df.copy()
df_preprocessed.head(10)
df_preprocessed.info()

# Separate dependent and independent variables
y = df_preprocessed.loc[:, ['Absenteeism time in hours']]
X = df_preprocessed.drop(['Absenteeism time in hours'], axis=1)
X.info()
