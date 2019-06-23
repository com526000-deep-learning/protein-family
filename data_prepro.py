#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[2]:


df_seq = pd.read_csv('protein-data-set/pdb_data_seq.csv')
df_no_dups = pd.read_csv('protein-data-set/pdb_data_no_dups.csv')


# In[3]:


df_seq.head(10)


# In[4]:


df_no_dups.head(10)


# In[5]:


# we only want proteins
protein_seq = df_seq[df_seq.macromoleculeType == 'Protein']
protein_no_dups = df_no_dups[df_no_dups.macromoleculeType == 'Protein']


# In[6]:


# we need structureId as index to combine two datasets
# we need sequence and classifition for our task
# residueCount told us sequence length 
protein_seq = protein_seq[['structureId','sequence', 'residueCount']]
protein_no_dups = protein_no_dups[['structureId','classification', 'residueCount']]


# In[7]:


protein_seq.residueCount.describe()


# In[8]:


protein_no_dups.residueCount.describe()


# In[9]:


protein_seq_2000 = protein_seq.loc[protein_seq.residueCount<2000]
protein_no_dups_2000 = protein_no_dups.loc[protein_no_dups.residueCount<2000]
protein_seq_1200 = protein_seq.loc[protein_seq.residueCount<1200]
protein_no_dups_1200 = protein_no_dups.loc[protein_no_dups.residueCount<1200]
protein_seq_2000 = protein_seq_2000[['structureId','sequence']]
protein_no_dups_2000 = protein_no_dups_2000[['structureId','classification']]
protein_seq_1200 = protein_seq_1200[['structureId','sequence']]
protein_no_dups_1200 = protein_no_dups_1200[['structureId','classification']]


# In[10]:


# join two dataset using structureId as index
data_max2000 = protein_no_dups_2000.set_index('structureId').join(protein_seq_2000.set_index('structureId'))
data_max2000.head(10)


# In[11]:


# join two dataset using structureId as index
data_max1200 = protein_no_dups_1200.set_index('structureId').join(protein_seq_1200.set_index('structureId'))
data_max1200.head(10)


# In[12]:


protein_seq = protein_seq[['structureId','sequence']]
protein_no_dups = protein_no_dups[['structureId','classification']]
# join two dataset using structureId as index
data = protein_no_dups.set_index('structureId').join(protein_seq.set_index('structureId'))
data.head(10)


# In[14]:


# count missing and drop rows with them 
print(data.isnull().sum())
data = data.dropna()
print("number of rows of data is %d"%data.shape[0])
data_max2000 = data_max2000.dropna()
print("number of rows of data_max2000 is %d"%data_max2000.shape[0])
data_max1200 = data_max1200.dropna()
print("number of rows of data_max1200 is %d"%data_max1200.shape[0])


# In[16]:


# count numbers for each classification type
type_counts_2000 = data_max2000.classification.value_counts()
print(type_counts_2000)


# In[17]:


# count numbers for each classification type
type_counts_1200 = data_max1200.classification.value_counts()
print(type_counts_1200)


# In[18]:


# count numbers for each classification type
type_counts = data.classification.value_counts()
print(type_counts)


# In[19]:


print("There are %d classification types (seq maxlength=2000)" %type_counts_2000.shape[0])
plt.title('distribution of classification types')
plt.hist(np.array(type_counts_2000), bins = np.arange(start=0, stop=47000, step=1000))
plt.xlabel('value counts')
plt.ylabel('# of types')
plt.show()


# In[20]:


print("There are %d classification types (seq maxlength=1200)" %type_counts_1200.shape[0])
plt.title('distribution of classification types')
plt.hist(np.array(type_counts_1200), bins = np.arange(start=0, stop=47000, step=1000))
plt.xlabel('value counts')
plt.ylabel('# of types')
plt.show()


# In[21]:


print("There are %d classification types" %type_counts.shape[0])
plt.title('distribution of classification types')
plt.hist(np.array(type_counts), bins = np.arange(start=0, stop=47000, step=1000))
plt.xlabel('value counts')
plt.ylabel('# of types')
plt.show()


# In[22]:


# using classification types which counts are over 500
# filter dataset and only remain classification types > 500
types_500 = np.asarray(type_counts[(type_counts > 500)].index)
data_500 = data[data.classification.isin(types_500)]
types_500_max2000 = np.asarray(type_counts_2000[(type_counts_2000 > 500)].index)
data_500_max2000 = data_max2000[data_max2000.classification.isin(types_500_max2000)]
types_500_max1200 = np.asarray(type_counts_1200[(type_counts_1200 > 500)].index)
data_500_max1200 = data_max1200[data_max1200.classification.isin(types_500_max1200)]
print("There are %d classification types with counts > 500" %types_500.shape[0])
#print()
#print(types_500)
#print()
print("There are %d rows of data" %data_500.shape[0])

print("There are %d classification types with counts > 500 (seq maxlength=2000)" %types_500_max2000.shape[0])
print("There are %d rows of data (seq maxlength=2000)" %data_500_max2000.shape[0])

print("There are %d classification types with counts > 500 (seq maxlength=1200)" %types_500_max1200.shape[0])
print("There are %d rows of data (seq maxlength=1200)" %data_500_max1200.shape[0])


# In[23]:


# using classification types which counts are over 1000
# filter dataset and only remain classification types > 1000
types_1000 = np.asarray(type_counts[(type_counts > 1000)].index)
data_1000 = data[data.classification.isin(types_1000)]
types_1000_max2000 = np.asarray(type_counts_2000[(type_counts_2000 > 1000)].index)
data_1000_max2000 = data_max2000[data_max2000.classification.isin(types_1000_max2000)]
types_1000_max1200 = np.asarray(type_counts_1200[(type_counts_1200 > 1000)].index)
data_1000_max1200 = data_max1200[data_max1200.classification.isin(types_1000_max1200)]
print("There are %d classification types with counts > 1000" %types_1000.shape[0])
#print()
#print(types_1000)
#print()
print("There are %d rows of data" %data_1000.shape[0])

print("There are %d classification types with counts > 1000 (seq maxlength=2000)" %types_1000_max2000.shape[0])
print("There are %d rows of data (seq maxlength=2000)" %data_1000_max2000.shape[0])

print("There are %d classification types with counts > 1000 (seq maxlength=1200)" %types_1000_max1200.shape[0])
print("There are %d rows of data (seq maxlength=1200)" %data_1000_max1200.shape[0])


# In[24]:


# using classification types which counts are over 1500
# filter dataset and only remain classification types > 1500
types_1500 = np.asarray(type_counts[(type_counts > 1500)].index)
data_1500 = data[data.classification.isin(types_1500)]
types_1500_max2000 = np.asarray(type_counts_2000[(type_counts_2000 > 1500)].index)
data_1500_max2000 = data_max2000[data_max2000.classification.isin(types_1500_max2000)]
types_1500_max1200 = np.asarray(type_counts_1200[(type_counts_1200 > 1500)].index)
data_1500_max1200 = data_max1200[data_max1200.classification.isin(types_1500_max1200)]
print("There are %d classification types with counts > 1500" %types_1500.shape[0])
#print()
#print(types_1500)
#print()
print("There are %d rows of data" %data_1500.shape[0])

print("There are %d classification types with counts > 1500 (seq maxlength=2000)" %types_1500_max2000.shape[0])
print("There are %d rows of data (seq maxlength=2000)" %data_1500_max2000.shape[0])

print("There are %d classification types with counts > 1500 (seq maxlength=1200)" %types_1500_max1200.shape[0])
print("There are %d rows of data (seq maxlength=1200)" %data_1500_max1200.shape[0])


# In[25]:


# using classification types which counts are over 5000
# filter dataset and only remain classification types > 5000
types_5000 = np.asarray(type_counts[(type_counts > 5000)].index)
data_5000 = data[data.classification.isin(types_5000)]
types_5000_max2000 = np.asarray(type_counts_2000[(type_counts_2000 > 5000)].index)
data_5000_max2000 = data_max2000[data_max2000.classification.isin(types_5000_max2000)]
types_5000_max1200 = np.asarray(type_counts_1200[(type_counts_1200 > 5000)].index)
data_5000_max1200 = data_max1200[data_max1200.classification.isin(types_5000_max1200)]
print("There are %d classification types with counts > 5000" %types_5000.shape[0])
#print()
#print(types_5000)
#print()
print("There are %d rows of data" %data_5000.shape[0])

print("There are %d classification types with counts > 5000 (seq maxlength=2000)" %types_5000_max2000.shape[0])
print("There are %d rows of data (seq maxlength=2000)" %data_5000_max2000.shape[0])

print("There are %d classification types with counts > 5000 (seq maxlength=1200)" %types_5000_max1200.shape[0])
print("There are %d rows of data (seq maxlength=1200)" %data_5000_max1200.shape[0])


# In[26]:


data_5000_max2000.to_csv('data_5000_max2000.csv')
data_1500_max2000.to_csv('data_1500_max2000.csv')
data_1000_max2000.to_csv('data_1000_max2000.csv')
data_500_max2000.to_csv('data_500_max2000.csv')


# In[27]:


data_5000_max1200.to_csv('data_5000_max1200.csv')
data_1500_max1200.to_csv('data_1500_max1200.csv')
data_1000_max1200.to_csv('data_1000_max1200.csv')
data_500_max1200.to_csv('data_500_max1200.csv')


# In[29]:


data_5000.to_csv('data_5000.csv')
data_1500.to_csv('data_1500.csv')
data_1000.to_csv('data_1000.csv')
data_500.to_csv('data_500.csv')


# In[ ]:




