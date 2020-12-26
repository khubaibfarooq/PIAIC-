#!/usr/bin/env python
# coding: utf-8

# ### Numpy_Assignment_2::
# 

# #### Question:1
# #### Convert a 1D array to a 2D array with 2 rows?
# #### Desired output::
# #### array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

# In[1]:


import numpy as np
arr1d=np.array([0, 1, 2, 3, 4,5, 6, 7, 8, 9])
arr2d= np.reshape(arr1d,(2,5))
arr2d


# #### Question:2
# ##### How to stack two arrays vertically?
# 
# ##### Desired Output::
# ##### array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

# In[2]:


x = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
y = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
np.vstack((x,y))


# #### Question:3
# ##### How to stack two arrays horizontally?
# #####  Desired Output::
# #####  array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1], [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])

# In[3]:


x = np.array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1]])
y= np.array([[5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
np.vstack((x,y))


# #### Question:4
# ##### How to convert an array of arrays into a flat 1d array?
# #####  Desired Output::
# ##### array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# 

# In[4]:


x = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
x.flatten()


# #### Question:5
# ##### How to Convert higher dimension into one dimension?
# ##### Desired Output::
# ##### array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

# In[9]:


arr=np.array([[ 0, 1, 2], [ 3, 4, 5], [ 6, 7, 8], [ 9, 10, 11], [12, 13, 14]])
arr.flatten()


# #### Question:6
# ##### Convert one dimension to higher dimension?
# ##### Desired Output::
# ##### array([[ 0, 1, 2], [ 3, 4, 5], [ 6, 7, 8], [ 9, 10, 11], [12, 13, 14]])

# In[14]:


arr=np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
newarr=arr.reshape(5, 3)
newarr


# #### Question:7
# ##### Create 5x5 an array and find the square of an array?
# 
# 

# In[20]:


arr=np.random.random((5,5))

np.square(arr)


# ##### Question:8
# ##### Create 5x6 an array and find the mean?
# 
# 

# In[22]:


arr=np.random.random((5,6))
print("Original array")
print(arr)
print("Mean")
np.mean(arr)


# ##### Question:9
# ##### Find the standard deviation of the previous array in Q8?
# 

# In[23]:


np.std(arr)


# ##### Question:10
# ##### Find the median of the previous array in Q8?
# 
# 

# In[24]:


np.median(arr)


# ##### Question:11
# ##### Find the transpose of the previous array in Q8?
# 

# In[26]:


print("Original array")
print(arr)
newarr=arr.transpose()
newarr


# 
# ##### Question:12
# ##### Create a 4x4 an array and find the sum of diagonal elements?
# 

# In[28]:


arr4x4=np.random.random((4,4))
print(arr4x4)
sum=np.trace(arr4x4)
sum


# ##### Question:13
# ##### Find the determinant of the previous array in Q12?
# 
# 

# In[29]:


np.linalg.det(arr4x4)


# ##### Question:14
# ##### Find the 5th and 95th percentile of an array?
# 
# 

# In[31]:



print("5th percentile")
print(np.percentile(arr4x4,5))
print("95th percentile")
print(np.percentile(arr4x4,95))


# ##### Question:15
# ##### How to find if a given array has any null values?

# In[43]:


a = np.array([1,2,3,np.nan])
np.isnan(a).any()


# In[ ]:




