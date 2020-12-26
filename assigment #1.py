#!/usr/bin/env python
# coding: utf-8

# ## Assignment For Numpy
# Import the numpy package under the name np

# In[1]:


import numpy as np


# ###### Create a null vector of size 10

# In[2]:


import numpy as np
x = np.zeros(10)
print (x)


# ###### Create a vector with values ranging from 10 to 49

# In[3]:


import numpy as np
arr = np.arange(10,49)
print (arr)


# ###### Find the shape of previous array in question 3

# In[4]:


print(arr.shape)


# ###### Print the type of the previous array in question 3
# 

# In[5]:


print(arr.dtype)


# ###### Print the numpy version and the configuration

# In[6]:


print(np.__version__)
print(np.show_config())


# ###### Print the dimension of the array in question 3

# In[7]:


print(arr.ndim)


# ###### Create a boolean array with all the True values

# In[8]:


bool_arr = np.ones(10, dtype=bool)
print('Numpy Array: ')
print(bool_arr)


# ###### Create a two dimensional array

# In[9]:


two_d_array=np.array([[1, 2, 3], [4, 5, 6]])
print(two_d_array)
print(two_d_array.ndim)


# ###### Create a three dimensional array

# In[10]:


three_d_array=np.array([[[1, 2, 3], [4, 5, 6],[6,7,8]]])
print(three_d_array)
print(three_d_array.ndim)


# # Difficulty Level Easy

# ###### Reverse a vector (first element becomes last)

# In[11]:


x = np.arange(10, 49)
print("Original array:")
print(x)
print("Reverse array:")
x = x[::-1]
print(x)


# ###### Create a null vector of size 10 but the fifth value which is 1

# In[12]:


Z = np.zeros(10)
Z[4] = 1
print(Z)


# ###### Create a 3x3 identity matrix

# In[13]:


identity = np.eye(3)
print(identity)


# ###### arr = np.array([1, 2, 3, 4, 5])

# ###### Convert the data type of the given array from int to float

# In[14]:


arr = np.array([1, 2, 3, 4, 5])
float_arr=arr.astype('float')
print(float_arr)


# # arr1 = np.array([[1., 2., 3.],[4., 5., 6.]])  
# 
# ###### arr2 = np.array([[0., 4., 1.],
# 
# ######  [7., 2., 12.]])
# ###### Multiply arr1 with arr2

# In[15]:


arr1 = np.array([[1., 2., 3.],[4., 5., 6.]])
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
result=np.multiply(arr1, arr2)
result


# ###### arr1 = np.array([[1., 2., 3.],
# 
# ######    [4., 5., 6.]]) 
# 
# ###### arr2 = np.array([[0., 4., 1.],
# ######  [7., 2., 12.]])
# ###### Make an array by comparing both the arrays provided above

# In[16]:


arr1 = np.array([[1., 2., 3.],
[4., 5., 6.]])
arr2 = np.array([[0., 4., 1.],
[7., 2., 12.]])
comparison = arr1 == arr2
equal_arrays = comparison.all()
equal_arrays


# ###### Extract all odd numbers from arr with values(0-9)

# In[17]:


arr=np.arange(1,10)

arr[arr % 2 == 1]


# ###### Replace all odd numbers to -1 from previous array

# In[18]:



print("Original array")
print(arr)
arr[arr%2==1]=-1
print("all odd numbers to replaced -1 ")
print(arr)


# ###### Replace the values of indexes 5,6,7 and 8 to 12

# In[19]:


arr = np.arange(10,49)
arr[5]=12
arr[6]=12
arr[7]=12
arr[8]=12
arr


# ###### Create a 2d array with 1 on the border and 0 inside

# In[20]:


x = np.ones((5,5))
print("Original array:")
print(x)
print("1 on the border and 0 inside in the array")
x[1:-1,1:-1] = 0
print(x)


# ###### Difficulty Level Medium
# 
# ###### arr2d = np.array([[1, 2, 3],
# 
# ######            [4, 5, 6], 
# 
# ######            [7, 8, 9]])
# ###### Replace the value 5 to 12

# In[21]:


arr2d = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
arr2d[1:-1:,1:-1:]=12
arr2d


# ###### arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# ###### Convert all the values of 1st array to 64

# In[22]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0:1:,0:1:]=64
arr3d


# ###### Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[23]:


arr2d = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])

arr1d=arr2d[0:1]
arr1d


# ###### Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[24]:


arr2d = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
arr1d=arr2d[1:2]
arr1d


# ###### Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[25]:


arr2d = np.array([[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
arr1d=arr2d[2:3]
arr1d


# ###### Create a 10x10 array with random values and find the minimum and maximum values

# In[26]:


x=np.random.random((10,10))
xmin, xmax = x.min(), x.max()
print("Minimum and Maximum Values:")
print(xmin, xmax)


# ###### a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])¶
# ###### Find the common items between a and b

# In[27]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a, b))


# ###### a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ###### Find the positions where elements of a and b match

# In[28]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(np.in1d(a, b))[0]


# ###### names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# ###### Find all the values from array data where the values from array names are not equal to Will

# In[29]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)


# ###### names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# ###### Find all the values from array data where the values from array names are not equal to Will and Joe

# In[ ]:





# ###### Difficulty Level Hard
# 
# ###### Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[30]:


arr = np.arange(15).reshape(5,3)
arr


# ###### Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[31]:


arr = np.arange(16).reshape(2,2,4)
arr


# ###### Swap axes of the array you created in Question 32

# In[34]:


np.swapaxes(arr,0,1)


# ###### Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[46]:


ary=np.array([0.5,.9,0.2,0.1,4,2.5,89,12,41,9])
np.where(ary<=0.5,0,ary)
ary


# ###### Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[48]:


a=np.arange(12)
b=np.arange(12)
a
b


# ###### names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# ###### Find the unique names and sort them out!

# In[51]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
uniq_names=np.unique(names)
uniq_names


# ###### a = np.array([1,2,3,4,5]) b = np.array([5,6,7,8,9])
# ###### From array a remove all items present in array b

# In[52]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
c=np.setdiff1d(a, b)
c


# ###### Following is the input NumPy array delete column two and insert following new column in its place.
# ###### sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]])
# 
# ###### newColumn = numpy.array([[10,10,10]])

# In[64]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = np.array([[10,10,10]])
result=np.delete(sampleArray, 1, axis=1) 
result=np.insert(sampleArray, 1, newColumn, axis=1)
result


# ###### x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# ###### Find the dot product of the above two matrix

# In[62]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
result=np.dot(x,y)
result


# ###### Generate a matrix of 20 random values and find its cumulative sum

# In[60]:


matrix=np.random.random((20))
cumulative_sum=matrix.sum()
cumulative_sum

