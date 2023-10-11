import numpy as np

a = np.array([1,2,3,4])
print(type(a))
b = np.array([5,6,7,8])
c = np.random.randint(0,10,size = 10)
print(c)
d = np.random.normal(10,4,(3,4))
print(d)
zeros = np.zeros(10,dtype=int)
print(zeros)

#numpy features
print(d.ndim)
print(d.shape)
print(d.size)
print(d.dtype)

#reshaping
d = d.reshape(2,6)
print(d)
#indexing
a = np.random.randint(10,size=5)
print(a[2])
print(a[0:3]) # slicing
b = np.random.normal(10,5,(5,5))
print(b)
print(b[2,3])
print(b[0:2,0:3])
#fancy index
a = np.arange(0,30,3)
print(a[[0,5,7,2]])
#conditional operations
print(a[a<15])
print(a[a==15])
print(a[a!=15])
print(a[a<=15])
print(a[a>15])
#mathematical operations
print(a-5)
print(a*2)
#np operations
print(np.subtract(a,1)  )
print(np.add(a,1)  )
print(np.mean(a)  )
print(np.var(a)  )
#
print(a)
print(a[-3:-1])
print(a % 2 == 0)