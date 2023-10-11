import numpy as np
import pandas as pd

a = np.arange(1,101,1)
print(a)

print(pd.cut(a,bins=[0,25,50,75,np.nan] , labels=["0_25","25_50","50_75","75_100"]))