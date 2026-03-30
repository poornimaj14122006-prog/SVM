import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
hours=np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
result=np.array([0,0,0,0,1,1,1,1])
model=svm.SVC(kernel="linear")
model.fit(hours,result)
prediction=model.predict([[5]])
print("prediction for 5 study hours:",prediction)
plt.scatter(hours,result)
plt.show()
