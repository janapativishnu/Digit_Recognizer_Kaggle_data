"""
Title: Kaggle Digit Recognition data (using scikit-Learn)

@author: Vishnuvardhan Janapati
"""

#import matplotlib.pyplot as plt                             # enable when platting image
import numpy as np
import pandas as pd
from sklearn import svm, preprocessing,
#from sklearn import cross_validation    # enable when testing
#import time

# loading training data and scaling
ImageId=pd.read_csv('sample_submission.csv')
ImageId=ImageId['ImageId']
digits=pd.read_csv('train.csv')
y=digits['label']
X=digits.drop(['label'],1)
X=preprocessing.scale(X)


# loading test data and scaling
digits_test=pd.read_csv('test.csv')
digits_test=preprocessing.scale(digits_test)


## ------------ Test model parameters and optimize for better performance before testing with unlabel data 
#clf=svm.SVC(gamma=0.001,C=100,kernel='linear')
#test_size=2000 # This is just for test purpose only. The complete data set is huge and takes a longer time.
#X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
#clf.fit(X_train[:test_size],y_train[:test_size])
#print("Accuracy of classifying training set is " + str(clf.score(X_train[:100],y_train[:100])*100) + " percent")
#print("Accuracy of classifying test set is " + str(clf.score(X_test[:100],y_test[:100])*100) + " percent")
#
## for comparison
#for i in range(1,20):
##    plt.imshow(np.reshape(digits_test[i,:],(28,28)))
#    if i==1:
#        print("         Actual digits          Predicted digit       ")
#    print("              ", str(y_test.values[i]) ,"                    " , str(clf.predict(np.reshape(X_test[i,:],(1,-1)))))
    



## ------------ Train with given label data (X,y), and test model accuracy with unlabel data 
clf=svm.SVC(gamma=0.001,C=100,kernel='linear')
clf.fit(X,y)





# prediction and writing output to a csv file

predictions=clf.predict(digits_test)
submission=pd.DataFrame({'ImageId':ImageId,'Label':predictions})
submission.to_csv('submission.csv',index=False)
