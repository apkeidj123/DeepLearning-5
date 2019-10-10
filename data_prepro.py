import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time

#examples_to_show = 10
#k = 30

t00 = time.time()
dfb = pd.read_csv("emnist-balanced-train_new.csv")  ## 0~46 (0~9:0~9, A~Z:10~35)
#dfl = pd.read_csv("emnist-letters-train_new.csv")   ## 1~26

#print(dfb[:5])
#print(dfl[:5])
t01 = time.time()
print("Load Time: ", t01 - t00)


count = 0
row = dfb.shape[0]
#row = test.shape[0]

t02 = time.time()
for i in dfb['label']:    
    if i > 35 or i < 10:
        #test.drop(count, axis = 0, inplace = True) 
        dfb.drop(count, axis = 0, inplace = True)  
        row -= 1
    count +=1
    if count % 1000 == 0:
        t04 = time.time()
        print('Count : ', count, 'Time: ', t04 - t02)

t03 = time.time() 
print("Process Time: ", t03 - t02)

#"""
y_train = dfb['label'].values
y_train = y_train.reshape(row,1)
dfb.drop(['label'], axis = 1, inplace = True)
x_train = dfb.values

np.save("x_train", x_train)
np.save("y_train", y_train)
#"""

print(dfb[:20])

#test = dfb[:1000].copy()
#print(test)

"""
print(test[:10])

y_train = test['label'].values
y_train = y_train.reshape(row, 1)
test.drop(['label'], axis = 1, inplace = True)
x_train = test.values
"""

"""
test2 = dfl[:100].copy()

label_test2 =test2['label'].values
label_test2 = label_test2.reshape(100,1)
test2.drop(['label'], axis=1,inplace=True)
img2 = test2.values
"""

#print(label_test[k:examples_to_show + k].reshape(1,examples_to_show ))
#print(label_test2[k:examples_to_show +k].reshape(1,examples_to_show ))


"""
plt.figure()

f, a = plt.subplots(2, 10, figsize=(10, 2))
#f2, a2 = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):  
    a[0][i].imshow(np.reshape(img[i+k], (28, 28)))    
    #a2[0][i].imshow(np.reshape(img2[i+k], (28, 28)))

plt.show()
"""

