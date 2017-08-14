
# coding: utf-8

# In[2]:

import tensorflow as tf
import numpy as np


# In[4]:

data = np.loadtxt("C:/Users/abc/Desktop/data.csv",delimiter = ",",unpack=True,dtype='float32')


# In[6]:

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])


# In[8]:

global_step = tf.Variable(0,trainable=False,name="global_step")


# In[9]:

X = tf.placeholder(tf.float32)
Y= tf.placeholder(tf.float32)


# In[13]:

W1 = tf.Variable(tf.random_uniform([2,10],-1.,1.))
L1 = tf.nn.relu(tf.matmul(X,W1))


# In[14]:

W2 = tf.Variable(tf.random_uniform([10,20],-1.,1.))
L2 = tf.nn.relu(tf.matmul(L1,W2))


# In[15]:

W3 = tf.Variable(tf.random_uniform([20,3],-1.,1.))
model = tf.matmul(L2,W3)


# In[16]:

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=model))


# In[18]:

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)


# In[19]:

train_op=optimizer.minimize(cost,global_step=global_step)


# In[20]:

sess = tf.Session()


# In[23]:

saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state("./model")
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
    


# In[26]:

for step in range(2):
    sess.run(train_op,feed_dict={X:x_data,Y:y_data})
    
    print("Step:%d"%sess.run(global_step),
         'Cost:%.3f' %sess.run(cost,feed_dict={X:x_data,Y:y_data}))


# In[28]:

saver.save(sess,'./modle/dnn.ckpt',global_step=global_step)


# In[29]:

prediction = tf.argmax(model,1)
target = tf.argmax(Y,1)


# In[30]:

print("예측값 :",sess.run(prediction,feed_dict={X:x_data}))


# In[31]:

print("실제값:",sess.run(target,feed_dict={Y:y_data}))


# In[32]:

check_prediction = tf.equal(prediction,target)


# In[33]:

accuracy = tf.reduce_mean(tf.cast(check_prediction,tf.float32))


# In[34]:

print("정확도:%.2f"%sess.run(accuracy*100,feed_dict={X:x_data,Y:y_data}))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



