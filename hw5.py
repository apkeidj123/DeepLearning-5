import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import time


t00 = time.time()
y_train = np.load('y_train.npy')
x_train = np.load('x_train.npy')
x_train = x_train / 255

t01 = time.time()
print('Load Time: ', t01 - t00)

tf.reset_default_graph()

# Parameters
LR = 1e-3              #learning_rate
training_epochs = 100
batch_size = 128
examples_to_show = 12
keep_rate = 0.75
n_examples = x_train.shape[0]
calc_iter = 500
alpha = 1e-3
latent_dim = 50

def get_batch(inputs, n_examples, batch_size):          
    #indices = np.random.choice(n_examples, n_examples, replace = False) # random indices
    
    for batch_i in range(n_examples // batch_size): # 25000 // 8
        start = batch_i * batch_size
        end = start + batch_size       
        batch_xs = inputs[start:end]
        #batch_ys = targets[start:end]

        yield batch_xs#, batch_ys

def random_batch(inputs, n_examples, batch_size):          
    indices = np.random.choice(n_examples, n_examples, replace = False)
    
    for batch_i in range(n_examples // batch_size): # 7411 // 2
        start = batch_i * batch_size
        end = start + batch_size       
        batch_xs = inputs[indices[start:end]]
        #batch_ys = targets[indices[start:end]]

        yield batch_xs#, batch_ys
        
import skimage
def add_noise(image):
    img = skimage.util.random_noise(image, mode = 'gaussian')
    return img
 
# Network Parameters
n_input = 784  # EMNIST data input (img shape: 28*28)    
    
#"""       
# tf Graph input (only pictures)
xs = tf.placeholder(tf.float32, [None, n_input])
xs_noise = tf.placeholder(tf.float32, [None, n_input])
#keep_prob = tf.placeholder(tf.float32) 

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape = shape, stddev = 1. / tf.sqrt(shape[0] / 2.))

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)

### hidden layer settings
n_hidden_1 = 512 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features

"""
weights = {
    'encoder_h1': weight_variable([n_input, n_hidden_1]),
    'encoder_h2': weight_variable([n_hidden_1, n_hidden_2]),
    'z_mean': weight_variable([n_hidden_2, latent_dim]),
    'z_std': weight_variable([n_hidden_2, latent_dim]),     
    'decoder_h1': weight_variable([latent_dim, n_hidden_1]),
    'decoder_h2': weight_variable([n_hidden_1, n_input]),
}
biases = {
    'encoder_b1': bias_variable([n_hidden_1]),
    'encoder_b2': bias_variable([n_hidden_2]),
    'z_mean': bias_variable([latent_dim]),
    'z_std': bias_variable([latent_dim]),    
    'decoder_b1': bias_variable([n_hidden_1]),
    'decoder_b2': bias_variable([n_input]),
}
"""

#"""
weights = {
    'encoder_h1': tf.Variable(glorot_init([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(glorot_init([n_hidden_1, n_hidden_2])),
    'z_mean': tf.Variable(glorot_init([n_hidden_2, latent_dim])),
    'z_std': tf.Variable(glorot_init([n_hidden_2, latent_dim])),     
    'decoder_h1': tf.Variable(glorot_init([latent_dim, n_hidden_1])),
    'decoder_h2': tf.Variable(glorot_init([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(glorot_init([n_hidden_1])),
    'encoder_b2': tf.Variable(glorot_init([n_hidden_2])),
    'z_mean': tf.Variable(glorot_init([latent_dim])),
    'z_std': tf.Variable(glorot_init([latent_dim])),    
    'decoder_b1': tf.Variable(glorot_init([n_hidden_1])),
    'decoder_b2': tf.Variable(glorot_init([n_input])),
}
#"""

# Building the encoder

# Encoder Hidden layer with sigmoid activation #1
#encoder_1 = tf.nn.sigmoid(tf.add(tf.matmul(xs, weights['encoder_h1']),
encoder_1 = tf.nn.sigmoid(tf.add(tf.matmul(xs_noise, weights['encoder_h1']),                                 
                                   biases['encoder_b1']))
L2_enco1 = tf.contrib.layers.l2_regularizer(alpha)(weights['encoder_h1']) 
#L2_enco1 = tf.nn.l2_loss(weights['encoder_h1']) 


# Decoder Hidden layer with sigmoid activation #2
encoder_2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
L2_enco2 = tf.contrib.layers.l2_regularizer(alpha)(weights['encoder_h2'])
#L2_enco2 = tf.nn.l2_loss(weights['encoder_h2'])    

z_mean = tf.matmul(encoder_2, weights['z_mean']) + biases['z_mean']
L2_zme = tf.contrib.layers.l2_regularizer(alpha)(weights['z_mean'])
#L2_zme = tf.nn.l2_loss(weights['z_mean'])

z_std = tf.matmul(encoder_2, weights['z_std']) + biases['z_std']
L2_zst = tf.contrib.layers.l2_regularizer(alpha)(weights['z_std'])
#L2_zst = tf.nn.l2_loss(weights['z_std']) 
   
# Sampler: Normal (gaussian) random distribution
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')
    
z = z_mean + tf.exp(z_std / 2) * eps
#z = z_mean + tf.sqrt(tf.exp(z_std)) * eps 

# Building the decoder

# Encoder Hidden layer with sigmoid activation #1
decoder_1 = tf.nn.sigmoid(tf.add(tf.matmul(z, weights['decoder_h1']),
                                   biases['decoder_b1']))
L2_deco1 = tf.contrib.layers.l2_regularizer(alpha)(weights['decoder_h1'])
#L2_deco1 = tf.nn.l2_loss(weights['decoder_h1'])    

# Decoder Hidden layer with sigmoid activation #2
decoder_2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
L2_deco2 = tf.contrib.layers.l2_regularizer(alpha)(weights['decoder_h2'])    
#L2_deco2 = tf.nn.l2_loss(weights['decoder_h2'])  #trash

# Define VAE Loss
def vae_loss(x_reconstructed, x_true):
    # Reconstruction loss
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                         + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)

# Construct model

# Prediction
y_pred = decoder_2
# Targets (Labels) are the input data.
y_true = xs


#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_pred, labels = y_true))

L2_Reg = L2_enco1 + L2_enco2 + L2_zme + L2_zst + L2_deco1 + L2_deco2

#total_loss = cross_entropy + L2_Reg

#loss = compute_loss(y_pred, y_true)
loss = vae_loss(y_pred, y_true)
#optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

total_loss = loss + L2_Reg
optimizer = tf.train.AdamOptimizer(LR).minimize(total_loss)



###--------------Start Training--------------------
train_loss = np.zeros([training_epochs,1])
lower_bound = np.zeros([training_epochs,1])

init = tf.global_variables_initializer()

save_path = 'checkpoints/dev'

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print('Start Training')
    tStart3 = time.time()
    for epoch in range(training_epochs):    
        print('Epoch: ', epoch + 1)
        ###----------Loss--------------
        loss_train = 0.0
        L2_loss1 = 0.0
        L2_loss2 = 0.0
        
        tStart = time.time()
        
        for batch_xs in random_batch(x_train, n_examples, batch_size): # 7411            
            sess.run(optimizer,feed_dict={xs: batch_xs,  #})
                                          xs_noise: add_noise(batch_xs)})
            
        tEnd = time.time()
        
        print("Training Time = ", tEnd - tStart)
        
        tStart2 = time.time()
        
        """
        for i in range(calc_iter):
            random_index_train = np.random.choice(y_train.shape[0], 2, replace = False)
            Loss = sess.run(loss, feed_dict={xs: x_train[random_index_train],   #})
                                             xs_noise: add_noise(x_train[random_index_train])})
            loss_train += Loss
        
        tEnd2 = time.time()
        print("Calculate Time = ", tEnd2 - tStart2)
        
        bound_lower = -loss_train/(calc_iter * n_examples)
        
        print('Train Loss: ', loss_train/calc_iter)
        print('Lower_bound: ', bound_lower)
        
        train_loss[epoch]= loss_train/calc_iter       
        lower_bound[epoch] = bound_lower
        """
        
        ### L2
        #"""
        for i in range(calc_iter):
            random_index_train = np.random.choice(y_train.shape[0], 2, replace = False)
            Loss, l2_loss1, l2_loss2 = sess.run([total_loss, loss, L2_Reg],
                                 feed_dict={xs: x_train[random_index_train],   #})
                                            xs_noise: add_noise(x_train[random_index_train])})
            L2_loss1 += l2_loss1
            L2_loss2 += l2_loss2
            loss_train += Loss
        
        tEnd2 = time.time()
        print("Calculate Time = ", tEnd2 - tStart2)
        
        print('Train Loss: ', loss_train/calc_iter)
        #print('L2_loss1: ', L2_loss1/calc_iter)
        #print('L2_loss2: ', L2_loss2/calc_iter)
        bound_lower = -loss_train/(calc_iter * n_examples)    
        print('Lower_bound: ', bound_lower)
              
        train_loss[epoch] = loss_train/calc_iter 
        
        lower_bound[epoch] = bound_lower
        #"""
        
    tEnd3 = time.time()
    print("Total Training Time = ", tEnd3 - tStart3)
       
    
    ### 5. Latent Space 
    
    #"""
    if latent_dim == 2 :
        n = 20
        x = np.linspace(-3, 3, n)
        y = np.linspace(-3, 3, n)
        h = 28
        w = 28
        
        I_latent = np.empty((h * n, w * n))
        for i, yi in enumerate(x):
            for j, xi in enumerate(y):
                zx = np.array([[xi, yi]] * batch_size)
                x_hat = sess.run(decoder_2, feed_dict = {z: zx})
                I_latent[(n-i-1) * h : (n-i) * h,
                         j * w : (j+1) * w] = x_hat[0].reshape(h, w)
        
        plt.figure(figsize=(8, 8))        
        plt.imshow(I_latent, cmap="gray")
        plt.savefig('output/Latent_space.jpg')
    #"""
    
    ### 6. Generate Images
    randoms = [np.random.normal(0, 1, latent_dim) for _ in range(examples_to_show)]
    imgs = sess.run(y_pred, feed_dict = {z: randoms})
    imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]
    
    """
    for img in imgs:
        plt.figure(figsize=(1,1))
        plt.axis('off')
        plt.imshow(img, cmap='gray')
    """
       
    ### 4. Plot original image & reconstructed image
    #"""
    encode_decode = sess.run(
        y_pred, feed_dict={xs: x_train[:examples_to_show],   #})
                           xs_noise: add_noise(x_train[:examples_to_show])})
    
    plt.figure()
    
    f, a = plt.subplots(4, examples_to_show, figsize=(examples_to_show, 4))    
    for i in range(examples_to_show):  
        a[0][i].imshow(np.reshape(x_train[i], (28, 28)), cmap='gray') 
        a[1][i].imshow(np.reshape(add_noise(x_train[i]), (28, 28)), cmap='gray')
        a[2][i].imshow(np.reshape(encode_decode[i], (28, 28)), cmap='gray')
        a[3][i].imshow(imgs[i], cmap='gray')
        plt.savefig('output/ALL.jpg')
    
    plt.show()
    #"""
    
    #"""
    
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
    
    ###----------Save result-------- -
 
    np.save("result/Loss_train", train_loss)
    np.save("result/Lower_bound", lower_bound)
    #"""
    
#"""
### Save Parameters

def save_params(params):
    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)


def load_params():
    with open('params.p', mode='rb') as in_file:
        return pickle.load(in_file)

# Save parameters for checkpoint
save_params(save_path)
#"""     

#"""

###------ 3. Plot------
### Loss
plt.figure()
y_range = range(0, training_epochs)       

plt.plot(y_range, train_loss, color='blue', label="training loss")   
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(loc='best')       
plt.savefig('output/Loss.jpg')
plt.show()
plt.clf()

plt.figure()
y_range = range(0, training_epochs)       

plt.plot(y_range, lower_bound, color='blue', label="train")   
plt.xlabel('epoch')
plt.ylabel('lower bound')
plt.title('Learning curve')
plt.legend(loc='best')       
plt.savefig('output/Lower_bound.jpg')
plt.show()
plt.clf()
#"""

