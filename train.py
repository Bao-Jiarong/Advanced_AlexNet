'''
  Author       : Bao Jiarong
  Creation Date: 2020-08-12
  email        : bao.salirong@gmail.com
  Task         : Denoise Autoencoder Implementation
  Dataset      : MNIST Digits (0,1,...,9)
'''
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import random
import cv2
import loader
import conv_ae

# np.random.seed(7)
# tf.random.set_seed(7)
# np.set_printoptions(threshold=np.inf)

# Input/Ouptut Parameters
width      = 24 << 1
height     = 24 << 1
channel    = 3
model_name = "models/flowers/flowers"
data_path  = "../../data_img/flowers/train/"

# Step 0: Global Parameters
epochs     = 60
lr_rate    = 0.0001
batch_size = 32

# Step 1: Create Model
model = conv_ae.Conv_AE((None,height, width, channel), latent = 200, units=32)

if sys.argv[1] == "train":

    print(model.summary())
    # sys.exit()

    # Load weights:
    model.load_weights(model_name)

    # Step 3: Load data
    X_train, Y_train, X_valid, Y_valid = loader.load_light(data_path,width,height,True,0.8,False)
    # Define The Optimizer
    optimizer= tf.keras.optimizers.Adam(learning_rate=lr_rate)
    # Define The Loss
    #---------------------
    @tf.function
    def my_loss(y_true, y_pred):
        return tf.keras.losses.MSE(y_true=y_true, y_pred=y_pred)

    # Define The Metrics
    tr_loss = tf.keras.metrics.MeanSquaredError(name = 'tr_loss')
    va_loss = tf.keras.metrics.MeanSquaredError(name = 'va_loss')

    #---------------------
    @tf.function
    def train_step(X, Y_true):
        with tf.GradientTape() as tape:
            Y_pred = model(X, training=True)
            loss   = my_loss(y_true=Y_true, y_pred=Y_pred )
        gradients= tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        tr_loss.update_state(y_true = Y_true, y_pred = Y_pred )

    #---------------------
    @tf.function
    def valid_step(X, Y_true):
        Y_pred= model(X, training=False)
        loss  = my_loss(y_true=Y_true, y_pred=Y_pred)
        va_loss.update_state(y_true = Y_true, y_pred = Y_pred)

    #---------------------
    # start training
    L = len(X_train)
    M = len(X_valid)
    steps  = int(L/batch_size)
    steps1 = int(M/batch_size)

    for epoch in range(epochs):
        # Run on training data + Update weights
        for step in range(steps):
            noisy_imgs, imgs = loader.get_batch_light(X_train, X_train, batch_size, width, height)
            train_step(noisy_imgs,imgs)

            print(epoch,"/",epochs,step,steps,
                  "loss:",tr_loss.result().numpy(),end="\r")

        # Run on validation data without updating weights
        for step in range(steps1):
            noisy_images, images_ = loader.get_batch_light(X_valid, X_valid, batch_size, width, height)
            valid_step(noisy_images, images_)

        print(epoch,"/",epochs,step,steps,
              "loss:",tr_loss.result().numpy(),
              "val_loss:",va_loss.result().numpy())

        # Save the model for each epoch
        model.save_weights(filepath=model_name, save_format='tf')

elif sys.argv[1] == "predict":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Prepare the input
    img    = cv2.imread(sys.argv[2])
    image  = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
    image  = loader.scaling_tech(image,method="normalization")
    image  = loader.noisy("s_and_p",image)
    images = np.array([image])


    # Step 5: Predict the class
    preds = my_model.predict(images)
    preds = (preds[0] - preds.min())/(preds.max() - preds.min())
    images = np.hstack((images[0],preds))
    images = cv2.resize(images,(width*4,height*2))
    cv2.imshow("imgs",images)
    cv2.waitKey(0)
elif sys.argv[1] == "predict_all":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Prepare the input
    # imgs_filenames = ["../../data_img/MNIST/test/img_2.jpg" , # 0
    #                   "../../data_img/MNIST/test/img_18.jpg", # 1
    #                   "../../data_img/MNIST/test/img_1.jpg" , # 2
    #                   "../../data_img/MNIST/test/img_5.jpg" , # 3
    #                   "../../data_img/MNIST/test/img_13.jpg", # 4
    #                   "../../data_img/MNIST/test/img_11.jpg", # 5
    #                   "../../data_img/MNIST/test/img_35.jpg", # 6
    #                   "../../data_img/MNIST/test/img_6.jpg" , # 7
    #                   "../../data_img/MNIST/test/img_45.jpg", # 8
    #                   "../../data_img/MNIST/test/img_3.jpg" ] # 9

    # imgs_filenames = [os.path.abspath(d) for d in os.listdir("../../data_img/flowers/test/daisy/")[10:12]]
    imgs_filenames = sorted([os.path.join("../../data_img/flowers/test/daisy/", file)
                             for file in os.listdir("../../data_img/flowers/test/daisy/")],
                             key=os.path.getctime)[2:12]
    # imgs_filenames.extend([os.path.abspath(d) for d in os.listdir("../../data_img/flowers/test/dandelion/")[10:12]])
    # imgs_filenames.extend([os.path.abspath(d) for d in os.listdir("../../data_img/flowers/test/rose/")[10:12]])
    # imgs_filenames.extend([os.path.abspath(d) for d in os.listdir("../../data_img/flowers/test/sunflower/")[10:12]])
    # imgs_filenames.extend([os.path.abspath(d) for d in os.listdir("../../data_img/flowers/test/tulip/")[10:12]])
    # print(imgs_filenames)

    images = []
    for filename in imgs_filenames:
        img = cv2.imread(filename)
        img = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
        img = loader.scaling_tech(img,method="normalization")
        method = np.random.choice(["gaussian","s_and_p","poisson","speckle"])
        noisy_img = loader.noisy(method, img)
        images.append(noisy_img)

    # True images
    images = np.array(images)

    # Predicted images
    preds = my_model.predict(images)
    preds = (preds - preds.min())/(preds.max() - preds.min())


    true_images = np.hstack(images)
    pred_images = np.hstack(preds)

    images = np.vstack((true_images, pred_images))
    h = images.shape[0]
    w = images.shape[1]
    images = cv2.resize(images,(w << 2, h << 2))

    cv2.imshow("imgs",images)
    cv2.waitKey(0)

elif sys.argv[1] == "test_noise":
    filename  = "../../data_img/MNIST/test/img_2.jpg"
    image = cv2.imread(filename)
    image = image / 255.0
    # image = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
    image_ = loader.noisy("gaussian",image)
    # image_ = loader.noisy("s_and_p",image)
    # image_ = loader.noisy("poisson",image)
    # image_ = loader.noisy("speckle",image)
    # images = np.hstack((image,image_))
    cv2.imshow("noisy_imgs",image_)
    cv2.waitKey(0)
