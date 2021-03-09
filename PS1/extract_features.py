import os
import pandas as pd
import numpy as np
import imageio
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

img_input = keras.Input(shape=(400,400,3))

base_model = VGG16(weights="imagenet",
                  include_top=False,
                  input_tensor = img_input,
                   pooling="max"
                  )
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('back').output)
all_images = []
img_folders = os.listdir('./google_image/')
img_folders.remove(".DS_Store")
for i in img_folders:
    folder_images = ["./google_image/"+i+"/"+j for j in os.listdir("./google_image/"+i)]
    all_images = all_images+folder_images

#test_x = preprocess_input(np.expand_dims(np.array(imageio.imread(all_images[0],pilmode="RGB")),axis=0))
#test = base_model.predict(test_x)
#print(test.shape)

features = []
for i in range(0,len(all_images)//1000+1):
    subset_end = min(len(all_images),1000*(i+1))
    print(subset_end)
    img_subset = [np.array(imageio.imread(all_images[j],pilmode="RGB")) for j in range(i*1000,subset_end)]
    img_subset = np.array(img_subset)
    img_subset = preprocess_input(img_subset)
    features_subset = base_model.predict(img_subset)
    features.append(features_subset)
    features_subset = np.squeeze(features_subset)
    np.savetxt("./output/google_image_features_"+str(i*1000)+"_"+str(subset_end)+"_cnn.csv",features_subset, delimiter = ",")

#features = np.array(features)
#features = np.squeeze(features)
#np.savetxt("./output/google_image_features_cnn.csv",features,delimiter = ",")

all_images = pd.DataFrame(all_images)
all_images.to_csv("./output/all_images.csv")
