import numpy as np
import pandas as pd

from PIL import Image, ImageTk
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from sklearn.metrics.pairwise import cosine_distances, pairwise_distances, cosine_similarity
from tensorflow.keras.preprocessing import image as keras_image


class Predictor:
    def __init__(self, embeddings_path, images_path):
        self.embeddings_path = embeddings_path
        self.embeddings = pd.read_csv(embeddings_path)
        self.model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
        self.similar_count = 5
        self.images_path = images_path
        self.image_fields = ['image']

    def get_image_embeddings(self, img):
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        model_input = preprocess_input(img_array)
        preds = self.model.predict(model_input)
        curr_image_df = pd.DataFrame(preds, columns=self.embeddings.columns[:-1])
        return curr_image_df

    def find_similar_images(self, img):
        self.embeddings = pd.read_csv(self.embeddings_path)

        curr_image_df = self.get_image_embeddings(img)
        curr_image_df['image'] = 'temp_image'
        temp_emmbeddings = pd.concat([self.embeddings, curr_image_df], ignore_index=True)
        curr_index = temp_emmbeddings[temp_emmbeddings['image'] == 'temp_image'].index[0]

        cosine_similarity_df = pd.DataFrame(cosine_similarity(temp_emmbeddings.drop('image', axis=1)))
        closest_images = pd.DataFrame(cosine_similarity_df.iloc[curr_index].nlargest(self.similar_count + 1)[1:])
        closest_images_path = [temp_emmbeddings.iloc[index]['image'] for index, imgs in closest_images.iterrows()]
        return closest_images_path