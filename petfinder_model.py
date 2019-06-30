# Libraries
import numpy as np
import pandas as pd
import gc
import os
import random
import glob
import pathlib
from joblib import Parallel, delayed
from PIL import Image
import json
import time
from functools import partial
import scipy as sp
from scipy import stats
from scipy.stats import kurtosis
from scipy.stats import skew

from math import sqrt
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from collections import Counter, defaultdict

import re
import string
from nltk.corpus import stopwords

import cv2
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.applications.densenet import preprocess_input, DenseNet121
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
from keras.layers import Dense, Dropout
import torch

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings("ignore")


start_time = time.time()

BATCH_SIZE = 256

os.listdir('../input/petfinder-adoption-prediction/test/')

# Load input file
train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
sample_submission = pd.read_csv(
    '../input/petfinder-adoption-prediction/test/sample_submission.csv')
labels_breed = pd.read_csv(
    '../input/petfinder-adoption-prediction/breed_labels.csv')
labels_color = pd.read_csv(
    '../input/petfinder-adoption-prediction/color_labels.csv')
labels_state = pd.read_csv(
    '../input/petfinder-adoption-prediction/state_labels.csv')

# Additional file
train_image_files = sorted(glob.glob(
    '../input/petfinder-adoption-prediction/train_images/*.jpg'))
train_metadata_files = sorted(glob.glob(
    '../input/petfinder-adoption-prediction/train_metadata/*.json'))
train_sentiment_files = sorted(glob.glob(
    '../input/petfinder-adoption-prediction/train_sentiment/*.json'))
print('num of train images files: {}'.format(len(train_image_files)))
print('num of train metadata files: {}'.format(len(train_metadata_files)))
print('num of train sentiment files: {}'.format(len(train_sentiment_files)))
test_image_files = sorted(glob.glob(
    '../input/petfinder-adoption-prediction/test_images/*.jpg'))
test_metadata_files = sorted(glob.glob(
    '../input/petfinder-adoption-prediction/test_metadata/*.json'))
test_sentiment_files = sorted(glob.glob(
    '../input/petfinder-adoption-prediction/test_sentiment/*.json'))
print('num of test images files: {}'.format(len(test_image_files)))
print('num of test metadata files: {}'.format(len(test_metadata_files)))
print('num of test sentiment files: {}'.format(len(test_sentiment_files)))

train_id = train['PetID']
test_id = test['PetID']
train_image_path = "../input/petfinder-adoption-prediction/train_images/"
test_image_path = "../input/petfinder-adoption-prediction/test_images/"
densenet_weights = "../input/densenet-keras/DenseNet-BC-121-32-no-top.h5"
mobilenet_weights = \
    "../input/keras-pretrain-model-weights/mobilenet_1_0_224_tf_no_top.h5"
nima_weights = \
    "../input/titu1994neuralimageassessment/inception_resnet_weights.h5"
train_image_path2 = pathlib.Path(
    '../input/petfinder-adoption-prediction/train_images')
test_image_path2 = pathlib.Path(
    '../input/petfinder-adoption-prediction/test_images')
nima_weights_3 = \
    '../input/titu1994neuralimageassessment/neural-image-assessment-master/neural-image-assessment-master/weights/mobilenet_weights.h5'

state_gdp = {
    41336: 116679,
    41325: 40596,
    41367: 23020,
    41401: 190075,
    41415: 5984,
    41324: 37274,
    41332: 42389,
    41335: 52452,
    41330: 67629,
    41380: 5642,
    41327: 81284,
    41345: 80167,
    41342: 121414,
    41326: 280698,
    41361: 32270
}

state_area = {
    41336: 18987,
    41325: 9425,
    41367: 15024,
    41401: 243,
    41415: 92,
    41324: 1652,
    41332: 6644,
    41335: 35965,
    41330: 21005,
    41380: 795,
    41327: 1031,
    41345: 73619,
    41342: 124450,
    41326: 7960,
    41361: 12995
}

state_population = {
    41336: 2740625,
    41325: 1649756,
    41367: 1313014,
    41401: 1379310,
    41415: 76067,
    41324: 635791,
    41332: 859924,
    41335: 1288376,
    41330: 2051236,
    41380: 204450,
    41327: 1313449,
    41345: 2603485,
    41342: 2071506,
    41326: 4188876,
    41361: 898825
}

state_population_per_area = {
    41336: 144.342181,
    41325: 175.040424,
    41367: 87.394436,
    41401: 5676.172840,
    41415: 826.815217,
    41324: 384.861380,
    41332: 129.428657,
    41335: 35.823050,
    41330: 97.654654,
    41380: 257.169811,
    41327: 1273.956353,
    41345: 35.364308,
    41342: 16.645287,
    41326: 526.240704,
    41361: 69.380548
}

# newly added
# https://en.wikipedia.org/wiki/States_and_federal_territories_of_Malaysia
state_HDI = {
    41336: 0.785,
    41325: 0.769,
    41367: 0.741,
    41401: 0.822,
    41415: 0.742,
    41324: 0.794,
    41332: 0.789,
    41335: 0.766,
    41330: 0.778,
    41380: 0.767,
    41327: 0.803,
    41345: 0.674,
    41342: 0.709,
    41326: 0.819,
    41361: 0.762
}


# Common Function
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class PetFinderParser(object):

    def __init__(self, debug=False):

        self.debug = debug
        self.sentence_sep = ' '
        self.extract_sentiment_text = False

    def open_metadata_file(self, filename):
        """
        Load metadata file.
        """
        with open(filename, 'r') as f:
            metadata_file = json.load(f)
        return metadata_file

    def open_sentiment_file(self, filename):
        """
        Load sentiment file.
        """
        with open(filename, 'r') as f:
            sentiment_file = json.load(f)
        return sentiment_file

    def open_image_file(self, filename):
        """
        Load image file.
        """
        image = np.asarray(Image.open(filename))
        return image

    def parse_sentiment_file(self, file):
        """
        Parse sentiment file. Output DF with sentiment features.
        """

        file_sentiment = file['documentSentiment']
        file_entities = [x['name'] for x in file['entities']]
        file_entities = self.sentence_sep.join(file_entities)

        if self.extract_sentiment_text:
            file_sentences_text = [x['text']['content'] for x in
                                   file['sentences']]
            file_sentences_text = self.sentence_sep.join(file_sentences_text)
        file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]

        file_sentences_sentiment = pd.DataFrame.from_dict(
            file_sentences_sentiment, orient='columns').sum()
        file_sentences_sentiment = file_sentences_sentiment.add_prefix(
            'document_').to_dict()

        file_sentiment.update(file_sentences_sentiment)

        df_sentiment = pd.DataFrame.from_dict(file_sentiment, orient='index').T
        if self.extract_sentiment_text:
            df_sentiment['text'] = file_sentences_text

        df_sentiment['entities'] = file_entities
        df_sentiment = df_sentiment.add_prefix('sentiment_')

        return df_sentiment

    def parse_metadata_file(self, file):
        """
        Parse metadata file. Output DF with metadata features.
        """

        file_keys = list(file.keys())

        if 'labelAnnotations' in file_keys:
            file_annots = file['labelAnnotations']
            file_top_score = np.asarray(
                [x['score'] for x in file_annots]).mean()
            file_top_desc = [x['description'] for x in file_annots]
        else:
            file_top_score = np.nan
            file_top_desc = ['']

        file_colors = file['imagePropertiesAnnotation']['dominantColors'][
            'colors']
        file_crops = file['cropHintsAnnotation']['cropHints']

        file_color_score = np.asarray([x['score'] for x in file_colors]).mean()
        file_color_pixelfrac = np.asarray(
            [x['pixelFraction'] for x in file_colors]).mean()

        file_crop_conf = np.asarray(
            [x['confidence'] for x in file_crops]).mean()

        if 'importanceFraction' in file_crops[0].keys():
            file_crop_importance = np.asarray(
                [x['importanceFraction'] for x in file_crops]).mean()
        else:
            file_crop_importance = np.nan

        df_metadata = {
            'annots_score': file_top_score,
            'color_score': file_color_score,
            'color_pixelfrac': file_color_pixelfrac,
            'crop_conf': file_crop_conf,
            'crop_importance': file_crop_importance,
            'annots_top_desc': self.sentence_sep.join(file_top_desc)
        }

        df_metadata = pd.DataFrame.from_dict(df_metadata, orient='index').T
        df_metadata = df_metadata.add_prefix('metadata_')

        return df_metadata


# Helper function for parallel data processing:
def extract_additional_features(pet_id, mode='train'):
    sentiment_filename = '../input/petfinder-adoption-prediction/{}_sentiment/{}.json'.format(mode, pet_id)
    try:
        sentiment_file = pet_parser.open_sentiment_file(sentiment_filename)
        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)
        df_sentiment['PetID'] = pet_id
    except FileNotFoundError:
        df_sentiment = []

    dfs_metadata = []
    metadata_filenames = sorted(glob.glob(
        '../input/petfinder-adoption-prediction/{}_metadata/{}*.json'.format(
            mode, pet_id)))
    if len(metadata_filenames) > 0:
        for f in metadata_filenames:
            metadata_file = pet_parser.open_metadata_file(f)
            df_metadata = pet_parser.parse_metadata_file(metadata_file)
            df_metadata['PetID'] = pet_id
            dfs_metadata.append(df_metadata)
        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)
    dfs = [df_sentiment, dfs_metadata]

    return dfs


def load_meta_sentiment(train_proc, test_proc, debug=False):
    # Images:
    train_df_ids = train_proc[['PetID']]
    print('Train: ', train_df_ids.shape)

    train_df_imgs = pd.DataFrame(train_image_files)
    train_df_imgs.columns = ['image_filename']
    train_imgs_pets = train_df_imgs['image_filename'].apply(
        lambda x: x.split('/')[-1].split('-')[0])
    train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)
    print(len(train_imgs_pets.unique()))

    pets_with_images = len(np.intersect1d(train_imgs_pets.unique(),
                                          train_df_ids['PetID'].unique()))
    print('fraction of pets with images: {:.3f}'.format(
        pets_with_images / train_df_ids.shape[0]))

    # Metadata:
    train_df_ids = train_proc[['PetID']]
    train_df_metadata = pd.DataFrame(train_metadata_files)
    train_df_metadata.columns = ['metadata_filename']
    train_metadata_pets = train_df_metadata['metadata_filename'].apply(
        lambda x: x.split('/')[-1].split('-')[0])
    print(len(train_metadata_pets.unique()))

    pets_with_metadatas = len(np.intersect1d(train_metadata_pets.unique(),
                                             train_df_ids['PetID'].unique()))
    print('fraction of pets with metadata: {:.3f}'.format(
        pets_with_metadatas / train_df_ids.shape[0]))

    # Sentiment:
    train_df_ids = train_proc[['PetID']]
    train_df_sentiment = pd.DataFrame(train_sentiment_files)
    train_df_sentiment.columns = ['sentiment_filename']
    train_sentiment_pets = train_df_sentiment['sentiment_filename'].apply(
        lambda x: x.split('/')[-1].split('.')[0])
    print(len(train_sentiment_pets.unique()))

    pets_with_sentiments = len(np.intersect1d(train_sentiment_pets.unique(),
                                              train_df_ids['PetID'].unique()))
    print('fraction of pets with sentiment: {:.3f}'.format(
        pets_with_sentiments / train_df_ids.shape[0]))

    # Images:
    test_df_ids = test_proc[['PetID']]
    print('Test: ', test_df_ids.shape)

    test_df_imgs = pd.DataFrame(test_image_files)
    test_df_imgs.columns = ['image_filename']
    test_imgs_pets = test_df_imgs['image_filename'].apply(
        lambda x: x.split('/')[-1].split('-')[0])
    test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)
    print(len(test_imgs_pets.unique()))

    pets_with_images = len(
        np.intersect1d(test_imgs_pets.unique(), test_df_ids['PetID'].unique()))
    print('fraction of pets with images: {:.3f}'.format(
        pets_with_images / test_df_ids.shape[0]))

    # Metadata:
    test_df_ids = test_proc[['PetID']]
    test_df_metadata = pd.DataFrame(test_metadata_files)
    test_df_metadata.columns = ['metadata_filename']
    test_metadata_pets = test_df_metadata['metadata_filename'].apply(
        lambda x: x.split('/')[-1].split('-')[0])
    print(len(test_metadata_pets.unique()))

    pets_with_metadatas = len(np.intersect1d(test_metadata_pets.unique(),
                                             test_df_ids['PetID'].unique()))
    print('fraction of pets with metadata: {:.3f}'.format(
        pets_with_metadatas / test_df_ids.shape[0]))

    # Sentiment:
    test_df_ids = test_proc[['PetID']]
    test_df_sentiment = pd.DataFrame(test_sentiment_files)
    test_df_sentiment.columns = ['sentiment_filename']
    test_sentiment_pets = test_df_sentiment['sentiment_filename'].apply(
        lambda x: x.split('/')[-1].split('.')[0])
    print(len(test_sentiment_pets.unique()))

    pets_with_sentiments = len(np.intersect1d(test_sentiment_pets.unique(),
                                              test_df_ids['PetID'].unique()))
    print('fraction of pets with sentiment: {:.3f}'.format(
        pets_with_sentiments / test_df_ids.shape[0]))

    # are distributions the same?
    print('images and metadata distributions the same? {}'.format(
        np.all(test_metadata_pets == test_imgs_pets)))

    # Unique IDs from train and test:
    train_pet_ids = train.PetID.unique()
    test_pet_ids = test.PetID.unique()

    if debug:
        train_pet_ids = train_pet_ids[:1000]
        test_pet_ids = test_pet_ids[:500]

    # Train set:
    # Parallel processing of data:
    dfs_train = Parallel(n_jobs=6, verbose=1)(
        delayed(extract_additional_features)(i, mode='train') for i in
        train_pet_ids)

    # Extract processed data and format them as DFs:
    train_dfs_sentiment = [x[0] for x in dfs_train if
                           isinstance(x[0], pd.DataFrame)]
    train_dfs_metadata = [x[1] for x in dfs_train if
                          isinstance(x[1], pd.DataFrame)]

    train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True,
                                    sort=False)
    train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True,
                                   sort=False)

    print(train_dfs_sentiment.shape, train_dfs_metadata.shape)

    # Test set:
    # Parallel processing of data:
    dfs_test = Parallel(n_jobs=6, verbose=1)(
        delayed(extract_additional_features)(i, mode='test') for i in
        test_pet_ids)

    # Extract processed data and format them as DFs:
    test_dfs_sentiment = [x[0] for x in dfs_test if
                          isinstance(x[0], pd.DataFrame)]
    test_dfs_metadata = [x[1] for x in dfs_test if
                         isinstance(x[1], pd.DataFrame)]

    test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True,
                                   sort=False)
    test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True,
                                  sort=False)

    print(test_dfs_sentiment.shape, test_dfs_metadata.shape)

    return train_df_imgs, train_dfs_sentiment, train_dfs_metadata, \
           test_df_imgs, test_dfs_sentiment, test_dfs_metadata


def load_image_color(train, test):
    print('Load image color')
    vertex_xs = []
    vertex_ys = []
    bounding_confidences = []
    bounding_importance_fracs = []
    dominant_blues = []
    dominant_greens = []
    dominant_reds = []
    dominant_pixel_fracs = []
    dominant_scores = []
    label_descriptions = []
    label_scores = []
    nf_count = 0
    nl_count = 0
    for pet in train_id:
        try:
            with open(
                    '../input/petfinder-adoption-prediction/train_metadata/' + pet + '-1.json',
                    'r') as f:
                data = json.load(f)
            vertex_x = \
            data['cropHintsAnnotation']['cropHints'][0]['boundingPoly'][
                'vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = \
            data['cropHintsAnnotation']['cropHints'][0]['boundingPoly'][
                'vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0][
                'confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = \
            data['cropHintsAnnotation']['cropHints'][0].get(
                'importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = \
            data['imagePropertiesAnnotation']['dominantColors']['colors'][0][
                'color']['blue']
            dominant_blues.append(dominant_blue)
            dominant_green = \
            data['imagePropertiesAnnotation']['dominantColors']['colors'][0][
                'color']['green']
            dominant_greens.append(dominant_green)
            dominant_red = \
            data['imagePropertiesAnnotation']['dominantColors']['colors'][0][
                'color']['red']
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = \
            data['imagePropertiesAnnotation']['dominantColors']['colors'][0][
                'pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = \
            data['imagePropertiesAnnotation']['dominantColors']['colors'][0][
                'score']
            dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                nl_count += 1
                label_descriptions.append('nothing')
                label_scores.append(-1)
        except FileNotFoundError:
            nf_count += 1
            vertex_xs.append(-1)
            vertex_ys.append(-1)
            bounding_confidences.append(-1)
            bounding_importance_fracs.append(-1)
            dominant_blues.append(-1)
            dominant_greens.append(-1)
            dominant_reds.append(-1)
            dominant_pixel_fracs.append(-1)
            dominant_scores.append(-1)
            label_descriptions.append('nothing')
            label_scores.append(-1)

    print(nf_count)
    print(nl_count)
    train.loc[:, 'vertex_x'] = vertex_xs
    train.loc[:, 'vertex_y'] = vertex_ys
    train.loc[:, 'bounding_confidence'] = bounding_confidences
    train.loc[:, 'bounding_importance'] = bounding_importance_fracs
    train.loc[:, 'dominant_blue'] = dominant_blues
    train.loc[:, 'dominant_green'] = dominant_greens
    train.loc[:, 'dominant_red'] = dominant_reds
    train.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
    train.loc[:, 'dominant_score'] = dominant_scores
    train.loc[:, 'label_description'] = label_descriptions
    train.loc[:, 'label_score'] = label_scores

    vertex_xs = []
    vertex_ys = []
    bounding_confidences = []
    bounding_importance_fracs = []
    dominant_blues = []
    dominant_greens = []
    dominant_reds = []
    dominant_pixel_fracs = []
    dominant_scores = []
    label_descriptions = []
    label_scores = []
    nf_count = 0
    nl_count = 0
    for pet in test_id:
        try:
            with open(
                    '../input/petfinder-adoption-prediction/test_metadata/' + pet + '-1.json',
                    'r') as f:
                data = json.load(f)
            vertex_x = \
            data['cropHintsAnnotation']['cropHints'][0]['boundingPoly'][
                'vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = \
            data['cropHintsAnnotation']['cropHints'][0]['boundingPoly'][
                'vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0][
                'confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = \
            data['cropHintsAnnotation']['cropHints'][0].get(
                'importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = \
            data['imagePropertiesAnnotation']['dominantColors']['colors'][0][
                'color']['blue']
            dominant_blues.append(dominant_blue)
            dominant_green = \
            data['imagePropertiesAnnotation']['dominantColors']['colors'][0][
                'color']['green']
            dominant_greens.append(dominant_green)
            dominant_red = \
            data['imagePropertiesAnnotation']['dominantColors']['colors'][0][
                'color']['red']
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = \
            data['imagePropertiesAnnotation']['dominantColors']['colors'][0][
                'pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = \
            data['imagePropertiesAnnotation']['dominantColors']['colors'][0][
                'score']
            dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                nl_count += 1
                label_descriptions.append('nothing')
                label_scores.append(-1)
        except FileNotFoundError:
            nf_count += 1
            vertex_xs.append(-1)
            vertex_ys.append(-1)
            bounding_confidences.append(-1)
            bounding_importance_fracs.append(-1)
            dominant_blues.append(-1)
            dominant_greens.append(-1)
            dominant_reds.append(-1)
            dominant_pixel_fracs.append(-1)
            dominant_scores.append(-1)
            label_descriptions.append('nothing')
            label_scores.append(-1)

    print(nf_count)
    test.loc[:, 'vertex_x'] = vertex_xs
    test.loc[:, 'vertex_y'] = vertex_ys
    test.loc[:, 'bounding_confidence'] = bounding_confidences
    test.loc[:, 'bounding_importance'] = bounding_importance_fracs
    test.loc[:, 'dominant_blue'] = dominant_blues
    test.loc[:, 'dominant_green'] = dominant_greens
    test.loc[:, 'dominant_red'] = dominant_reds
    test.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
    test.loc[:, 'dominant_score'] = dominant_scores
    test.loc[:, 'label_description'] = label_descriptions
    test.loc[:, 'label_score'] = label_scores

    return train, test


# Image DenseNet
def resize_to_square(im, IMG_SIZE):
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(IMG_SIZE) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = IMG_SIZE - new_size[1]
    delta_h = IMG_SIZE - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return new_im


def load_image(path):  # firstではない
    image = cv2.imread(str(path))
    return image


def make_densenet(weights):
    inp = Input((256, 256, 3))
    backbone = DenseNet121(input_tensor=inp,
                           weights=weights,
                           include_top=False)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
    x = AveragePooling1D(4)(x)
    out = Lambda(lambda x: x[:, :, 0])(x)

    return Model(inp, out)


def make_mobilenet(weights):
    inp = Input((224, 224, 3))
    backbone = MobileNet(input_tensor=inp,
                         weights=weights,
                         include_top=False)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
    x = AveragePooling1D(4)(x)
    out = Lambda(lambda x: x[:, :, 0])(x)

    return Model(inp, out)


def make_nima(weights):
    backbone = InceptionResNetV2(input_shape=(None, None, 3),
                                 include_top=False,
                                 pooling='avg',
                                 weights=None)
    x = Dropout(0.75)(backbone.output)
    x = Dense(10, activation='softmax')(x)
    m = Model(backbone.input, x)
    m.load_weights(weights)
    return m


def make_nima3(weights):
    backbone = MobileNet((None, None, 3),
                         alpha=1,
                         include_top=False,
                         pooling='avg',
                         weights=None)
    x = Dropout(0.75)(backbone.output)
    x = Dense(10, activation='softmax')(x)
    m = Model(backbone.input, x)
    m.load_weights(weights)
    return m


def mean_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si, axis=1)
    return mean


def std_score(scores):
    mean = mean_score(scores)
    si = np.arange(1, 11, 1)
    std = np.sqrt(
        np.sum((si - mean.values.reshape(-1, 1))**2 * scores.values, axis=1))
    return std


def image_models_prediction(mode, dense_model, mobile_model, nima_model_3):
    print('Run Dense Net')
    IMG_SIZE = 256

    if mode is 'train':
        img_list = np.array(sorted(list(train_image_path2.glob("*.jpg"))))
        img_no_list = np.array(
            [p.name.split('.')[0].split('-')[1] for p in img_list])
        img_list_no1 = img_list[img_no_list == '1']
        img_list_oth = img_list[img_no_list != '1']
        n_batches_no1 = len(img_list_no1) // BATCH_SIZE + (
                len(img_list_no1) % BATCH_SIZE != 0)
        n_batches_oth = len(img_list_oth) // BATCH_SIZE + (
                len(img_list_oth) % BATCH_SIZE != 0)

    elif mode is 'test':
        img_list = np.array(sorted(list(test_image_path2.glob("*.jpg"))))
        img_no_list = np.array(
            [p.name.split('.')[0].split('-')[1] for p in img_list])
        img_list_no1 = img_list[img_no_list == '1']
        img_list_oth = img_list[img_no_list != '1']
        n_batches_no1 = len(img_list_no1) // BATCH_SIZE + (
                    len(img_list_no1) % BATCH_SIZE != 0)
        n_batches_oth = len(img_list_oth) // BATCH_SIZE + (
                len(img_list_oth) % BATCH_SIZE != 0)

    imageid_no1 = []
    blur_no1 = []
    dense_preds_no1 = {}
    mobile_preds_no1 = {}
    nima3_preds_no1 = {}

    DENSE_IMG_SIZE = 256
    MOBILE_IMG_SIZE = 224

    # no1
    for b in range(n_batches_no1):
        start = b * BATCH_SIZE
        end = (b + 1) * BATCH_SIZE
        batch_pets = img_list_no1[start:end]

        dense_batch_images = np.zeros(
            (len(batch_pets), DENSE_IMG_SIZE, DENSE_IMG_SIZE, 3))
        mobile_batch_images = np.zeros(
            (len(batch_pets), MOBILE_IMG_SIZE, MOBILE_IMG_SIZE, 3))
        for i, pet_id in enumerate(batch_pets):

            try:
                image = load_image(str(pet_id))

                # image quality (blur)
                imageid_no1.append(
                    str(pet_id).replace('.jpg', '').split('/')[-1])
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blur_no1.append(cv2.Laplacian(gray, cv2.CV_64F).var())

                # resize for dense
                dense_image = resize_to_square(image, DENSE_IMG_SIZE)
                dense_batch_images[i] = preprocess_input(dense_image)

                # resize for mobile, nima
                mobile_image = resize_to_square(image, MOBILE_IMG_SIZE)
                mobile_batch_images[i] = preprocess_input(mobile_image)

            except:
                pass

        # dense net
        batch_preds = dense_model.predict(dense_batch_images)
        for i, pet_id in enumerate(batch_pets):
            dense_preds_no1[str(pet_id).split('/')[-1].split('.')[0]] = \
            batch_preds[i]

        # mobile net
        batch_preds = mobile_model.predict(mobile_batch_images)
        for i, pet_id in enumerate(batch_pets):
            mobile_preds_no1[str(pet_id).split('/')[-1].split('.')[0]] = \
            batch_preds[i]

        # nima3
        batch_preds = nima_model_3.predict(mobile_batch_images) # mobile==nima
        for i, pet_id in enumerate(batch_pets):
            nima3_preds_no1[str(pet_id).split('/')[-1].split('.')[0]] = \
                batch_preds[i]

    imageid_oth = []
    blur_oth = []
    dense_preds_oth = {}
    image_quality_oth = {}

    # other
    for b in range(n_batches_oth):
        start = b * BATCH_SIZE
        end = (b + 1) * BATCH_SIZE
        batch_pets = img_list_oth[start:end]

        dense_batch_images = np.zeros(
            (len(batch_pets), DENSE_IMG_SIZE, DENSE_IMG_SIZE, 3))
        mobile_batch_images = np.zeros(
            (len(batch_pets), MOBILE_IMG_SIZE, MOBILE_IMG_SIZE, 3))
        for i, pet_id in enumerate(batch_pets):
            try:
                image = load_image(str(pet_id))

                # image quality (blur)
                imageid_oth.append(
                    str(pet_id).replace('.jpg', '').split('/')[-1])
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blur_oth.append(cv2.Laplacian(gray, cv2.CV_64F).var())

                # resize for dense
                dense_image = resize_to_square(image, DENSE_IMG_SIZE)
                dense_batch_images[i] = preprocess_input(dense_image)

                # resize for mobile
                mobile_image = resize_to_square(image, MOBILE_IMG_SIZE)
                mobile_batch_images[i] = preprocess_input(mobile_image)

            except:
                pass

        # dense net
        batch_preds = dense_model.predict(dense_batch_images)
        for i, pet_id in enumerate(batch_pets):
            dense_preds_oth[str(pet_id).split('/')[-1].split('.')[0]] = \
            batch_preds[i]

    # no1
    image_quality_no1 = pd.concat(
        [pd.DataFrame(imageid_no1, columns=['ImageId']),
         pd.DataFrame(blur_no1, columns=['blur'])], axis=1)
    image_quality_oth = pd.concat(
        [pd.DataFrame(imageid_oth, columns=['ImageId']),
         pd.DataFrame(blur_oth, columns=['blur'])], axis=1)
    image_quality = pd.concat([image_quality_no1, image_quality_oth])

    image_quality['PetID'] = image_quality['ImageId'].str.split('-').str[0]
    image_quality['blur_mean'] = image_quality.groupby(['PetID'])[
        'blur'].transform('mean')
    image_quality['blur_min'] = image_quality.groupby(['PetID'])[
        'blur'].transform('min')
    image_quality['blur_max'] = image_quality.groupby(['PetID'])[
        'blur'].transform('max')
    image_quality = image_quality.drop(['blur', 'ImageId'], 1)
    image_quality = image_quality.drop_duplicates('PetID')

    dense_preds_no1 = pd.DataFrame.from_dict(dense_preds_no1, orient='index')
    dense_preds_no1.columns = [f'pic_{i}' for i in
                               range(dense_preds_no1.shape[1])]
    dense_preds_no1 = dense_preds_no1.reset_index()
    dense_preds_no1['PetID'] = dense_preds_no1['index'].apply(
        lambda x: str(x).split('-')[0])
    dense_preds_no1 = dense_preds_no1.drop('index', axis=1)

    mobile_preds_no1 = pd.DataFrame.from_dict(mobile_preds_no1, orient='index')
    mobile_preds_no1.columns = [f'mobile_pic_{i}' for i in
                                range(mobile_preds_no1.shape[1])]
    mobile_preds_no1 = mobile_preds_no1.reset_index()
    mobile_preds_no1['PetID'] = mobile_preds_no1['index'].apply(
        lambda x: str(x).split('-')[0])
    mobile_preds_no1 = mobile_preds_no1.drop('index', axis=1)

    nima3_preds_no1 = pd.DataFrame.from_dict(nima3_preds_no1, orient='index')
    nima3_columns = [f'nima3_pic_{i}' for i in range(nima3_preds_no1.shape[1])]
    nima3_preds_no1.columns = nima3_columns
    nima3_preds_no1['nima3_mean'] = mean_score(nima3_preds_no1[nima3_columns])
    nima3_preds_no1['nima3_std'] = std_score(nima3_preds_no1[nima3_columns])
    nima3_preds_no1.drop(nima3_columns, axis=1, inplace=True)
    nima3_preds_no1 = nima3_preds_no1.reset_index()
    nima3_preds_no1['PetID'] = nima3_preds_no1['index'].apply(
        lambda x: str(x).split('-')[0])
    nima3_preds_no1 = nima3_preds_no1.drop('index', axis=1)

    dense_preds_oth = pd.DataFrame.from_dict(dense_preds_oth, orient='index')
    dense_preds_oth.columns = [f'pic_{i}' for i in
                               range(dense_preds_oth.shape[1])]
    dense_preds_oth = dense_preds_oth.reset_index()
    dense_preds_oth['PetID'] = dense_preds_oth['index'].apply(
        lambda x: str(x).split('-')[0])
    dense_preds_oth = dense_preds_oth.drop('index', axis=1)

    return image_quality, dense_preds_no1, mobile_preds_no1, \
           nima3_preds_no1, dense_preds_oth


def img_first_svd(train_feats_mobile, test_feats_mobile):
    df = pd.concat([train_feats_mobile, test_feats_mobile])
    df = df.set_index('PetID')

    n_components = 16
    svd_ = TruncatedSVD(n_components=n_components, random_state=1337)

    features = df[[f'mobile_pic_{i}' for i in range(256)]].values

    svd_col = svd_.fit_transform(features)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix('IMG_First_SVD_')
    svd_col.index = df.index

    mobilenet_features_first = svd_col
    mobilenet_features_first.reset_index(inplace=True)

    return mobilenet_features_first


def aggregate_meta_sentiment(train_dfs_sentiment, train_dfs_metadata,
                             test_dfs_sentiment, test_dfs_metadata):
    # Extend aggregates and improve column naming
    aggregates = ['mean', 'sum', 'std']
    sentiment_agg = ['mean']

    # Train
    train_metadata_desc = train_dfs_metadata.groupby(['PetID'])[
        'metadata_annots_top_desc'].unique()
    train_metadata_desc = train_metadata_desc.reset_index()
    train_metadata_desc[
        'metadata_annots_top_desc'] = train_metadata_desc[
        'metadata_annots_top_desc'].apply(lambda x: ' '.join(x))

    prefix = 'metadata'
    train_metadata_gr = train_dfs_metadata.drop(['metadata_annots_top_desc'],
                                                axis=1)
    for i in train_metadata_gr.columns:
        if 'PetID' not in i:
            train_metadata_gr[i] = train_metadata_gr[i].astype(float)
    train_metadata_gr = train_metadata_gr.groupby(['PetID']).agg(aggregates)
    train_metadata_gr.columns = pd.Index(['{}_{}_{}'.format(
        prefix, c[0], c[1].upper()) for c in
        train_metadata_gr.columns.tolist()])
    train_metadata_gr = train_metadata_gr.reset_index()

    train_sentiment_desc = train_dfs_sentiment.groupby(['PetID'])[
        'sentiment_entities'].unique()
    train_sentiment_desc = train_sentiment_desc.reset_index()
    train_sentiment_desc[
        'sentiment_entities'] = train_sentiment_desc[
        'sentiment_entities'].apply(lambda x: ' '.join(x))

    prefix = 'sentiment'
    train_sentiment_gr = train_dfs_sentiment.drop(['sentiment_entities'],
                                                  axis=1)
    for i in train_sentiment_gr.columns:
        if 'PetID' not in i:
            train_sentiment_gr[i] = train_sentiment_gr[i].astype(float)
    train_sentiment_gr = train_sentiment_gr.groupby(['PetID']).agg(
        sentiment_agg)
    train_sentiment_gr.columns = pd.Index(['{}_{}_{}'.format(
        prefix, c[0], c[1].upper()) for c in
        train_sentiment_gr.columns.tolist()])
    train_sentiment_gr = train_sentiment_gr.reset_index()

    # Test
    test_metadata_desc = test_dfs_metadata.groupby(['PetID'])[
        'metadata_annots_top_desc'].unique()
    test_metadata_desc = test_metadata_desc.reset_index()
    test_metadata_desc[
        'metadata_annots_top_desc'] = test_metadata_desc[
        'metadata_annots_top_desc'].apply(lambda x: ' '.join(x))

    prefix = 'metadata'
    test_metadata_gr = test_dfs_metadata.drop(['metadata_annots_top_desc'],
                                              axis=1)
    for i in test_metadata_gr.columns:
        if 'PetID' not in i:
            test_metadata_gr[i] = test_metadata_gr[i].astype(float)
    test_metadata_gr = test_metadata_gr.groupby(['PetID']).agg(aggregates)
    test_metadata_gr.columns = pd.Index(['{}_{}_{}'.format(
        prefix, c[0], c[1].upper()) for c in
        test_metadata_gr.columns.tolist()])
    test_metadata_gr = test_metadata_gr.reset_index()

    test_sentiment_desc = test_dfs_sentiment.groupby(['PetID'])[
        'sentiment_entities'].unique()
    test_sentiment_desc = test_sentiment_desc.reset_index()
    test_sentiment_desc[
        'sentiment_entities'] = test_sentiment_desc[
        'sentiment_entities'].apply(lambda x: ' '.join(x))

    prefix = 'sentiment'
    test_sentiment_gr = test_dfs_sentiment.drop(['sentiment_entities'], axis=1)
    for i in test_sentiment_gr.columns:
        if 'PetID' not in i:
            test_sentiment_gr[i] = test_sentiment_gr[i].astype(float)
    test_sentiment_gr = test_sentiment_gr.groupby(['PetID']).agg(sentiment_agg)
    test_sentiment_gr.columns = pd.Index(['{}_{}_{}'.format(
        prefix, c[0], c[1].upper()) for c in
        test_sentiment_gr.columns.tolist()])
    test_sentiment_gr = test_sentiment_gr.reset_index()

    return train_metadata_gr, train_sentiment_gr, train_metadata_desc, \
           train_sentiment_desc, test_metadata_gr, test_sentiment_gr, \
           test_metadata_desc, test_sentiment_desc


def merge_meta(train_proc, train_sentiment_gr, train_metadata_gr,
               train_metadata_desc, train_sentiment_desc,
               test_proc, test_sentiment_gr, test_metadata_gr,
               test_metadata_desc, test_sentiment_desc):
    # Train merges:
    train_proc = train_proc.merge(
        train_sentiment_gr, how='left', on='PetID')
    train_proc = train_proc.merge(
        train_metadata_gr, how='left', on='PetID')
    train_proc = train_proc.merge(
        train_metadata_desc, how='left', on='PetID')
    train_proc = train_proc.merge(
        train_sentiment_desc, how='left', on='PetID')

    # Test merges:
    test_proc = test_proc.merge(
        test_sentiment_gr, how='left', on='PetID')
    test_proc = test_proc.merge(
        test_metadata_gr, how='left', on='PetID')
    test_proc = test_proc.merge(
        test_metadata_desc, how='left', on='PetID')
    test_proc = test_proc.merge(
        test_sentiment_desc, how='left', on='PetID')

    print(train_proc.shape, test_proc.shape)
    assert train_proc.shape[0] == train.shape[0]
    assert test_proc.shape[0] == test.shape[0]

    return train_proc, test_proc


def merge_label(train_proc, test_proc):
    train_breed_main = train_proc[['Breed1']].merge(
        labels_breed, how='left',
        left_on='Breed1', right_on='BreedID',
        suffixes=('', '_main_breed'))

    train_breed_main = train_breed_main.iloc[:, 2:]
    train_breed_main = train_breed_main.add_prefix('main_breed_')

    train_breed_second = train_proc[['Breed2']].merge(
        labels_breed, how='left',
        left_on='Breed2', right_on='BreedID',
        suffixes=('', '_second_breed'))

    train_breed_second = train_breed_second.iloc[:, 2:]
    train_breed_second = train_breed_second.add_prefix('second_breed_')

    train_proc = pd.concat(
        [train_proc, train_breed_main, train_breed_second], axis=1)

    test_breed_main = test_proc[['Breed1']].merge(
        labels_breed, how='left',
        left_on='Breed1', right_on='BreedID',
        suffixes=('', '_main_breed'))

    test_breed_main = test_breed_main.iloc[:, 2:]
    test_breed_main = test_breed_main.add_prefix('main_breed_')

    test_breed_second = test_proc[['Breed2']].merge(
        labels_breed, how='left',
        left_on='Breed2', right_on='BreedID',
        suffixes=('', '_second_breed'))

    test_breed_second = test_breed_second.iloc[:, 2:]
    test_breed_second = test_breed_second.add_prefix('second_breed_')

    test_proc = pd.concat(
        [test_proc, test_breed_main, test_breed_second], axis=1)

    print(train_proc.shape, test_proc.shape)

    return train_proc, test_proc


# Image(kaeru)
def getSize(filename):
    st = os.stat(filename)
    return st.st_size


def getDimensions(filename):
    img_size = Image.open(filename).size
    return img_size


def kaeru_feature(train_df_imgs, test_df_imgs):
    train_df_imgs['image_size'] = train_df_imgs['image_filename'].apply(
        getSize)
    train_df_imgs['temp_size'] = train_df_imgs['image_filename'].apply(
        getDimensions)
    train_df_imgs['width'] = train_df_imgs['temp_size'].apply(lambda x: x[0])
    train_df_imgs['height'] = train_df_imgs['temp_size'].apply(lambda x: x[1])
    train_df_imgs = train_df_imgs.drop(['temp_size'], axis=1)

    test_df_imgs['image_size'] = test_df_imgs['image_filename'].apply(getSize)
    test_df_imgs['temp_size'] = test_df_imgs['image_filename'].apply(
        getDimensions)
    test_df_imgs['width'] = test_df_imgs['temp_size'].apply(lambda x: x[0])
    test_df_imgs['height'] = test_df_imgs['temp_size'].apply(lambda x: x[1])
    test_df_imgs = test_df_imgs.drop(['temp_size'], axis=1)

    aggs = {
        'image_size': ['sum', 'mean', 'std', 'min'],
        'width': ['sum', 'mean', 'std'],
        'height': ['sum', 'mean', 'std'],
    }

    agg_train_imgs = train_df_imgs.groupby('PetID').agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_train_imgs.columns = new_columns
    agg_train_imgs = agg_train_imgs.reset_index()

    agg_test_imgs = test_df_imgs.groupby('PetID').agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_test_imgs.columns = new_columns
    agg_test_imgs = agg_test_imgs.reset_index()

    agg_imgs = pd.concat([agg_train_imgs, agg_test_imgs], axis=0).reset_index(
        drop=True)

    return agg_imgs


# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating = None
    max_rating = None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


# put some numerical values to bins
def to_bins(x, borders):
    for i in range(len(borders)):
        if x <= borders[i]:
            return i
    return len(borders)


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _loss(self, coef, X, y, idx):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        ll = -quadratic_weighted_kappa(y, X_p)
        return ll

    def fit(self, X, y):
        coef = [1.5, 2.0, 2.5, 3.0]
        golden1 = 0.618
        golden2 = 1 - golden1
        ab_start = [(1, 2), (1.5, 2.5), (2, 3), (2.5, 3.5)]
        for it1 in range(10):
            for idx in range(4):
                # golden section search
                a, b = ab_start[idx]
                # calc losses
                coef[idx] = a
                la = self._loss(coef, X, y, idx)
                coef[idx] = b
                lb = self._loss(coef, X, y, idx)
                for it in range(20):
                    # choose value
                    if la > lb:
                        a = b - (b - a) * golden1
                        coef[idx] = a
                        la = self._loss(coef, X, y, idx)
                    else:
                        b = b - (b - a) * golden2
                        coef[idx] = b
                        lb = self._loss(coef, X, y, idx)
        self.coef_ = {'x': coef}

    def predict(self, X, coef):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        return X_p

    def coefficients(self):
        return self.coef_['x']


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std(
                [y_counts_per_fold[i][label] / y_distr[label] for i in
                 range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts,
                              key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def get_distribution(y_vals):
    y_distr = Counter(y_vals)
    y_vals_sum = sum(y_distr.values())
    return [f'{y_distr[i] / y_vals_sum:.2%}' for i in range(np.max(y_vals) + 1)]


def eval_predict(y=[], y_pred=[], coeffs=None, ret_coeffs=False):
    optR = OptimizedRounder()
    if not coeffs:
        optR.fit(y_pred.reshape(-1,), y)
        coeffs = optR.coefficients()
    if ret_coeffs:
        return optR.coefficients()
    return optR.predict(y_pred, coeffs).reshape(-1,)


def ridge_stack(train_stack, test_stack, x_train):
    print('\nRun Ridge Stack')
    oof = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])
    qwk_scores = []

    groups = x_train['RescuerID']
    for folds, (trn_idx, val_idx) in enumerate(
            stratified_group_k_fold(x_train,
                                    x_train['AdoptionSpeed'].astype(int),
                                    groups, k=10, seed=1)):

        trn_data, trn_y = train_stack[trn_idx], x_train['AdoptionSpeed'].iloc[
            trn_idx].values
        val_data, val_y = train_stack[val_idx], x_train['AdoptionSpeed'].iloc[
            val_idx].values

        clf = Ridge(alpha=30, random_state=1)
        clf.fit(trn_data, trn_y)
        val_pred = clf.predict(val_data)
        pred_val_y_k = eval_predict(y=val_y, y_pred=val_pred,
                                    coeffs=None).astype(int)

        print("Valid Counts = ", Counter(val_y))
        print("Predicted Counts = ", Counter(pred_val_y_k))
        optR = OptimizedRounder()
        optR.fit(val_pred, val_y)
        coefficients = optR.coefficients()
        print('coefficients = ', coefficients)
        qwk = quadratic_weighted_kappa(val_y, pred_val_y_k)
        qwk_scores.append(qwk)
        print("QWK = ", qwk)

        oof[val_idx] = clf.predict(val_data)
        predictions += clf.predict(test_stack) / 10

    print('{} cv mean QWK score : {}'.format(
        'Stacked_model_ridge', np.mean(qwk_scores)))
    print(clf.coef_)

    return oof, predictions


# Fujita Function
def fujita_img_svd(train_feats, test_feats):
    print('\nRun Fujita image SVD')

    n_components = 64
    svd_ = TruncatedSVD(n_components=n_components, random_state=916)
    features_df = pd.concat([train_feats, test_feats], axis=0)
    features_df = features_df.set_index('PetID')
    features = features_df[[f'pic_{i}' for i in range(256)]].values

    svd_col = svd_.fit_transform(features)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix('IMG_SVD_')
    svd_col.index = features_df.index

    img_features = svd_col
    img_features.reset_index(inplace=True)

    return img_features


def fujita_text_svd(train_proc, test_proc):

    n_components = 32

    text_columns = ['Description', 'metadata_annots_top_desc',
                    'sentiment_entities']

    tfidf_para = {
        'strip_accents': 'unicode',
        'analyzer': 'word',
        'token_pattern': r'\w{1,}',
        "norm": 'l2',
        'use_idf': True,
        'smooth_idf': True,
        'sublinear_tf': True,
        'stop_words': 'english'}

    # Generate text features:
    for i in text_columns:

        train_proc.loc[:, i] = train_proc.loc[:, i].fillna('none')
        test_proc.loc[:, i] = test_proc.loc[:, i].fillna('none')
        # Initialize decomposition methods:
        print('generating features from: {}'.format(i))
        svd_ = TruncatedSVD(n_components=n_components, random_state=916)

        vectorizer = TfidfVectorizer(min_df=3,
                                     ngram_range=(1, 1),
                                     max_features=None,
                                     **tfidf_para)

        vectorizer.fit(train_proc.loc[:, i].values)
        tfidf_tr = vectorizer.transform(train_proc.loc[:, i].values)
        tfidf_te = vectorizer.transform(test_proc.loc[:, i].values)

        svd = svd_.fit(tfidf_tr)
        svd_tr = svd.transform(tfidf_tr)
        svd_te = svd.transform(tfidf_te)

        svd_tr = pd.DataFrame(svd_tr)
        svd_tr = svd_tr.add_prefix('SVD_{}_'.format(i))
        svd_te = pd.DataFrame(svd_te)
        svd_te = svd_te.add_prefix('SVD_{}_'.format(i))

        # Concatenate with main DF:
        train_proc = pd.concat([train_proc, svd_tr], axis=1)
        test_proc = pd.concat([test_proc, svd_te], axis=1)

    return train_proc, test_proc


def pure_breed(x):
    if (x.Breed2 == 0) & (x.Breed1 not in [307, 264, 265, 266]):
        return 1
    elif (x.Breed1 == 0) & (x.Breed2 not in [307, 264, 265, 266]):
        return 1
    else:
        return 0


def single_color(x):
    if (x.Color2 == 0) & (x.Color3 == 0):
        return 1
    else:
        return 0


def fujita_run_lgb(X_train, X_test):

    print('\nRun Fujita LGB')

    categorical = [
        'Type',
        'Gender',
        'Breed1',
        'Breed2',
        'Color1',
        'Color2',
        'Color3',
        'MaturitySize',
        'FurLength',
        'Vaccinated',
        'Dewormed',
        'Sterilized',
        'Health',
        'State'
    ]

    params = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'num_leaves': 16,
              'max_depth': 5,
              'learning_rate': 0.019298097588045453,
              'bagging_fraction': 0.7120164522653443,
              'feature_fraction': 0.20275048053384834,
              'min_split_gain': 0.00023999254924688356,
              'min_child_samples': 68,
              'min_child_weight': 58.51375036732489,
              'lambda_l1': 0.00039543733409789415,
              'lambda_l2': 0.00030252403763729056,
              'cat_l2': 0.011421982120831599,
              'cat_smooth': 0.03853533731179398,
              'max_cat_to_onehot': 3,
              'verbosity': -1,
              'data_random_seed': 916}

    # Additional parameters:
    early_stop = 1000
    verbose_eval = 1000
    num_rounds = 10000
    n_splits = 10

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    cv_scores = []
    qwk_scores = []
    feature_importance_df = pd.DataFrame()

    i = 0

    groups = X_train['RescuerID']
    for fold_ind, (train_idx, valid_idx) in enumerate(
            stratified_group_k_fold(X_train,
                                    X_train['AdoptionSpeed'].astype(int),
                                    groups, k=10, seed=916)):
        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        del X_tr['RescuerID'], X_val['RescuerID']

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        print('\ny_tr distribution: {}'.format(Counter(y_tr)))
        d_train = lgb.Dataset(X_tr, label=y_tr,
                              categorical_feature=categorical)
        d_valid = lgb.Dataset(X_val, label=y_val,
                              categorical_feature=categorical)
        watchlist = [d_train, d_valid]

        print('training LGB:')
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)

        valid_pred = model.predict(X_val, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)

        optR = OptimizedRounder()
        optR.fit(valid_pred, y_val)
        coefficients = optR.coefficients()
        pred_test_y_k = optR.predict(valid_pred, coefficients)
        print("Valid Counts = ", Counter(y_val))
        print("Predicted Counts = ", Counter(pred_test_y_k))
        print("Coefficients = ", coefficients)
        qwk = quadratic_weighted_kappa(y_val, pred_test_y_k)
        rmse_score = rmse(y_val, valid_pred)
        print("QWK = ", qwk)
        cv_scores.append(rmse_score)
        qwk_scores.append(qwk)
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = X_tr.columns.values
        fold_importance_df['importance'] = model.feature_importance(
            importance_type='gain')
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df,
                                           fold_importance_df], axis=0)

        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1

    print('\ncv RMSE scores : {}'.format(cv_scores))
    print('cv mean RMSE score : {}'.format(np.mean(cv_scores)))
    print('cv std RMSE score : {}'.format(np.std(cv_scores)))
    print('cv QWK scores : {}'.format(qwk_scores))
    print('cv mean QWK score : {}'.format(np.mean(qwk_scores)))
    print('cv std QWK score : {}'.format(np.std(qwk_scores)))

    return feature_importance_df, oof_train, oof_test


def fujita_run_xen(X_train, X_test):

    print('\nRun Fujita LGB Xentropy')

    categorical = [
        'Type',
        'Gender',
        'Breed1',
        'Breed2',
        'Color1',
        'Color2',
        'Color3',
        'MaturitySize',
        'FurLength',
        'Vaccinated',
        'Dewormed',
        'Sterilized',
        'Health',
        'State'
    ]

    params = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': 'xentropy',
              'num_leaves': 56,
              'max_depth': 5,
              'learning_rate': 0.020449499447419583,
              'bagging_fraction': 0.850459908830826,
              'feature_fraction': 0.20871912929997272,
              'min_split_gain': 2.9436810935800595,
              'min_child_samples': 49,
              'min_child_weight': 34.21536512518751,
              'lambda_l1': 0.8429441386236972,
              'lambda_l2': 0.16465057772464275,
              'cat_l2': 7.709378557748177,
              'cat_smooth': 0.00010528323430233814,
              'max_cat_to_onehot': 1,
              'verbosity': -1,
              'data_random_seed': 1984}

    # Additional parameters:
    early_stop = 1000
    verbose_eval = 1000
    num_rounds = 10000
    n_splits = 10

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    cv_scores = []
    qwk_scores = []
    feature_importance_df = pd.DataFrame()

    i = 0

    groups = X_train['RescuerID']
    for fold_ind, (train_idx, valid_idx) in enumerate(
            stratified_group_k_fold(X_train,
                                    X_train['AdoptionSpeed'].astype(int),
                                    groups, k=10, seed=1984)):
        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        del X_tr['RescuerID'], X_val['RescuerID']

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        print('\ny_tr distribution: {}'.format(Counter(y_tr)))
        d_train = lgb.Dataset(X_tr, label=y_tr * (1/4),
                              categorical_feature=categorical)
        d_valid = lgb.Dataset(X_val, label=y_val * (1/4),
                              categorical_feature=categorical)
        watchlist = [d_train, d_valid]

        print('training LGB:')
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)

        valid_pred = model.predict(X_val,
                                   num_iteration=model.best_iteration) * 4
        test_pred = model.predict(X_test,
                                  num_iteration=model.best_iteration) * 4

        optR = OptimizedRounder()
        optR.fit(valid_pred, y_val)
        coefficients = optR.coefficients()
        pred_test_y_k = optR.predict(valid_pred, coefficients)
        print("Valid Counts = ", Counter(y_val))
        print("Predicted Counts = ", Counter(pred_test_y_k))
        print("Coefficients = ", coefficients)
        qwk = quadratic_weighted_kappa(y_val, pred_test_y_k)
        rmse_score = rmse(y_val, valid_pred)
        print("QWK = ", qwk)
        cv_scores.append(rmse_score)
        qwk_scores.append(qwk)
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = X_tr.columns.values
        fold_importance_df['importance'] = model.feature_importance(
            importance_type='gain')
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df,
                                           fold_importance_df], axis=0)

        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1

    print('\ncv RMSE scores : {}'.format(cv_scores))
    print('cv mean RMSE score : {}'.format(np.mean(cv_scores)))
    print('cv std RMSE score : {}'.format(np.std(cv_scores)))
    print('cv QWK scores : {}'.format(qwk_scores))
    print('cv mean QWK score : {}'.format(np.mean(qwk_scores)))
    print('cv std QWK score : {}'.format(np.std(qwk_scores)))

    return oof_train, oof_test


# Nakama Function
# RescuerID_COUNT
def RescuerID_COUNT(df):
    concat = df.groupby(['RescuerID'], as_index=False)['PetID'].count()
    concat = pd.DataFrame(concat).rename(columns={'PetID': 'RescuerID_COUNT'})
    df = df.merge(concat, on=['RescuerID'], how='left')
    return df


def add_health(df):
    df['health'] = df['Vaccinated'].astype(str) \
                   + '_' + df['Dewormed'].astype(str) \
                   + '_' + df['Sterilized'].astype(str) \
                   + '_' + df['Health'].astype(str)
    df['health1'] = ((df[['Vaccinated', 'Dewormed', 'Sterilized']] == 1)*1
                     ).sum(axis=1)
    df['health3'] = ((df[['Vaccinated', 'Dewormed', 'Sterilized']] == 3)*1
                     ).sum(axis=1)

    return df


def pure_breed_Description_length(df):
    df['pure_breed'] = (df['Breed2'] != 0)*1
    df['Description_length'] = df['Description'].apply(lambda x: len(str(x)))
    return df


# only_cat_RescuerID
def only_cat_RescuerID(df):
    df['only_cat_RescuerID'] = 0
    index = df.groupby(['RescuerID']).nunique()[df.groupby(
        ['RescuerID'])['Type'].nunique() == 1].index
    index = df[df['RescuerID'].isin(index)].groupby(
        ['RescuerID']).mean()[df[df['RescuerID'].isin(index)].groupby(
        ['RescuerID'])['Type'].mean() == 2].index
    index = df[df['RescuerID'].isin(index)].loc[
        df[df['RescuerID'].isin(index)]['RescuerID_COUNT'] >= 10].groupby(
        ['RescuerID'])['Type'].nunique().index
    df.loc[df['RescuerID'].isin(index), 'only_cat_RescuerID'] = 1

    return df


# only_dog_RescuerID
def only_dog_RescuerID(df):
    df['only_dog_RescuerID'] = 0
    index = df.groupby(['RescuerID']).nunique()[df.groupby(
        ['RescuerID'])['Type'].nunique() == 1].index
    index = df[df['RescuerID'].isin(index)].groupby(
        ['RescuerID']).mean()[df[df['RescuerID'].isin(index)].groupby(
        ['RescuerID'])['Type'].mean() == 1].index
    index = df[df['RescuerID'].isin(index)].loc[
        df[df['RescuerID'].isin(index)]['RescuerID_COUNT'] >= 10].groupby(
        ['RescuerID'])['Type'].nunique().index
    df.loc[df['RescuerID'].isin(index), 'only_dog_RescuerID'] = 1

    return df


# big_return_RescuerID
def big_return_RescuerID(df):
    col = ['Age', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated',
           'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee']
    aggs = {}

    for col in col:
        aggs[col] = ['var']

    concat_df = df.groupby('RescuerID').agg(aggs)

    # fillna
    concat_df = concat_df.fillna(-1)

    # change column name
    concat_df.columns = pd.Index(
        [e[0] + "_" + e[1] for e in concat_df.columns.tolist()])
    concat_df.columns = ['RescuerID_' + c for c in concat_df.columns]

    # concat
    concat_df = concat_df.reset_index()
    df = df.merge(concat_df, on='RescuerID', how='left')

    # fix
    index = df[df['RescuerID_COUNT'] < 10].index
    for c in concat_df.columns:
        if c != 'RescuerID':
            df.loc[index, c] = -1

    return df


def health_var(df):
    for c in ['RescuerID_Age_var',
              'RescuerID_Gender_var', 'RescuerID_MaturitySize_var',
              'RescuerID_FurLength_var', 'RescuerID_Vaccinated_var',
              'RescuerID_Dewormed_var', 'RescuerID_Sterilized_var',
              'RescuerID_Health_var', 'RescuerID_Quantity_var',
              'RescuerID_Fee_var']:
        df[c+'_health1'] = df[c]*df['health1']
        df[c+'_health3'] = df[c]*df['health3']

    return df


def refundable_in_Description(df):
    df['refundable_in_Description'] = 0
    df.loc[df[df['Description'].fillna('').str.contains('refundable')].index,
           'refundable_in_Description'] = 1

    return df


def chinese_in_Description(df):
    df['chinese_in_Description'] = 0
    df.loc[df[df['Description'].fillna('').str.contains('。')].index,
           'chinese_in_Description'] = 1

    return df


def RM_in_Description(df):
    df['RM_in_Description'] = 0
    df.loc[df[df['Description'].fillna('').str.contains('RM')].index,
           'RM_in_Description'] = 1

    return df


def add_breed_prefer(train_proc, test_proc):
    # breed_prefer
    breed_prefer = train_proc.groupby(['Breed1', 'Breed2'], as_index=False)[
        'AdoptionSpeed'].mean()
    breed_prefer = pd.DataFrame(breed_prefer).rename(
        columns={'AdoptionSpeed': 'breed_prefer'})
    train_proc = train_proc.merge(
        breed_prefer, on=['Breed1', 'Breed2'], how='left')
    test_proc = test_proc.merge(
        breed_prefer, on=['Breed1', 'Breed2'], how='left')

    # breed_prefer_count
    breed_prefer_count = train_proc.groupby(
        ['Breed1', 'Breed2'], as_index=False)['AdoptionSpeed'].count()
    breed_prefer_count = pd.DataFrame(breed_prefer_count).rename(
        columns={'AdoptionSpeed': 'breed_prefer_count'})
    train_proc = train_proc.merge(
        breed_prefer_count, on=['Breed1', 'Breed2'], how='left')
    test_proc = test_proc.merge(
        breed_prefer_count, on=['Breed1', 'Breed2'], how='left')

    # breed_prefer_var
    breed_prefer_var = train_proc.groupby(
        ['Breed1', 'Breed2'], as_index=False)['AdoptionSpeed'].var()
    breed_prefer_var = pd.DataFrame(breed_prefer_var).rename(
        columns={'AdoptionSpeed': 'breed_prefer_var'})
    train_proc = train_proc.merge(
        breed_prefer_var, on=['Breed1', 'Breed2'], how='left')
    test_proc = test_proc.merge(
        breed_prefer_var, on=['Breed1', 'Breed2'], how='left')

    # filtering by breed_prefer_count
    train_proc.loc[
        train_proc[
            train_proc['breed_prefer_count'] < 100].index, 'breed_prefer'] = -1
    train_proc.drop(['breed_prefer_count'], axis=1, inplace=True)
    test_proc.loc[
        test_proc[
            test_proc['breed_prefer_count'] < 100].index, 'breed_prefer'] = -1
    test_proc.drop(['breed_prefer_count'], axis=1, inplace=True)

    # filtering by breed_prefer_var
    train_proc.loc[
        train_proc[
            train_proc['breed_prefer_var'] > 1.5].index, 'breed_prefer'] = -1
    train_proc.drop(['breed_prefer_var'], axis=1, inplace=True)
    test_proc.loc[
        test_proc[
            test_proc['breed_prefer_var'] > 1.5].index, 'breed_prefer'] = -1
    test_proc.drop(['breed_prefer_var'], axis=1, inplace=True)

    return train_proc, test_proc


def nakama_img_svd(train_feats, test_feats):
    print('\nRun Nakama image SVD')

    n_components = 32
    svd_ = TruncatedSVD(n_components=n_components, random_state=1337)
    features_df = pd.concat([train_feats, test_feats], axis=0)
    features_df = features_df.set_index('PetID')
    features = features_df[[f'pic_{i}' for i in range(256)]].values

    svd_col = svd_.fit_transform(features)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix('IMG_SVD_')
    svd_col.index = features_df.index

    img_features = svd_col
    img_features.reset_index(inplace=True)

    return img_features


def nakama_desc_svd(train_proc, test_proc):
    print('\nRun Nakama desc SVD')
    # WITHOUT ERROR FIXED
    train_desc = train_proc.Description.fillna("none").values
    test_desc = test_proc.Description.fillna("none").values

    tfv = TfidfVectorizer(min_df=3, max_features=10000,
                          strip_accents='unicode', analyzer='word',
                          token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1,
                          sublinear_tf=1,
                          stop_words='english')

    # Fit TFIDF
    tfv.fit(list(train_desc))
    X = tfv.transform(train_desc)
    X_test = tfv.transform(test_desc)
    print("X (tfidf):", X.shape)

    svd = TruncatedSVD(n_components=120, random_state=416)
    svd.fit(X)
    X = svd.transform(X)
    X_test = svd.transform(X_test)
    print("X (svd):", X.shape)

    X = pd.DataFrame(X, columns=['svd_{}'.format(i) for i in range(120)])
    train_proc = pd.concat((train_proc, X), axis=1)
    X_test = pd.DataFrame(X_test,
                          columns=['svd_{}'.format(i) for i in range(120)])
    test_proc = pd.concat((test_proc, X_test), axis=1)
    print("train:", train_proc.shape)
    print("test:", test_proc.shape)
    return train_proc, test_proc


def nakama_text_svd(X, X_text):
    print('\nRun Nakama text SVD')
    n_components = 16
    text_features = []

    # Generate text features:
    for i in X_text.columns:
        # Initialize decomposition methods:
        print(f'generating features from: {i}')
        tfv = TfidfVectorizer(min_df=2, max_features=None,
                              strip_accents='unicode', analyzer='word',
                              token_pattern=r'(?u)\b\w+\b',
                              ngram_range=(1, 3), use_idf=1, smooth_idf=1,
                              sublinear_tf=1)
        svd_ = TruncatedSVD(
            n_components=n_components, random_state=1337)

        tfidf_col = tfv.fit_transform(X_text.loc[:, i].values)

        svd_col = svd_.fit_transform(tfidf_col)
        svd_col = pd.DataFrame(svd_col)
        svd_col = svd_col.add_prefix('TFIDF_{}_'.format(i))

        text_features.append(svd_col)

    text_features = pd.concat(text_features, axis=1)

    X = pd.concat([X, text_features], axis=1)

    for i in X_text.columns:
        X = X.drop(i, axis=1)

    return X


def nakama_run_lgb(X_train, X_test):

    print('\nRun Nakama LGB Xentropy')

    # Nakama LGB Xentropy
    params = {
        'objective': 'regression',
        'metric': 'xentropy',
        'verbosity': -1,
        'data_random_seed': 1234,
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 30,
        'subsample': 0.9856150674268281,
        'max_depth': 7,
        'min_child_weight': 2.8287386998242416,
        'reg_alpha': 0.0006585344942513311,
        'colsample_bytree': 0.26965553021836786,
        'min_split_gain': 0.0020016677776033584,
        'reg_lambda': 1.437854102380511,
        'min_data_in_leaf': 9
    }

    # Additional parameters:
    early_stop = 1000
    verbose_eval = 1000
    num_rounds = 10000
    n_splits = 10

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    cv_scores = []
    qwk_scores = []
    feature_importance_df = pd.DataFrame()

    i = 0

    groups = X_train['RescuerID']
    for fold_ind, (train_idx, valid_idx) in enumerate(
            stratified_group_k_fold(X_train,
                                    X_train['AdoptionSpeed'].astype(int),
                                    groups, k=10, seed=416)):
        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        del X_tr['RescuerID'], X_val['RescuerID']

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        print('\ny_tr distribution: {}'.format(Counter(y_tr)))
        d_train = lgb.Dataset(X_tr, label=y_tr*(1/4))
        d_valid = lgb.Dataset(X_val, label=y_val*(1/4))
        watchlist = [d_train, d_valid]

        print('training LGB:')
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)

        valid_pred = model.predict(X_val,
                                   num_iteration=model.best_iteration) * 4
        test_pred = model.predict(X_test,
                                  num_iteration=model.best_iteration) * 4

        optR = OptimizedRounder()
        optR.fit(valid_pred, y_val)
        coefficients = optR.coefficients()
        pred_test_y_k = optR.predict(valid_pred, coefficients)
        print("Valid Counts = ", Counter(y_val))
        print("Predicted Counts = ", Counter(pred_test_y_k))
        print("Coefficients = ", coefficients)
        qwk = quadratic_weighted_kappa(y_val, pred_test_y_k)
        rmse_score = rmse(y_val, valid_pred)
        print("QWK = ", qwk)
        cv_scores.append(rmse_score)
        qwk_scores.append(qwk)
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = X_tr.columns.values
        fold_importance_df['importance'] = model.feature_importance(
            importance_type='gain')
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df,
                                           fold_importance_df], axis=0)
        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1

    print('\ncv RMSE scores : {}'.format(cv_scores))
    print('cv mean RMSE score : {}'.format(np.mean(cv_scores)))
    print('cv std RMSE score : {}'.format(np.std(cv_scores)))
    print('cv QWK scores : {}'.format(qwk_scores))
    print('cv mean QWK score : {}'.format(np.mean(qwk_scores)))
    print('cv std QWK score : {}'.format(np.std(qwk_scores)))

    return feature_importance_df, oof_train, oof_test


def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]


def nakama_run_cat(X_train, X_test):

    print('\nRun Nakama CatBoost')
    n_splits = 10
    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    cv_scores = []
    qwk_scores = []
    i = 0
    groups = X_train['RescuerID']
    for fold_ind, (train_idx, valid_idx) in enumerate(
            stratified_group_k_fold(X_train,
                                    X_train['AdoptionSpeed'].astype(int),
                                    groups, k=10, seed=2)):
        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        del X_tr['RescuerID'], X_val['RescuerID']

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        categorical = ['Type', 'Gender', 'Breed1', 'Breed2',
                       'Color1', 'Color2', 'Color3',
                       'MaturitySize', 'FurLength', 'Vaccinated',
                       'Dewormed', 'Sterilized', 'Health', 'State',
                       'pure_breed', 'only_cat_RescuerID', 'health']

        categorical_features_pos = column_index(X_tr, categorical)

        print('\ny_tr distribution: {}'.format(Counter(y_tr)))

        print('training CatBoost:')
        model = CatBoostRegressor(loss_function='RMSE',
                                  bootstrap_type='Poisson',
                                  iterations=20000,
                                  learning_rate=0.03,
                                  max_depth=5,
                                  eval_metric='RMSE',
                                  random_seed=1225,
                                  subsample=0.16339653191092432,
                                  bagging_temperature=0.32509913175866006,
                                  random_strength=1,
                                  l2_leaf_reg=21,
                                  od_type='Iter',
                                  metric_period=1000,
                                  task_type="GPU",
                                  od_wait=1000,
                                  border_count=32,
                                  max_ctr_complexity=6,
                                  boosting_type='Plain'
                                  )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                  cat_features=categorical_features_pos,
                  use_best_model=True)

        valid_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        optR = OptimizedRounder()
        optR.fit(valid_pred, y_val)
        coefficients = optR.coefficients()
        pred_test_y_k = optR.predict(valid_pred, coefficients)
        print("Valid Counts = ", Counter(y_val))
        print("Predicted Counts = ", Counter(pred_test_y_k))
        print("Coefficients = ", coefficients)
        qwk = quadratic_weighted_kappa(y_val, pred_test_y_k)
        rmse_score = rmse(y_val, valid_pred)
        print("QWK = ", qwk)
        cv_scores.append(rmse_score)
        qwk_scores.append(qwk)
        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1

    print('\ncv RMSE scores : {}'.format(cv_scores))
    print('cv mean RMSE score : {}'.format(np.mean(cv_scores)))
    print('cv std RMSE score : {}'.format(np.std(cv_scores)))
    print('cv QWK scores : {}'.format(qwk_scores))
    print('cv mean QWK score : {}'.format(np.mean(qwk_scores)))
    print('cv std QWK score : {}'.format(np.std(qwk_scores)))

    del model
    torch.cuda.empty_cache()

    return oof_train, oof_test


def nakama_run_xgb(X_train, X_test):

    print('\nRun Nakama XGB')

    X_train_tmp = X_train.copy()
    X_test_tmp = X_test.copy()

    xgb_params = {
        'booster': "gbtree",
        'eval_metric': 'rmse',
        'eta': 0.01,
        'min_child_weight': 73,
        'gamma': 0,  # 0
        'subsample': 0.6901557493429121,
        'colsample_bytree': 0.5643126765887346,
        'colsample_bylevel': 0.7740802106949766,
        'alpha': 0,  # 0
        'lambda': 1,  # 1
        'seed': 11,
        'tree_method': 'gpu_hist',
        'device': 'gpu',
        'silent': 1
    }

    n_splits = 10
    verbose_eval = 1000
    num_rounds = 20000
    early_stop = 1000

    oof_train = np.zeros((X_train_tmp.shape[0]))
    oof_test = np.zeros((X_test_tmp.shape[0], n_splits))
    cv_scores = []
    qwk_scores = []

    i = 0

    groups = X_train['RescuerID']
    for fold_ind, (train_idx, valid_idx) in enumerate(
            stratified_group_k_fold(X_train,
                                    X_train['AdoptionSpeed'].astype(int),
                                    groups, k=10, seed=3)):
        X_tr = X_train_tmp.iloc[train_idx, :]
        X_val = X_train_tmp.iloc[valid_idx, :]

        del X_tr['RescuerID'], X_val['RescuerID']

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        d_train = xgb.DMatrix(data=X_tr, label=y_tr,
                              feature_names=X_tr.columns)
        d_valid = xgb.DMatrix(data=X_val, label=y_val,
                              feature_names=X_val.columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds,
                          evals=watchlist,
                          early_stopping_rounds=early_stop,
                          verbose_eval=verbose_eval, params=xgb_params)

        valid_pred = model.predict(
            xgb.DMatrix(X_val, feature_names=X_val.columns),
            ntree_limit=model.best_ntree_limit)
        test_pred = model.predict(
            xgb.DMatrix(X_test_tmp, feature_names=X_test.columns),
            ntree_limit=model.best_ntree_limit)

        optR = OptimizedRounder()
        optR.fit(valid_pred, y_val)
        coefficients = optR.coefficients()
        pred_test_y_k = optR.predict(valid_pred, coefficients)
        print("Valid Counts = ", Counter(y_val))
        print("Predicted Counts = ", Counter(pred_test_y_k))
        print("Coefficients = ", coefficients)
        qwk = quadratic_weighted_kappa(y_val, pred_test_y_k)
        rmse_score = rmse(y_val, valid_pred)
        print("QWK = ", qwk)
        cv_scores.append(rmse_score)
        qwk_scores.append(qwk)

        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1

    print('\ncv RMSE scores : {}'.format(cv_scores))
    print('cv mean RMSE score : {}'.format(np.mean(cv_scores)))
    print('cv std RMSE score : {}'.format(np.std(cv_scores)))
    print('cv QWK scores : {}'.format(qwk_scores))
    print('cv mean QWK score : {}'.format(np.mean(qwk_scores)))
    print('cv std QWK score : {}'.format(np.std(qwk_scores)))
    model.__del__()

    return oof_train, oof_test


def coppe_desc_svd(train_proc, test_proc, vectorizer, svd, prefix):
    print('\nRun copypaste desc decomposition')
    # WITHOUT ERROR FIXED
    train_desc = train_proc.Description.fillna("none").values
    test_desc = test_proc.Description.fillna("none").values

    # Fit TFIDF
    vectorizer.fit(list(train_desc) + list(test_desc))
    X = vectorizer.transform(train_desc)
    X_test = vectorizer.transform(test_desc)
    print("X (vectorize):", X.shape)

    svd.fit(sp.sparse.vstack([X, X_test]))
    X = svd.transform(X)
    X_test = svd.transform(X_test)
    print("X (svd):", X.shape)

    n_components = svd.n_components
    X = pd.DataFrame(X, columns=[f'{prefix}_{i}' for i in range(n_components)])
    train_proc = pd.concat((train_proc, X), axis=1)
    X_test = pd.DataFrame(X_test,
                          columns=[f'{prefix}_{i}' for i in range(n_components)])
    test_proc = pd.concat((test_proc, X_test), axis=1)
    print("train:", train_proc.shape)
    print("test:", test_proc.shape)
    return train_proc, test_proc


def get_mode(df, groupkey, col):
    df_mode = df.groupby(groupkey)[col].agg(lambda x: stats.mode(x)[0][0]).reset_index()
    df_mode = df_mode.rename(columns={col:f'{col}_mode'})
    return df_mode


def get_median(df, groupkey, col):
    df_mode = df.groupby(groupkey)[col].mean().reset_index()
    df_mode = df_mode.rename(columns={col:f'{col}_median'})
    return df_mode


def coppe_rescureid_stats(train_proc, test_proc):

    groupkey = 'RescuerID'
    usecols_mode = ['Type', 'Breed1', 'Gender', 'MaturitySize',
                    'Vaccinated', 'Dewormed', 'Sterilized']
    usecols_median = ['Age', 'Fee']

    len_train_proc = len(train_proc)
    df_proc = pd.concat(
        [train_proc.copy(), test_proc.copy()]).reset_index(drop=True)
    df_proc = df_proc[[groupkey]+usecols_mode+usecols_median]

    df_proc['RescuerID_cnt'] = (df_proc.RescuerID.map(
        df_proc.RescuerID.value_counts()))
    df_proc['RescuerID_cnt_dog'] = (df_proc.RescuerID.map(
        df_proc.query('Type==1').RescuerID.value_counts()))
    df_proc['RescuerID_cnt_cat'] = (df_proc.RescuerID.map(
        df_proc.query('Type==2').RescuerID.value_counts()))
    df_proc[['RescuerID_cnt', 'RescuerID_cnt_dog', 'RescuerID_cnt_cat']] = df_proc[['RescuerID_cnt', 'RescuerID_cnt_dog', 'RescuerID_cnt_cat']].fillna(0)
    df_proc['RescuerID_cnt_dog_ratio'] = df_proc['RescuerID_cnt_dog'] / (df_proc['RescuerID_cnt'])
    df_proc['RescuerID_cnt_dog_ratio_sub_0.5_abs'] = np.abs(0.5 - df_proc['RescuerID_cnt_dog_ratio'])
    df_proc['RescuerID_cnt_cat_times_dog_log'] = np.log1p(df_proc['RescuerID_cnt_cat']) * np.log1p(df_proc['RescuerID_cnt_dog'])
    df_proc['RescuerID_cnt_cat_sub_dog_log'] = np.log1p(df_proc['RescuerID_cnt_cat']) - np.log1p(df_proc['RescuerID_cnt_dog'])

    for col in usecols_mode:
        df_mode = get_mode(df_proc, groupkey, col)
        df_proc = pd.merge(df_proc, df_mode, how='left', on=groupkey)
        df_proc[f'{col}_is_mode'] = (df_proc[col] == df_proc[f'{col}_mode']).astype(int)
    df_proc['is_mode_sum'] = df_proc.filter(regex=('is_mode$')).sum(axis=1)

    for col in usecols_median:
        df_median = get_median(df_proc, groupkey, col)
        df_proc = pd.merge(df_proc, df_median, how='left', on=groupkey)
        df_proc[f'{col}_sub'] = df_proc[col] - df_proc[f'{col}_median']
        df_proc[f'{col}_sub_abs'] = np.abs(df_proc[col] - df_proc[f'{col}_median'])
        df_proc[f'{col}_sub_ratio'] = (df_proc[col] - df_proc[f'{col}_median'])/df_proc[f'{col}_median']
        df_proc[f'{col}_sub_abs_ratio'] = np.abs(df_proc[col] - df_proc[f'{col}_median'])/df_proc[f'{col}_median']

    df_proc.drop(usecols_mode+usecols_median, axis=1, inplace=True)

    train_proc = pd.concat([train_proc.reset_index(drop=True), df_proc.iloc[:len_train_proc, :].drop(groupkey, axis=1).reset_index(drop=True)], axis=1)
    test_proc = pd.concat([test_proc.reset_index(drop=True), df_proc.iloc[len_train_proc:, :].drop(groupkey, axis=1).reset_index(drop=True)], axis=1)
    return train_proc, test_proc


cont_patterns = [
    (b'US', b'United States'),
    (b'IT', b'Information Technology'),
    (b'(W|w)on\'t', b'will not'),
    (b'(C|c)an\'t', b'can not'),
    (b'(I|i)\'m', b'i am'),
    (b'(A|a)in\'t', b'is not'),
    (b'(\w+)\'ll', b'\g<1> will'),
    (b'(\w+)n\'t', b'\g<1> not'),
    (b'(\w+)\'ve', b'\g<1> have'),
    (b'(\w+)\'s', b'\g<1> is'),
    (b'(\w+)\'re', b'\g<1> are'),
    (b'(\w+)\'d', b'\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]


def prepare_for_char_n_gram(text):
    """ Simple text clean up process"""
    # 1. Go to lower case (only good for english)
    # Go to bytes_strings as I had issues removing all \n in r""
    clean = bytes(text.lower(), encoding="utf-8")
    # 2. Drop \n and  \t
    clean = clean.replace(b"\n", b" ")
    clean = clean.replace(b"\t", b" ")
    clean = clean.replace(b"\b", b" ")
    clean = clean.replace(b"\r", b" ")

    clean = clean.replace(b"\p", b" ")
    # 3. Replace english contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    # 4. Drop puntuation
    # I could have used regex package with regex.sub(b"\p{P}", " ")
    exclude = re.compile(b'[%s]' % re.escape(
        bytes(string.punctuation, encoding='utf-8')))
    clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
    # 5. Drop numbers - as a scientist I don't think numbers are toxic ;-)
    clean = re.sub(b"\d+", b" ", clean)
    # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences
    clean = re.sub(b'\s+', b' ', clean)
    # Remove ending space if any
    clean = re.sub(b'\s+$', b'', clean)
    # 7. Now replace words by words surrounded by # signs
    # e.g. my name is bond would become #my# #name# #is# #bond#
    # clean = re.sub(b"([a-z]+)", b"#\g<1>#", clean)
    #clean = re.sub(b" ", b"# #", clean)  # Replace space
    #clean = b"#" + clean + b"#"  # add leading and trailing #

    return str(clean, 'utf-8')


def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))


def coppe_img_quality(path):
    image_quality =sorted(glob.glob(path))

    blur=[]
    imageid =[]
    for filename in image_quality:
        #Blur
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        #image id
        imageid.append(filename.replace('.jpg', '').split('/')[-1])
    # Join Pixel, Blur and Image ID
    image_quality = pd.concat([pd.DataFrame(imageid, columns=['ImageId']),
                               pd.DataFrame(blur, columns=['blur'])], axis=1)

    # create the PetId variable
    image_quality['PetID'] = image_quality['ImageId'].str.split('-').str[0]

    #Mean of the Mean
    image_quality['blur_mean'] = image_quality.groupby(['PetID'])['blur'].transform('mean')
    image_quality['blur_min'] = image_quality.groupby(['PetID'])['blur'].transform('min')
    image_quality['blur_max'] = image_quality.groupby(['PetID'])['blur'].transform('max')

    image_quality = image_quality.drop(['blur', 'ImageId'], 1)
    image_quality = image_quality.drop_duplicates('PetID')
    return image_quality


# Curry function
def dense_expect1_group_max(train_feats_ex1, test_feats_ex1):
    n_components = 32
    df = pd.concat([train_feats_ex1, test_feats_ex1], axis=0)
    df.set_index('PetID', inplace=True)

    svd_ = TruncatedSVD(n_components=n_components, random_state=12345)

    features = df[[f'pic_{i}' for i in range(256)]].values

    svd_col = svd_.fit_transform(features)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix('dense_IMG_SVD_')
    svd_col = svd_col.add_suffix('_expect1')

    svd_col.index = df.index

    svd_col.reset_index(inplace=True)
    dense_img_features_exclude1 = svd_col.groupby('PetID').max()

    dense_img_features_exclude1.reset_index(inplace=True)
    print(dense_img_features_exclude1.shape)

    return dense_img_features_exclude1


def age_henkan_curry(age):
    b = 0
    if age == 0:
        b = 0
    elif age <= 6:
        b = 1
    elif age <= 12:
        b = 2
    elif age <= 12*3:
        b = 3
    elif age <= 12*6:
        b = 4
    elif age <= 12*9:
        b = 5
    else:
        b = 6
    return b


def curry_run_lgb(X_train, X_test):

    print('\nRun Curry LGB')
    categorical = ['Breed1', 'Breed2', 'Color1', 'Color2', 'Color3',
                   'RescureIDType_3', 'Health']

    # lightGBMパラメータ
    params = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'num_leaves': 100,
              'max_depth': 5,
              'learning_rate': 0.0115908699404395,
              'bagging_fraction': 0.85,
              'feature_fraction': 0.8366869513109573,
              'min_split_gain': 1.048996698247088,
              'min_child_samples': 192,
              'min_child_weight': 0.07897274227325708,
              'lambda_l2': 0.0014357035087754625,
              'lambda_l1': 0.0045713744601731466,
              'verbosity': -1,
              'data_random_seed': 1
              }

    # Additional parameters:
    early_stop = 1000
    verbose_eval = 1000
    num_rounds = 10000
    n_splits = 10  # n_splits

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    cv_scores = []
    qwk_scores = []
    feature_importance_df = pd.DataFrame()

    i = 0

    groups = X_train['RescuerID']
    for fold_ind, (train_idx, valid_idx) in enumerate(
            stratified_group_k_fold(X_train,
                                    X_train['AdoptionSpeed'].astype(int),
                                    groups, k=10, seed=123)):
        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        del X_tr['RescuerID'], X_val['RescuerID']

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        print('\ny_tr distribution: {}'.format(Counter(y_tr)))
        d_train = lgb.Dataset(X_tr, label=y_tr, categorical_feature=categorical)
        d_valid = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical)
        watchlist = [d_train, d_valid]

        print('training LGB:')
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)

        valid_pred = model.predict(X_val, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)

        optR = OptimizedRounder()
        optR.fit(valid_pred, y_val)
        coefficients = optR.coefficients()
        pred_test_y_k = optR.predict(valid_pred, coefficients)
        print("Valid Counts = ", Counter(y_val))
        print("Predicted Counts = ", Counter(pred_test_y_k))
        print("Coefficients = ", coefficients)
        qwk = quadratic_weighted_kappa(y_val, pred_test_y_k)
        rmse_score = rmse(y_val, valid_pred)
        print("QWK = ", qwk)
        cv_scores.append(rmse_score)
        qwk_scores.append(qwk)
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = X_tr.columns.values
        fold_importance_df['importance'] = model.feature_importance(
            importance_type='gain')
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df,
                                           fold_importance_df], axis=0)
        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1

    print('\ncv RMSE scores : {}'.format(cv_scores))
    print('cv mean RMSE score : {}'.format(np.mean(cv_scores)))
    print('cv std RMSE score : {}'.format(np.std(cv_scores)))
    print('cv QWK scores : {}'.format(qwk_scores))
    print('cv mean QWK score : {}'.format(np.mean(qwk_scores)))
    print('cv std QWK score : {}'.format(np.std(qwk_scores)))

    return feature_importance_df, oof_train, oof_test


def curry_run_xen(X_train, X_test):

    print('\nRun Curry LGB Xentropy')
    categorical = ['Breed1', 'Breed2', 'Color1', 'Color2', 'Color3',
                   'RescureIDType_3', 'Health']

    # lightGBMパラメータ
    params = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': 'xentropy',
              'num_leaves': 100,
              'max_depth': 5,
              'learning_rate': 0.0115908699404395,
              'bagging_fraction': 0.85,
              'feature_fraction': 0.8366869513109573,
              'min_split_gain': 1.048996698247088,
              'min_child_samples': 192,
              'min_child_weight': 0.07897274227325708,
              'lambda_l2': 0.0014357035087754625,
              'lambda_l1': 0.0045713744601731466,
              'verbosity': -1,
              'data_random_seed': 12321
              }

    # Additional parameters:
    early_stop = 1000
    verbose_eval = 1000
    num_rounds = 10000
    n_splits = 10  # n_splits

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    cv_scores = []
    qwk_scores = []
    feature_importance_df = pd.DataFrame()

    i = 0

    groups = X_train['RescuerID']
    for fold_ind, (train_idx, valid_idx) in enumerate(
            stratified_group_k_fold(X_train,
                                    X_train['AdoptionSpeed'].astype(int),
                                    groups, k=10, seed=12321)):
        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        del X_tr['RescuerID'], X_val['RescuerID']

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        print('\ny_tr distribution: {}'.format(Counter(y_tr)))
        d_train = lgb.Dataset(X_tr, label=y_tr*(1/4),
                              categorical_feature=categorical)
        d_valid = lgb.Dataset(X_val, label=y_val*(1/4),
                              categorical_feature=categorical)
        watchlist = [d_train, d_valid]

        print('training LGB:')
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)

        valid_pred = model.predict(X_val,
                                   num_iteration=model.best_iteration) * 4
        test_pred = model.predict(X_test,
                                  num_iteration=model.best_iteration) * 4

        optR = OptimizedRounder()
        optR.fit(valid_pred, y_val)
        coefficients = optR.coefficients()
        pred_test_y_k = optR.predict(valid_pred, coefficients)
        print("Valid Counts = ", Counter(y_val))
        print("Predicted Counts = ", Counter(pred_test_y_k))
        print("Coefficients = ", coefficients)
        qwk = quadratic_weighted_kappa(y_val, pred_test_y_k)
        rmse_score = rmse(y_val, valid_pred)
        print("QWK = ", qwk)
        cv_scores.append(rmse_score)
        qwk_scores.append(qwk)
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = X_tr.columns.values
        fold_importance_df['importance'] = model.feature_importance(
            importance_type='gain')
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df,
                                           fold_importance_df], axis=0)
        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1

    print('\ncv RMSE scores : {}'.format(cv_scores))
    print('cv mean RMSE score : {}'.format(np.mean(cv_scores)))
    print('cv std RMSE score : {}'.format(np.std(cv_scores)))
    print('cv QWK scores : {}'.format(qwk_scores))
    print('cv mean QWK score : {}'.format(np.mean(qwk_scores)))
    print('cv std QWK score : {}'.format(np.std(qwk_scores)))

    return oof_train, oof_test


def curry_run_gos(X_train, X_test):

    print('\nRun Curry GOSS')
    categorical = ['Breed1', 'Breed2', 'Color1', 'Color2', 'Color3',
                   'RescureIDType_3', 'Health']

    # lightGBMパラメータ
    params = {'application': 'regression',
              'boosting': 'goss',
              'metric': 'rmse',
              'num_leaves': 100,
              'max_depth': 5,
              'learning_rate': 0.0045908699404395,
              'bagging_fraction': 0.85,
              'feature_fraction': 0.8366869513109573,
              'min_split_gain': 1.048996698247088,
              'min_child_samples': 192,
              'min_child_weight': 0.07897274227325708,
              'lambda_l2': 0.0014357035087754625,
              'lambda_l1': 0.0045713744601731466,
              'verbosity': -1,
              'data_random_seed': 121,
              'top_rate': 0.2,
              'other_rate': 0.1
              }

    # Additional parameters:
    early_stop = 1000
    verbose_eval = 1000
    num_rounds = 10000
    n_splits = 10  # n_splits

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    cv_scores = []
    qwk_scores = []
    feature_importance_df = pd.DataFrame()

    i = 0

    groups = X_train['RescuerID']
    for fold_ind, (train_idx, valid_idx) in enumerate(
            stratified_group_k_fold(X_train,
                                    X_train['AdoptionSpeed'].astype(int),
                                    groups, k=10, seed=121)):
        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        del X_tr['RescuerID'], X_val['RescuerID']

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        print('\ny_tr distribution: {}'.format(Counter(y_tr)))
        d_train = lgb.Dataset(X_tr, label=y_tr,
                              categorical_feature=categorical)
        d_valid = lgb.Dataset(X_val, label=y_val,
                              categorical_feature=categorical)
        watchlist = [d_train, d_valid]

        print('training LGB:')
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)

        valid_pred = model.predict(X_val,
                                   num_iteration=model.best_iteration)
        test_pred = model.predict(X_test,
                                  num_iteration=model.best_iteration)

        optR = OptimizedRounder()
        optR.fit(valid_pred, y_val)
        coefficients = optR.coefficients()
        pred_test_y_k = optR.predict(valid_pred, coefficients)
        print("Valid Counts = ", Counter(y_val))
        print("Predicted Counts = ", Counter(pred_test_y_k))
        print("Coefficients = ", coefficients)
        qwk = quadratic_weighted_kappa(y_val, pred_test_y_k)
        rmse_score = rmse(y_val, valid_pred)
        print("QWK = ", qwk)
        cv_scores.append(rmse_score)
        qwk_scores.append(qwk)
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = X_tr.columns.values
        fold_importance_df['importance'] = model.feature_importance(
            importance_type='gain')
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df,
                                           fold_importance_df], axis=0)
        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1

    print('\ncv RMSE scores : {}'.format(cv_scores))
    print('cv mean RMSE score : {}'.format(np.mean(cv_scores)))
    print('cv std RMSE score : {}'.format(np.std(cv_scores)))
    print('cv QWK scores : {}'.format(qwk_scores))
    print('cv mean QWK score : {}'.format(np.mean(qwk_scores)))
    print('cv std QWK score : {}'.format(np.std(qwk_scores)))

    return oof_train, oof_test


def curry_run_cat(X_train, X_test):

    print('\nRun Curry CatBoost')
    n_splits = 10
    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))
    cv_scores = []
    qwk_scores = []
    i = 0
    groups = X_train['RescuerID']
    for fold_ind, (train_idx, valid_idx) in enumerate(
            stratified_group_k_fold(X_train,
                                    X_train['AdoptionSpeed'].astype(int),
                                    groups, k=10, seed=111)):
        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        del X_tr['RescuerID'], X_val['RescuerID']

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        categorical = ['Breed1', 'Breed2',
                       'Color1', 'Color2', 'Color3',
                       'RescureIDType_3', 'Health'
                       ]

        categorical_features_pos = column_index(X_tr, categorical)

        d_train = cb.Pool(X_tr, label=y_tr,
                          cat_features=categorical_features_pos) # weight=trn_weight
        d_valid = cb.Pool(X_val, label=y_val,
                          cat_features=categorical_features_pos)

        print('\ny_tr distribution: {}'.format(Counter(y_tr)))

        print('training CatBoost:')
        # catboost パラメータ
        model = cb.CatBoostRegressor(loss_function='RMSE',
                                     max_depth=4,
                                     learning_rate=0.02710676343253643,
                                     eval_metric='RMSE',
                                     iterations=10000,
                                     early_stopping_rounds=800,
                                     task_type="GPU",
                                     random_seed=0,
                                     l2_leaf_reg=0.0002394765700985016,
                                     bootstrap_type='Bayesian',
                                     bagging_temperature=0.18030025567019173,
                                     random_strength=1,
                                     use_best_model=True,
                                     border_count=32,
                                     boosting_type='Plain')

        model.fit(d_train, eval_set=d_valid, use_best_model=True,
                  verbose_eval=1000)

        valid_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        optR = OptimizedRounder()
        optR.fit(valid_pred, y_val)
        coefficients = optR.coefficients()
        pred_test_y_k = optR.predict(valid_pred, coefficients)
        print("Valid Counts = ", Counter(y_val))
        print("Predicted Counts = ", Counter(pred_test_y_k))
        print("Coefficients = ", coefficients)
        qwk = quadratic_weighted_kappa(y_val, pred_test_y_k)
        rmse_score = rmse(y_val, valid_pred)
        print("QWK = ", qwk)
        cv_scores.append(rmse_score)
        qwk_scores.append(qwk)
        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1

    print('\ncv RMSE scores : {}'.format(cv_scores))
    print('cv mean RMSE score : {}'.format(np.mean(cv_scores)))
    print('cv std RMSE score : {}'.format(np.std(cv_scores)))
    print('cv QWK scores : {}'.format(qwk_scores))
    print('cv mean QWK score : {}'.format(np.mean(qwk_scores)))
    print('cv std QWK score : {}'.format(np.std(qwk_scores)))

    del model
    torch.cuda.empty_cache()

    return oof_train, oof_test


# curry_xgb_run
def curry_run_xgb(X_train, X_test):

    print('\nRun Curry XGB')

    X_train_tmp = X_train.copy()
    X_test_tmp = X_test.copy()

    xgb_params = {
        'eval_metric': 'rmse',
        'seed': 11111,
        'eta': 0.010043709293497441,
        'subsample': 0.8408695712091527,
        'colsample_bytree': 0.2382534303476766,
        'min_child_weight': 286,
        'alpha': 0.26074467275533564,
        'lambda': 59.07131602437971,
        'tree_method': 'gpu_hist',
        'device': 'gpu',
        'silent': 1
        }

    n_splits = 10
    verbose_eval = 1000
    num_rounds = 20000
    early_stop = 1000

    oof_train = np.zeros((X_train_tmp.shape[0]))
    oof_test = np.zeros((X_test_tmp.shape[0], n_splits))
    cv_scores = []
    qwk_scores = []

    i = 0

    groups = X_train['RescuerID']
    for fold_ind, (train_idx, valid_idx) in enumerate(
            stratified_group_k_fold(X_train,
                                    X_train['AdoptionSpeed'].astype(int),
                                    groups, k=10, seed=11111)):
        X_tr = X_train_tmp.iloc[train_idx, :]
        X_val = X_train_tmp.iloc[valid_idx, :]

        del X_tr['RescuerID'], X_val['RescuerID']

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        d_train = xgb.DMatrix(data=X_tr, label=y_tr,
                              feature_names=X_tr.columns)
        d_valid = xgb.DMatrix(data=X_val, label=y_val,
                              feature_names=X_val.columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds,
                          evals=watchlist,
                          early_stopping_rounds=early_stop,
                          verbose_eval=verbose_eval, params=xgb_params)

        valid_pred = model.predict(
            xgb.DMatrix(X_val, feature_names=X_val.columns),
            ntree_limit=model.best_ntree_limit)
        test_pred = model.predict(
            xgb.DMatrix(X_test_tmp, feature_names=X_test.columns),
            ntree_limit=model.best_ntree_limit)

        optR = OptimizedRounder()
        optR.fit(valid_pred, y_val)
        coefficients = optR.coefficients()
        pred_test_y_k = optR.predict(valid_pred, coefficients)
        print("Valid Counts = ", Counter(y_val))
        print("Predicted Counts = ", Counter(pred_test_y_k))
        print("Coefficients = ", coefficients)
        qwk = quadratic_weighted_kappa(y_val, pred_test_y_k)
        rmse_score = rmse(y_val, valid_pred)
        print("QWK = ", qwk)
        cv_scores.append(rmse_score)
        qwk_scores.append(qwk)

        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1

    print('\ncv RMSE scores : {}'.format(cv_scores))
    print('cv mean RMSE score : {}'.format(np.mean(cv_scores)))
    print('cv std RMSE score : {}'.format(np.std(cv_scores)))
    print('cv QWK scores : {}'.format(qwk_scores))
    print('cv mean QWK score : {}'.format(np.mean(qwk_scores)))
    print('cv std QWK score : {}'.format(np.std(qwk_scores)))
    model.__del__()

    return oof_train, oof_test


# Fujita Process
def FujitaProcess(train_proc, test_proc, train_feats, test_feats):

    print('\nRun Fujita Process')
    img_features = fujita_img_svd(train_feats, test_feats)

    train_proc, test_proc = fujita_text_svd(train_proc, test_proc)

    # text sparce embedding 1
    vectorizer = TfidfVectorizer(min_df=3, max_features=10000,
                                 strip_accents='unicode', analyzer='word',
                                 token_pattern=r'\w{1,}',
                                 ngram_range=(1, 3),
                                 binary=True,
                                 stop_words='english')
    svd = TruncatedSVD(n_components=16, random_state=916)
    train_proc, test_proc = coppe_desc_svd(train_proc, test_proc, vectorizer,
                                           svd, 'tfidf_svd')

    # text sparce embedding 3
    vectorizer = CountVectorizer(min_df=3, max_features=10000,
                                 strip_accents='unicode', analyzer='char',
                                 token_pattern=r'\w{1,}',
                                 ngram_range=(1, 1),
                                 binary=False,
                                 stop_words='english')
    svd = TruncatedSVD(n_components=16, random_state=916)
    train_proc, test_proc = coppe_desc_svd(train_proc, test_proc, vectorizer,
                                           svd, 'count_svd')

    # text sparce embedding 2
    vectorizer = TfidfVectorizer(min_df=3, max_features=10000,
                                 strip_accents='unicode', analyzer='char',
                                 token_pattern=r'\w{1,}',
                                 ngram_range=(1, 3),
                                 binary=False,
                                 stop_words='english')
    nmf = NMF(n_components=16, random_state=916)
    train_proc, test_proc = coppe_desc_svd(train_proc, test_proc, vectorizer,
                                           nmf, 'tfidf_nmf')

    # text sparce embedding 4
    vectorizer = CountVectorizer(min_df=3, max_features=10000,
                                 strip_accents='unicode', analyzer='word',
                                 token_pattern=r'\w{1,}',
                                 ngram_range=(1, 1),
                                 binary=True,
                                 stop_words='english')
    nmf = NMF(n_components=16, random_state=916)
    train_proc, test_proc = coppe_desc_svd(train_proc, test_proc, vectorizer,
                                           nmf, 'count_nmf')

    X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)

    # Select subsets of columns:
    text_columns = ['Description', 'metadata_annots_top_desc',
                    'sentiment_entities']

    # RescuerID will also be dropped, as a feature based on this column
    # will be extracted independently
    X = X.merge(img_features, how='left', on='PetID')

    # Count RescuerID occurrences:
    rescuer1 = X.groupby(['RescuerID'])['PetID'].count().reset_index()
    rescuer1.columns = ['RescuerID', 'RescuerID_COUNT']
    rescuer2 = X.groupby(['RescuerID'])['Age'].mean().reset_index()
    rescuer2.columns = ['RescuerID', 'RescuerID_Age_MEAN']

    # Merge as another feature onto main DF:
    X = X.merge(rescuer1, how='left', on='RescuerID')
    X = X.merge(rescuer2, how='left', on=['RescuerID'])

    # Subset text features:
    X_text = X[text_columns]

    for i in X_text.columns:
        X_text.loc[:, i] = X_text.loc[:, i].fillna('none')

    # text statistics
    for i in X_text.columns:
        X[i] = X[i].astype(str)
        X[i] = X[i].str.lower()
        X[i + '_num_chars'] = X[i].apply(lambda x: len(x))
        X[i + '_num_words'] = X[i].apply(lambda x: len(x.split()))
        X[i + '_num_unique_words'] = X[i].apply(
            lambda x: len(set(w for w in x.split())))
        X[i + '_words_vs_unique'] = X[i + '_num_unique_words'] / (X[i + '_num_words'] + 1) * 100

    X['RescuerID_Age_MEAN_DIFF'] = X['Age'] - X['RescuerID_Age_MEAN']
    X['Fee'] = np.log1p(X['Fee'])
    X['RescuerID_COUNT'] = np.log1p(X['RescuerID_COUNT'])
    X['Quantity'] = np.log1p(X['Quantity'])
    X['image_size_sum'] = np.log1p(X['image_size_sum'])
    X['image_size_mean'] = np.log1p(X['image_size_mean'])
    X['image_size_std'] = np.log1p(X['image_size_std'])
    X['width_sum'] = np.log1p(X['width_sum'])
    X['width_mean'] = np.log1p(X['width_mean'])
    X['width_std'] = np.log1p(X['width_std'])
    X['height_sum'] = np.log1p(X['height_sum'])
    X['height_mean'] = np.log1p(X['height_mean'])
    X['height_std'] = np.log1p(X['height_std'])

    X['Pure_Breed'] = X.apply(lambda x: pure_breed(x), axis=1)
    X['Single_Color'] = X.apply(lambda x: single_color(x), axis=1)

    X["state_gdp"] = X.State.map(state_gdp)
    X["state_population"] = X.State.map(state_population)
    X["state_area"] = X.State.map(state_area)
    X["state_population_per_area"] = X.State.map(state_population_per_area)
    # newly added
    X["state_HDI"] = X.State.map(state_HDI)

    X['Description_num_chars'] = np.log1p(X['Description_num_chars'])
    X['Description_num_words'] = np.log1p(X['Description_num_words'])
    X['Description_num_unique_words'] = np.log1p(
        X['Description_num_unique_words'])
    X['metadata_annots_top_desc_num_chars'] = np.log1p(
        X['metadata_annots_top_desc_num_chars'])
    X['metadata_annots_top_desc_num_words'] = np.log1p(
        X['metadata_annots_top_desc_num_words'])
    X['metadata_annots_top_desc_num_unique_words'] = np.log1p(
        X['metadata_annots_top_desc_num_unique_words'])
    X['sentiment_entities_num_chars'] = np.log1p(
        X['sentiment_entities_num_chars'])
    X['sentiment_entities_num_words'] = np.log1p(
        X['sentiment_entities_num_words'])
    X['sentiment_entities_num_unique_words'] = np.log1p(
        X['sentiment_entities_num_unique_words'])

    # Remove raw text columns:
    for i in X_text.columns:
        X = X.drop(i, axis=1)

    to_drop_columns = ['PetID', 'Name', 'VideoAmt', 'main_breed_Type',
                       'second_breed_Type', 'main_breed_BreedName',
                       'second_breed_BreedName',
                       'metadata_metadata_crop_conf_SUM',
                       'metadata_metadata_crop_importance_SUM',
                       'PhotoAmt', 'bounding_confidence',
                       'metadata_metadata_crop_conf_MEAN',
                       'RescuerID_Age_MEAN', 'label_description',
                       ]

    X = X.drop(to_drop_columns, axis=1)

    # Check final df shape:
    print('X shape: {}'.format(X.shape))

    # Split into train and test again:
    X_train = X.loc[np.isfinite(X.AdoptionSpeed), :]
    X_test = X.loc[~np.isfinite(X.AdoptionSpeed), :]

    # Remove missing target column from test:
    X_test = X_test.drop(['AdoptionSpeed'], axis=1)
    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
    assert X_train.shape[0] == train.shape[0]
    assert X_test.shape[0] == test.shape[0]

    # Check if columns between the two DFs are the same:
    train_cols = X_train.columns.tolist()
    train_cols.remove('AdoptionSpeed')
    test_cols = X_test.columns.tolist()
    assert np.all(train_cols == test_cols)

    del X_test['RescuerID']

    print(X_train.shape)
    print(X_test.shape)

    feature_importance_df, fujita_oof_train_lgb, fujita_oof_test_lgb = \
        fujita_run_lgb(X_train, X_test)

    imports = feature_importance_df.groupby('feature')[
        'feature', 'importance'].mean().reset_index()
    print(imports.sort_values('importance', ascending=False))

    fujita_oof_train_xen, fujita_oof_test_xen = fujita_run_xen(X_train, X_test)

    return fujita_oof_train_lgb, fujita_oof_test_lgb, \
           fujita_oof_train_xen, fujita_oof_test_xen, X_train


# Nakama Process
def NakamaProcess(train_proc, test_proc, train_feats, test_feats):
    print('\nRun Nakama Process')
    # RescuerID_COUNT
    concat = train_proc.groupby(['RescuerID'], as_index=False)['PetID'].count()
    concat = pd.DataFrame(concat).rename(columns={'PetID': 'RescuerID_COUNT'})
    train_proc = train_proc.merge(concat, on=['RescuerID'], how='left')
    concat = test_proc.groupby(['RescuerID'], as_index=False)['PetID'].count()
    concat = pd.DataFrame(concat).rename(columns={'PetID': 'RescuerID_COUNT'})
    test_proc = test_proc.merge(concat, on=['RescuerID'], how='left')

    train_proc['health'] = train_proc['Vaccinated'].astype(str) + '_' \
                           + train_proc['Dewormed'].astype(str) + '_' \
                           + train_proc['Sterilized'].astype(str) + '_' \
                           + train_proc['Health'].astype(str)
    test_proc['health'] = test_proc['Vaccinated'].astype(str) + '_' \
                          + test_proc['Dewormed'].astype(str) + '_' \
                          + test_proc['Sterilized'].astype(str) + '_' \
                          + test_proc['Health'].astype(str)

    train_proc['health1'] = ((train_proc[['Health', 'Vaccinated',
                                          'Dewormed', 'Sterilized']] == 1) * 1).sum(axis=1)
    test_proc['health1'] = ((test_proc[['Health', 'Vaccinated',
                                        'Dewormed', 'Sterilized']] == 1) * 1).sum(axis=1)
    train_proc['health3'] = ((train_proc[['Vaccinated', 'Dewormed',
                                          'Sterilized']] == 3) * 1).sum(axis=1)
    test_proc['health3'] = ((test_proc[['Vaccinated', 'Dewormed',
                                        'Sterilized']] == 3) * 1).sum(axis=1)

    # ['pure_breed', 'Description_length']
    train_proc['pure_breed'] = (train_proc['Breed2'] != 0) * 1
    test_proc['pure_breed'] = (test_proc['Breed2'] != 0) * 1
    train_proc['Description_length'] = train_proc['Description'].apply(
        lambda x: len(str(x)))
    test_proc['Description_length'] = test_proc['Description'].apply(
        lambda x: len(str(x)))

    train_proc = only_cat_RescuerID(train_proc)
    test_proc = only_cat_RescuerID(test_proc)
    train_proc = big_return_RescuerID(train_proc)
    test_proc = big_return_RescuerID(test_proc)
    train_proc = health_var(train_proc)
    test_proc = health_var(test_proc)

    img_features = nakama_img_svd(train_feats, test_feats)
    train_proc, test_proc = add_breed_prefer(train_proc, test_proc)
    train_proc, test_proc = nakama_desc_svd(train_proc, test_proc)

    X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)

    # Select subsets of columns:
    text_columns = ['Description', 'metadata_annots_top_desc',
                    'sentiment_entities', 'label_description']
    categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName',
                           'Color1', 'Color2', 'Color3', 'health']

    # Factorize categorical columns:
    for i in categorical_columns:
        X.loc[:, i] = pd.factorize(X.loc[:, i])[0]

    # Subset text features:
    X_text = X[text_columns]

    for i in X_text.columns:
        X_text.loc[:, i] = X_text.loc[:, i].fillna('none')

    X['Length_metadata_annots_top_desc'] = \
        X_text['metadata_annots_top_desc'].map(len)
    X['Lengths_sentiment_entities'] = X_text['sentiment_entities'].map(len)

    X = nakama_text_svd(X, X_text)

    X = X.merge(img_features, how='left', on='PetID')
    X['Fee'] = np.log1p(X['Fee'])
    X['RescuerID_COUNT'] = np.log1p(X['RescuerID_COUNT'])
    X['Quantity'] = np.log1p(X['Quantity'])
    X['image_size_sum'] = np.log1p(X['image_size_sum'])
    X['image_size_mean'] = np.log1p(X['image_size_mean'])
    X['image_size_std'] = np.log1p(X['image_size_std'])
    X['width_sum'] = np.log1p(X['width_sum'])
    X['width_mean'] = np.log1p(X['width_mean'])
    X['width_std'] = np.log1p(X['width_std'])
    X['height_sum'] = np.log1p(X['height_sum'])
    X['height_mean'] = np.log1p(X['height_mean'])
    X['height_std'] = np.log1p(X['height_std'])
    X["state_gdp"] = X.State.map(state_gdp)
    X["state_population"] = X.State.map(state_population)
    X["state_area"] = X.State.map(state_area)
    X["state_population_per_area"] = X.State.map(state_population_per_area)
    # newly added
    X["state_HDI"] = X.State.map(state_HDI)

    # Split into train and test again:
    X_train = X.loc[np.isfinite(X.AdoptionSpeed), :]
    X_test = X.loc[~np.isfinite(X.AdoptionSpeed), :]

    # Remove missing target column from test:
    X_test = X_test.drop(['AdoptionSpeed'], axis=1)

    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
    assert X_train.shape[0] == train.shape[0]
    assert X_test.shape[0] == test.shape[0]

    # Check if columns between the two DFs are the same:
    train_cols = X_train.columns.tolist()
    train_cols.remove('AdoptionSpeed')
    test_cols = X_test.columns.tolist()
    assert np.all(train_cols == test_cols)

    X_train['Age_better'] = X_train['Age'] / (X_train['RescuerID_Age_var'] + 2)
    X_train.loc[X_train['RescuerID_Age_var'] < 0, 'Age_better'] = -1
    X_test['Age_better'] = X_test['Age'] / (X_test['RescuerID_Age_var'] + 2)
    X_test.loc[X_test['RescuerID_Age_var'] < 0, 'Age_better'] = -1

    del X_test['RescuerID']

    FEATS_EXCLUDED = ['Name', 'main_breed_BreedName', 'second_breed_BreedName',
                      'PetID',
                      'main_breed_Type', 'second_breed_Type',
                      'bounding_confidence',
                      'VideoAmt']

    for i in X_train.columns:
        if i.find('TFIDF_label_description') == -1:
            None
        else:
            FEATS_EXCLUDED.append(i)

    X_train = X_train.drop(FEATS_EXCLUDED, axis=1)
    X_test = X_test.drop(FEATS_EXCLUDED, axis=1)

    print(X_train.shape)
    print(X_test.shape)

    feature_importance_df, nakama_oof_train_lgb, nakama_oof_test_lgb = \
        nakama_run_lgb(X_train, X_test)

    imports = feature_importance_df.groupby('feature')[
        'feature', 'importance'].mean().reset_index()
    print(imports.sort_values('importance', ascending=False))

    nakama_oof_train_cat, nakama_oof_test_cat = nakama_run_cat(X_train, X_test)
    nakama_oof_train_xgb, nakama_oof_test_xgb = nakama_run_xgb(X_train, X_test)

    return nakama_oof_train_lgb, nakama_oof_test_lgb, \
           nakama_oof_train_cat, nakama_oof_test_cat, \
           nakama_oof_train_xgb, nakama_oof_test_xgb


# curry Process
def CurryProcess(train_proc, test_proc, train_feats, test_feats):
    print('\nRun Curry Process')
    # RescuerID_COUNT
    concat = train_proc.groupby(['RescuerID'], as_index=False)['PetID'].count()
    concat = pd.DataFrame(concat).rename(columns={'PetID': 'RescuerID_COUNT'})
    train_proc = train_proc.merge(concat, on=['RescuerID'], how='left')
    concat = test_proc.groupby(['RescuerID'], as_index=False)['PetID'].count()
    concat = pd.DataFrame(concat).rename(columns={'PetID': 'RescuerID_COUNT'})
    test_proc = test_proc.merge(concat, on=['RescuerID'], how='left')

    train_proc['Description_length'] = train_proc['Description'].apply(
        lambda x: len(str(x)))
    test_proc['Description_length'] = test_proc['Description'].apply(
        lambda x: len(str(x)))

    img_features = nakama_img_svd(train_feats, test_feats)

    X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)

    # Select subsets of columns:
    text_columns = ['Description', 'metadata_annots_top_desc',
                    'sentiment_entities']
    categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName']

    # Factorize categorical columns:
    for i in categorical_columns:
        X.loc[:, i] = pd.factorize(X.loc[:, i])[0]

    # Subset text features:
    X_text = X[text_columns]

    for i in X_text.columns:
        X_text.loc[:, i] = X_text.loc[:, i].fillna('none')

    X['Length_metadata_annots_top_desc'] = \
        X_text['metadata_annots_top_desc'].map(len)

    X = nakama_text_svd(X, X_text)

    X = X.merge(img_features, how='left', on='PetID')

    # dominantcolorの特徴の統計量
    X_metadata_cr = X.loc[:, ['dominant_blue', 'dominant_green', 'dominant_red']]
    X['dominant_color_std'] = X_metadata_cr.apply(lambda x: x.std(), axis=1).values
    X['dominant_color_mean'] = X_metadata_cr.apply(lambda x: x.mean(), axis=1).values
    X['dominant_color_max'] = X_metadata_cr.apply(lambda x: x.max(), axis=1).values
    X['dominant_color_min'] = X_metadata_cr.apply(lambda x: x.min(), axis=1).values

    # RescuerIDでグループ
    aggs = {'Type': ['mean', 'std', 'median', skew, kurtosis],
            'height_mean': ['mean', 'std'],
            'PhotoAmt': ['sum', 'std', 'mean', 'max', 'min', skew, kurtosis],
            'Gender': [skew],
            'Breed1': ['mean', 'std']}

    X_agg = X.groupby('RescuerID').agg(aggs)
    X_agg.columns = [f'RescuerID_{c[0]}_{c[1].upper()}' for c in X_agg.columns]
    X_agg['RescureIDType_3'] = [3 if (i % 1) > 0 else i for i in X_agg['RescuerID_Type_MEAN']]
    X = X.merge(X_agg.reset_index(), on='RescuerID', how='left')

    X["state_gdp"] = X.State.map(state_gdp)
    X["state_population"] = X.State.map(state_population)
    X["state_population_per_area"] = X.State.map(state_population_per_area)

    X['Free'] = [0 if i > 0 else 1 for i in X['Fee']]

    X['Categorical_Age'] = X['Age'].map(age_henkan_curry)

    # state_count
    state_count = X.groupby(['State'])['PetID'].count().reset_index()
    state_count.columns = ['State', 'State_COUNT']
    X = X.merge(state_count, how='left', on='State')

    # Split into train and test again:
    X_train = X.loc[np.isfinite(X.AdoptionSpeed), :]
    X_test = X.loc[~np.isfinite(X.AdoptionSpeed), :]

    # Remove missing target column from test:
    X_test = X_test.drop(['AdoptionSpeed'], axis=1)

    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
    assert X_train.shape[0] == train.shape[0]
    assert X_test.shape[0] == test.shape[0]

    # Check if columns between the two DFs are the same:
    train_cols = X_train.columns.tolist()
    train_cols.remove('AdoptionSpeed')
    test_cols = X_test.columns.tolist()
    assert np.all(train_cols == test_cols)

    del X_test['RescuerID']

    FEATS_EXCLUDED = ['Name', #'RescuerID',
                      'PetID', #'Breed1', 'Breed2',
                      # 'main_breed_Type', 'second_breed_Type',
                      # 'only_dog_RescuerID',  'only_cat_RescuerID',
                      'bounding_confidence', #'State',
                      # 'refundable_in_Description',
                      # 'RM_in_Description', 'chinese_in_Description',
                      'label_description',  #  'VideoAmt'
                      'State', 'metadata_metadata_color_score_MEAN',
                      'metadata_metadata_color_score_SUM',
                      'metadata_metadata_color_score_STD',
                      'metadata_metadata_color_pixelfrac_MEAN',
                      'metadata_metadata_color_pixelfrac_SUM',
                      'metadata_metadata_color_pixelfrac_STD',
                      'metadata_metadata_crop_conf_MEAN',
                      'metadata_metadata_crop_conf_SUM',
                      'metadata_metadata_crop_conf_STD',
                      'metadata_metadata_crop_importance_MEAN',
                      'metadata_metadata_crop_importance_SUM',
                      'metadata_metadata_crop_importance_STD',
                      'Length_metadata_annots_top_desc',
                      'vertex_x', 'vertex_y',
                      'bounding_importance', 'dominant_pixel_frac',
                      'dominant_score', 'label_score',
                      'sentiment_sentiment_magnitude_MEAN',
                      'sentiment_sentiment_score_MEAN',
                      'sentiment_sentiment_document_magnitude_MEAN',
                      'sentiment_sentiment_document_score_MEAN']

    for i in X_train.columns:
        if i.find('TFIDF_label_description') == -1:
            None
        else:
            FEATS_EXCLUDED.append(i)

    X_train = X_train.drop(FEATS_EXCLUDED, axis=1)
    X_test = X_test.drop(FEATS_EXCLUDED, axis=1)
    print(X_train.shape)
    print(X_test.shape)

    feature_importance_df, curry_oof_train_lgb, curry_oof_test_lgb = \
        curry_run_lgb(X_train, X_test)

    imports = feature_importance_df.groupby('feature')[
        'feature', 'importance'].mean().reset_index()
    print(imports.sort_values('importance', ascending=False))

    curry_oof_train_cat, curry_oof_test_cat = curry_run_cat(X_train, X_test)
    curry_oof_train_xgb, curry_oof_test_xgb = curry_run_xgb(X_train, X_test)
    curry_oof_train_xen, curry_oof_test_xen = curry_run_xen(X_train, X_test)

    return curry_oof_train_lgb, curry_oof_test_lgb, \
           curry_oof_train_cat, curry_oof_test_cat, \
           curry_oof_train_xgb, curry_oof_test_xgb, \
           curry_oof_train_xen, curry_oof_test_xen


pet_parser = PetFinderParser()
seed_everything(seed=1)


# Main
def main():
    train_proc, test_proc = load_image_color(train, test)
    print(train_proc.shape)
    print(test_proc.shape)

    train_df_imgs, train_dfs_sentiment, train_dfs_metadata,\
    test_df_imgs, test_dfs_sentiment, test_dfs_metadata\
        = load_meta_sentiment(train_proc, test_proc, debug=False)
    print(train_df_imgs.shape)
    print(test_df_imgs.shape)

    train_metadata_gr, train_sentiment_gr,\
    train_metadata_desc, train_sentiment_desc,\
    test_metadata_gr, test_sentiment_gr, \
    test_metadata_desc, test_sentiment_desc\
        = aggregate_meta_sentiment(train_dfs_sentiment,
                                   train_dfs_metadata,
                                   test_dfs_sentiment,
                                   test_dfs_metadata)
    print(train_metadata_gr.shape)
    print(train_sentiment_gr.shape)
    print(train_metadata_desc.shape)
    print(train_sentiment_desc.shape)
    print(test_metadata_gr.shape)
    print(test_sentiment_gr.shape)
    print(test_metadata_desc.shape)
    print(test_sentiment_desc.shape)

    densenet = make_densenet(densenet_weights)
    mobilenet = make_mobilenet(mobilenet_weights)
    nima3 = make_nima3(nima_weights_3)

    image_quality_train, train_feats, train_feats_mobile, train_feats_nima3,\
    train_feats_ex1 \
        = image_models_prediction('train', densenet, mobilenet, nima3)

    image_quality_test, test_feats, test_feats_mobile, test_feats_nima3,\
    test_feats_ex1 = \
        image_models_prediction('test', densenet, mobilenet, nima3)

    del densenet, mobilenet, nima3
    K.clear_session()

    mobilenet_img_first = img_first_svd(train_feats_mobile, test_feats_mobile)
    print(mobilenet_img_first.shape)
    train_proc = train_proc.merge(mobilenet_img_first, how='left', on='PetID')
    test_proc = test_proc.merge(mobilenet_img_first, how='left', on='PetID')
    print(train_proc.shape)
    print(test_proc.shape)

    train_proc, test_proc = merge_meta(train_proc, train_sentiment_gr,
                                       train_metadata_gr, train_metadata_desc,
                                       train_sentiment_desc,
                                       test_proc, test_sentiment_gr,
                                       test_metadata_gr, test_metadata_desc,
                                       test_sentiment_desc)
    print(train_proc.shape)
    print(test_proc.shape)

    train_proc, test_proc = merge_label(train_proc, test_proc)
    print(train_proc.shape)
    print(test_proc.shape)

    train_proc = train_proc.merge(image_quality_train, how='left', on='PetID')
    test_proc = test_proc.merge(image_quality_test, how='left', on='PetID')
    print(train_proc.shape)
    print(test_proc.shape)

    img_features_ex1 = dense_expect1_group_max(train_feats_ex1, test_feats_ex1)
    print(img_features_ex1.shape)
    train_proc = train_proc.merge(img_features_ex1, how='left', on='PetID')
    test_proc = test_proc.merge(img_features_ex1, how='left', on='PetID')
    print(train_proc.shape)
    print(test_proc.shape)

    agg_imgs = kaeru_feature(train_df_imgs, test_df_imgs)
    print(agg_imgs.shape)
    train_proc = train_proc.merge(agg_imgs, how='left', on='PetID')
    test_proc = test_proc.merge(agg_imgs, how='left', on='PetID')
    print(train_proc.shape)
    print(test_proc.shape)

    train_proc = train_proc.merge(train_feats_nima3, how='left', on='PetID')
    test_proc = test_proc.merge(test_feats_nima3, how='left', on='PetID')
    print(train_proc.shape)
    print(test_proc.shape)

    del agg_imgs, image_quality_train, image_quality_test
    del mobilenet_img_first
    del img_features_ex1, train_feats_ex1, test_feats_ex1
    del train_feats_nima3, test_feats_nima3
    gc.collect()

    fujita_oof_train_lgb, fujita_oof_test_lgb, \
    fujita_oof_train_xen, fujita_oof_test_xen, X_train = \
        FujitaProcess(train_proc, test_proc, train_feats, test_feats)
    gc.collect()

    nakama_oof_train_lgb, nakama_oof_test_lgb, \
    nakama_oof_train_cat, nakama_oof_test_cat,\
    nakama_oof_train_xgb, nakama_oof_test_xgb = \
        NakamaProcess(train_proc, test_proc, train_feats, test_feats)
    gc.collect()

    curry_oof_train_lgb, curry_oof_test_lgb, \
    curry_oof_train_cat, curry_oof_test_cat, \
    curry_oof_train_xgb, curry_oof_test_xgb, \
    curry_oof_train_xen, curry_oof_test_xen = \
        CurryProcess(train_proc, test_proc, train_feats, test_feats)
    gc.collect()

    fujita_oof_test_lgb = np.mean(fujita_oof_test_lgb, axis=1)
    fujita_oof_test_xen = np.mean(fujita_oof_test_xen, axis=1)
    nakama_oof_test_lgb = np.mean(nakama_oof_test_lgb, axis=1)
    nakama_oof_test_cat = np.mean(nakama_oof_test_cat, axis=1)
    nakama_oof_test_xgb = np.mean(nakama_oof_test_xgb, axis=1)
    curry_oof_test_cat = np.mean(curry_oof_test_cat, axis=1)
    curry_oof_test_lgb = np.mean(curry_oof_test_lgb, axis=1)
    curry_oof_test_xgb = np.mean(curry_oof_test_xgb, axis=1)
    curry_oof_test_xen = np.mean(curry_oof_test_xen, axis=1)

    train_stack = np.vstack([fujita_oof_train_lgb, fujita_oof_train_xen,
                             nakama_oof_train_lgb, nakama_oof_train_cat,
                             nakama_oof_train_xgb,
                             curry_oof_train_lgb, curry_oof_train_cat,
                             curry_oof_train_xgb, curry_oof_train_xen
                             ]
                            ).transpose()
    test_stack = np.vstack([fujita_oof_test_lgb, fujita_oof_test_xen,
                            nakama_oof_test_lgb, nakama_oof_test_cat,
                            nakama_oof_test_xgb,
                            curry_oof_test_lgb, curry_oof_test_cat,
                            curry_oof_test_xgb, curry_oof_test_xen
                            ]
                           ).transpose()

    print('\nTrain Corr Matrix', np.corrcoef(train_stack.transpose()))
    print('\nTest CorrMatrix', np.corrcoef(test_stack.transpose()))

    oof, predictions = ridge_stack(train_stack, test_stack, X_train)

    print('\nFinal OOF RMSE = ', rmse(X_train['AdoptionSpeed'], oof))

    # Compute QWK based on OOF train predictions:
    optR = OptimizedRounder()
    optR.fit(oof, X_train['AdoptionSpeed'].values)
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(oof, coefficients)
    print("Final Valid Counts = ", Counter(X_train['AdoptionSpeed'].values))
    print("Final Predicted Counts = ", Counter(pred_test_y_k))
    print("Final Coefficients = ", coefficients)
    qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values,
                                   pred_test_y_k)
    print("Final OOF QWK = ", qwk)

    # print(coefficients_)
    pred_final =  optR.predict(predictions, coefficients)
    print(Counter(pred_final))

    # Generate submission:
    submission = pd.DataFrame({'PetID': test['PetID'].values,
                               'AdoptionSpeed': pred_final.astype(
                                   np.int32)})
    submission.head()
    submission.to_csv('submission.csv', index=False)
    print('End')


if __name__ == '__main__':
    main()
