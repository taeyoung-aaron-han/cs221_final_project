#All imports
import json
import pandas as pd
import csv
from collections import defaultdict
from collections import OrderedDict
import numpy
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os, random, operator, sys
from collections import Counter
import sklearn
import math
import timeit
import pickle

def get_train_and_test_data():
  feature_type = input("TF-IDF or n-grams (type exactly)?").lower()
  while True:
    if feature_type == "tf-idf": break
    if feature_type == "n-grams": break
    feature_type = lower(input("Invalid input: TF-IDF or n-grams (type exactly)? (ctrl + c to exit)"))\

  start = timeit.default_timer()

  #take filename as input, use default file if not provided
  filename = input("Filename: (enter for default) ")
  if filename == "":
    filename = 'politicians.csv'

  #take the minimum and the maximum range for n-grams if n-gram training is chosen as input, use default figures if not provided
  #determine pickle filename to use provided with feature type and appropriate params
  if feature_type == "n-grams":
    min = input("Minimum for n-grams (enter for default): ")
    #set default range as (1,1) if no minimum input
    if min == "":
      min = 1
      max = 1
    else:
      max = input("Maximum for n-grams (enter for default): ")
      #set default max range equal to min if no input
      if max == "":
        min = int(min)
        max = min
      else:
        min = int(min)
        max = int(max)
    filename_template = "pickle_files/" + filename + "_range_" + str(min) + str(max)
    train_pickle_file_name = filename_template +  "_train.pickle"
    test_pickle_file_name = filename_template + "_test.pickle"
  else:
    train_pickle_file_name = "pickle_files/" + filename + "tf_idf_train.pickle"
    test_pickle_file_name = "pickle_files/" + filename + "tf_idf_test.pickle"

  #use pickle file if it exists
  if os.path.exists(train_pickle_file_name) and os.path.exists(test_pickle_file_name):
    with open(train_pickle_file_name, 'rb') as handle:
        train_data = pickle.load(handle)
    with open(test_pickle_file_name, 'rb') as handle:
        test_data = pickle.load(handle)
    print("Using pickle file previously created.")
    return train_data, test_data



  print("Creating features...")

  # Key = screen_name, Val = []
  # take in data for csv file to use as well as the range for n-gram features
  # Read in Senator/Congressman info and put into dict
  congress_dict = OrderedDict()
  congress_df = pd.read_csv(filename)
  screen_names = congress_df['screen_name'].tolist()

  for name in screen_names:
    congress_dict[name] = ''

  # Read in Tweet info and match with Senator/Congressman screen name
  for line in open('US_PoliticalTweets/tweets.json', 'r'):
    tweets_dict = json.loads(line)
    screen_name = tweets_dict['screen_name']
    # print(screen_name)
    tweet = tweets_dict['text']
    # print(tweet)
    if screen_name in congress_dict.keys():
      congress_dict[screen_name] += tweet

  #N-grams implementation
  vector_dict = OrderedDict()
  if feature_type == "n-grams":
    vectorizer = CountVectorizer(ngram_range=(min,max))
    all_vectors = vectorizer.fit_transform(congress_dict.values()).toarray()
    all_congress = list(congress_dict.keys())
  
  # TF-IDF implementation
  else:
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    all_vectors = tv.fit_transform(congress_dict.values()).toarray()
    all_congress = list(congress_dict.keys())

  # Match vectors with Senators/Congressmen (using OrderedDict, so should be in order)
  for i in range(len(all_congress)):
    vector_dict[all_congress[i]] = [all_vectors[i]]

  # Append other relevant Senator/Congressman info to vector_dict
  for index, row in congress_df.iterrows():
    screen_name = row['screen_name']
    #vector_dict[screen_name] += [row['bioname'], row['economic_simple_pc'], row['social_simple_pc']]
    vector_dict[screen_name] += [row['bioname']]

    # Economic: left = -1, right = 1
    if row['economic_simple_tr'] == 'left':
      vector_dict[screen_name] += [-1]
    else:
      vector_dict[screen_name] += [1]

    # Social: left = -1, right = 1
    if row['social_simple_tr'] == 'left':
      vector_dict[screen_name] += [-1]
    else:
      vector_dict[screen_name] += [1]

    # Social and economic for multiclass classifications:
    vector_dict[screen_name] += [row['econ_and_social']]

  # vector_dict is a dictonary with key = screen_name and val = [BOW vector, bioname, econ_pc, social_pc]

  #normalize word vectors into percentage of each Senator/Congressman's corpus
  remove_keys = []
  for key in vector_dict.keys():
      #check if the Senator/Congressman is in the data, remove them if not
      if (sum(vector_dict[key][0])) == 0:
          remove_keys.append(key)
      else:
          #doesn't work for n-grams with min range g.e.q 4 due to memory constraints (16GB RAM)
          if min < 4:
              vector_dict[key][0] = vector_dict[key][0] / sum(vector_dict[key][0])
          else:
              sum_dict = sum(vector_dict[key][0])
              for i in range(len(vector_dict[key][0])):
                  temp = vector_dict[key][0][i]
                  temp = temp / sum_dict
                  vector_dict[key][0][i] = temp

  #remove current senators/congressmen that are not in our dataset
  for key in remove_keys:
      vector_dict.pop(key)

  #split train and test sets
  random.seed(123)
  keys = list(vector_dict.keys())
  random.shuffle(keys)
  train_key = keys[:math.floor(0.8 * len(keys))] #80% training
  test_key = keys[math.floor(0.8 * len(keys)):] #20% testing
  train_data = OrderedDict()
  test_data = OrderedDict()
  for key in train_key:
      train_data[key] = vector_dict[key]
  for key in test_key:
      test_data[key] = vector_dict[key]

  #create pickle files with train and test data
  #doesn't work with 16GB RAM beyond (2,2)

  # with open(train_pickle_file_name, 'wb') as handle:
  #   pickle.dump(train_data, handle, protocol = pickle.HIGHEST_PROTOCOL)
  # with open(test_pickle_file_name, 'wb') as handle:
  #   pickle.dump(test_data, handle, protocol = pickle.HIGHEST_PROTOCOL)

  # print("Pickle files created for future usage.")

  return train_data, test_data
