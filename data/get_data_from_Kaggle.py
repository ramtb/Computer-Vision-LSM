import kaggle
import os
import json

#### You need to have a Kaggle API key to download the dataset 
#### You can get it from your Kaggle account settings
#### Save the API key in a file named kaggle.json in the - /root/.kaggle/ directory
#### The kaggle.json file should have the following format:
# {
#   "username
#   "key":
# }

kaggle.api.dataset_download_files('keetsekkan/vision-artificial-lsm-emociones-y-voz', path='dataset', unzip=True)


#### The dataset will be downloaded in the /content/dataset directory
#### The dataset contains the following files:
# - Images and csv files for training and testing

# The dataset can be found at the following link:
### https://www.kaggle.com/datasets/keetsekkan/vision-artificial-lsm-emociones-y-voz