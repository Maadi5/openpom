shrunk_dataset = False

import warnings
warnings.filterwarnings('ignore')

TASKS = [
'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]

TASKS_shrunk = ['alcoholic',
 'burnt',
 'citrus',
 'earthy',
 'green',
 'mint',
 'odorless',
 'phenolic',
 'spicy',
 'sulfurous',
 'sweet',
 'woody']


if shrunk_dataset:
    TASKS = TASKS_shrunk

print("No of tasks: ", len(TASKS))
n_tasks = len(TASKS)

import warnings
import json

# Ignore all warnings
warnings.filterwarnings("ignore")

perfurmershub2labels_updated = {'Wood': {'main': 'woody', 'accessory':['cedar', 'sandalwood']},
 'Green': {'main': 'green', 'accessory': ['grassy', 'leafy']},
 'Linalool': {'main': 'floral', 'accessory': ['lavender', 'citrus']},#['floral', 'lavender'],
 'Fruity': {'main': 'fruity', 'accessory': ['apple', 'apricot', 'banana', 'berry', 'black currant', 'cherry', 'grape', 'grapefruit', 'melon', 'orange', 'peach', 'pear', 'pineapple', 'plum', 'raspberry', 'strawberry']},#['fruity', 'apple', 'apricot', 'banana', 'berry', 'black currant', 'cherry', 'grape', 'grapefruit', 'melon', 'orange', 'peach', 'pear', 'pineapple', 'plum', 'raspberry', 'strawberry'],
 'Zolvent': {'main': 'solvent', 'accessory': []},
 'Herb': {'main': 'herbal', 'accessory': ['anisic', 'clove', 'mint']},
 'Spice': {'main': 'spicy', 'accessory': ['spicy', 'cinnamon', 'clove']},
 'Citrus': {'main': 'citrus','accessory': ['lemon', 'bergamot', 'grapefruit', 'orange']},
 'Balsamic': {'main': 'balsamic', 'accessory': []},
 'Narcotic': {'main': 'ethereal' , 'accessory': []},
 'Vanilla': {'main': 'vanilla' , 'accessory': []},
 'Rose': {'main': 'rose' , 'accessory': []},
 'Orchid': {'main': 'floral' , 'accessory': []},#['floral'],
 'Animalic': {'main': 'meaty' , 'accessory': ['animal', 'musk','beefy']}, #'animal', 'beefy', 'meaty', 'musk'],
 'Phenolic': {'main': 'phenolic' , 'accessory': []},#['phenolic'],
 'Iris': {'main': 'orris' , 'accessory': []},#['orris'],
 'Musk': {'main': 'musk' , 'accessory': []},#['musk'],
 'Dairy': {'main': 'dairy'  , 'accessory': ['buttery', 'cheesy', 'creamy', 'milky']},#['dairy', 'buttery', 'cheesy', 'creamy', 'milky'],
 'Muguet': {'main': 'muguet', 'accessory': []},#['muguet'],
 'iceBerg': {'main': 'fresh', 'accessory': ['cooling']},#['cooling', 'fresh'],
 'Edible': {'main': 'vanilla', 'accessory': ['chocolate', 'caramellic', 'honey', 'almond', 'cinnamon', 'cocoa', 'coffee', 'fruity', 'sweet', 'buttery', 'creamy', 'milky']},#['almond', 'chocolate', 'cocoa', 'coconut', 'coffee', 'hazelnut', 'honey', 'malty', 'nutty', 'popcorn', 'roasted', 'tea'],
 'Aliphatic': {'main': 'fatty', 'accessory': ['aldehydic', 'oily']},#['aldehydic', 'fatty', 'oily'],
 'Jasmine': {'main': 'jasmin', 'accessory': []}, #['jasmin'],
 'Konifer': {'main': 'pine', 'accessory': ['woody']}}#['pine', 'woody']}

import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.utils.data_utils import get_class_imbalance_ratio, IterativeStratifiedSplitter
from openpom.models.mpnn_pom5 import MPNNPOMModel, MPNNPOM
from datetime import datetime
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from rdkit import Chem
import rdkit
from PIL import Image
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# uncomment and run if no splits saved yet

# download curated dataset
# !wget https://raw.githubusercontent.com/ARY2260/openpom/main/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv

# The curated dataset can also found at `openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv` in the repo.

import pandas as pd

if shrunk_dataset:
    input_file = r'shrunkdataset_v2.csv' # or new downloaded file path
else:
    input_file = r'openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv' # or new downloaded file path
# input_file = r'shrunkdataset_v1.csv'

shrunkdataset_v1 = pd.read_csv(r'shrunkdataset_v2.csv', index_col = False)

datasetdf = pd.read_csv(input_file, index_col = False)


# get dataset

featurizer = GraphFeaturizer()
smiles_field = 'nonStereoSMILES'
loader = dc.data.CSVLoader(tasks=TASKS,
                   feature_field=smiles_field,
                   featurizer=featurizer)
dataset = loader.create_dataset(inputs=[input_file])
n_tasks = len(dataset.tasks)

# get train valid test splits
splitter = IterativeStratifiedSplitter(order=2)
train_dataset, test_dataset = splitter.train_test_split(dataset, frac_train=0.8, train_dir='./splits/train_data', test_dir='./splits/test_data')

# print("train_dataset: ", len(train_dataset))
# print("test_dataset: ", len(test_dataset))

from skmultilearn.model_selection import IterativeStratification
from typing import List, Optional, Tuple, Iterator

def iterative_splitter_by_indices(
    x_dataset: pd.DataFrame,
    y_dataset: pd.DataFrame,
    frac_train: float = 0.8,
    frac_valid: float = 0.0,
    frac_test: float = 1-0.8,
    order = 2, #order of iterative stratifications
    seed: Optional[int] = None,
    log_every_n: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    X1: pd.DataFrame
    y1: pd.DataFrame
    X1, y1 = x_dataset, y_dataset
    stratifier1: IterativeStratification = IterativeStratification(
        n_splits=2,
        order=order,
        sample_distribution_per_fold=[frac_test + frac_valid, frac_train],
        # shuffle=True,
        random_state=seed,
    )

    train_indices: np.ndarray
    other_indices: np.ndarray

    X2: pd.DataFrame
    y2: pd.DataFrame
    X2, y2 = pd.DataFrame(other_dataset.X), pd.DataFrame(other_dataset.y)
    new_split_ratio: float = round(frac_test / (frac_test + frac_valid), 2)
    stratifier2: IterativeStratification = IterativeStratification(
        n_splits=2,
        order=self.order,
        sample_distribution_per_fold=[
            new_split_ratio, 1 - new_split_ratio
        ],
        random_state=seed,
    )

    valid_indices: np.ndarray
    test_indices: np.ndarray
    valid_indices, test_indices = next(stratifier2.split(X2, y2))
        

    return train_indices, valid_indices, test_indices


train_indices, _, test_indices = iterative_splitter_by_indices(x_dataset=datasetdf[datasetdf.columns[0]], y_dataset=datasetdf[datasetdf.columns[2:]])