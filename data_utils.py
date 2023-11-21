import torch
import pytorch_lightning as pl
import numpy as np
import os
import pickle
import pandas as pd
from ptls.frames.coles import CoLESModule
from sklearn.model_selection import train_test_split

from ptls.preprocessing import PandasDataPreprocessor
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule

def download_data():
    if not os.path.exists('data/transactions_train.csv'):
        os.system('mkdir -p data')
        os.system('wget --progress=bar:force:noscroll https://storage.yandexcloud.net/ptls-datasets/age-prediction-nti-sbebank-2019.zip')
        os.system('unzip -j -o age-prediction-nti-sbebank-2019.zip \'data/*.csv\' -d data')
        os.system('mv age-prediction-nti-sbebank-2019.zip data/')
        
    print('Data saved in <data> folder')
    return 'data/transactions_train.csv'
        
def preprocess_data(data_file):
    df = pd.read_csv(data_file)

    if not os.path.exists('preprocessor.p'):
        preprocessor = PandasDataPreprocessor(
            col_id='client_id',
            col_event_time='trans_date',
            event_time_transformation='none',
            cols_category=['small_group'],
            cols_numerical=['amount_rur'],
            return_records=True,
        )
        dataset = preprocessor.fit_transform(df)
        with open('preprocessor.p', 'wb') as f:
            pickle.dump(preprocessor, f)
        print('Preprocessor saved')

    else:
        with open('preprocessor.p', 'rb') as f:
            preprocessor = pickle.load(f)
        print('Preprocessor loaded')
        dataset = preprocessor.transform(df)
        
    return dataset


def split_data(dataset, test_size=0.25):
    dataset = sorted(dataset, key=lambda x: x['client_id'])
    train, test = train_test_split(dataset, test_size=0.25, random_state=3407)
    print(f'{len(train)} users in train, {len(test)} users in test') 
    return train, test


def get_loader(data):
    dl = PtlsDataModule(
        train_data=ColesDataset(
            MemoryMapDataset(
                data=data,
                i_filters=[
                    SeqLenFilter(min_seq_len=25),
                ],
            ),
            splitter=SampleSlices(
                split_count=5,
                cnt_min=25,
                cnt_max=200,
            ),
        ),
        train_num_workers=16,
        train_batch_size=256,
    )

    return dl