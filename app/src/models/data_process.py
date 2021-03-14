import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataReader:  
    def standard_scaler(self, df):
        '''
        This function scales all the features included in the dataframe using Standard Scaler

        Arguments:
        ----------
        df: a panda dataframe with all the features to be scaled

        Return:
        -------
        scaled_df: a panda dataframe with all the scaled features 
        '''
        sc = StandardScaler()
        df_scaled = sc.fit_transform(df)
        
        # Set as pd.dataframe and re-apply column names
        scaled_df = pd.DataFrame(df_scaled)
        scaled_df.columns = df.columns
        return scaled_df
             
    def format_features(header: str, brewery_name: str, review_aroma:float, review_appearance:float, review_palate: float, review_taste: float):
        return {
        'brewery_name': [brewery_name],
        'review_aroma' : [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste]
        }

# end of DataReader
