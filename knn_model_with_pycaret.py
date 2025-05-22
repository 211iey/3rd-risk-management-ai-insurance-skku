import pandas as pd
import numpy as np

df = pd.read_csv('gamlss_data.csv')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np

from pycaret.regression import *

df_pycaret = df.copy()
df_pycaret = df_pycaret.drop(columns=['expected_loss'])
df_pycaret['frequency'] = df_pycaret['frequency']

reg_setup = setup(
    data=df_pycaret,
    target='frequency',
    train_size=0.8,
    session_id=42,
    use_gpu=True  # GPU 지원 (XGBoost, CatBoost 등)
)

best_model = compare_models(n_select=3, sort='R2')
ensemble_model = blend_models(estimator_list=best_model, choose_better=True)
final_model = finalize_model(ensemble_model)
preds = predict_model(final_model)

knn_model = create_model('knn')

import os
print("현재 작업 디렉토리:", os.getcwd())


from pycaret.regression import save_model
save_model(knn_model, 'knn_model_final')
