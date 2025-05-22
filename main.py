import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pycaret.regression import load_model
# import joblib
from sklearn.preprocessing import LabelEncoder

def load_file(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_df(df):
    df[['deployer_category', 'developer_category']] = df[['deployer_category', 'developer_category']].replace('error', np.nan)
    df['severity'] = df['severity'].replace(-1, np.nan)
    df = df[~df['risk_domain'].isin([
        '4. Malicious Actors & Misuse',
        '6. Socioeconomic & Environmental Harms'
    ])].reset_index(drop=True)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df.drop(columns = 'date', inplace = True)
    
    df = df.dropna()
    return df

def weight(df, weight_dict):
    distribution = df['deployer_category'].value_counts(normalize=True)
    weighted_dist = {k: weight_dict[k] * distribution.get(k, 0) for k in weight_dict}
    total = sum(weighted_dist.values())
    deployer_weights = {k: v / total for k, v in weighted_dist.items()}
    for k, v in deployer_weights.items():
        print(f"{k}: {v:.4f}")
    df['deployer_weight'] = df['deployer_category'].map(deployer_weights)

    distribution = df['developer_category'].value_counts(normalize=True)
    weighted_dist = {k: weight_dict[k] * distribution.get(k, 0) for k in weight_dict}
    total = sum(weighted_dist.values())
    developer_weights = {k: v / total for k, v in weighted_dist.items()}
    for k, v in developer_weights.items():
        print(f"{k}: {v:.4f}")
    df['developer_weight'] = df['developer_category'].map(developer_weights)

    return df

def predict(model, df):
    df = df.copy()

    features = ['Alleged deployer of AI system', 'risk_domain', 'Entity', 'Intent', 'developer_category', 'year', 'month']
    cat_cols = ['Alleged deployer of AI system', 'risk_domain', 'Entity', 'Intent', 'developer_category']
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df[features]
    return model.predict(X)

def raf(df):
    X = df[['developer_weight', 'deployer_weight']]
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1)
    raf = pca.fit_transform(X_scaled)

    df['RAF'] = abs(raf)
    return df

def predict_freq(df, model):
    X = df
    frequency = predict(model, X)
    df['frequency'] = frequency
    # risk_freq = df['Alleged deployer of AI system'].value_counts(normalize=False).to_dict()
    # df['frequency'] = df['Alleged deployer of AI system'].map(risk_freq)
    return df

def final(df):
    # expected loss 계산 
    basic_severity_map = df.groupby('risk_domain')['severity'].mean().to_dict()
    df['basic_severity'] = df['risk_domain'].map(basic_severity_map)
    df['expected_loss'] = df['frequency'] * df['basic_severity'] * df['RAF']
    # print(f"\n\n====\nexpected loss minimum: {df['expected_loss'].min()}\n\nfrequency minimum: {df['frequency'].min()}\n\nbasic_severity minimum: {df['basic_severity'].min()}\n\nraf minimum: {df['RAF'].min()}\n====\n\n")

    df.drop(columns = ['deployer_category', 'incident_id', 'severity', 'developer_weight', 'deployer_weight', 'RAF', 'basic_severity'], inplace = True)
    # deployer_category

    epsilon = 1e-6
    # shift = abs(df['expected_loss'].min()) + epsilon
    df['log_expected_loss'] = np.log(df['expected_loss']+epsilon)
    df.drop(columns = 'expected_loss', inplace=True)  
    return df

def risk_level(df):
    log_expected_loss_risk = 39.795
    frequency_risk = 16.0

    for i in range(len(df)):
        sev = df['log_expected_loss'].iloc[i]
        freq = df['frequency'].iloc[i]

        if sev > log_expected_loss_risk and freq > frequency_risk:
            print(f"{i+1}번째 리스크는 severity와 frequency 측면에서 모두 고위험군 리스크입니다.\n- severity: {sev} > {log_expected_loss_risk}\n- frequency: {freq} > {frequency_risk}")
        elif sev > log_expected_loss_risk:
            print(f"{i+1}번째 리스크는 severity 측면에서 고위험군 리스크입니다.\n- severity: {sev} > {log_expected_loss_risk}\n- frequency: {freq} <= {frequency_risk}")
        elif freq > frequency_risk:
            print(f"{i+1}번째 리스크는 frequency 측면에서 고위험군 리스크입니다.\n- severity: {sev} <= {log_expected_loss_risk}\n- frequency: {freq} > {frequency_risk}")
        else:
            print(f"{i+1}번째 리스크는 고위험군 리스크가 아닙니다.\n- severity: {sev} <= {log_expected_loss_risk}\n- frequency: {freq} <= {frequency_risk}")



def main():
    file_path = "활용데이터_madeby.csv"
    freq_model = load_model('knn_model_final')
    social_impact = {
    'regulated_corporation': 5,
    'unregulated_startup': 7,
    'public_agency': 3,
    'academic_lab': 2,
    'freelancer_or_individual': 6,
    'open_source_community': 4
    }

    df = load_file(file_path)
    df = clean_df(df)
    df = weight(df, social_impact)
    df = raf(df)
    df = predict_freq(df, freq_model)
    df = final(df)

    risk_level(df)

if __name__=="__main__":
    main() 