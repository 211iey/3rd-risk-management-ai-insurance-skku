import pandas as pd
import numpy as np

df = pd.read_csv('training_input_data.csv', encoding='latin1')

# 결측값 제거 
df[['deployer_category', 'developer_category']] = df[['deployer_category', 'developer_category']].replace('error', np.nan)
df['severity'] = df['severity'].replace(-1, np.nan)
df = df[~df['risk_domain'].isin([
    '4. Malicious Actors & Misuse',
    '6. Socioeconomic & Environmental Harms'
])].reset_index(drop=True)

social_impact = {
    'regulated_corporation': 5,
    'unregulated_startup': 7,
    'public_agency': 3,
    'academic_lab': 2,
    'freelancer_or_individual': 6,
    'open_source_community': 4
}
distribution = df['deployer_category'].value_counts(normalize=True)
weighted_dist = {k: social_impact[k] * distribution.get(k, 0) for k in social_impact}
total = sum(weighted_dist.values())
deployer_weights = {k: v / total for k, v in weighted_dist.items()}
for k, v in deployer_weights.items():
    print(f"{k}: {v:.4f}")
df['deployer_weight'] = df['deployer_category'].map(deployer_weights)

distribution = df['developer_category'].value_counts(normalize=True)
weighted_dist = {k: social_impact[k] * distribution.get(k, 0) for k in social_impact}
total = sum(weighted_dist.values())
developer_weights = {k: v / total for k, v in weighted_dist.items()}
for k, v in developer_weights.items():
    print(f"{k}: {v:.4f}")
df['developer_weight'] = df['developer_category'].map(developer_weights)

# 전체 데이터 결측값 제거
df = df.dropna()

# PCA 기반 Risk Adjustment Ractor(RAF) 계산
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = df[['developer_weight', 'deployer_weight']]
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=1)
raf = pca.fit_transform(X_scaled)

df['RAF'] = abs(raf)

# risk domain별 frequency 
risk_freq = df['deployer_category'].value_counts(normalize=False).to_dict()
df['frequency'] = df['deployer_category'].map(risk_freq)

# expected loss 계산 
basic_severity_map = df.groupby('risk_domain')['severity'].mean().to_dict()
df['basic_severity'] = df['risk_domain'].map(basic_severity_map)
df['expected_loss'] = df['frequency'] * df['basic_severity'] * df['RAF']

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df.drop(columns = 'date', inplace = True)
df.drop(columns = ['incident_id', 'severity', 'developer_weight', 'deployer_weight', 'RAF', 'basic_severity'], inplace = True)
# deployer_category

epsilon = 1e-6
shift = abs(df['expected_loss'].min()) + epsilon
df['log_expected_loss'] = np.log(df['expected_loss']+shift)

first = df['log_expected_loss'].describe()
second = df['expected_loss'].describe()
print(f'\n===log_expected: {first}\n\nexpected: {second}\n')
df.drop(columns = 'expected_loss', inplace=True)

object_col = ['risk_domain', 'Entity', 'Intent', 'deployer_category', 'developer_category']
# df = pd.get_dummies(df, columns=object_col, drop_first=False)

df.to_csv('gamlss_data.csv', index = False)