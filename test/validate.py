from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import polars as pl

# Assuming 'df_full' is your original unsplit dataset
df_full = pl.read_csv("/home/biostat/Register_data/data_request/RA/Stratafit/05_data_ready_to_use/mockClientData/realSplitData/*.csv")

columns = ['DAS28', 'CRP', 'ESR', 'SJC28', 'TJC28']
central_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42)
df_central_imputed = pd.DataFrame(
    central_imputer.fit_transform(df_full[columns]), 
    columns=columns
)