from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, List

class ImputationStrategy(ABC):

    @abstractmethod
    def compute(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """compute imputation metric

        Args:
            df (pd.DataFrame): pandas dataframe 
            columns (List[str]): list of columns the metric should be computed for

        Returns:
            pd.DataFrame: with ID and column means
        """
        pass

    @abstractmethod
    def impute(self, df: pd.DataFrame, global_metric: pd.DataFrame) -> pd.DataFrame:
        """imputes the given metrics into df

        Args:
            df (pd.DataFrame): pandas dataframe the metrics should imputed into
            global_metric (Dict[str, Any]): column, metric pairs

        Returns:
            pd.DataFrame: the imputed dataframe
        """
        pass