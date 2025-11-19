from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, List, Type

STRATEGY_REGISTRY: Dict[str, Type['ImputationStrategy']] = {}

class ImputationStrategy(ABC):

    @classmethod
    def register(cls, subclass: Type['ImputationStrategy']) -> Type['ImputationStrategy']:
        """
        Register a subclass using its class name automatically.
        """
        STRATEGY_REGISTRY[subclass.__name__] = subclass
        return subclass

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
    def aggregate(self, node_metrics: List[Dict[Any, Any]], columns: List[str]) -> pd.DataFrame:
        """aggregates node means into global means

        Args:
            df (pd.DataFrame): dataframe of means from all nodes        

        Returns:
            pd.DataFrame: a pandas dataframe with global mean per ID and column
        """
        pass

    @abstractmethod
    def impute(self, df: pd.DataFrame, global_metric: Dict) -> pd.DataFrame:
        """imputes the given metrics into df

        Args:
            df (pd.DataFrame): pandas dataframe the metrics should imputed into
            global_metric (Dict[str, Any]): column, metric pairs

        Returns:
            pd.DataFrame: the imputed dataframe
        """
        pass