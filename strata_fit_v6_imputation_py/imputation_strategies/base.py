from abc import ABC, abstractmethod
import pandas as pd
from enum import Enum
from typing import Dict, Any, List, Type

class ImputationStrategyEnum(str, Enum):  
    MEAN_IMPUTER = "mean_imputer"  
    MEDIAN_IMPUTER = "median_imputer"  
    CONSTANT_IMPUTER = "constant_imputer"


def register_imputation_strategy(key: ImputationStrategyEnum):  
    """  
    Decorator: register `ImputationStrategy` subclasses under a given enum key.  
    """  
    def decorator(cls: Type[ImputationStrategy]) -> Type[ImputationStrategy]:
        if not issubclass(cls, ImputationStrategy):  
            raise TypeError(  
                f"{cls.__name__} must inherit from ImputationStrategy "  
                f"to be registered under {key}"  
            )  
        STRATEGY_REGISTRY[key] = cls  
        return cls  

    return decorator 

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
    def aggregate(self, node_metrics: List[Dict[Any, Any]], columns: List[str]) -> Dict:
        """aggregates node metrics into global metrics

        Args:
            df (pd.DataFrame): dataframe of means from all nodes        

        Returns:
            Dict: a dictionary with global mean per ID and column
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


STRATEGY_REGISTRY: Dict[ImputationStrategyEnum, Type[ImputationStrategy]] = {}