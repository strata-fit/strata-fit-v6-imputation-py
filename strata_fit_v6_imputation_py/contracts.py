from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, RootModel

from .imputation_strategies.base import ImputationStrategyEnum


class ImputationParameters(BaseModel):
    columns: List[str] = Field(min_length=1)


class ImputationConfig(BaseModel):
    schema_version: int = 1
    strategy: ImputationStrategyEnum
    parameters: ImputationParameters


class CentralInput(BaseModel):
    imputation_config: ImputationConfig
    organizations_to_include: Optional[List[int]] = None


class PartialComputeInput(BaseModel):
    columns: List[str] = Field(min_length=1)
    imputation_strategy: ImputationStrategyEnum = ImputationStrategyEnum.MEAN_IMPUTER


class PartialComputeOutput(RootModel[Dict[str, Dict[Any, Any]]]):
    pass


class CentralOutput(BaseModel):
    schema_version: int
    type: Literal["imputation"]
    strategy: str
    fitted: bool
    parameters: Dict[str, Any]
    state: Dict[str, Any]
    metadata: Dict[str, Any]
