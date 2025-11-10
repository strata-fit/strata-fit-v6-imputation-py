
# strata-fit-v6-kmeans-py

Run suited imputation methods in a federated environment.

Current considerations on imputation methods:

- Different missingness mechanisms require different handling:

  - MCAR (Missing Completely At Random): simple imputers (mean/median, kNN) can perform adequately.

  - MAR (Missing At Random): methods that model joint distributions (e.g. MICE, missForest, MIWAE) are generally considered good.

  - MNAR (Missing Not At Random): cannot be identified from observed data alone — must be addressed with sensitivity analyses or model-based approaches (selection/pattern-mixture models). Evtl. adding config files with domain knowledge for each center/node

- Robust methods that perform well across various patterns:

  - Nonparametric: missForest, IterativeImputer, miceforest.

  - Model-based: Multiple Imputation by Chained Equations (MICE).

  - Deep generative: MIWAE / VAE-based approaches (and federated variants like Fed-MIWAE).
    - Could not find any easy to re-use implementation of federated imputation

  - Simple baselines: mean, median, kNN (useful for benchmarking).

- Automatic imputer selection using a mask-and-recover strategy:

  - Simulate missingness on the observed part of the data (both MCAR and MAR-like masks).

  - Benchmark multiple imputers locally.

  - Select the best-performing one per variable or dataset.

  - Optionally share only aggregated scores for global consensus (no raw data transfer).

- MNAR handling: where domain knowledge or diagnostics suggest MNAR, apply sensitivity analyses (e.g. delta-adjusted MI) and report uncertainty ranges rather than single imputations.

<br>

This algorithm is designed to be run with the [vantage6](https://vantage6.ai)
infrastructure for distributed analysis and learning.

The base code for this algorithm has been created via the
[v6-algorithm-template](https://github.com/vantage6/v6-algorithm-template)
template generator.

