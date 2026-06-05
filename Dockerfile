FROM python:3.11-slim

WORKDIR /app
COPY . /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PKG_NAME=strata_fit_v6_imputation_py

RUN pip install \
    numpy \
    pandas \
    pydantic \
    pyarrow \
    requests \
    PyJWT \
    scikit-learn

RUN pip install --no-deps \
    "v6-federated-algo-core-py @ https://github.com/mdw-nl/v6-federated-algo-core-v6/archive/c29dd63f40c6e3997a0865cb0cbc81dd9ce02a60.tar.gz"

RUN pip install --no-deps /app

CMD ["python", "-m", "strata_fit_v6_imputation_py.container"]
