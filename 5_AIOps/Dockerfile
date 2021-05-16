FROM nishkarshraj/mlops-ubuntu
RUN pip install --upgrade pip
RUN pip install torch
RUN pip install torchvision
RUN pip install opencv-python
RUN pip install requests
RUN pip install pillow
ADD model.pkl .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY model_dir/ /opt/ml/model
                RUN python -c                 'from mlflow.models.container import _install_pyfunc_deps;                _install_pyfunc_deps("/opt/ml/model", install_mlflow=False)'
                ENV MLFLOW_DISABLE_ENV_CREATION="true"
ENTRYPOINT ["python", "-c", "from mlflow.models import container as C; C._serve()"]
