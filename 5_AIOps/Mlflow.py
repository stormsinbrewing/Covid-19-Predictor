from verta import Client
import pickle
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os

def downloadArtifact(proj,exp_name,exp_run, serialization):
    client = Client("http://localhost:3000")
    proj = client.set_project(proj)
    expt = client.set_experiment(exp_name) 
    run = client.set_experiment_run(exp_run)
    if serialization.lower() == 'pickle':
        run.download_model('model.pkl')
    
def logModel(library, modelName):
    infile = open('./model.pkl','rb')
    model = pickle.load(infile)
    print ('Loaded Model')
    infile.close()
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    if library.lower() == 'pytorch':
    	mlflow.pytorch.log_model (model, "covid-predictor",registered_model_name=modelName)
#        mlflow.tensorflow.log_model (tf_saved_model_dir='.',registered_model_name=modelName,tf_meta_graph_tags=[],tf_signature_def_key='covid-predictor', artifact_path='model_dir/')
    client = MlflowClient()
    client.transition_model_version_stage(
    name=modelName,
    version=1,
    stage="Production"
    )
    print ('Logged model')
    
def serveModel(modelName):
    os.environ["MLFLOW_TRACKING_URI"]="sqlite:///mlruns.db"
    os.system("mlflow models serve -m models:/CovidPredictor/production -p 2000 --no-conda")

# Function Calls ("MajorII","CovidPredictor","Version 1","model.pkl","pickle","pytorch")
downloadArtifact("MajorII","CovidPredictor","Version 1","pickle")
logModel("pytorch","CovidPredictor")
#serveModel("CovidPredictor")

