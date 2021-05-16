from verta import Client
client = Client("http://localhost:3000")

def uploadSerializedObject(proj,exp_name,exp_run,objName,serialization,library):
    proj = client.set_project(proj) 
    expt = client.set_experiment(exp_name) 
    run = client.set_experiment_run(exp_run) 
    run.log_model('./' + objName ,overwrite=True)
    print("Username:", proj)
    print("Experiment:",exp_name)
    print("Experiment Run:", exp_run)
    print("Serialization:", serialization)
    print("Library:",library)

uploadSerializedObject("MajorII","CovidPredictor","Version 1","model.pkl","pickle","pytorch")
