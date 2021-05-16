# End-to-End Workflow (ModelDB x MLFlow)

## Start ModelDB Server

```
docker-compose up
```

## Upload Model in ModelDB Store

```
python3 ModelDB.py
```

## Fetch Model from ModelDB, Log into Mlflow and Move into Production

```
python3 Mlflow.py
```

## Build Base Docker Image

```
docker build -t nishkarshraj/mlops-ubunu .
```

Or Pull from DockerHub

```
docker pull nishkarshraj/mlops-ubuntu
```

## Dockerization

```
mkdir model_dir/

mv mlruns/0/<run id>/artifacts/<model name>

docker build -t [Final Image] .

docker run --rm -it -p 5000:8080 nish

curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{"data":{"ndarray":["https://content.steward.org/sites/default/files/image01.jpg",1]}}'
```
