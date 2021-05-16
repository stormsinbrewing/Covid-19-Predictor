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

curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{ 
    "columns": ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age"], 
    "data": [[10, 101, 76, 48, 180, 32.9, 0.171, 63], [0, 137, 40, 35, 168, 43.1, 2.288, 33]]
}'
```
