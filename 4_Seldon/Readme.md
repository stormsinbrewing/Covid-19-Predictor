## Creating Seldon Endpoints

### Build Base Docker Image with Dockerfile

```
docker build -t [image name] .
```

### Build S2i Image for Seldon

```
s2i build . [base image] [seldon image]
```

### Expose Seldon Image as local endpoint

```
docker run --rm --name test -p 9000:9000 -p 5000:5000 -p 6000:6000 base
```

### Hit the Local Endpoint with cURL

```
curl -X POST -H 'Content-Type: application/json' -d '{"data":{"ndarray":["https://content.steward.org/sites/default/files/image01.jpg",1]}}' http://localhost:9000/api/v1.0/predictions
```

### Hosting Seldon Image as REST Endpoint using Kubernetes

[Kubernetes Workflow](kubernetes/)

