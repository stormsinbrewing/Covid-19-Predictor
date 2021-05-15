# Installing Seldon on K8s

## Install Istio Service Mesh in "istio-system" namespace

```
curl -L https://istio.io/downloadIstio | sh -
cd istio-1.9.1
export PATH=$PWD/bin:$PATH
istioctl install --set profile=demo -y
kubectl create namespace nish
kubectl label namespace nish istio-injection=enabled
```

## Create Seldon Gateway on Istio

```
kubectl apply -f gateway.yaml -n nish
```

## Install Seldon-Core in "seldon-system" namespace

```
kubectl create namespace seldon-system
helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --namespace seldon-system \
    --set istio.enabled=true \
    --set istio.gateway=istio-system/seldon-gateway
```

## Docker Build, Push and Deploy on K8s

```
kubectl apply -f deployment.yaml -n jio
kubectl apply -f service.yaml -n jio
kubectl apply -f istio-service.yaml -n jio
```

## Hit the Local Endpoint with cURL

```
curl -X POST -H 'Content-Type: application/json' -d '{"data": {"ndarray": [["https://content.steward.org/sites/default/files/image01.jpg"],[1]]}}' http://<External IP>/api/v1.0/predictions
```

