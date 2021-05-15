FROM seldonio/seldon-core-s2i-python3:1.6.0
RUN pip install --upgrade pip
RUN pip install torch
RUN pip install torchvision
RUN pip install opencv-python
RUN pip install requests
RUN pip install pillow
ADD model.pkl .
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
