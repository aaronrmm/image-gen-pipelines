FROM pytorch/pytorch:latest
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 imagemagick -y
RUN pip install matplotlib opencv-python pillow tensorflow
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
COPY ./server/requirements.txt ./server_requirements.txt
RUN pip install -r ./server_requirements.txt
RUN pip install uvicorn
RUN pip install jupyterlab
RUN pip install fastprogress
RUN pip install ipywidgets
RUN pip install pydantic