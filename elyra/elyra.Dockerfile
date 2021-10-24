FROM elyra/elyra:dev
USER root
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 imagemagick -y
USER jovyan
RUN pip install matplotlib opencv-python pillow tensorflow