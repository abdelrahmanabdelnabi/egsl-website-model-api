FROM ubuntu:14.04
FROM python:3.6

# copy ffmpeg binaries from a public docker image
ENV LD_LIBRARY_PATH=/usr/local/lib
COPY --from=jrottenberg/ffmpeg /usr/local /usr/local/

# make directories suited to your application
RUN mkdir -p /home/project/app
RUN mkdir -p /home/project/app/models
WORKDIR /home/project/app

# copy and install packages for flask
COPY requirements.txt /home/project/app
RUN pip3 install --no-cache-dir -r requirements.txt --ignore-installed

# # install dependencies for av
RUN apt-get update -y
RUN apt-get install -y \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev

RUN pip install av

# copy contents from your local to your docker container
COPY . /home/project/app

EXPOSE 6000

ENV MODEL_PATH best_params_100_deep_64_arabic.pth

CMD ["flask", "run", "--port=6000"]

# av==6.2.0
# pytorch=1.1.0