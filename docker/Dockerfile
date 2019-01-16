FROM ubuntu:16.04

RUN apt-get update --fix-missing && \
    apt-get -y install wget bzip2 ca-certificates curl git sudo git python3 tar cmake libopenmpi-dev python3-dev zlib1g-dev

RUN apt-get install -y python-pyglet python3-opengl zlib1g-dev libjpeg-dev patchelf \
        cmake swig libboost-all-dev libsdl2-dev libosmesa6-dev xvfb ffmpeg

SHELL ["/bin/bash", "-c"]

# user
RUN useradd --create-home --home-dir /home/docker --shell /bin/bash docker
USER docker
ENV HOME /home/docker

# install conda
RUN wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O $HOME/miniconda.sh
RUN chmod 0755 $HOME/miniconda.sh
RUN $HOME/miniconda.sh -b -p $HOME/conda
ENV PATH="/home/docker/conda/bin:$PATH"
RUN rm $HOME/miniconda.sh
RUN conda update conda

# install umrl w/o mujoco, gym, baselines
COPY ./docker/environment.yml $HOME
RUN conda env create -f $HOME/environment.yml
RUN rm $HOME/environment.yml

# install mujoco
COPY ./docker/.mujoco/ $HOME/.mujoco
ENV LD_LIBRARY_PATH $HOME/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
RUN source activate umrl && pip install mujoco-py

# install gym
RUN source activate umrl && pip install gym

# install baselines
RUN mkdir $HOME/install && cd $HOME/install && git clone https://github.com/openai/baselines.git
RUN source activate umrl && pip install -e $HOME/install/baselines

# delay loading code until last
WORKDIR $HOME/experiments/umrl
COPY . .