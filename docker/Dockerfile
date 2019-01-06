#FROM ubuntu:16.04
#
#RUN apt-get update --fix-missing && \
#    apt-get -y install wget bzip2 ca-certificates curl git sudo git python3 tar
#
#RUN useradd --create-home --shell /bin/bash docker && echo "docker:docker" | chpasswd && adduser docker sudo
#USER docker
#WORKDIR /home/docker/experiments
#
#RUN wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O $HOME/miniconda.sh
#RUN chmod 0755 $HOME/miniconda.sh
#RUN $HOME/miniconda.sh -b -p $HOME/conda
#ENV PATH="/home/docker/conda/bin:$PATH"
#RUN rm $HOME/miniconda.sh
#RUN conda update conda
#
#RUN conda config --append channels conda-forge && \
#    conda config --append channels pytorch
#COPY . $HOME
#RUN conda create --name umrl --file requirements_conda.txt

#FROM kylehsu/umrl:0.1
#
#USER root
#RUN apt-get -y install cmake libopenmpi-dev python3-dev zlib1g-dev
#USER docker
#
#ENV PATH $HOME/conda/envs/testenv/bin:$PATH
#COPY . $HOME
#RUN ["/bin/bash", "-c", "source activate umrl && pip install --upgrade pip && pip install -r requirements_pip.txt"]

#RUN ["/bin/bash", "-c", "mkdir ~/install && cd ~/install && git clone https://github.com/openai/baselines.git"]
#RUN ["/bin/bash", "-c", "source activate umrl && pip install -e ~/install/baselines"]

FROM kylehsu/umrl:0.2
COPY ./ /home/docker/experiments/umrl
ENTRYPOINT ["/bin/bash", "-c", "cd umrl"]