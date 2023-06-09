# use pytorch image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# set path to conda
ENV PATH /opt/conda/bin:$PATH

# install sudo
RUN apt-get update && apt-get install sudo

# install dependancies
RUN conda update -n base conda && \
    conda install -c anaconda pip && \
    conda install -c conda-forge jupyterlab
RUN pip install scikit-learn && \
    pip install seaborn && \
    pip install tensorboard && \
    pip install torchdata==0.6.0 && \
    pip install torchmetrics==0.11.4 && \
    pip install optuna

# install conda-token to root env to use commercial eddition
# RUN conda install conda-token -n root

# add user to run jupyter notebook (uid and gid can be modified by entrypoint.sh with envvalue)
ARG NB_USER=user
ARG NB_GROUP=usergroup
ARG NB_UID=1000
ARG NB_GID=1000
RUN groupadd -g $NB_GID $NB_GROUP && \
    useradd -m -u $NB_UID -g $NB_GID $NB_USER
RUN mkdir /home/$NB_USER/work

# give write permission so that users can add conda package by themselves
RUN chmod -R o+w /opt/conda

# set work directory
WORKDIR ../home/$NB_USER/work

# copy entrypoint shell script
COPY entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/$NB_USER/.bashrc
USER ${NB_UID}

