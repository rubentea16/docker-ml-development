FROM tensorflow/tensorflow:1.15.2-gpu-py3

# Set Environment
ENV MAIN_PATH=/usr/local/bin/jpl_config
ENV LIBS_PATH=${MAIN_PATH}/libs
ENV CONFIG_PATH=${MAIN_PATH}/config
ENV NOTEBOOK_PATH=${MAIN_PATH}/notebooks

# System packages 
RUN apt-get update && apt-get install -y \
    apt-utils \
    liblzma-dev \
    git \
    screen \
    curl

# requirements file for installation
COPY config/common_libs.txt .
COPY config/specific_libs.txt .

# pip install
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r common_libs.txt
RUN pip install --no-cache-dir -r specific_libs.txt

# Install jupyternotebook & jupyterlab
RUN pip install jupyter --upgrade
RUN pip install jupyterlab --upgrade

# Install NodeJS
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash
RUN apt-get install -y nodejs

# Install jupyterlab-manager
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension install @jupyterlab/debugger
RUN jupyter labextension install jupyterlab_tensorboard

# Expose port
EXPOSE 8888
# RUN
CMD cd ${MAIN_PATH} && sh config/run_jupyter.sh
