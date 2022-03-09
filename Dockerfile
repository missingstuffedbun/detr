FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime
MAINTAINER author "missingstuffedbun@hotmail.com"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev && \
    rm -rf /var/cache/apk/*

RUN pip --no-cache-dir install Cython

COPY requirements.txt /workspace

RUN pip --no-cache-dir install -r /workspace/requirements.txt

RUN pip install jupyterlab
RUN jupyter lab --generate-config
RUN python -c "from notebook.auth import passwd; print(\"c.NotebookApp.password = u'\" +  passwd('3582521') + \"'\")" >> ~/.jupyter/jupyter_lab_config.py
ENV PORT=9005
ENV NOTEBOOK_DIR=/workspace

CMD jupyter lab --ip='0.0.0.0' --allow-root --no-browser --port=${PORT} --allow-root --notebook-dir=${NOTEBOOK_DIR}
