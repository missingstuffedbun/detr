FROM yihui8776/detr:v0.2
MAINTAINER author "missingstuffedbun@hotmail.com"


RUN pip install jupyterlab
RUN jupyter lab --generate-config
RUN python -c "from notebook.auth import passwd; print(\"c.NotebookApp.password = u'\" +  passwd('3582521') + \"'\")" >> ~/.jupyter/jupyter_lab_config.py
ENV PORT=9005
ENV NOTEBOOK_DIR=/workspace

CMD jupyter lab --ip='0.0.0.0' --allow-root --no-browser --port=${PORT} --allow-root --notebook-dir=${NOTEBOOK_DIR}
