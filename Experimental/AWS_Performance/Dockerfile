FROM pytorch/pytorch
WORKDIR /opt/work
COPY main.py .
RUN pip3 install sentencepiece transformers
RUN python3 main.py
CMD [ "python3", "main.py" ]