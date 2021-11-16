FROM debian AS pydev
RUN apt-get update && apt-get install -y python3 python3-pip
RUN useradd -ms /bin/bash user

FROM pydev AS web
COPY web/requirements.txt .
RUN pip3 install -r requirements.txt
USER user

FROM pydev AS translation_worker
COPY translation_worker/requirements.txt .
RUN pip3 install -r requirements.txt
USER user

FROM pydev AS file_worker
COPY file_worker/requirements.txt .
RUN pip3 install -r requirements.txt
USER user