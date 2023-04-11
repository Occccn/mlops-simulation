FROM python:latest

ENV PROJECT_DIR test_env
WORKDIR /${PROJECT_DIR}
COPY src/ /${PROJECT_DIR}/src

RUN apt update\
    && apt install -y vim

COPY requirements.txt .
RUN python -m pip install -r /${PROJECT_DIR}/requirements.txt

# CMDだと動かない。execで入って、下記コマンドを実行して127.0.0.1:8000/docs見ると見れるはず
#CMD ["uvicorn", "src.api.app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]


# ADD requirements.txt /${PROJECT_DIR}/
# RUN apt-get -y update && \
#     apt-get -y install \
#     apt-utils \
#     gcc && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/* && \
#     pip install --no-cache-dir -r requirements.txt

# COPY src/ /${PROJECT_DIR}/src/

# COPY run.sh /${PROJECT_DIR}/run.sh
#RUN chmod +x /${PROJECT_DIR}/run.sh
#CMD [ "./run.sh" ]
