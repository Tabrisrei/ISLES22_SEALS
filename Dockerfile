FROM continuumio/miniconda3

RUN conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

RUN mkdir -p /opt/algorithm /input /output

COPY ./ /opt/algorithm/

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm \
    &&chown -R algorithm:algorithm /opt/algorithm/ /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    &&pip install -e .

ENTRYPOINT ["bash", "nnunet_launcher.sh"]
