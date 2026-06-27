FROM continuumio/miniconda3:latest

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN conda create -y -n llm_algo_cnb_dev \
        python=3.10 \
        pip \
        jupyterlab \
        ipywidgets \
        nbformat \
        nbconvert \
        jupyterlab-mathjax3 \
        matplotlib \
        pandas \
        -c conda-forge \
        -c pytorch \
    && conda clean -afy

RUN /opt/conda/envs/llm_algo_cnb_dev/bin/pip install --no-cache-dir \
        torch \
        einops \
        transformers \
        triton \
        termcolor \
        pytest

SHELL ["/bin/bash", "-lc"]

RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate llm_algo_cnb_dev" >> /etc/bash.bashrc

ENV PATH=/opt/conda/envs/llm_algo_cnb_dev/bin:/opt/conda/bin:$PATH
ENV CONDA_DEFAULT_ENV=llm_algo_cnb_dev
ENV CONDA_PREFIX=/opt/conda/envs/llm_algo_cnb_dev

CMD ["bash"]
