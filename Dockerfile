FROM mambaorg/micromamba:2.0.5

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /app/environment.yml

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba install -y -n base -f /app/environment.yml && \
    micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

ENV PYTHONPATH=/app
ENV WANDB_MODE=disabled

CMD ["python", "run_all.py", "--suite", "replication", "--overwrite_results", "--no_wandb"]
