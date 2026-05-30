ARG BASE_IMAGE=vastai/pytorch_cuda-12.1.1-auto
FROM ${BASE_IMAGE}

ARG TRADUZAI_REPO_URL=https://github.com/OTheVyzx/TraduzAi.git
ARG TRADUZAI_REPO_BRANCH=main

ENV TRADUZAI_PROJECT_ROOT=/workspace/TraduzAI
ENV TRADUZAI_MODELS_DIR=/workspace/TraduzAI/pipeline/models
ENV TRADUZAI_VENV_DIR=/workspace/TraduzAI/.venv

COPY bootstrap.sh /tmp/traduzai-vast-bootstrap.sh
RUN chmod +x /tmp/traduzai-vast-bootstrap.sh \
    && TRADUZAI_REPO_URL="${TRADUZAI_REPO_URL}" \
       TRADUZAI_REPO_BRANCH="${TRADUZAI_REPO_BRANCH}" \
       /tmp/traduzai-vast-bootstrap.sh

WORKDIR /workspace/TraduzAI
CMD ["bash", "scripts/vast/start-worker.sh"]
