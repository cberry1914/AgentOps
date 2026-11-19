
FROM node:24-bookworm-slim

ENV NODE_OPTIONS=""
ENV NPM_CONFIG_UPDATE_NOTIFIER=false
ENV NPM_CONFIG_FUND=false

RUN groupadd --gid 10001 agentops \
    && useradd \
        --uid 10001 \
        --gid 10001 \
        --no-create-home \
        --shell /usr/sbin/nologin \
        agentops

WORKDIR /workspace

USER 10001:10001

CMD ["node", "--version"]
