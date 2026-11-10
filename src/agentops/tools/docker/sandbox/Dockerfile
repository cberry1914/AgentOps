
# AgentOps - Python Test Sandbox
#
# Minimal Python environment for isolated pytest execution.
# The container runs as an unprivileged user.
#
# This image is built separately from the AgentOps application.

FROM python:3.12-slim

# Prevent Python from creating bytecode files and buffering output.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install only the test framework.
# Plugin autoload is disabled at runtime by DockerSandbox.
RUN pip install --no-cache-dir pytest==8.3.5

# Create a dedicated non-root execution account.
RUN groupadd --gid 10001 agentops \
    && useradd \
        --uid 10001 \
        --gid 10001 \
        --no-create-home \
        --shell /usr/sbin/nologin \
        agentops

# Project files are mounted read-only at runtime.
WORKDIR /workspace

# Run all container processes without root privileges.
USER 10001:10001

# DockerSandbox supplies the pytest command.
CMD ["python", "--version"]
