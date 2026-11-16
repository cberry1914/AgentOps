
# AgentOps - Python Test Sandbox
#
# Reproducible Python environment for isolated pytest execution.
# Dependencies are installed during image construction.
# Runtime execution does not require network access.
#
# The container runs as an unprivileged user.

FROM python:3.12-slim

# Prevent Python from creating bytecode files and buffering output.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Copy the pinned dependency manifest into the image.
COPY requirements.lock /tmp/requirements.lock

# Install the complete pinned AgentOps test environment.
RUN pip install --no-cache-dir \
        --requirement /tmp/requirements.lock \
    && rm /tmp/requirements.lock

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
