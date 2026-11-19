
"""
AgentOps - Node.js Sandbox Image Builder

Builds the reusable Node.js sandbox base image.

Reuses the existing SandboxImageBuilder implementation while
keeping the Node.js image and Dockerfile separate from Python.
"""

from pathlib import Path

from agentops.tools.sandbox_image_builder import SandboxImageBuilder


class NodeSandboxImageBuilder(SandboxImageBuilder):
    """Build and manage the reusable Node.js sandbox image."""

    IMAGE_NAME = "agentops-node-sandbox:latest"

    DOCKERFILE_RELATIVE_PATH = (
        Path("src")
        / "agentops"
        / "tools"
        / "docker"
        / "node-sandbox"
        / "Dockerfile"
    )
