
"""
AgentOps - Node.js Docker Sandbox

Executes Node.js project tests inside a restricted Docker container.

Inherits DockerSandbox's process management, output capture,
timeout handling, cancellation, and container cleanup.

Node.js dependencies are installed in the project-specific image
at /workspace/node_modules.

The project source is mounted read-only at /workspace/project,
allowing Node.js module resolution to find the dependencies
without modifying the project workspace.
"""

from pathlib import Path

from agentops.tools.docker_sandbox import (
    DockerSandbox,
    SandboxConfig,
)


class NodeDockerSandbox(DockerSandbox):
    """
    Executes Node.js tests using the existing Docker sandbox controls.

    The only supported test command is npm test.
    Arbitrary commands and Docker arguments are not accepted.
    """

    DEFAULT_IMAGE = "agentops-node-sandbox:latest"

    def __init__(
        self,
        config: SandboxConfig | None = None,
    ):
        if config is None:
            config = SandboxConfig(
                image=self.DEFAULT_IMAGE,
            )

        super().__init__(config)

    def _build_command(
        self,
        workspace: Path,
        container_name: str,
    ) -> list[str]:
        """
        Construct the restricted Node.js Docker execution command.

        Project source:
            /workspace/project (read-only)

        Project dependencies:
            /workspace/node_modules (inside the image)

        Node.js resolves modules from /workspace/project by
        searching parent directories, including /workspace/node_modules.
        """

        return [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--pids-limit",
            str(self.config.process_limit),
            "--memory",
            self.config.memory_limit,
            "--memory-swap",
            self.config.memory_swap_limit,
            "--cpus",
            self.config.cpu_limit,
            "--ulimit",
            "nofile=1024:1024",
            "--tmpfs",
            (
                f"/tmp:rw,noexec,nosuid,nodev,"
                f"size={self.config.tmpfs_size}"
            ),
            "--mount",
            (
                f"type=bind,source={workspace},"
                "target=/workspace/project,readonly"
            ),
            "--workdir",
            "/workspace/project",
            "--user",
            "10001:10001",
            "--env",
            "HOME=/tmp",
            "--env",
            "TMPDIR=/tmp",
            "--env",
            "NPM_CONFIG_CACHE=/tmp/.npm",
            "--env",
            "NPM_CONFIG_UPDATE_NOTIFIER=false",
            "--env",
            "NPM_CONFIG_FUND=false",
            self.config.image,
            "npm",
            "--prefix",
            "/workspace/project",
            "test",
        ]
