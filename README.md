# AgentOps

**A local-first, cloud-portable, multi-agent AI development platform.**

AgentOps coordinates specialized AI agents to assist with software development, quality assurance, security review, and architecture analysis. It is designed to work locally using Ollama, with a path toward future deployment on AWS or Azure.

## Project Goals

- Coordinate multiple specialized AI agents through a defined workflow.
- Use a local language model to minimize external API costs.
- Support development tasks against isolated project workspaces.
- Execute discovered Python tests inside a restricted Docker sandbox.
- Apply deterministic QA and security policies rather than allowing an LLM to make final approval decisions.
- Generate architecture and dependency diagrams from actual project files.
- Provide a browser dashboard for task submission, progress monitoring, and review.
- Preserve a modular architecture that can evolve toward cloud deployment.

## Architecture

AgentOps uses LangGraph to coordinate the workflow and specialized agents.

### Agent Responsibilities

| Agent | Responsibility |
|---|---|
| Project Manager | Coordinates agents, tracks progress, manages retries, and handles cancellation. |
| Developer | Analyzes development tasks and proposes workspace file changes. |
| QA | Performs static review, coordinates automated tests, and applies deterministic QA policy. |
| Security | Combines static security scanning with AI-assisted review and deterministic security policy. |
| Architect | Inspects project structure and generates architecture and dependency diagrams. |
| Instructor | Answers questions about an existing project using its source files and documentation. |

### Workflow

The primary development workflow follows this sequence:

1. Receive a development task.
2. Create an isolated task workspace.
3. Have the Developer Agent inspect the workspace and implement proposed changes.
4. Have the QA Agent perform static review and coordinate automated test execution.
5. Apply deterministic QA policy to the collected evidence.
6. Have the Security Agent perform static scanning and AI-assisted review.
7. Apply deterministic security policy.
8. Return bounded correction requests to the Developer Agent when appropriate.
9. Have the Architect Agent inspect the resulting project and generate diagrams.
10. Finalize the workflow and publish the results.

The Project Manager tracks observable workflow checkpoints, execution history, retry counts, and cancellation requests.

QA and security decisions require human review. A successful automated test run or clean security scan does not independently authorize production approval.

## Technology Stack

- **Language:** Python 3.12
- **Agent orchestration:** LangGraph
- **LLM integration:** LangChain and Ollama
- **Local model:** Qwen3 4B Instruct
- **API:** FastAPI
- **Dashboard:** HTML, CSS, and JavaScript
- **Automated testing:** pytest
- **Execution isolation:** Docker
- **Architecture diagrams:** Mermaid
- **Version control:** Git

## Project Structure

```text
AgentOps/
├── src/
│   └── agentops/
│       ├── agents/
│       │   ├── developer.py
│       │   ├── qa.py
│       │   ├── security.py
│       │   ├── architect.py
│       │   └── instructor.py
│       ├── models/
│       │   ├── state.py
│       │   ├── execution.py
│       │   ├── qa_review.py
│       │   └── security_review.py
│       ├── orchestrator/
│       │   └── workflow.py
│       ├── tools/
│       │   ├── cancellation.py
│       │   ├── context_manager.py
│       │   ├── file_manager.py
│       │   ├── project_importer.py
│       │   ├── diagram_generator.py
│       │   ├── test_runner.py
│       │   ├── test_discovery.py
│       │   ├── test_report_parser.py
│       │   ├── docker_sandbox.py
│       │   ├── sandbox_image_builder.py
│       │   ├── sandbox_manager.py
│       │   ├── qa_execution_manager.py
│       │   ├── qa_policy.py
│       │   ├── qa_review_service.py
│       │   ├── qa_finding_parser.py
│       │   ├── qa_report_formatter.py
│       │   ├── qa_review_history.py
│       │   ├── security_scanner.py
│       │   ├── security_sanitizer.py
│       │   ├── security_finding_parser.py
│       │   ├── security_policy.py
│       │   ├── security_review_service.py
│       │   ├── security_report_formatter.py
│       │   └── security_review_history.py
│       ├── static/
│       │   └── index.html
│       ├── api.py
│       └── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

The structure above describes the primary application components; additional supporting files may exist.

## Local Development Setup

### Requirements

- Python 3.12
- Ollama
- Docker Engine
- Git

### 1. Clone the repository

```bash
git clone <repository-url>
cd AgentOps
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Ollama

Install and start Ollama, then download the configured model:

```bash
ollama pull qwen3:4b-instruct
```

The model is used by the specialized agents with different instructions and responsibilities.

### 5. Start the API

From the project root, with the virtual environment activated:

```bash
PYTHONPATH=src uvicorn agentops.api:app --host 127.0.0.1 --port 8000
```

Open the dashboard:

```text
http://127.0.0.1:8000/
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Run the CLI

```bash
PYTHONPATH=src python -m agentops.main
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health. |
| POST | `/tasks` | Submit a development task. |
| GET | `/tasks` | List registered tasks. |
| GET | `/tasks/{task_id}` | Retrieve task progress and results. |
| POST | `/tasks/{task_id}/cancel` | Request task cancellation. |

Example task submission:

```json
{
  "task": "Create a Python REST API with input validation and unit tests.",
  "max_retries": 2
}
```

The API returns a task ID that can be used to monitor progress and retrieve results.

## Quality Assurance

The QA subsystem separates automated execution evidence from AI-assisted static review.

Its responsibilities include:

- Discovering supported pytest files.
- Identifying incomplete test discovery.
- Executing tests inside Docker.
- Parsing pytest results.
- Tracking individual test cases separately from test files.
- Recording execution status, errors, timeouts, and cancellation.
- Applying deterministic QA policy.
- Producing structured review reports.

Automated test execution is isolated from the main AgentOps process.

The sandbox is configured with restrictions including disabled networking, a read-only workspace mount, resource limits, a non-root execution user, and dropped Linux capabilities.

## Security Review

The security subsystem combines deterministic scanning with AI-assisted analysis.

Current capabilities include:

- Detecting selected hardcoded credential patterns.
- Identifying common API token formats.
- Redacting selected sensitive values before AI review.
- Parsing structured security findings.
- Applying deterministic severity-based policy.
- Maintaining structured security review history.

Security review is deliberately conservative. A clean scan does not constitute a guarantee that a project is secure.

## Architecture Analysis

The Architect Agent inspects project files and generates Mermaid diagrams.

Current diagram types include:

- Project structure.
- Python module dependencies.
- Component architecture.

Dependency analysis uses Python AST inspection. Dynamic imports and runtime relationships are not fully represented.

## Current Limitations

AgentOps is under active development.

- Task and review history are currently stored in memory.
- Data does not survive application restarts.
- API authentication has not yet been implemented.
- The API is intended for trusted local development, not public exposure.
- The local LLM operates within available hardware resources.
- Automated code execution is restricted to the Docker sandbox.
- Test execution currently supports pytest.
- Security scanning uses selected deterministic patterns and is not a comprehensive security audit.
- Architecture diagrams represent statically discoverable relationships rather than complete runtime behavior.
- Multi-file development changes are not transactional.
- Cloud deployment and persistent storage are future development goals.

## Roadmap

1. Complete integration of the multi-agent workflow.
2. Verify end-to-end task execution, retries, and cancellation.
3. Validate Docker sandbox construction and execution.
4. Verify QA and security decision handling.
5. Improve task and review persistence.
6. Add API authentication and access controls.
7. Improve project import and workspace isolation.
8. Expand supported languages and testing frameworks.
9. Develop cloud deployment options for AWS and Azure.
10. Introduce more advanced agent coordination and observability.

## Development Philosophy

AgentOps is designed around a separation of responsibilities:

- LLMs provide reasoning and recommendations.
- Deterministic application logic enforces workflow and policy decisions.
- Isolated execution environments handle untrusted code.
- Human reviewers retain final approval authority.

The objective is to build an extensible multi-agent engineering platform that combines AI-assisted development with controlled execution, measurable evidence, and accountable decision-making.
