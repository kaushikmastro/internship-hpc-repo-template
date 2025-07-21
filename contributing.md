# How to contribute

Here are some important resources:

- [How to Write a Git Commit Message](https://cbea.ms/git-commit/)
- [Addressing Merge Conflicts](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts)

## Conventions

<!-- markdownlint-disable MD033 -->
<p>

1. We use [GitHub Issues](https://github.com/rodrigobdz/hugging-face-voice-assistant/issues) with [labels](https://github.com/rodrigobdz/hugging-face-voice-assistant/issues/labels) to track issues and tasks.

2. We follow the [Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow) and [make atomic commits](https://dev.to/samuelfaure/how-atomic-git-commits-dramatically-increased-my-productivity-and-will-increase-yours-too-4a84) and atomic PRs.

3. We strive to write meaningful and clear commit messages. One-line messages are fine for small changes, but bigger changes should look like this ):

    ```sh
    git commit --message "Scope: A brief summary of the commit
    >
    > A paragraph describing what changed, why it was necessary, and its impact.
    > At /some_path/some_file
    > This and that changed,
    > because this and that was required.
    > Test it like so and so."
    ```

    The example above loosely follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

4. Coding Conventions

   - Shell scripts: [Google Styleguide](https://google.github.io/styleguide/shellguide.html)
   - Python: [black](https://black.readthedocs.io/en/stable/), [Pydantic](https://docs.pydantic.dev/latest/)

5. Documentation

    - Document in the ```README.md``` or in the ```/doc``` folder.
    - Documentation is required to cover everything from repository set-up to running ```main.py``` or another script showcasing the project.

## Getting Started

### Requirements

Before you commit please install the pre-commit hooks. Only commit once they all pass.

- `pre-commit`

  ```sh
  # Install pre-commit binary
  brew install pre-commit

  # Install pre-commit hooks
  pre-commit install
  ```

#### Pre-commit restrictions

- You can't commit to main, please create a feature based branch and use a Pull Request to integrate your features to main.
  - Branch naming convention is ```feature/feature_name```, ```bug/bug_name```, ```refactoring/refactoring_name```, etc.
- Test coverage must be sufficient. 90% of your codebase must be covered by the tests in ```/tests```.

### Testing

**Unit Tests**: Unit tests verify individual functions, methods, or classes in isolation. They test components independently without external dependencies.

**Integration Tests**: Integration tests verify that different components work correctly when combined. They ensure proper data flow and system behavior across multiple integrated parts.

**Testing Requirements**:
- One unit test per function/method/class to ensure individual components work.
- One integration test for the whole application to verify end-to-end functionality.
- All tests must be placed in the ```/tests``` directory


### Development Environment

We recommend the following development environment:

- [VSCode](https://code.visualstudio.com)

## Workflow

Follow this 6-step procedure to contribute to the project.

### 1. Branch off
Create a feature branch using naming convention `feature/name`, `bug/name`, or `refactoring/name`.

```sh
git checkout main && git pull origin main
git checkout -b feature/feature_name
```

### 2. Code + Tests
Write code following Python conventions and ensure 90% test coverage.

```sh
# Write unit tests in /tests/unit/ and integration tests in /tests/integration/
```

### 3. Commit (ensure hooks work locally)
Run pre-commit hooks before committing with meaningful messages.

```sh
pre-commit run --all-files
git add . && git commit -m "Scope: Brief summary"
```

### 4. CI checks for compliance
Push branch and let CI validate code style, test coverage, and compliance.

```sh
git push origin feature/feature_name
```

### 5. Create PR to main
Create a Pull Request with clear title and description, referencing GitHub Issues.

### 6. Approved and merge
Address review feedback, get approval, merge to main, then delete feature branch.
