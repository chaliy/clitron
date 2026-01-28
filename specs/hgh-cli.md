# HGH CLI Specification

## Overview

`hgh` (Human GitHub) is a demonstration CLI that wraps the official `gh` CLI with a natural language interface powered by clitron.

## Design Philosophy

1. **Transparent Wrapper**: hgh translates natural language to gh commands
2. **Fallback Support**: Users can always use traditional gh syntax
3. **Learning Tool**: Shows users the equivalent gh command
4. **Zero Configuration**: Works out of the box with gh installed
5. **Context-Aware**: Uses git state to improve interpretation

## Context Awareness

hgh gathers git context to enable smarter interpretations:

### Detected Context

| Context | Source | Use Case |
|---------|--------|----------|
| Current branch | `git branch --show-current` | Infer PR for "merge this" |
| Uncommitted changes | `git status --porcelain` | Warn before destructive ops |
| Staged changes | `git diff --cached` | Suggest commit before push |
| Repo owner/name | `git remote get-url origin` | Scope to current repo |
| Current PR | `gh pr view --json number` | Enable "this pr" references |
| Is worktree | `git rev-parse --git-dir` | Handle worktree-specific ops |

### Context-Aware Examples

```bash
# On branch 'feature/login' with PR #42
hgh "merge this"
# → Interprets as: gh pr merge 42

hgh "show the pr"
# → Interprets as: gh pr view 42

# With uncommitted changes
hgh "push this"
# → Warning: You have uncommitted changes. Commit first?

# In acme/webapp repo
hgh "list issues"
# → Interprets as: gh issue list (scoped to current repo)
```

### Implementation

```rust
fn gather_context() -> Context {
    let git = GitContext::detect_with_pr(
        std::env::var("GITHUB_TOKEN").ok().as_deref()
    ).ok().flatten();

    Context {
        git,
        custom: HashMap::new(),
    }
}

fn main() {
    let context = gather_context();
    let result = interpreter.interpret_with_context(&input, &context)?;
    // ...
}
```

## Installation

```bash
# Install hgh
cargo install hgh

# Ensure gh is installed and authenticated
gh auth status
```

## Usage Examples

### Pull Requests

```bash
# Natural language
hgh "show my open pull requests"
hgh "list merged prs from last week"
hgh "view pr 123"
hgh "create a pr for this branch"
hgh "merge my pr"

# Equivalent gh commands shown
# → gh pr list --author @me --state open
# → gh pr list --state merged --search "merged:>2024-01-01"
# → gh pr view 123
# → gh pr create
# → gh pr merge
```

### Issues

```bash
# Natural language
hgh "show open issues assigned to me"
hgh "create an issue about the login bug"
hgh "close issue 456"
hgh "list issues with bug label"

# Equivalent gh commands
# → gh issue list --assignee @me --state open
# → gh issue create --title "Login bug"
# → gh issue close 456
# → gh issue list --label bug
```

### Repository

```bash
# Natural language
hgh "clone the react repo"
hgh "fork this repository"
hgh "show repo info"
hgh "list my repos"

# Equivalent gh commands
# → gh repo clone facebook/react
# → gh repo fork
# → gh repo view
# → gh repo list
```

### Workflow/Actions

```bash
# Natural language
hgh "show workflow runs"
hgh "run the tests workflow"
hgh "show failed actions"

# Equivalent gh commands
# → gh run list
# → gh workflow run tests.yml
# → gh run list --status failure
```

## Command Schema

```yaml
# schemas/gh.yaml
cli_name: "gh"
description: "GitHub CLI - Work seamlessly with GitHub from the command line"

commands:
  - name: "pr"
    description: "Work with GitHub pull requests"
    aliases: ["pull-request", "pull request", "pr", "prs"]
    subcommands:
      - name: "list"
        description: "List pull requests in a repository"
        aliases: ["show", "display", "get", "find"]
        args:
          - name: "state"
            type: "enum"
            values: ["open", "closed", "merged", "all"]
            default: "open"
            description: "Filter by state"
            natural_language:
              - "open" -> ["open", "active", "pending"]
              - "closed" -> ["closed", "done", "finished"]
              - "merged" -> ["merged", "completed", "landed"]
          - name: "author"
            type: "string"
            description: "Filter by author"
            natural_language:
              - "@me" -> ["my", "mine", "i created", "i made"]
          - name: "assignee"
            type: "string"
            description: "Filter by assignee"
          - name: "label"
            type: "string"
            description: "Filter by label"
          - name: "limit"
            type: "integer"
            default: 30
            description: "Maximum number of PRs to fetch"
            natural_language:
              - pattern: "last (\\d+)"
              - pattern: "top (\\d+)"
              - pattern: "(\\d+) most recent"
        flags:
          - name: "web"
            short: "w"
            description: "Open in browser"
            natural_language: ["in browser", "on web", "open website"]

      - name: "view"
        description: "View a pull request"
        aliases: ["show", "display", "get", "open"]
        args:
          - name: "number"
            type: "integer"
            required: false
            description: "PR number"
            natural_language:
              - pattern: "#?(\\d+)"
              - pattern: "number (\\d+)"
        flags:
          - name: "web"
            short: "w"
            description: "Open in browser"

      - name: "create"
        description: "Create a pull request"
        aliases: ["make", "new", "open", "submit"]
        args:
          - name: "title"
            type: "string"
            description: "PR title"
          - name: "body"
            type: "string"
            description: "PR body"
          - name: "base"
            type: "string"
            description: "Base branch"
        flags:
          - name: "draft"
            short: "d"
            description: "Create as draft"
            natural_language: ["as draft", "draft pr", "wip"]
          - name: "web"
            short: "w"
            description: "Open in browser after creating"

      - name: "merge"
        description: "Merge a pull request"
        aliases: ["land", "complete", "finish"]
        args:
          - name: "number"
            type: "integer"
            required: false
            description: "PR number"
        flags:
          - name: "squash"
            short: "s"
            description: "Squash commits"
            natural_language: ["squash", "squash commits"]
          - name: "rebase"
            short: "r"
            description: "Rebase commits"
          - name: "delete-branch"
            short: "d"
            description: "Delete branch after merge"

  - name: "issue"
    description: "Work with GitHub issues"
    aliases: ["issues", "bug", "bugs", "ticket", "tickets"]
    subcommands:
      - name: "list"
        description: "List issues"
        aliases: ["show", "display", "get", "find"]
        args:
          - name: "state"
            type: "enum"
            values: ["open", "closed", "all"]
            default: "open"
          - name: "assignee"
            type: "string"
            natural_language:
              - "@me" -> ["assigned to me", "my issues", "mine"]
          - name: "author"
            type: "string"
          - name: "label"
            type: "string"
            natural_language:
              - pattern: "with label (\\w+)"
              - pattern: "labeled (\\w+)"
          - name: "limit"
            type: "integer"
            default: 30

      - name: "view"
        description: "View an issue"
        args:
          - name: "number"
            type: "integer"

      - name: "create"
        description: "Create an issue"
        aliases: ["make", "new", "open", "file", "report"]
        args:
          - name: "title"
            type: "string"
          - name: "body"
            type: "string"
          - name: "label"
            type: "string"

      - name: "close"
        description: "Close an issue"
        aliases: ["resolve", "complete", "finish"]
        args:
          - name: "number"
            type: "integer"
            required: true

  - name: "repo"
    description: "Work with GitHub repositories"
    aliases: ["repository", "project"]
    subcommands:
      - name: "view"
        description: "View repository details"
        aliases: ["show", "info", "details"]
        args:
          - name: "repository"
            type: "string"
            description: "Repository in owner/repo format"
        flags:
          - name: "web"
            short: "w"

      - name: "clone"
        description: "Clone a repository"
        aliases: ["download", "get", "fetch"]
        args:
          - name: "repository"
            type: "string"
            required: true

      - name: "fork"
        description: "Fork a repository"
        aliases: ["copy"]
        args:
          - name: "repository"
            type: "string"

      - name: "list"
        description: "List repositories"
        aliases: ["show", "my repos"]
        flags:
          - name: "public"
            description: "Show only public repos"
          - name: "private"
            description: "Show only private repos"

  - name: "run"
    description: "Work with GitHub Actions runs"
    aliases: ["workflow", "action", "actions", "ci"]
    subcommands:
      - name: "list"
        description: "List workflow runs"
        args:
          - name: "workflow"
            type: "string"
            description: "Workflow name or file"
          - name: "status"
            type: "enum"
            values: ["queued", "in_progress", "completed", "failure", "success"]
            natural_language:
              - "failure" -> ["failed", "failing", "broken", "red"]
              - "success" -> ["passed", "passing", "green", "successful"]

      - name: "view"
        description: "View a workflow run"
        args:
          - name: "run_id"
            type: "string"

      - name: "watch"
        description: "Watch a workflow run"
        args:
          - name: "run_id"
            type: "string"
```

## Implementation

### Project Structure

```
hgh/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── executor.rs
│   └── display.rs
└── schemas/
    └── gh.yaml
```

### Main Entry Point

```rust
// hgh/src/main.rs

use clitron::{Interpreter, CommandSchema, InterpretedCommand};
use std::process::Command;

fn main() -> anyhow::Result<()> {
    // Parse command line
    let args: Vec<String> = std::env::args().skip(1).collect();

    // If no args or help, show usage
    if args.is_empty() {
        print_usage();
        return Ok(());
    }

    // Check for raw mode (bypass interpretation)
    if args[0] == "--raw" || args[0] == "-r" {
        return execute_gh(&args[1..]);
    }

    // Check for explain mode
    let explain = args.iter().any(|a| a == "--explain" || a == "-e");
    let args: Vec<_> = args.into_iter()
        .filter(|a| a != "--explain" && a != "-e")
        .collect();

    // Combine args into natural language input
    let input = args.join(" ");

    // Load interpreter
    let interpreter = load_interpreter()?;

    // Interpret the input
    match interpreter.interpret(&input) {
        Ok(cmd) if cmd.is_confident(0.7) => {
            let gh_args = cmd.to_args();

            if explain {
                println!("Interpreted as: gh {}", gh_args.join(" "));
                println!("Confidence: {:.0}%", cmd.confidence * 100.0);
                println!();
            }

            execute_gh(&gh_args)
        }
        Ok(cmd) => {
            eprintln!("Low confidence ({:.0}%)", cmd.confidence * 100.0);
            eprintln!("Did you mean: gh {}", cmd.to_args().join(" "));
            eprintln!();
            eprintln!("Run with the suggested command? [y/N]");

            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;

            if input.trim().to_lowercase() == "y" {
                execute_gh(&cmd.to_args())
            } else {
                Ok(())
            }
        }
        Err(e) => {
            eprintln!("Could not interpret: {}", e);
            eprintln!();
            eprintln!("Try using traditional gh syntax:");
            eprintln!("  hgh --raw pr list");
            Err(e.into())
        }
    }
}

fn execute_gh(args: &[String]) -> anyhow::Result<()> {
    let status = Command::new("gh")
        .args(args)
        .status()?;

    std::process::exit(status.code().unwrap_or(1));
}

fn load_interpreter() -> anyhow::Result<Interpreter> {
    let schema = CommandSchema::from_yaml(include_str!("../schemas/gh.yaml"))?;
    Interpreter::with_schema(schema)
}

fn print_usage() {
    println!("hgh - Human GitHub CLI");
    println!();
    println!("Usage: hgh <natural language command>");
    println!();
    println!("Examples:");
    println!("  hgh show my open prs");
    println!("  hgh create a pr for this branch");
    println!("  hgh list issues with bug label");
    println!("  hgh merge pr 123");
    println!();
    println!("Options:");
    println!("  --explain, -e    Show interpreted command before running");
    println!("  --raw, -r        Bypass interpretation, pass args to gh directly");
    println!();
    println!("For traditional gh usage:");
    println!("  hgh --raw pr list --state open");
}
```

### Display Formatting

```rust
// hgh/src/display.rs

use colored::Colorize;
use clitron::InterpretedCommand;

pub fn show_interpretation(cmd: &InterpretedCommand) {
    println!("{}", "Interpreted command:".dimmed());
    println!("  gh {}", cmd.to_args().join(" ").green());
    println!();
    println!("{} {:.0}%",
        "Confidence:".dimmed(),
        cmd.confidence * 100.0
    );
    println!();
}

pub fn show_suggestions(suggestions: &[String]) {
    println!("{}", "Did you mean:".yellow());
    for (i, suggestion) in suggestions.iter().enumerate() {
        println!("  {}. gh {}", i + 1, suggestion);
    }
}

pub fn show_error(msg: &str) {
    eprintln!("{} {}", "Error:".red(), msg);
}
```

## Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pr_list_interpretation() {
        let interpreter = load_interpreter().unwrap();

        let test_cases = vec![
            ("show my open prs", vec!["pr", "list", "--author", "@me", "--state", "open"]),
            ("list merged pull requests", vec!["pr", "list", "--state", "merged"]),
            ("show prs", vec!["pr", "list"]),
        ];

        for (input, expected) in test_cases {
            let cmd = interpreter.interpret(input).unwrap();
            assert_eq!(cmd.to_args(), expected, "Failed for: {}", input);
        }
    }

    #[test]
    fn test_issue_interpretation() {
        let interpreter = load_interpreter().unwrap();

        let cmd = interpreter.interpret("show issues assigned to me").unwrap();
        assert_eq!(cmd.command, "issue");
        assert_eq!(cmd.subcommand, Some("list".into()));
        assert_eq!(cmd.args.get("assignee"), Some(&json!("@me")));
    }
}
```

### Integration Tests

```bash
#!/bin/bash
# test_hgh.sh

set -e

echo "Testing hgh..."

# Test PR commands
hgh --explain "show my open prs" | grep -q "gh pr list"
hgh --explain "view pr 123" | grep -q "gh pr view 123"

# Test Issue commands
hgh --explain "list issues with bug label" | grep -q "gh issue list"
hgh --explain "create issue about login" | grep -q "gh issue create"

# Test raw mode
hgh --raw pr list --limit 1

echo "All tests passed!"
```

## Cargo.toml

```toml
[package]
name = "hgh"
version = "0.1.0"
edition = "2021"
description = "Human-friendly GitHub CLI powered by clitron"
license = "MIT"
repository = "https://github.com/chaliy/clitron"

[[bin]]
name = "hgh"
path = "src/main.rs"

[dependencies]
clitron = { path = "../clitron" }
anyhow = "1.0"
colored = "2.0"

[dev-dependencies]
assert_cmd = "2.0"
predicates = "3.0"
```

## Distribution

### Binary Releases

Build for multiple platforms:

```bash
# Linux
cargo build --release --target x86_64-unknown-linux-gnu

# macOS (Intel)
cargo build --release --target x86_64-apple-darwin

# macOS (Apple Silicon)
cargo build --release --target aarch64-apple-darwin

# Windows
cargo build --release --target x86_64-pc-windows-msvc
```

### Homebrew Formula

```ruby
class Hgh < Formula
  desc "Human-friendly GitHub CLI"
  homepage "https://github.com/chaliy/clitron"
  version "0.1.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/chaliy/clitron/releases/download/v0.1.0/hgh-aarch64-apple-darwin.tar.gz"
      sha256 "..."
    else
      url "https://github.com/chaliy/clitron/releases/download/v0.1.0/hgh-x86_64-apple-darwin.tar.gz"
      sha256 "..."
    end
  end

  depends_on "gh"

  def install
    bin.install "hgh"
  end
end
```

## Training Data Examples

Training examples specific to the `gh` CLI. See [training.md](training.md) for the generic training pipeline.

### Dataset Size

For gh CLI with ~30 subcommands: **~3,000 examples**

| Category | Count | Description |
|----------|-------|-------------|
| PR commands | ~800 | list, view, create, merge, checkout, close, ready |
| Issue commands | ~600 | list, view, create, close, reopen, comment |
| Repo commands | ~400 | view, clone, fork, list, create |
| Run/Actions | ~400 | list, view, watch, rerun, cancel |
| Gist commands | ~200 | list, view, create |
| Release commands | ~200 | list, view, create |
| Negative cases | ~400 | ambiguous, incomplete, off-topic |

### Example Training Data

#### Positive Cases (with context)

```json
{
  "instruction": "merge this and delete the branch",
  "context": {
    "git": {
      "current_branch": "feature/add-login",
      "repo_owner": "acme",
      "repo_name": "webapp",
      "current_pr": 123,
      "has_uncommitted_changes": false
    }
  },
  "output": "{\"type\": \"command\", \"command\": \"pr\", \"subcommand\": \"merge\", \"args\": {\"number\": 123}, \"flags\": [\"delete-branch\"], \"confidence\": 0.95}"
}
```

```json
{
  "instruction": "show the pr",
  "context": {
    "git": {
      "current_branch": "fix/bug-456",
      "current_pr": 42
    }
  },
  "output": "{\"type\": \"command\", \"command\": \"pr\", \"subcommand\": \"view\", \"args\": {\"number\": 42}, \"flags\": [], \"confidence\": 0.95}"
}
```

```json
{
  "instruction": "close this issue",
  "context": null,
  "output": "{\"type\": \"clarification\", \"message\": \"Which issue would you like to close?\", \"suggestions\": [], \"confidence\": 0.3}"
}
```

#### Positive Cases (without context)

```json
{
  "instruction": "list my open pull requests",
  "context": null,
  "output": "{\"type\": \"command\", \"command\": \"pr\", \"subcommand\": \"list\", \"args\": {\"author\": \"@me\", \"state\": \"open\"}, \"flags\": [], \"confidence\": 0.95}"
}
```

```json
{
  "instruction": "show issues labeled bug",
  "context": null,
  "output": "{\"type\": \"command\", \"command\": \"issue\", \"subcommand\": \"list\", \"args\": {\"label\": \"bug\"}, \"flags\": [], \"confidence\": 0.92}"
}
```

```json
{
  "instruction": "create a draft pr",
  "context": null,
  "output": "{\"type\": \"command\", \"command\": \"pr\", \"subcommand\": \"create\", \"args\": {}, \"flags\": [\"draft\"], \"confidence\": 0.9}"
}
```

#### Negative Cases (clarification needed)

```json
{
  "instruction": "pr",
  "context": null,
  "output": "{\"type\": \"clarification\", \"message\": \"What would you like to do with pull requests?\", \"suggestions\": [{\"label\": \"List PRs\", \"command\": \"pr\", \"subcommand\": \"list\"}, {\"label\": \"Create PR\", \"command\": \"pr\", \"subcommand\": \"create\"}, {\"label\": \"View current PR\", \"command\": \"pr\", \"subcommand\": \"view\"}], \"confidence\": 0.3}"
}
```

```json
{
  "instruction": "show 123",
  "context": null,
  "output": "{\"type\": \"clarification\", \"message\": \"Did you mean PR #123 or Issue #123?\", \"suggestions\": [{\"label\": \"PR #123\", \"command\": \"pr\", \"subcommand\": \"view\", \"args\": {\"number\": 123}}, {\"label\": \"Issue #123\", \"command\": \"issue\", \"subcommand\": \"view\", \"args\": {\"number\": 123}}], \"confidence\": 0.4}"
}
```

```json
{
  "instruction": "what's the weather",
  "context": null,
  "output": "{\"type\": \"clarification\", \"message\": \"I can help with GitHub commands like pull requests, issues, repos, and workflows. What would you like to do?\", \"suggestions\": [], \"confidence\": 0.1}"
}
```

### Context-Dependent Interpretations

| Input | Without Context | With Context (PR #42) |
|-------|-----------------|----------------------|
| "merge this" | Clarification: which PR? | `pr merge 42` |
| "show the pr" | Clarification: which PR? | `pr view 42` |
| "close it" | Clarification: close what? | `pr close 42` |
| "checkout the pr" | Clarification: which PR? | `pr checkout 42` |
| "mark as ready" | Clarification: which PR? | `pr ready 42` |

### Phrasing Variations

Examples of how the same command can be expressed:

**List open PRs:**
- "show my open pull requests"
- "list open prs"
- "what prs do i have open"
- "my pending pull requests"
- "open prs by me"
- "display my active prs"

**Create a PR:**
- "create a pull request"
- "make a pr"
- "open a new pr"
- "submit pr"
- "new pull request"
- "create pr for this branch"

**Merge PR #123:**
- "merge pr 123"
- "merge #123"
- "land pr 123"
- "complete pull request 123"
- "merge pull request number 123"
