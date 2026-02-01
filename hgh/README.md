# hgh - Human GitHub CLI

A natural language wrapper around the GitHub CLI (`gh`) powered by a small language model.

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/chaliy/clitron.git
cd clitron

# Install hgh
cargo install --path hgh

# Download the model (~770MB, one-time)
hgh model-download
```

### Requirements

- [GitHub CLI](https://cli.github.com/) (`gh`) must be installed and authenticated
- Rust 1.70+ (for building from source)

## Usage

```bash
# Natural language commands
hgh "show my open pull requests"
hgh "list merged prs from last week"
hgh "view pr 123"
hgh "create a pr for this branch"
hgh "show issues assigned to me"

# See what command will be executed
hgh --dry-run "show my prs"
# Output: gh pr list

# See interpretation details
hgh --explain "show my prs"
# Output:
# Input: show my prs
# Interpreted as: gh pr list
# Confidence: 90%

# Always confirm before executing
hgh --confirm "merge this pr"

# Bypass interpretation, use gh directly
hgh --raw pr list --state open
```

## Options

| Flag | Description |
|------|-------------|
| `-e, --explain` | Show interpreted command before running |
| `-n, --dry-run` | Don't execute, just show the interpreted command |
| `-c, --confirm` | Always ask for confirmation before executing |
| `-t, --threshold <N>` | Confidence threshold for auto-execution (default: 0.7) |
| `-r, --raw` | Bypass interpretation, pass args directly to gh |

## Model Management

```bash
# Download the model
hgh model-download

# Check model status
hgh model-status
```

The model is downloaded to `~/.clitron/models/` and runs fully offline.

## How It Works

1. Your natural language input is sent to a small fine-tuned LLM (~770MB)
2. The model interprets your intent and outputs a structured command
3. The command is validated against the GitHub CLI schema
4. If confidence is high enough, it executes `gh` with the interpreted arguments

## Examples

| Natural Language | Interpreted Command |
|-----------------|---------------------|
| "show my prs" | `gh pr list` |
| "view pr 123" | `gh pr view 123` |
| "create a new pull request" | `gh pr create` |
| "list open issues" | `gh issue list --state open` |
| "show failed workflow runs" | `gh run list --status failure` |
| "merge this pr" | `gh pr merge` |

## License

MIT
