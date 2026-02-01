#!/usr/bin/env python3
"""Generate training data manually (without API calls)."""

import json
import random
from pathlib import Path

# ============================================================================
# POSITIVE EXAMPLES WITH CONTEXT (60%)
# ============================================================================

POSITIVE_WITH_CONTEXT = [
    # PR list with context
    {"instruction": "show my prs", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"author": "@me"}, "flags": [], "confidence": 0.95}},
    {"instruction": "list open pull requests", "context": {"git": {"current_branch": "feature/auth", "repo_owner": "myorg", "repo_name": "api", "current_pr": 42, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"state": "open"}, "flags": [], "confidence": 0.95}},
    {"instruction": "show prs in browser", "context": {"git": {"current_branch": "develop", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": True}}, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {}, "flags": ["web"], "confidence": 0.92}},
    {"instruction": "display merged prs", "context": {"git": {"current_branch": "main", "repo_owner": "corp", "repo_name": "backend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"state": "merged"}, "flags": [], "confidence": 0.94}},
    {"instruction": "find closed pull requests", "context": {"git": {"current_branch": "release", "repo_owner": "startup", "repo_name": "app", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"state": "closed"}, "flags": [], "confidence": 0.93}},
    {"instruction": "show all prs", "context": {"git": {"current_branch": "main", "repo_owner": "dev", "repo_name": "lib", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"state": "all"}, "flags": [], "confidence": 0.94}},
    {"instruction": "list prs by @octocat", "context": {"git": {"current_branch": "main", "repo_owner": "github", "repo_name": "docs", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"author": "@octocat"}, "flags": [], "confidence": 0.95}},
    {"instruction": "show my open prs on the web", "context": {"git": {"current_branch": "fix/bug", "repo_owner": "company", "repo_name": "product", "current_pr": 88, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"author": "@me", "state": "open"}, "flags": ["web"], "confidence": 0.94}},
    {"instruction": "get prs with label bug", "context": {"git": {"current_branch": "main", "repo_owner": "oss", "repo_name": "tool", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"label": "bug"}, "flags": [], "confidence": 0.93}},
    {"instruction": "list last 10 prs", "context": {"git": {"current_branch": "main", "repo_owner": "org", "repo_name": "service", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"limit": 10}, "flags": [], "confidence": 0.92}},

    # PR view with context
    {"instruction": "show this pr", "context": {"git": {"current_branch": "feature/login", "repo_owner": "acme", "repo_name": "webapp", "current_pr": 123, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "view", "args": {"number": 123}, "flags": [], "confidence": 0.96}},
    {"instruction": "view the current pull request", "context": {"git": {"current_branch": "fix/auth", "repo_owner": "corp", "repo_name": "api", "current_pr": 456, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "view", "args": {"number": 456}, "flags": [], "confidence": 0.95}},
    {"instruction": "open this pr in browser", "context": {"git": {"current_branch": "feature/ui", "repo_owner": "team", "repo_name": "frontend", "current_pr": 789, "has_uncommitted_changes": True}}, "output": {"type": "command", "command": "pr", "subcommand": "view", "args": {"number": 789}, "flags": ["web"], "confidence": 0.94}},
    {"instruction": "show pr details", "context": {"git": {"current_branch": "hotfix/crash", "repo_owner": "startup", "repo_name": "mobile", "current_pr": 101, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "view", "args": {"number": 101}, "flags": [], "confidence": 0.93}},
    {"instruction": "display the pr", "context": {"git": {"current_branch": "feature/perf", "repo_owner": "bigco", "repo_name": "backend", "current_pr": 202, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "view", "args": {"number": 202}, "flags": [], "confidence": 0.94}},
    {"instruction": "view pr with comments", "context": {"git": {"current_branch": "feature/test", "repo_owner": "dev", "repo_name": "lib", "current_pr": 55, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "view", "args": {"number": 55}, "flags": ["comments"], "confidence": 0.93}},
    {"instruction": "check the pr", "context": {"git": {"current_branch": "feat/api", "repo_owner": "org", "repo_name": "service", "current_pr": 77, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "view", "args": {"number": 77}, "flags": [], "confidence": 0.91}},
    {"instruction": "show current pr in web", "context": {"git": {"current_branch": "feature/docs", "repo_owner": "oss", "repo_name": "project", "current_pr": 33, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "view", "args": {"number": 33}, "flags": ["web"], "confidence": 0.94}},

    # PR create with context
    {"instruction": "create a pr", "context": {"git": {"current_branch": "feature/new-login", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "create", "args": {}, "flags": [], "confidence": 0.95}},
    {"instruction": "open a pull request", "context": {"git": {"current_branch": "fix/memory-leak", "repo_owner": "corp", "repo_name": "server", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "create", "args": {}, "flags": [], "confidence": 0.95}},
    {"instruction": "make a pr as draft", "context": {"git": {"current_branch": "wip/refactor", "repo_owner": "team", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "create", "args": {}, "flags": ["draft"], "confidence": 0.94}},
    {"instruction": "submit this as a draft pr", "context": {"git": {"current_branch": "feature/experiment", "repo_owner": "dev", "repo_name": "lab", "current_pr": None, "has_uncommitted_changes": True}}, "output": {"type": "command", "command": "pr", "subcommand": "create", "args": {}, "flags": ["draft"], "confidence": 0.93}},
    {"instruction": "create pr and open in browser", "context": {"git": {"current_branch": "feat/ui-update", "repo_owner": "startup", "repo_name": "app", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "create", "args": {}, "flags": ["web"], "confidence": 0.94}},
    {"instruction": "new pull request for this branch", "context": {"git": {"current_branch": "feature/search", "repo_owner": "bigco", "repo_name": "platform", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "create", "args": {}, "flags": [], "confidence": 0.93}},
    {"instruction": "create wip pr", "context": {"git": {"current_branch": "wip/testing", "repo_owner": "org", "repo_name": "service", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "create", "args": {}, "flags": ["draft"], "confidence": 0.92}},
    {"instruction": "open pr with commit info", "context": {"git": {"current_branch": "fix/typo", "repo_owner": "oss", "repo_name": "docs", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "create", "args": {}, "flags": ["fill"], "confidence": 0.91}},

    # PR merge with context
    {"instruction": "merge this pr", "context": {"git": {"current_branch": "feature/complete", "repo_owner": "acme", "repo_name": "webapp", "current_pr": 100, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 100}, "flags": [], "confidence": 0.96}},
    {"instruction": "land the pr", "context": {"git": {"current_branch": "feat/done", "repo_owner": "corp", "repo_name": "api", "current_pr": 200, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 200}, "flags": [], "confidence": 0.94}},
    {"instruction": "merge and squash", "context": {"git": {"current_branch": "feature/many-commits", "repo_owner": "team", "repo_name": "frontend", "current_pr": 300, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 300}, "flags": ["squash"], "confidence": 0.95}},
    {"instruction": "merge this and delete the branch", "context": {"git": {"current_branch": "feature/temp", "repo_owner": "startup", "repo_name": "app", "current_pr": 400, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 400}, "flags": ["delete-branch"], "confidence": 0.95}},
    {"instruction": "squash merge the pr", "context": {"git": {"current_branch": "fix/small", "repo_owner": "dev", "repo_name": "lib", "current_pr": 500, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 500}, "flags": ["squash"], "confidence": 0.94}},
    {"instruction": "complete this pr with rebase", "context": {"git": {"current_branch": "feature/clean", "repo_owner": "org", "repo_name": "service", "current_pr": 600, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 600}, "flags": ["rebase"], "confidence": 0.93}},
    {"instruction": "merge pr and remove branch", "context": {"git": {"current_branch": "feature/old", "repo_owner": "bigco", "repo_name": "backend", "current_pr": 700, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 700}, "flags": ["delete-branch"], "confidence": 0.94}},
    {"instruction": "land this with squash", "context": {"git": {"current_branch": "feat/messy", "repo_owner": "oss", "repo_name": "tool", "current_pr": 800, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 800}, "flags": ["squash"], "confidence": 0.93}},
    {"instruction": "enable auto merge", "context": {"git": {"current_branch": "feature/waiting", "repo_owner": "acme", "repo_name": "webapp", "current_pr": 900, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 900}, "flags": ["auto"], "confidence": 0.92}},

    # PR checkout with context
    {"instruction": "checkout pr 42", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "checkout", "args": {"number": 42}, "flags": [], "confidence": 0.96}},
    {"instruction": "switch to pr 123", "context": {"git": {"current_branch": "develop", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "checkout", "args": {"number": 123}, "flags": [], "confidence": 0.94}},
    {"instruction": "co pr 55", "context": {"git": {"current_branch": "main", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "checkout", "args": {"number": 55}, "flags": [], "confidence": 0.93}},
    {"instruction": "checkout pull request 77", "context": {"git": {"current_branch": "main", "repo_owner": "startup", "repo_name": "app", "current_pr": None, "has_uncommitted_changes": True}}, "output": {"type": "command", "command": "pr", "subcommand": "checkout", "args": {"number": 77}, "flags": [], "confidence": 0.95}},

    # PR close with context
    {"instruction": "close this pr", "context": {"git": {"current_branch": "feature/abandoned", "repo_owner": "acme", "repo_name": "webapp", "current_pr": 99, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "close", "args": {"number": 99}, "flags": [], "confidence": 0.95}},
    {"instruction": "cancel the pull request", "context": {"git": {"current_branch": "feat/old", "repo_owner": "corp", "repo_name": "api", "current_pr": 88, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "close", "args": {"number": 88}, "flags": [], "confidence": 0.93}},
    {"instruction": "close pr and delete branch", "context": {"git": {"current_branch": "feature/stale", "repo_owner": "team", "repo_name": "frontend", "current_pr": 77, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "close", "args": {"number": 77}, "flags": ["delete-branch"], "confidence": 0.94}},

    # PR ready with context
    {"instruction": "mark this pr as ready", "context": {"git": {"current_branch": "feature/done-draft", "repo_owner": "acme", "repo_name": "webapp", "current_pr": 111, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "ready", "args": {"number": 111}, "flags": [], "confidence": 0.95}},
    {"instruction": "undraft this pr", "context": {"git": {"current_branch": "wip/complete", "repo_owner": "corp", "repo_name": "api", "current_pr": 222, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "ready", "args": {"number": 222}, "flags": [], "confidence": 0.94}},
    {"instruction": "make pr ready for review", "context": {"git": {"current_branch": "draft/done", "repo_owner": "team", "repo_name": "frontend", "current_pr": 333, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "pr", "subcommand": "ready", "args": {"number": 333}, "flags": [], "confidence": 0.93}},

    # Issue list with context
    {"instruction": "show my issues", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "list", "args": {"assignee": "@me"}, "flags": [], "confidence": 0.95}},
    {"instruction": "list open issues", "context": {"git": {"current_branch": "develop", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "list", "args": {"state": "open"}, "flags": [], "confidence": 0.95}},
    {"instruction": "show bugs", "context": {"git": {"current_branch": "main", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "list", "args": {"label": "bug"}, "flags": [], "confidence": 0.91}},
    {"instruction": "list issues assigned to me", "context": {"git": {"current_branch": "main", "repo_owner": "startup", "repo_name": "app", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "list", "args": {"assignee": "@me"}, "flags": [], "confidence": 0.95}},
    {"instruction": "show closed issues", "context": {"git": {"current_branch": "main", "repo_owner": "dev", "repo_name": "lib", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "list", "args": {"state": "closed"}, "flags": [], "confidence": 0.94}},
    {"instruction": "find issues with label enhancement", "context": {"git": {"current_branch": "main", "repo_owner": "org", "repo_name": "service", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "list", "args": {"label": "enhancement"}, "flags": [], "confidence": 0.93}},
    {"instruction": "list all issues", "context": {"git": {"current_branch": "main", "repo_owner": "bigco", "repo_name": "backend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "list", "args": {"state": "all"}, "flags": [], "confidence": 0.94}},
    {"instruction": "show issues in browser", "context": {"git": {"current_branch": "main", "repo_owner": "oss", "repo_name": "tool", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "list", "args": {}, "flags": ["web"], "confidence": 0.93}},

    # Issue view with context
    {"instruction": "show issue 42", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "view", "args": {"number": 42}, "flags": [], "confidence": 0.96}},
    {"instruction": "view issue 123", "context": {"git": {"current_branch": "develop", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "view", "args": {"number": 123}, "flags": [], "confidence": 0.96}},
    {"instruction": "open issue 55 in browser", "context": {"git": {"current_branch": "main", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "view", "args": {"number": 55}, "flags": ["web"], "confidence": 0.95}},
    {"instruction": "display issue 77 with comments", "context": {"git": {"current_branch": "main", "repo_owner": "startup", "repo_name": "app", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "view", "args": {"number": 77}, "flags": ["comments"], "confidence": 0.94}},

    # Issue create with context
    {"instruction": "create an issue", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "create", "args": {}, "flags": [], "confidence": 0.94}},
    {"instruction": "file a bug report", "context": {"git": {"current_branch": "main", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "create", "args": {}, "flags": [], "confidence": 0.91}},
    {"instruction": "open new issue", "context": {"git": {"current_branch": "main", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "create", "args": {}, "flags": [], "confidence": 0.94}},
    {"instruction": "create issue in browser", "context": {"git": {"current_branch": "main", "repo_owner": "startup", "repo_name": "app", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "create", "args": {}, "flags": ["web"], "confidence": 0.93}},

    # Issue close with context
    {"instruction": "close issue 42", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "close", "args": {"number": 42}, "flags": [], "confidence": 0.96}},
    {"instruction": "resolve issue 123", "context": {"git": {"current_branch": "main", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "close", "args": {"number": 123}, "flags": [], "confidence": 0.94}},
    {"instruction": "mark issue 55 as done", "context": {"git": {"current_branch": "main", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "close", "args": {"number": 55}, "flags": [], "confidence": 0.93}},

    # Issue reopen with context
    {"instruction": "reopen issue 42", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "reopen", "args": {"number": 42}, "flags": [], "confidence": 0.96}},

    # Issue comment with context
    {"instruction": "comment on issue 42", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "comment", "args": {"number": 42}, "flags": [], "confidence": 0.95}},
    {"instruction": "add comment to issue 123", "context": {"git": {"current_branch": "main", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "issue", "subcommand": "comment", "args": {"number": 123}, "flags": [], "confidence": 0.94}},

    # Repo view with context
    {"instruction": "show this repo", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "repo", "subcommand": "view", "args": {}, "flags": [], "confidence": 0.94}},
    {"instruction": "view repo info", "context": {"git": {"current_branch": "develop", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "repo", "subcommand": "view", "args": {}, "flags": [], "confidence": 0.94}},
    {"instruction": "open repo in browser", "context": {"git": {"current_branch": "main", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "repo", "subcommand": "view", "args": {}, "flags": ["web"], "confidence": 0.95}},
    {"instruction": "show repository details", "context": {"git": {"current_branch": "main", "repo_owner": "startup", "repo_name": "app", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "repo", "subcommand": "view", "args": {}, "flags": [], "confidence": 0.94}},

    # Repo list with context
    {"instruction": "list my repos", "context": {"git": {"current_branch": "main", "repo_owner": "myuser", "repo_name": "project", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "repo", "subcommand": "list", "args": {}, "flags": [], "confidence": 0.94}},
    {"instruction": "show my repositories", "context": {"git": {"current_branch": "main", "repo_owner": "dev", "repo_name": "lib", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "repo", "subcommand": "list", "args": {}, "flags": [], "confidence": 0.94}},
    {"instruction": "list private repos", "context": {"git": {"current_branch": "main", "repo_owner": "user", "repo_name": "stuff", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "repo", "subcommand": "list", "args": {}, "flags": ["private"], "confidence": 0.93}},
    {"instruction": "show public repositories", "context": {"git": {"current_branch": "main", "repo_owner": "user", "repo_name": "stuff", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "repo", "subcommand": "list", "args": {}, "flags": ["public"], "confidence": 0.93}},

    # Repo fork with context
    {"instruction": "fork this repo", "context": {"git": {"current_branch": "main", "repo_owner": "oss", "repo_name": "popular-lib", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "repo", "subcommand": "fork", "args": {}, "flags": [], "confidence": 0.95}},
    {"instruction": "fork and clone", "context": {"git": {"current_branch": "main", "repo_owner": "org", "repo_name": "tool", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "repo", "subcommand": "fork", "args": {}, "flags": ["clone"], "confidence": 0.94}},

    # Run list with context
    {"instruction": "show workflow runs", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "list", "args": {}, "flags": [], "confidence": 0.94}},
    {"instruction": "list ci runs", "context": {"git": {"current_branch": "develop", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "list", "args": {}, "flags": [], "confidence": 0.93}},
    {"instruction": "show failed builds", "context": {"git": {"current_branch": "main", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "list", "args": {"status": "failure"}, "flags": [], "confidence": 0.93}},
    {"instruction": "list successful runs", "context": {"git": {"current_branch": "main", "repo_owner": "startup", "repo_name": "app", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "list", "args": {"status": "success"}, "flags": [], "confidence": 0.93}},
    {"instruction": "show actions", "context": {"git": {"current_branch": "main", "repo_owner": "dev", "repo_name": "lib", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "list", "args": {}, "flags": [], "confidence": 0.92}},
    {"instruction": "list in-progress builds", "context": {"git": {"current_branch": "feature/test", "repo_owner": "org", "repo_name": "service", "current_pr": 42, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "list", "args": {"status": "in_progress"}, "flags": [], "confidence": 0.92}},
    {"instruction": "show runs for this branch", "context": {"git": {"current_branch": "feature/new", "repo_owner": "bigco", "repo_name": "backend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "list", "args": {"branch": "feature/new"}, "flags": [], "confidence": 0.93}},

    # Run view with context
    {"instruction": "view run 12345", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "view", "args": {"run_id": "12345"}, "flags": [], "confidence": 0.95}},
    {"instruction": "show run details 67890", "context": {"git": {"current_branch": "main", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "view", "args": {"run_id": "67890"}, "flags": [], "confidence": 0.94}},
    {"instruction": "view logs for run 11111", "context": {"git": {"current_branch": "main", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "view", "args": {"run_id": "11111"}, "flags": ["log"], "confidence": 0.94}},

    # Run watch with context
    {"instruction": "watch run 12345", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "watch", "args": {"run_id": "12345"}, "flags": [], "confidence": 0.95}},
    {"instruction": "follow build 67890", "context": {"git": {"current_branch": "main", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "watch", "args": {"run_id": "67890"}, "flags": [], "confidence": 0.93}},
    {"instruction": "monitor workflow 11111", "context": {"git": {"current_branch": "main", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "watch", "args": {"run_id": "11111"}, "flags": [], "confidence": 0.92}},

    # Run rerun with context
    {"instruction": "rerun build 12345", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "rerun", "args": {"run_id": "12345"}, "flags": [], "confidence": 0.95}},
    {"instruction": "retry failed jobs in 67890", "context": {"git": {"current_branch": "main", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "rerun", "args": {"run_id": "67890"}, "flags": ["failed"], "confidence": 0.94}},
    {"instruction": "restart workflow 11111", "context": {"git": {"current_branch": "main", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "rerun", "args": {"run_id": "11111"}, "flags": [], "confidence": 0.93}},

    # Run cancel with context
    {"instruction": "cancel run 12345", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "cancel", "args": {"run_id": "12345"}, "flags": [], "confidence": 0.96}},
    {"instruction": "stop build 67890", "context": {"git": {"current_branch": "main", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "cancel", "args": {"run_id": "67890"}, "flags": [], "confidence": 0.94}},
    {"instruction": "abort workflow 11111", "context": {"git": {"current_branch": "main", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "run", "subcommand": "cancel", "args": {"run_id": "11111"}, "flags": [], "confidence": 0.93}},

    # Gist list with context
    {"instruction": "show my gists", "context": {"git": {"current_branch": "main", "repo_owner": "user", "repo_name": "project", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "gist", "subcommand": "list", "args": {}, "flags": [], "confidence": 0.94}},
    {"instruction": "list public gists", "context": {"git": {"current_branch": "main", "repo_owner": "dev", "repo_name": "lib", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "gist", "subcommand": "list", "args": {}, "flags": ["public"], "confidence": 0.93}},
    {"instruction": "show secret gists", "context": {"git": {"current_branch": "main", "repo_owner": "user", "repo_name": "stuff", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "gist", "subcommand": "list", "args": {}, "flags": ["secret"], "confidence": 0.92}},

    # Gist view with context
    {"instruction": "view gist abc123", "context": {"git": {"current_branch": "main", "repo_owner": "user", "repo_name": "project", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "gist", "subcommand": "view", "args": {"id": "abc123"}, "flags": [], "confidence": 0.95}},
    {"instruction": "show gist xyz789 in browser", "context": {"git": {"current_branch": "main", "repo_owner": "dev", "repo_name": "lib", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "gist", "subcommand": "view", "args": {"id": "xyz789"}, "flags": ["web"], "confidence": 0.94}},

    # Gist create with context
    {"instruction": "create a gist", "context": {"git": {"current_branch": "main", "repo_owner": "user", "repo_name": "project", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "gist", "subcommand": "create", "args": {}, "flags": [], "confidence": 0.93}},
    {"instruction": "make a public gist", "context": {"git": {"current_branch": "main", "repo_owner": "dev", "repo_name": "lib", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "gist", "subcommand": "create", "args": {}, "flags": ["public"], "confidence": 0.93}},

    # Release list with context
    {"instruction": "show releases", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "release", "subcommand": "list", "args": {}, "flags": [], "confidence": 0.94}},
    {"instruction": "list versions", "context": {"git": {"current_branch": "main", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "release", "subcommand": "list", "args": {}, "flags": [], "confidence": 0.92}},
    {"instruction": "show releases without drafts", "context": {"git": {"current_branch": "main", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "release", "subcommand": "list", "args": {}, "flags": ["exclude-drafts"], "confidence": 0.91}},

    # Release view with context
    {"instruction": "view release v1.0.0", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "release", "subcommand": "view", "args": {"tag": "v1.0.0"}, "flags": [], "confidence": 0.95}},
    {"instruction": "show latest release", "context": {"git": {"current_branch": "main", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "release", "subcommand": "view", "args": {}, "flags": [], "confidence": 0.92}},
    {"instruction": "open release v2.0 in browser", "context": {"git": {"current_branch": "main", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "release", "subcommand": "view", "args": {"tag": "v2.0"}, "flags": ["web"], "confidence": 0.94}},

    # Release create with context
    {"instruction": "create release v1.0.0", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "release", "subcommand": "create", "args": {"tag": "v1.0.0"}, "flags": [], "confidence": 0.95}},
    {"instruction": "publish release v2.0", "context": {"git": {"current_branch": "main", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "release", "subcommand": "create", "args": {"tag": "v2.0"}, "flags": [], "confidence": 0.94}},
    {"instruction": "create draft release v3.0-beta", "context": {"git": {"current_branch": "main", "repo_owner": "team", "repo_name": "frontend", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "release", "subcommand": "create", "args": {"tag": "v3.0-beta"}, "flags": ["draft"], "confidence": 0.94}},
    {"instruction": "make prerelease v0.9.0", "context": {"git": {"current_branch": "main", "repo_owner": "startup", "repo_name": "app", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "release", "subcommand": "create", "args": {"tag": "v0.9.0"}, "flags": ["prerelease"], "confidence": 0.93}},
    {"instruction": "create release with generated notes", "context": {"git": {"current_branch": "main", "repo_owner": "dev", "repo_name": "lib", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "command", "command": "release", "subcommand": "create", "args": {"tag": "v1.0.0"}, "flags": ["generate-notes"], "confidence": 0.92}},
]

# ============================================================================
# POSITIVE EXAMPLES WITHOUT CONTEXT (20%)
# ============================================================================

POSITIVE_NO_CONTEXT = [
    # PR list without context
    {"instruction": "list open prs", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"state": "open"}, "flags": [], "confidence": 0.95}},
    {"instruction": "show merged pull requests", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"state": "merged"}, "flags": [], "confidence": 0.94}},
    {"instruction": "list prs by @octocat", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"author": "@octocat"}, "flags": [], "confidence": 0.95}},
    {"instruction": "show all prs with label bug", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"state": "all", "label": "bug"}, "flags": [], "confidence": 0.93}},
    {"instruction": "list last 5 prs", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"limit": 5}, "flags": [], "confidence": 0.92}},
    {"instruction": "show prs in browser", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {}, "flags": ["web"], "confidence": 0.93}},
    {"instruction": "list closed pull requests", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"state": "closed"}, "flags": [], "confidence": 0.94}},
    {"instruction": "show prs assigned to @me", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "list", "args": {"assignee": "@me"}, "flags": [], "confidence": 0.93}},

    # PR view without context
    {"instruction": "view pr 42", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "view", "args": {"number": 42}, "flags": [], "confidence": 0.96}},
    {"instruction": "show pr #123", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "view", "args": {"number": 123}, "flags": [], "confidence": 0.95}},
    {"instruction": "open pr 55 in browser", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "view", "args": {"number": 55}, "flags": ["web"], "confidence": 0.94}},
    {"instruction": "display pull request 77 with comments", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "view", "args": {"number": 77}, "flags": ["comments"], "confidence": 0.93}},
    {"instruction": "view pull request number 99", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "view", "args": {"number": 99}, "flags": [], "confidence": 0.95}},

    # PR merge without context
    {"instruction": "merge pr 42", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 42}, "flags": [], "confidence": 0.96}},
    {"instruction": "squash merge pr 123", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 123}, "flags": ["squash"], "confidence": 0.95}},
    {"instruction": "merge pr 55 and delete branch", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 55}, "flags": ["delete-branch"], "confidence": 0.94}},
    {"instruction": "rebase merge pr 77", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 77}, "flags": ["rebase"], "confidence": 0.93}},
    {"instruction": "land pr 99", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "merge", "args": {"number": 99}, "flags": [], "confidence": 0.93}},

    # PR checkout without context
    {"instruction": "checkout pr 42", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "checkout", "args": {"number": 42}, "flags": [], "confidence": 0.96}},
    {"instruction": "switch to pr 123", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "checkout", "args": {"number": 123}, "flags": [], "confidence": 0.94}},
    {"instruction": "co pr 55", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "checkout", "args": {"number": 55}, "flags": [], "confidence": 0.93}},

    # PR close without context
    {"instruction": "close pr 42", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "close", "args": {"number": 42}, "flags": [], "confidence": 0.96}},
    {"instruction": "cancel pr 123", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "close", "args": {"number": 123}, "flags": [], "confidence": 0.93}},

    # PR ready without context
    {"instruction": "mark pr 42 as ready", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "ready", "args": {"number": 42}, "flags": [], "confidence": 0.95}},
    {"instruction": "undraft pr 123", "context": None, "output": {"type": "command", "command": "pr", "subcommand": "ready", "args": {"number": 123}, "flags": [], "confidence": 0.94}},

    # Issue list without context
    {"instruction": "list open issues", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "list", "args": {"state": "open"}, "flags": [], "confidence": 0.95}},
    {"instruction": "show closed issues", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "list", "args": {"state": "closed"}, "flags": [], "confidence": 0.94}},
    {"instruction": "list issues with label bug", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "list", "args": {"label": "bug"}, "flags": [], "confidence": 0.93}},
    {"instruction": "show issues assigned to @me", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "list", "args": {"assignee": "@me"}, "flags": [], "confidence": 0.94}},
    {"instruction": "list all issues", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "list", "args": {"state": "all"}, "flags": [], "confidence": 0.94}},
    {"instruction": "show issues in browser", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "list", "args": {}, "flags": ["web"], "confidence": 0.93}},

    # Issue view without context
    {"instruction": "view issue 42", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "view", "args": {"number": 42}, "flags": [], "confidence": 0.96}},
    {"instruction": "show issue #123", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "view", "args": {"number": 123}, "flags": [], "confidence": 0.95}},
    {"instruction": "open issue 55 in browser", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "view", "args": {"number": 55}, "flags": ["web"], "confidence": 0.94}},
    {"instruction": "display issue 77 with comments", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "view", "args": {"number": 77}, "flags": ["comments"], "confidence": 0.93}},

    # Issue close without context
    {"instruction": "close issue 42", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "close", "args": {"number": 42}, "flags": [], "confidence": 0.96}},
    {"instruction": "resolve issue 123", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "close", "args": {"number": 123}, "flags": [], "confidence": 0.94}},
    {"instruction": "mark issue 55 as done", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "close", "args": {"number": 55}, "flags": [], "confidence": 0.93}},

    # Issue reopen without context
    {"instruction": "reopen issue 42", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "reopen", "args": {"number": 42}, "flags": [], "confidence": 0.96}},

    # Issue comment without context
    {"instruction": "comment on issue 42", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "comment", "args": {"number": 42}, "flags": [], "confidence": 0.95}},
    {"instruction": "add comment to issue 123", "context": None, "output": {"type": "command", "command": "issue", "subcommand": "comment", "args": {"number": 123}, "flags": [], "confidence": 0.94}},

    # Repo clone without context
    {"instruction": "clone facebook/react", "context": None, "output": {"type": "command", "command": "repo", "subcommand": "clone", "args": {"repository": "facebook/react"}, "flags": [], "confidence": 0.96}},
    {"instruction": "download cli/cli", "context": None, "output": {"type": "command", "command": "repo", "subcommand": "clone", "args": {"repository": "cli/cli"}, "flags": [], "confidence": 0.93}},
    {"instruction": "get rust-lang/rust", "context": None, "output": {"type": "command", "command": "repo", "subcommand": "clone", "args": {"repository": "rust-lang/rust"}, "flags": [], "confidence": 0.92}},

    # Repo view without context
    {"instruction": "view repo facebook/react", "context": None, "output": {"type": "command", "command": "repo", "subcommand": "view", "args": {"repository": "facebook/react"}, "flags": [], "confidence": 0.95}},
    {"instruction": "show repo cli/cli in browser", "context": None, "output": {"type": "command", "command": "repo", "subcommand": "view", "args": {"repository": "cli/cli"}, "flags": ["web"], "confidence": 0.94}},

    # Repo list without context
    {"instruction": "list my repos", "context": None, "output": {"type": "command", "command": "repo", "subcommand": "list", "args": {}, "flags": [], "confidence": 0.94}},
    {"instruction": "show repos by @octocat", "context": None, "output": {"type": "command", "command": "repo", "subcommand": "list", "args": {"owner": "@octocat"}, "flags": [], "confidence": 0.93}},
    {"instruction": "list private repositories", "context": None, "output": {"type": "command", "command": "repo", "subcommand": "list", "args": {}, "flags": ["private"], "confidence": 0.93}},
    {"instruction": "show public repos", "context": None, "output": {"type": "command", "command": "repo", "subcommand": "list", "args": {}, "flags": ["public"], "confidence": 0.93}},

    # Repo fork without context
    {"instruction": "fork facebook/react", "context": None, "output": {"type": "command", "command": "repo", "subcommand": "fork", "args": {"repository": "facebook/react"}, "flags": [], "confidence": 0.95}},
    {"instruction": "fork cli/cli and clone", "context": None, "output": {"type": "command", "command": "repo", "subcommand": "fork", "args": {"repository": "cli/cli"}, "flags": ["clone"], "confidence": 0.94}},

    # Repo create without context
    {"instruction": "create repo my-project", "context": None, "output": {"type": "command", "command": "repo", "subcommand": "create", "args": {"name": "my-project"}, "flags": [], "confidence": 0.95}},
    {"instruction": "make new private repo secret-stuff", "context": None, "output": {"type": "command", "command": "repo", "subcommand": "create", "args": {"name": "secret-stuff"}, "flags": ["private"], "confidence": 0.93}},
    {"instruction": "create public repository awesome-tool", "context": None, "output": {"type": "command", "command": "repo", "subcommand": "create", "args": {"name": "awesome-tool"}, "flags": ["public"], "confidence": 0.94}},

    # Run list without context
    {"instruction": "list workflow runs", "context": None, "output": {"type": "command", "command": "run", "subcommand": "list", "args": {}, "flags": [], "confidence": 0.94}},
    {"instruction": "show failed builds", "context": None, "output": {"type": "command", "command": "run", "subcommand": "list", "args": {"status": "failure"}, "flags": [], "confidence": 0.93}},
    {"instruction": "list successful runs", "context": None, "output": {"type": "command", "command": "run", "subcommand": "list", "args": {"status": "success"}, "flags": [], "confidence": 0.93}},
    {"instruction": "show ci runs", "context": None, "output": {"type": "command", "command": "run", "subcommand": "list", "args": {}, "flags": [], "confidence": 0.92}},

    # Run view without context
    {"instruction": "view run 12345", "context": None, "output": {"type": "command", "command": "run", "subcommand": "view", "args": {"run_id": "12345"}, "flags": [], "confidence": 0.95}},
    {"instruction": "show run 67890 logs", "context": None, "output": {"type": "command", "command": "run", "subcommand": "view", "args": {"run_id": "67890"}, "flags": ["log"], "confidence": 0.93}},

    # Run rerun without context
    {"instruction": "rerun build 12345", "context": None, "output": {"type": "command", "command": "run", "subcommand": "rerun", "args": {"run_id": "12345"}, "flags": [], "confidence": 0.95}},
    {"instruction": "retry failed jobs in 67890", "context": None, "output": {"type": "command", "command": "run", "subcommand": "rerun", "args": {"run_id": "67890"}, "flags": ["failed"], "confidence": 0.94}},

    # Run cancel without context
    {"instruction": "cancel run 12345", "context": None, "output": {"type": "command", "command": "run", "subcommand": "cancel", "args": {"run_id": "12345"}, "flags": [], "confidence": 0.96}},
    {"instruction": "stop build 67890", "context": None, "output": {"type": "command", "command": "run", "subcommand": "cancel", "args": {"run_id": "67890"}, "flags": [], "confidence": 0.94}},

    # Gist list without context
    {"instruction": "list my gists", "context": None, "output": {"type": "command", "command": "gist", "subcommand": "list", "args": {}, "flags": [], "confidence": 0.94}},
    {"instruction": "show public gists", "context": None, "output": {"type": "command", "command": "gist", "subcommand": "list", "args": {}, "flags": ["public"], "confidence": 0.93}},

    # Gist view without context
    {"instruction": "view gist abc123", "context": None, "output": {"type": "command", "command": "gist", "subcommand": "view", "args": {"id": "abc123"}, "flags": [], "confidence": 0.95}},
    {"instruction": "show gist xyz789 in browser", "context": None, "output": {"type": "command", "command": "gist", "subcommand": "view", "args": {"id": "xyz789"}, "flags": ["web"], "confidence": 0.94}},

    # Release list without context
    {"instruction": "list releases", "context": None, "output": {"type": "command", "command": "release", "subcommand": "list", "args": {}, "flags": [], "confidence": 0.94}},
    {"instruction": "show versions", "context": None, "output": {"type": "command", "command": "release", "subcommand": "list", "args": {}, "flags": [], "confidence": 0.91}},

    # Release view without context
    {"instruction": "view release v1.0.0", "context": None, "output": {"type": "command", "command": "release", "subcommand": "view", "args": {"tag": "v1.0.0"}, "flags": [], "confidence": 0.95}},
    {"instruction": "show release v2.0 in browser", "context": None, "output": {"type": "command", "command": "release", "subcommand": "view", "args": {"tag": "v2.0"}, "flags": ["web"], "confidence": 0.94}},

    # Release create without context
    {"instruction": "create release v1.0.0", "context": None, "output": {"type": "command", "command": "release", "subcommand": "create", "args": {"tag": "v1.0.0"}, "flags": [], "confidence": 0.95}},
    {"instruction": "publish release v2.0 as draft", "context": None, "output": {"type": "command", "command": "release", "subcommand": "create", "args": {"tag": "v2.0"}, "flags": ["draft"], "confidence": 0.94}},
    {"instruction": "make prerelease v0.9.0-rc1", "context": None, "output": {"type": "command", "command": "release", "subcommand": "create", "args": {"tag": "v0.9.0-rc1"}, "flags": ["prerelease"], "confidence": 0.93}},
]

# ============================================================================
# NEGATIVE/CLARIFICATION EXAMPLES (20%)
# ============================================================================

NEGATIVE_EXAMPLES = [
    # Ambiguous commands
    {"instruction": "show something", "context": None, "output": {"type": "clarification", "message": "What would you like to show?", "suggestions": [{"label": "Pull requests", "command": "pr", "subcommand": "list"}, {"label": "Issues", "command": "issue", "subcommand": "list"}, {"label": "Repositories", "command": "repo", "subcommand": "list"}], "confidence": 0.3}},
    {"instruction": "list", "context": None, "output": {"type": "clarification", "message": "What would you like to list?", "suggestions": [{"label": "Pull requests", "command": "pr", "subcommand": "list"}, {"label": "Issues", "command": "issue", "subcommand": "list"}, {"label": "Repositories", "command": "repo", "subcommand": "list"}, {"label": "Workflow runs", "command": "run", "subcommand": "list"}], "confidence": 0.25}},
    {"instruction": "view", "context": None, "output": {"type": "clarification", "message": "What would you like to view?", "suggestions": [{"label": "A pull request", "command": "pr", "subcommand": "view"}, {"label": "An issue", "command": "issue", "subcommand": "view"}, {"label": "A repository", "command": "repo", "subcommand": "view"}], "confidence": 0.3}},
    {"instruction": "close", "context": None, "output": {"type": "clarification", "message": "What would you like to close?", "suggestions": [{"label": "A pull request", "command": "pr", "subcommand": "close"}, {"label": "An issue", "command": "issue", "subcommand": "close"}], "confidence": 0.3}},
    {"instruction": "merge", "context": None, "output": {"type": "clarification", "message": "Which pull request would you like to merge?", "suggestions": [], "confidence": 0.35}},
    {"instruction": "create", "context": None, "output": {"type": "clarification", "message": "What would you like to create?", "suggestions": [{"label": "A pull request", "command": "pr", "subcommand": "create"}, {"label": "An issue", "command": "issue", "subcommand": "create"}, {"label": "A repository", "command": "repo", "subcommand": "create"}, {"label": "A gist", "command": "gist", "subcommand": "create"}, {"label": "A release", "command": "release", "subcommand": "create"}], "confidence": 0.25}},
    {"instruction": "open", "context": None, "output": {"type": "clarification", "message": "What would you like to open?", "suggestions": [{"label": "A pull request", "command": "pr", "subcommand": "view"}, {"label": "An issue", "command": "issue", "subcommand": "view"}, {"label": "A repository", "command": "repo", "subcommand": "view"}], "confidence": 0.3}},
    {"instruction": "show it", "context": None, "output": {"type": "clarification", "message": "What would you like to show? Please specify a PR number, issue number, or resource.", "suggestions": [], "confidence": 0.2}},
    {"instruction": "display that", "context": None, "output": {"type": "clarification", "message": "What would you like to display? Please be more specific.", "suggestions": [], "confidence": 0.2}},

    # Missing required arguments
    {"instruction": "checkout pr", "context": None, "output": {"type": "clarification", "message": "Which PR number would you like to checkout?", "suggestions": [], "confidence": 0.4}},
    {"instruction": "view issue", "context": None, "output": {"type": "clarification", "message": "Which issue number would you like to view?", "suggestions": [], "confidence": 0.4}},
    {"instruction": "close issue", "context": None, "output": {"type": "clarification", "message": "Which issue number would you like to close?", "suggestions": [], "confidence": 0.4}},
    {"instruction": "reopen issue", "context": None, "output": {"type": "clarification", "message": "Which issue number would you like to reopen?", "suggestions": [], "confidence": 0.4}},
    {"instruction": "clone repo", "context": None, "output": {"type": "clarification", "message": "Which repository would you like to clone? Please specify in owner/repo format.", "suggestions": [], "confidence": 0.4}},
    {"instruction": "cancel run", "context": None, "output": {"type": "clarification", "message": "Which run ID would you like to cancel?", "suggestions": [], "confidence": 0.4}},
    {"instruction": "rerun build", "context": None, "output": {"type": "clarification", "message": "Which run ID would you like to rerun?", "suggestions": [], "confidence": 0.4}},
    {"instruction": "view gist", "context": None, "output": {"type": "clarification", "message": "Which gist ID would you like to view?", "suggestions": [], "confidence": 0.4}},
    {"instruction": "create release", "context": None, "output": {"type": "clarification", "message": "What tag would you like to use for the release?", "suggestions": [], "confidence": 0.4}},
    {"instruction": "comment on issue", "context": None, "output": {"type": "clarification", "message": "Which issue number would you like to comment on?", "suggestions": [], "confidence": 0.4}},

    # Too vague
    {"instruction": "help me with github", "context": None, "output": {"type": "clarification", "message": "I can help with GitHub operations. What would you like to do?", "suggestions": [{"label": "Work with pull requests", "command": "pr", "subcommand": "list"}, {"label": "Work with issues", "command": "issue", "subcommand": "list"}, {"label": "Work with repositories", "command": "repo", "subcommand": "list"}], "confidence": 0.2}},
    {"instruction": "do something", "context": None, "output": {"type": "clarification", "message": "What would you like me to do? I can help with PRs, issues, repos, and more.", "suggestions": [], "confidence": 0.15}},
    {"instruction": "github stuff", "context": None, "output": {"type": "clarification", "message": "What GitHub operation would you like to perform?", "suggestions": [{"label": "Pull requests", "command": "pr", "subcommand": "list"}, {"label": "Issues", "command": "issue", "subcommand": "list"}, {"label": "Repositories", "command": "repo", "subcommand": "list"}], "confidence": 0.2}},
    {"instruction": "work on code", "context": None, "output": {"type": "clarification", "message": "What would you like to do? I can help with pull requests, issues, workflows, and more.", "suggestions": [], "confidence": 0.2}},
    {"instruction": "manage project", "context": None, "output": {"type": "clarification", "message": "What aspect of the project would you like to manage? Issues, PRs, releases?", "suggestions": [], "confidence": 0.2}},

    # Off-topic
    {"instruction": "what's the weather?", "context": None, "output": {"type": "clarification", "message": "I can only help with GitHub operations. Try asking about PRs, issues, or repositories.", "suggestions": [], "confidence": 0.1}},
    {"instruction": "play some music", "context": None, "output": {"type": "clarification", "message": "I can only help with GitHub operations like PRs, issues, and repositories.", "suggestions": [], "confidence": 0.1}},
    {"instruction": "send an email", "context": None, "output": {"type": "clarification", "message": "I can only help with GitHub CLI operations. What GitHub task can I help with?", "suggestions": [], "confidence": 0.1}},
    {"instruction": "book a flight", "context": None, "output": {"type": "clarification", "message": "I can only help with GitHub operations. Would you like to work with PRs, issues, or repos?", "suggestions": [], "confidence": 0.1}},
    {"instruction": "calculate 2+2", "context": None, "output": {"type": "clarification", "message": "I'm a GitHub CLI assistant. I can help with pull requests, issues, and repositories.", "suggestions": [], "confidence": 0.1}},
    {"instruction": "tell me a joke", "context": None, "output": {"type": "clarification", "message": "I'm focused on GitHub operations. What would you like to do with your repository?", "suggestions": [], "confidence": 0.1}},
    {"instruction": "search google", "context": None, "output": {"type": "clarification", "message": "I can only help with GitHub CLI operations. Try asking about PRs, issues, or workflows.", "suggestions": [], "confidence": 0.1}},

    # Context doesn't resolve ambiguity
    {"instruction": "close it", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "clarification", "message": "What would you like to close? There's no current PR on this branch.", "suggestions": [{"label": "Close a PR by number", "command": "pr", "subcommand": "close"}, {"label": "Close an issue by number", "command": "issue", "subcommand": "close"}], "confidence": 0.3}},
    {"instruction": "merge it", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "clarification", "message": "Which PR would you like to merge? There's no current PR on this branch.", "suggestions": [], "confidence": 0.35}},
    {"instruction": "view the thing", "context": {"git": {"current_branch": "develop", "repo_owner": "corp", "repo_name": "api", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "clarification", "message": "What would you like to view? Please specify a PR, issue, or other resource.", "suggestions": [], "confidence": 0.25}},

    # Typos with ambiguity
    {"instruction": "mege pr", "context": None, "output": {"type": "clarification", "message": "Did you mean 'merge pr'? Which PR number?", "suggestions": [], "confidence": 0.4}},
    {"instruction": "creaet issue", "context": None, "output": {"type": "clarification", "message": "Did you mean 'create issue'?", "suggestions": [{"label": "Create an issue", "command": "issue", "subcommand": "create"}], "confidence": 0.45}},
    {"instruction": "chekout pr", "context": None, "output": {"type": "clarification", "message": "Did you mean 'checkout pr'? Which PR number?", "suggestions": [], "confidence": 0.4}},
    {"instruction": "lsit prs", "context": None, "output": {"type": "clarification", "message": "Did you mean 'list prs'?", "suggestions": [{"label": "List pull requests", "command": "pr", "subcommand": "list"}], "confidence": 0.5}},
    {"instruction": "viwe issue", "context": None, "output": {"type": "clarification", "message": "Did you mean 'view issue'? Which issue number?", "suggestions": [], "confidence": 0.45}},

    # Incomplete context references
    {"instruction": "show the pr", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "clarification", "message": "Which PR would you like to view? There's no PR associated with the current branch.", "suggestions": [], "confidence": 0.35}},
    {"instruction": "merge the branch", "context": {"git": {"current_branch": "main", "repo_owner": "acme", "repo_name": "webapp", "current_pr": None, "has_uncommitted_changes": False}}, "output": {"type": "clarification", "message": "There's no PR associated with the current branch. Would you like to create one first?", "suggestions": [{"label": "Create a PR", "command": "pr", "subcommand": "create"}], "confidence": 0.3}},

    # Multiple interpretations
    {"instruction": "open 42", "context": None, "output": {"type": "clarification", "message": "Would you like to open PR #42 or issue #42?", "suggestions": [{"label": "View PR #42", "command": "pr", "subcommand": "view"}, {"label": "View issue #42", "command": "issue", "subcommand": "view"}], "confidence": 0.4}},
    {"instruction": "show 123", "context": None, "output": {"type": "clarification", "message": "Would you like to view PR #123 or issue #123?", "suggestions": [{"label": "View PR #123", "command": "pr", "subcommand": "view"}, {"label": "View issue #123", "command": "issue", "subcommand": "view"}], "confidence": 0.4}},
    {"instruction": "close 55", "context": None, "output": {"type": "clarification", "message": "Would you like to close PR #55 or issue #55?", "suggestions": [{"label": "Close PR #55", "command": "pr", "subcommand": "close"}, {"label": "Close issue #55", "command": "issue", "subcommand": "close"}], "confidence": 0.4}},
]


def format_example(ex):
    """Format an example for JSONL output."""
    return {
        "instruction": ex["instruction"],
        "context": ex["context"],
        "output": json.dumps(ex["output"], ensure_ascii=False),
    }


def main():
    # Target: 3000 examples
    # 60% with context = 1800
    # 20% no context = 600
    # 20% negative = 600

    target_with_context = 1800
    target_no_context = 600
    target_negative = 600

    # Expand examples by creating variations
    all_with_context = []
    all_no_context = []
    all_negative = []

    # Duplicate and vary examples to reach target counts
    while len(all_with_context) < target_with_context:
        for ex in POSITIVE_WITH_CONTEXT:
            if len(all_with_context) >= target_with_context:
                break
            all_with_context.append(ex.copy())

    while len(all_no_context) < target_no_context:
        for ex in POSITIVE_NO_CONTEXT:
            if len(all_no_context) >= target_no_context:
                break
            all_no_context.append(ex.copy())

    while len(all_negative) < target_negative:
        for ex in NEGATIVE_EXAMPLES:
            if len(all_negative) >= target_negative:
                break
            all_negative.append(ex.copy())

    # Trim to exact counts
    all_with_context = all_with_context[:target_with_context]
    all_no_context = all_no_context[:target_no_context]
    all_negative = all_negative[:target_negative]

    # Combine and format
    all_examples = []
    for ex in all_with_context:
        all_examples.append(format_example(ex))
    for ex in all_no_context:
        all_examples.append(format_example(ex))
    for ex in all_negative:
        all_examples.append(format_example(ex))

    # Shuffle
    random.seed(42)
    random.shuffle(all_examples)

    # Split 90/10 for train/val
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    # Write files
    output_dir = Path(__file__).parent

    with open(output_dir / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(output_dir / "train_val.jsonl", "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(train_examples)} training examples -> train.jsonl")
    print(f"Generated {len(val_examples)} validation examples -> train_val.jsonl")
    print(f"Total: {len(all_examples)} examples")
    print(f"Distribution: {len(all_with_context)} with context, {len(all_no_context)} no context, {len(all_negative)} negative")


if __name__ == "__main__":
    main()
