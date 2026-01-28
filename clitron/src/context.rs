//! Context module for environmental information.
//!
//! Provides context about the current environment (git state, etc.)
//! to improve interpretation accuracy.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;

use crate::error::Result;

/// Environmental context for interpretation.
///
/// Context provides information about the current environment that helps
/// the model make better interpretations. For example, knowing the current
/// branch and PR number allows "merge this" to be interpreted correctly.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Context {
    /// Git repository context.
    pub git: Option<GitContext>,
    /// Custom key-value pairs for CLI-specific context.
    pub custom: HashMap<String, String>,
}

/// Git repository state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitContext {
    /// Current branch name.
    pub current_branch: Option<String>,
    /// Whether there are uncommitted changes.
    pub has_uncommitted_changes: bool,
    /// Whether there are staged changes.
    pub has_staged_changes: bool,
    /// Whether we're in a worktree.
    pub is_worktree: bool,
    /// Repository owner (from remote URL).
    pub repo_owner: Option<String>,
    /// Repository name (from remote URL).
    pub repo_name: Option<String>,
    /// PR number associated with current branch (if any).
    pub current_pr: Option<u64>,
    /// Upstream tracking branch.
    pub upstream_branch: Option<String>,
}

impl Context {
    /// Create empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create context with git information (auto-detected).
    pub fn with_git() -> Result<Self> {
        Ok(Self {
            git: GitContext::detect()?,
            custom: HashMap::new(),
        })
    }

    /// Add custom context value.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.custom.insert(key.into(), value.into());
        self
    }

    /// Get custom context value.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.custom.get(key).map(|s| s.as_str())
    }

    /// Format context for model prompt.
    pub fn to_prompt_string(&self) -> String {
        let mut parts = Vec::new();

        if let Some(ref git) = self.git {
            if let (Some(owner), Some(name)) = (&git.repo_owner, &git.repo_name) {
                parts.push(format!("repo: {}/{}", owner, name));
            }
            if let Some(ref branch) = git.current_branch {
                parts.push(format!("branch: {}", branch));
            }
            if let Some(pr) = git.current_pr {
                parts.push(format!("pr: #{}", pr));
            }
            parts.push(format!("uncommitted: {}", git.has_uncommitted_changes));
            parts.push(format!("staged: {}", git.has_staged_changes));
        }

        for (key, value) in &self.custom {
            parts.push(format!("{}: {}", key, value));
        }

        parts.join("\n")
    }
}

impl GitContext {
    /// Detect git context from current directory.
    pub fn detect() -> Result<Option<Self>> {
        // Check if we're in a git repo
        let is_repo = Command::new("git")
            .args(["rev-parse", "--git-dir"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        if !is_repo {
            return Ok(None);
        }

        // Get current branch
        let current_branch = Command::new("git")
            .args(["branch", "--show-current"])
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
                } else {
                    None
                }
            })
            .filter(|s| !s.is_empty());

        // Check for uncommitted changes
        let has_uncommitted_changes = Command::new("git")
            .args(["status", "--porcelain"])
            .output()
            .map(|o| !o.stdout.is_empty())
            .unwrap_or(false);

        // Check for staged changes
        let has_staged_changes = Command::new("git")
            .args(["diff", "--cached", "--quiet"])
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(false);

        // Check if worktree
        let is_worktree = Command::new("git")
            .args(["rev-parse", "--git-dir"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.contains(".git/worktrees"))
            .unwrap_or(false);

        // Parse remote URL for owner/repo
        let (repo_owner, repo_name) = Command::new("git")
            .args(["remote", "get-url", "origin"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|url| parse_github_url(&url))
            .unwrap_or((None, None));

        // Get upstream branch
        let upstream_branch = Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "@{upstream}"])
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
                } else {
                    None
                }
            });

        Ok(Some(Self {
            current_branch,
            has_uncommitted_changes,
            has_staged_changes,
            is_worktree,
            repo_owner,
            repo_name,
            current_pr: None, // Requires GitHub API
            upstream_branch,
        }))
    }

    /// Detect with PR lookup via GitHub API.
    ///
    /// Requires `gh` CLI to be installed and authenticated.
    pub fn detect_with_pr() -> Result<Option<Self>> {
        let mut context = Self::detect()?;

        if let Some(ref mut ctx) = context {
            // Try to get current PR using gh CLI
            if ctx.current_branch.is_some() {
                ctx.current_pr = Command::new("gh")
                    .args(["pr", "view", "--json", "number", "-q", ".number"])
                    .output()
                    .ok()
                    .and_then(|o| {
                        if o.status.success() {
                            String::from_utf8(o.stdout)
                                .ok()
                                .and_then(|s| s.trim().parse().ok())
                        } else {
                            None
                        }
                    });
            }
        }

        Ok(context)
    }
}

/// Parse GitHub URL to extract owner and repo name.
fn parse_github_url(url: &str) -> Option<(Option<String>, Option<String>)> {
    let url = url.trim();

    // Handle SSH format: git@github.com:owner/repo.git
    if url.starts_with("git@github.com:") {
        let path = url.strip_prefix("git@github.com:")?;
        let path = path.strip_suffix(".git").unwrap_or(path);
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() >= 2 {
            return Some((Some(parts[0].to_string()), Some(parts[1].to_string())));
        }
    }

    // Handle HTTPS format: https://github.com/owner/repo.git
    if url.contains("github.com") {
        let parts: Vec<&str> = url.split('/').collect();
        if parts.len() >= 2 {
            let repo = parts.last()?.strip_suffix(".git").unwrap_or(parts.last()?);
            let owner = parts.get(parts.len() - 2)?;
            return Some((Some(owner.to_string()), Some(repo.to_string())));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_github_ssh_url() {
        let (owner, repo) = parse_github_url("git@github.com:acme/webapp.git").unwrap();
        assert_eq!(owner, Some("acme".to_string()));
        assert_eq!(repo, Some("webapp".to_string()));
    }

    #[test]
    fn test_parse_github_https_url() {
        let (owner, repo) = parse_github_url("https://github.com/acme/webapp.git").unwrap();
        assert_eq!(owner, Some("acme".to_string()));
        assert_eq!(repo, Some("webapp".to_string()));
    }

    #[test]
    fn test_context_to_prompt_string() {
        let mut ctx = Context::new();
        ctx.git = Some(GitContext {
            current_branch: Some("feature/login".to_string()),
            has_uncommitted_changes: true,
            has_staged_changes: false,
            is_worktree: false,
            repo_owner: Some("acme".to_string()),
            repo_name: Some("webapp".to_string()),
            current_pr: Some(42),
            upstream_branch: None,
        });

        let prompt = ctx.to_prompt_string();
        assert!(prompt.contains("repo: acme/webapp"));
        assert!(prompt.contains("branch: feature/login"));
        assert!(prompt.contains("pr: #42"));
        assert!(prompt.contains("uncommitted: true"));
    }
}
