//! Interpreted command representation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The result of interpreting a natural language command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretedCommand {
    /// The main command (e.g., "pr", "issue").
    pub command: String,

    /// The subcommand if any (e.g., "list", "view").
    #[serde(default)]
    pub subcommand: Option<String>,

    /// Parsed arguments as key-value pairs.
    #[serde(default)]
    pub args: HashMap<String, serde_json::Value>,

    /// Flags that were set.
    #[serde(default)]
    pub flags: Vec<String>,

    /// Confidence score (0.0 to 1.0).
    #[serde(default = "default_confidence")]
    pub confidence: f32,

    /// Raw model output for debugging.
    #[serde(default, skip_serializing)]
    pub raw_output: String,
}

fn default_confidence() -> f32 {
    1.0
}

impl InterpretedCommand {
    /// Create a new interpreted command.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            subcommand: None,
            args: HashMap::new(),
            flags: Vec::new(),
            confidence: 1.0,
            raw_output: String::new(),
        }
    }

    /// Add a subcommand.
    pub fn with_subcommand(mut self, subcommand: impl Into<String>) -> Self {
        self.subcommand = Some(subcommand.into());
        self
    }

    /// Add an argument.
    pub fn with_arg(
        mut self,
        name: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.args.insert(name.into(), value.into());
        self
    }

    /// Add a flag.
    pub fn with_flag(mut self, flag: impl Into<String>) -> Self {
        self.flags.push(flag.into());
        self
    }

    /// Set confidence.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Check if interpretation confidence is above threshold.
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Get an argument value as a specific type.
    pub fn get_arg<T: serde::de::DeserializeOwned>(&self, name: &str) -> Option<T> {
        self.args
            .get(name)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Get an argument as a string.
    pub fn get_arg_str(&self, name: &str) -> Option<&str> {
        self.args.get(name).and_then(|v| v.as_str())
    }

    /// Check if a flag is set.
    pub fn has_flag(&self, name: &str) -> bool {
        self.flags.iter().any(|f| f == name)
    }

    /// Convert to clap-style argument vector.
    ///
    /// Returns arguments in the format: `["command", "subcommand", "--arg", "value", "--flag"]`
    pub fn to_args(&self) -> Vec<String> {
        let mut result = Vec::new();

        // Add command
        result.push(self.command.clone());

        // Add subcommand
        if let Some(ref sub) = self.subcommand {
            result.push(sub.clone());
        }

        // Add arguments
        for (name, value) in &self.args {
            result.push(format!("--{}", name));

            // Convert value to string representation
            match value {
                serde_json::Value::String(s) => result.push(s.clone()),
                serde_json::Value::Number(n) => result.push(n.to_string()),
                serde_json::Value::Bool(b) => result.push(b.to_string()),
                serde_json::Value::Null => {}
                _ => result.push(value.to_string()),
            }
        }

        // Add flags
        for flag in &self.flags {
            result.push(format!("--{}", flag));
        }

        result
    }

    /// Convert to a shell command string.
    pub fn to_shell_command(&self, cli_name: &str) -> String {
        let args = self.to_args();
        format!("{} {}", cli_name, args.join(" "))
    }
}

impl Default for InterpretedCommand {
    fn default() -> Self {
        Self {
            command: String::new(),
            subcommand: None,
            args: HashMap::new(),
            flags: Vec::new(),
            confidence: 0.0,
            raw_output: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_pattern() {
        let cmd = InterpretedCommand::new("pr")
            .with_subcommand("list")
            .with_arg("state", "open")
            .with_arg("author", "@me")
            .with_flag("web")
            .with_confidence(0.95);

        assert_eq!(cmd.command, "pr");
        assert_eq!(cmd.subcommand, Some("list".into()));
        assert_eq!(cmd.get_arg_str("state"), Some("open"));
        assert!(cmd.has_flag("web"));
        assert!(cmd.is_confident(0.9));
    }

    #[test]
    fn test_to_args() {
        let cmd = InterpretedCommand::new("pr")
            .with_subcommand("list")
            .with_arg("state", "open")
            .with_flag("web");

        let args = cmd.to_args();

        assert!(args.contains(&"pr".to_string()));
        assert!(args.contains(&"list".to_string()));
        assert!(args.contains(&"--state".to_string()));
        assert!(args.contains(&"open".to_string()));
        assert!(args.contains(&"--web".to_string()));
    }

    #[test]
    fn test_to_shell_command() {
        let cmd = InterpretedCommand::new("pr")
            .with_subcommand("list")
            .with_arg("state", "open");

        let shell = cmd.to_shell_command("gh");
        assert!(shell.starts_with("gh pr list"));
        assert!(shell.contains("--state"));
        assert!(shell.contains("open"));
    }

    #[test]
    fn test_confidence() {
        let cmd = InterpretedCommand::new("pr").with_confidence(0.75);

        assert!(cmd.is_confident(0.7));
        assert!(!cmd.is_confident(0.8));
    }
}
