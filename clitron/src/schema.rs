//! Command schema definitions.
//!
//! Schemas define the structure of CLI commands that clitron can interpret.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{ClitronError, Result};

/// Represents a complete CLI command schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandSchema {
    /// Name of the CLI (e.g., "gh").
    pub cli_name: String,

    /// Description of the CLI.
    pub description: String,

    /// Top-level commands.
    pub commands: Vec<Command>,
}

impl CommandSchema {
    /// Load schema from a YAML string.
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(yaml).map_err(ClitronError::from)
    }

    /// Load schema from a YAML file.
    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_yaml(&content)
    }

    /// Find a command by name.
    pub fn find_command(&self, name: &str) -> Option<&Command> {
        self.commands.iter().find(|c| {
            c.name == name
                || c.aliases
                    .as_ref()
                    .map_or(false, |a| a.contains(&name.to_string()))
        })
    }

    /// Generate a summary for model context.
    pub fn to_summary(&self) -> String {
        let mut summary = format!(
            "CLI: {} - {}\n\nCommands:\n",
            self.cli_name, self.description
        );

        for cmd in &self.commands {
            summary.push_str(&format!("  {} - {}\n", cmd.name, cmd.description));
            for sub in &cmd.subcommands {
                summary.push_str(&format!("    {} - {}\n", sub.name, sub.description));
            }
        }

        summary
    }
}

/// A top-level command (e.g., "pr", "issue").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Command {
    /// Command name.
    pub name: String,

    /// Description.
    pub description: String,

    /// Alternative names/aliases.
    #[serde(default)]
    pub aliases: Option<Vec<String>>,

    /// Subcommands.
    #[serde(default)]
    pub subcommands: Vec<Subcommand>,

    /// Direct arguments (if no subcommand).
    #[serde(default)]
    pub args: Vec<Argument>,

    /// Direct flags (if no subcommand).
    #[serde(default)]
    pub flags: Vec<Flag>,
}

impl Command {
    /// Find a subcommand by name.
    pub fn find_subcommand(&self, name: &str) -> Option<&Subcommand> {
        self.subcommands.iter().find(|s| {
            s.name == name
                || s.aliases
                    .as_ref()
                    .map_or(false, |a| a.contains(&name.to_string()))
        })
    }
}

/// A subcommand (e.g., "list", "view", "create").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subcommand {
    /// Subcommand name.
    pub name: String,

    /// Description.
    pub description: String,

    /// Alternative names/aliases.
    #[serde(default)]
    pub aliases: Option<Vec<String>>,

    /// Arguments.
    #[serde(default)]
    pub args: Vec<Argument>,

    /// Flags.
    #[serde(default)]
    pub flags: Vec<Flag>,
}

/// An argument definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Argument {
    /// Argument name.
    pub name: String,

    /// Argument type.
    #[serde(rename = "type")]
    pub arg_type: ArgType,

    /// Whether this argument is required.
    #[serde(default)]
    pub required: bool,

    /// Default value.
    #[serde(default)]
    pub default: Option<serde_json::Value>,

    /// Description.
    #[serde(default)]
    pub description: String,

    /// Natural language mappings for this argument.
    #[serde(default)]
    pub natural_language: Option<Vec<NaturalLanguageMapping>>,
}

/// A flag definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flag {
    /// Flag name (long form).
    pub name: String,

    /// Short form (single character).
    #[serde(default)]
    pub short: Option<char>,

    /// Description.
    #[serde(default)]
    pub description: String,

    /// Natural language triggers.
    #[serde(default)]
    pub natural_language: Option<Vec<String>>,
}

/// Argument types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ArgType {
    /// Enumeration with fixed values (must be first for untagged to work).
    Enum(EnumType),

    /// Simple type specified as string.
    Simple(SimpleArgType),
}

/// Simple argument types (string, integer, etc.).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SimpleArgType {
    /// String argument.
    String,
    /// Integer argument.
    Integer,
    /// Floating point argument.
    Float,
    /// Boolean argument.
    Boolean,
    /// File path argument.
    Path,
}

/// Enum type wrapper for YAML format: `type: enum: [values]`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnumType {
    /// The allowed enum values.
    #[serde(rename = "enum")]
    pub values: Vec<String>,
}

impl ArgType {
    /// Check if this is a string type.
    pub fn is_string(&self) -> bool {
        matches!(self, ArgType::Simple(SimpleArgType::String))
    }

    /// Get enum values if this is an enum type.
    pub fn enum_values(&self) -> Option<&[String]> {
        match self {
            ArgType::Enum(e) => Some(&e.values),
            _ => None,
        }
    }
}

impl std::fmt::Display for ArgType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArgType::Simple(s) => write!(f, "{}", s),
            ArgType::Enum(e) => write!(f, "enum({})", e.values.join("|")),
        }
    }
}

impl std::fmt::Display for SimpleArgType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimpleArgType::String => write!(f, "string"),
            SimpleArgType::Integer => write!(f, "integer"),
            SimpleArgType::Float => write!(f, "float"),
            SimpleArgType::Boolean => write!(f, "boolean"),
            SimpleArgType::Path => write!(f, "path"),
        }
    }
}

/// Natural language mapping for argument values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NaturalLanguageMapping {
    /// Simple value mapping: "open" -> ["open", "active", "pending"]
    ValueMapping {
        /// The canonical value.
        value: String,
        /// Natural language alternatives.
        alternatives: Vec<String>,
    },

    /// Pattern-based mapping: "last (\\d+)" captures a number
    Pattern {
        /// Regex pattern.
        pattern: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SCHEMA: &str = r#"
cli_name: "test"
description: "Test CLI"
commands:
  - name: "cmd"
    description: "A command"
    subcommands:
      - name: "sub"
        description: "A subcommand"
        args:
          - name: "arg1"
            type: string
            required: false
        flags:
          - name: "flag1"
            short: "f"
"#;

    #[test]
    fn test_parse_schema() {
        let schema = CommandSchema::from_yaml(TEST_SCHEMA).unwrap();
        assert_eq!(schema.cli_name, "test");
        assert_eq!(schema.commands.len(), 1);
        assert_eq!(schema.commands[0].subcommands.len(), 1);
    }

    #[test]
    fn test_find_command() {
        let schema = CommandSchema::from_yaml(TEST_SCHEMA).unwrap();
        assert!(schema.find_command("cmd").is_some());
        assert!(schema.find_command("unknown").is_none());
    }
}
