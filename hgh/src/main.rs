//! hgh - Human GitHub CLI
//!
//! A natural language wrapper around the GitHub CLI (`gh`) powered by clitron.
//!
//! # Examples
//!
//! ```bash
//! hgh "show my open pull requests"
//! hgh "list merged prs from last week"
//! hgh "create a pr for this branch"
//! ```

use std::io::{self, BufRead, Write};
use std::process::{Command, ExitCode};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use clitron::{
    progress::format_bytes, CommandSchema, InterpretedCommand, Interpreter, ModelManager,
    ModelStatus, TerminalOutput,
};

/// Human-friendly GitHub CLI
#[derive(Parser)]
#[command(name = "hgh")]
#[command(about = "Human-friendly GitHub CLI powered by clitron")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    subcommand: Option<Commands>,

    /// Natural language command (or pass via stdin)
    #[arg(trailing_var_arg = true)]
    command: Vec<String>,

    /// Show interpreted command before running
    #[arg(short, long)]
    explain: bool,

    /// Bypass interpretation, pass args directly to gh
    #[arg(short, long)]
    raw: bool,

    /// Don't execute, just show the interpreted command
    #[arg(short = 'n', long)]
    dry_run: bool,

    /// Always ask for confirmation before executing
    #[arg(short = 'c', long)]
    confirm: bool,

    /// Confidence threshold (0.0-1.0) for automatic execution
    #[arg(short = 't', long, default_value = "0.7")]
    threshold: f32,
}

#[derive(Subcommand)]
enum Commands {
    /// Download the language model
    #[command(name = "model-download")]
    ModelDownload,

    /// Show model status
    #[command(name = "model-status")]
    ModelStatus,
}

fn main() -> ExitCode {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::WARN.into()),
        )
        .init();

    let cli = Cli::parse();

    match run(cli) {
        Ok(code) => code,
        Err(e) => {
            let output = TerminalOutput::new();
            output.error(&format!("{e:#}"));
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    // Handle subcommands first
    if let Some(cmd) = &cli.subcommand {
        return match cmd {
            Commands::ModelDownload => model_download(),
            Commands::ModelStatus => model_status(),
        };
    }

    // Get input
    let input = if cli.command.is_empty() {
        // Read from stdin if no args
        read_stdin()?
    } else {
        cli.command.join(" ")
    };

    let input = input.trim();

    if input.is_empty() {
        print_usage();
        return Ok(ExitCode::SUCCESS);
    }

    // Raw mode: pass directly to gh
    if cli.raw {
        return execute_gh(&cli.command);
    }

    // Load interpreter
    let interpreter = load_interpreter().context("Failed to load interpreter")?;

    // Interpret the input
    let result = interpreter.interpret_lenient(input);

    match result {
        Ok(cmd) => handle_interpretation(&cli, &cmd, input),
        Err(e) => {
            let output = TerminalOutput::new();
            output.error(&format!("Could not interpret: {e}"));
            eprintln!();
            eprintln!("Try using traditional gh syntax:");
            eprintln!("  hgh --raw pr list");
            Ok(ExitCode::FAILURE)
        }
    }
}

fn model_download() -> Result<ExitCode> {
    let output = TerminalOutput::new();
    let manager = ModelManager::new();

    // Check if already downloaded
    if let ModelStatus::Downloaded { path, size } = manager.model_status() {
        output.info("Model already downloaded");
        output.dim(&format!("Location: {}", path.display()));
        output.dim(&format!("Size: {}", format_bytes(size)));
        return Ok(ExitCode::SUCCESS);
    }

    // Download with progress
    match manager.ensure_default_model() {
        Ok(_) => {
            output.success("Model ready for use!");
            Ok(ExitCode::SUCCESS)
        }
        Err(e) => {
            output.error(&format!("Failed to download model: {e}"));
            Ok(ExitCode::FAILURE)
        }
    }
}

fn model_status() -> Result<ExitCode> {
    let output = TerminalOutput::new();
    let manager = ModelManager::new();
    let status = manager.model_status();

    match status {
        ModelStatus::Downloaded { path, size } => {
            output.success("Model downloaded");
            output.dim(&format!("Location: {}", path.display()));
            output.dim(&format!("Size: {}", format_bytes(size)));
        }
        ModelStatus::NotDownloaded { expected_path } => {
            output.warning("Model not downloaded");
            output.dim(&format!("Expected location: {}", expected_path.display()));
            eprintln!();
            output.info("Run 'hgh model-download' to download the model.");
            output.info(
                "Or just run 'hgh' with a command - the model will be downloaded automatically.",
            );
        }
    }

    Ok(ExitCode::SUCCESS)
}

fn handle_interpretation(cli: &Cli, cmd: &InterpretedCommand, input: &str) -> Result<ExitCode> {
    let gh_args = cmd.to_args();
    let gh_command = format!("gh {}", gh_args.join(" "));

    // Dry run: just show the command
    if cli.dry_run {
        println!("{gh_command}");
        return Ok(ExitCode::SUCCESS);
    }

    // Explain mode: show interpretation
    if cli.explain {
        println!("Input: {input}");
        println!("Interpreted as: {gh_command}");
        println!("Confidence: {:.0}%", cmd.confidence * 100.0);
        println!();
    }

    // Check confidence
    let needs_confirmation = cli.confirm || !cmd.is_confident(cli.threshold);

    if needs_confirmation {
        if !cli.explain {
            println!("Interpreted as: {gh_command}");
            println!("Confidence: {:.0}%", cmd.confidence * 100.0);
        }
        println!();

        if !confirm("Execute this command?")? {
            println!("Cancelled.");
            return Ok(ExitCode::SUCCESS);
        }
    }

    // Execute
    execute_gh(&gh_args)
}

fn execute_gh(args: &[String]) -> Result<ExitCode> {
    let status = Command::new("gh")
        .args(args)
        .status()
        .context("Failed to execute gh. Is it installed?")?;

    Ok(if status.success() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(status.code().unwrap_or(1) as u8)
    })
}

fn load_interpreter() -> Result<Interpreter> {
    // Load embedded schema
    let schema_yaml = include_str!("../../schemas/gh.yaml");
    let schema = CommandSchema::from_yaml(schema_yaml).context("Failed to parse schema")?;

    Interpreter::with_schema(schema).context("Failed to create interpreter")
}

fn read_stdin() -> Result<String> {
    let stdin = io::stdin();
    let mut input = String::new();

    // Check if stdin has data (not a terminal)
    if !stdin_is_terminal() {
        for line in stdin.lock().lines() {
            input.push_str(&line?);
            input.push(' ');
        }
    }

    Ok(input)
}

fn confirm(prompt: &str) -> Result<bool> {
    print!("{prompt} [y/N] ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    Ok(input.trim().eq_ignore_ascii_case("y"))
}

fn print_usage() {
    println!(
        r#"hgh - Human GitHub CLI

Usage: hgh <natural language command>

Examples:
  hgh "show my open prs"
  hgh "list merged pull requests"
  hgh "create a pr for this branch"
  hgh "view pr 123"
  hgh "merge my pr"
  hgh "show issues assigned to me"
  hgh "list failed workflow runs"

Options:
  -e, --explain     Show interpreted command before running
  -n, --dry-run     Don't execute, just show the interpreted command
  -c, --confirm     Always ask for confirmation
  -t, --threshold   Confidence threshold for auto-execution (default: 0.7)
  -r, --raw         Bypass interpretation, pass args to gh directly

Model Commands:
  model-download    Download the language model
  model-status      Show model cache status

For traditional gh usage:
  hgh --raw pr list --state open

Requires gh CLI: https://cli.github.com/
"#
    );
}

/// Check if stdin is a terminal (not piped input)
fn stdin_is_terminal() -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        unsafe { libc::isatty(std::io::stdin().as_raw_fd()) != 0 }
    }
    #[cfg(not(unix))]
    {
        // Default to true on non-Unix platforms
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_parsing() {
        // Just check schema parsing works
        let schema_yaml = include_str!("../../schemas/gh.yaml");
        let schema = CommandSchema::from_yaml(schema_yaml);
        assert!(schema.is_ok());
    }
}
