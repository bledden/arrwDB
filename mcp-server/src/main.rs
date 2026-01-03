//! arrwDB MCP Server
//!
//! Model Context Protocol server for arrwDB vector database.
//! Exposes arrwDB's search and management capabilities to AI assistants.

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod client;
mod server;
mod tools;

use server::ArrwDBServer;

/// arrwDB MCP Server - Connect AI assistants to arrwDB vector database
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// arrwDB server URL
    #[arg(short, long, env = "ARRWDB_URL", default_value = "http://localhost:8000")]
    url: String,

    /// API key for authentication (optional)
    #[arg(short = 'k', long, env = "ARRWDB_API_KEY")]
    api_key: Option<String>,

    /// Enable debug logging
    #[arg(short, long)]
    debug: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.debug { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| format!("arrwdb_mcp={}", log_level).into()))
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .init();

    tracing::info!("Starting arrwDB MCP server");
    tracing::info!("Connecting to arrwDB at: {}", args.url);

    // Create the MCP server
    let server = ArrwDBServer::new(&args.url, args.api_key)?;

    // Run the server using stdio transport
    server.run_stdio().await?;

    Ok(())
}
