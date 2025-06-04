# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains tools for working with Claude and the Model Context Protocol (MCP):
- **claude-code/**: Official Anthropic Claude Code CLI (installed separately via npm)
- **claude_monitor.py**: Real-time terminal-based monitor for Claude Code sessions
- **claude_costs.py**: Comprehensive cost analysis tool for Claude Code usage
- **mcp-demo/**: Model Context Protocol demonstration and examples

## Common Commands

### Running the monitoring tools
```bash
# Monitor Claude Code sessions in real-time
python claude_monitor.py [--session SESSION_ID] [--timezone TIMEZONE]

# Analyze Claude Code costs
python claude_costs.py [--period today|yesterday|week|month|all] [--sessions N] [--sort cost|date|duration]

# Export cost data
python claude_costs.py --export csv|json [--output filename]
```

### MCP Demo
```bash
cd mcp-demo/
pip install -r requirements.txt  # Install mcp>=1.0.0

# Run the complete demo (server + client)
python run_working_demo.py

# Or run components separately
python working_example.py  # Start MCP server
python demo_client.py      # Run client (in another terminal)
```

### Adding MCP Servers to Claude Code
```bash
# List configured MCP servers
claude mcp list

# Add a stdio MCP server (local scope by default)
claude mcp add <name> -- python /path/to/server.py

# Add with environment variables
claude mcp add my-server -e API_KEY=123 -- python working_example.py

# Add with different scopes
claude mcp add shared-server -s project -- python server.py  # Shared with team
claude mcp add my-server -s user -- python server.py         # Available across projects

# Add an SSE server
claude mcp add --transport sse <name> <url>

# Get server details
claude mcp get my-server

# Remove a server
claude mcp remove my-server
```

## Architecture Notes

### Claude Cost/Monitor Tools
Both tools read from Claude Code log files located at `~/.claude/projects/*/logs/*.jsonl`. They parse the JSON Lines format to extract:
- Message exchanges between user and assistant
- Token usage (input_tokens, cache_creation_input_tokens, cache_read_input_tokens, output_tokens)
- Tool usage and associated costs
- Session timing and metadata

Key implementation details:
- Token costs are calculated based on Claude's pricing model (built-in pricing for claude-sonnet-4-20250514, claude-opus-4-20250514, and legacy models)
- Cache tokens are tracked separately as they have different pricing
- Sessions are identified by UUID in the log filenames
- Local timezone support for accurate time display
- Backward compatibility: handles both legacy log format (with costUSD field) and newer format (calculates costs from token usage)
- Auto-detects Claude Code version changes and adapts cost calculation accordingly

### MCP Demo Structure
The MCP demo demonstrates the Model Context Protocol with:
- **Server** (`working_example.py`): Exposes tools via MCP protocol
- **Client** (`demo_client.py`): Connects to server and calls tools
- **Transport**: Uses stdio for inter-process communication

Available demo tools:
- `get_time`: Returns current time
- `calculate`: Performs arithmetic calculations
- `roll_dice`: Simulates dice rolls
- `get_weather`: Returns mock weather data

## Development Notes

When modifying the cost analyzer or monitor:
- Token tracking includes separate categories for input, cache creation, cache read, and output tokens
- Session display truncates long filenames to maintain table alignment
- All timestamps are converted to local timezone for display
- Cost calculations use Decimal for precision

When working with MCP:
- Servers must implement at least `list_tools()` and `call_tool()` handlers
- The initialization handshake is required before any tool calls
- Tool schemas follow JSON Schema format for validation

### MCP Server Scopes in Claude Code:
- **Local scope** (default): Project-specific, stored in project user settings
- **Project scope** (`-s project`): Shared with team via `.mcp.json` file (check into version control)
- **User scope** (`-s user`): Available across all projects for the user

Example: Add the demo server to Claude Code:
```bash
claude mcp add demo-server -- python /Users/tijs/projects/claude/mcp-demo/working_example.py
```