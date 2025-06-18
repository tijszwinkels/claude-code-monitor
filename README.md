# Claude Code Monitoring Tools

A collection of Python utilities for monitoring and analyzing [Claude Code](https://claude.ai/code) usage, costs, and session activity.

## Quick Start

Get instant insights into your Claude Code usage:

```bash
# Real-time monitoring (hint: try with `watch`)
python claude_monitor.py

# Cost analysis for the past week
python claude_costs.py --period week

# Track Anthropic sessions (50/month limit)
python claude_costs.py --anthropic-sessions --billing-day 25
```

**Sample Output:**
```
================================================================================
CLAUDE CODE COST ANALYSIS SUMMARY
================================================================================

Total Messages: 3,681
Total Sessions: 44
Date Range: 2025-05-25 00:35 - 2025-05-28 11:24

Token Usage:                   Input:          16,391
                               Cache Creation: 6,118,276
                               Cache Read:     68,679,549
                               Output:         585,894
                               Total:          75,400,110

Total Cost:                    $149.45
Average Cost per Message:      $0.04
Average Cost per Session:      $3.40

--------------------------------------------------------------------------------
COST BY MODEL
--------------------------------------------------------------------------------
claude-opus-4-20250514                     $129.30 (1,054 messages)
claude-sonnet-4-20250514                    $20.15 (654 messages)

--------------------------------------------------------------------
TOP 5 MOST EXPENSIVE SESSIONS
--------------------------------------------------------------------
Session                    Date      Start   End      Cost  Messages
my-project/session-abc123  2025-05-26  22:02  10:53  $23.47      244
another-project/xyz789     2025-05-25  23:07  23:31  $10.11      189
data-analysis/session-456  2025-05-25  23:50  00:21   $9.13      177
```

**Daily Breakdown:**
```bash
python claude_costs.py --days 3
```
```
DAILY STATISTICS (Last 3 days)
Date        Sessions    Cost  Total Msgs │ Sonnet Msgs  Sonnet Cost │ Opus Msgs  Opus Cost
2025-05-28        11  $25.60         574 │          78        $2.83 │       192     $22.76
2025-05-27        13  $50.50         776 │          73        $2.81 │       272     $47.69
2025-05-26        12  $44.42       1,484 │         390       $12.10 │       305     $32.33
```

## Tools

### 🔍 `claude_monitor.py` - Real-time Session Monitor

Monitor Claude Code message flows and session activity in real-time with colored terminal output.

**Features:**
- Real-time monitoring of multiple Claude Code sessions simultaneously
- Color-coded output for different message types (user, assistant, system)
- Tool usage tracking with cost and duration information
- Session management with automatic detection of new/ended sessions
- Support for both single-file and multi-session monitoring modes

**Usage:**
```bash
# Monitor all active sessions (default mode)
python claude_monitor.py

# Monitor a specific log file
python claude_monitor.py -f ~/.claude/projects/my-project/session.jsonl

# Show detailed tool inputs/outputs
python claude_monitor.py --tools

# Show full content without truncation
python claude_monitor.py --full

# Demo mode with sample data
python claude_monitor.py --demo
```

### 💰 `claude_costs.py` - Cost Analysis Tool

Comprehensive cost analysis and reporting for Claude Code usage with detailed statistics and export capabilities.

**Features:**
- Detailed cost breakdown by session, model, and time period
- Token usage tracking (input, cache creation, cache read, output)
- Daily aggregated statistics with model breakdowns
- Tool usage analysis and costs
- Multiple export formats (CSV, JSON)
- Flexible filtering by date range, project, or session
- GPU hours calculation (cost ÷ 8) for rough comparison (not accurate)
- **Anthropic 5-hour session tracking** with billing cycle management and limit monitoring

**Usage:**
```bash
# Basic cost analysis
python claude_costs.py

# Analyze specific time periods
python claude_costs.py --period today
python claude_costs.py --period yesterday
python claude_costs.py --period week
python claude_costs.py --period month

# Custom date range
python claude_costs.py --from 2024-01-01 --to 2024-01-31

# Show daily breakdown for last 7 days
python claude_costs.py --days 7

# Sort sessions by different criteria
python claude_costs.py --sort cost      # Most expensive first (default)
python claude_costs.py --sort date      # Most recent first
python claude_costs.py --sort duration  # Longest sessions first

# Show top N sessions
python claude_costs.py --sessions 20

# Filter by project or session
python claude_costs.py --project my-project
python claude_costs.py --session abc123

# Include tool usage statistics
python claude_costs.py --tools

# Export data
python claude_costs.py --csv costs.csv
python claude_costs.py --json costs.json

# Show GPU hours column (cost ÷ 8, rough estimate only)
python claude_costs.py --gpu-hours

# Anthropic session tracking (5-hour windows)
python claude_costs.py --anthropic-sessions --billing-day 25
```

#### Anthropic Session Tracking

Track your usage against Anthropic's 50 sessions per month limit. Anthropic defines a "session" as a 5-hour window from your first message, where all messages within that window count as part of the same session.

**Key Features:**
- **5-hour session windows**: Automatically groups messages into Anthropic-defined sessions
- **Billing cycle tracking**: Monitor sessions per billing cycle based on your billing day
- **50-session limit monitoring**: Visual warnings when approaching or exceeding limits
- **Model breakdown**: Separate tracking for Sonnet vs Opus usage within each session
- **Project correlation**: Shows which Claude Code projects/sessions are included

**Usage:**
```bash
# Show Anthropic sessions with billing cycles (required: specify your billing day)
python claude_costs.py --anthropic-sessions --billing-day 25

# Sort by cost/date/duration
python claude_costs.py --anthropic-sessions --billing-day 1 --sort cost

# Show more sessions
python claude_costs.py --anthropic-sessions --billing-day 15 --sessions 20
```

**Sample Output:**
```
--------------------------------------------------------------------------------
TOP 10 MOST EXPENSIVE ANTHROPIC SESSIONS (5-hour windows)
--------------------------------------------------------------------------------
Session Window                       Date     First     Last      Cost  Total Msgs  Total Tokens │ Sonnet Msgs  Sonnet Cost  Sonnet Tokens │ Opus Msgs  Opus Cost  Opus Tokens
Anthropic-01                    2025-06-18     09:15    13:45    $45.67        123        45,234 │          89        $12.34         23,567 │        34     $33.33       21,667
Anthropic-02                    2025-06-17     14:22    18:30    $38.91         89        38,901 │          67        $15.67         28,432 │        22     $23.24       10,469
...

================================================================================
BILLING CYCLE BREAKDOWN (Billing day: 25)
================================================================================
2025-05-25 to 2025-06-25                    23 sessions    $234.56  ✅
2025-04-25 to 2025-05-25                    45 sessions    $445.67  ⚠️  APPROACHING
2025-03-25 to 2025-04-25                    52 sessions    $567.89  ⚠️  OVER LIMIT
--------------------------------------------------------------------------------
TOTAL across all cycles:                    120 sessions   $1248.12

⚠️  BILLING CYCLES OVER 50 SESSION LIMIT:
   2025-03-25 to 2025-04-25: 52 sessions (2 over limit)
```

## Installation

### Prerequisites
- Python 3.7+
- [Claude Code CLI](https://claude.ai/code) installed and configured

### Setup
1. Clone this repository:
```bash
git clone https://github.com/yourusername/claude-monitoring-tools.git
cd claude-monitoring-tools
```

2. Make scripts executable (optional):
```bash
chmod +x claude_monitor.py claude_costs.py
```

3. Run the tools:
```bash
python claude_monitor.py
python claude_costs.py
```

## How It Works

Both tools read from Claude Code's log files located at `~/.claude/projects/*/logs/*.jsonl`. These logs contain:
- Message exchanges between user and assistant
- Token usage statistics and costs
- Tool usage and execution results
- Session timing and metadata

### Log File Structure
Claude Code stores logs in JSON Lines format with entries like:
```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "type": "assistant",
  "message": {
    "role": "assistant",
    "content": "...",
    "model": "claude-3-5-sonnet-20241022",
    "usage": {
      "input_tokens": 150,
      "cache_read_input_tokens": 0,
      "output_tokens": 75
    }
  },
  "costUSD": 0.001125,
  "durationMs": 2500
}
```

## Output Examples

### Cost Analysis Summary
```
================================================================================
CLAUDE CODE COST ANALYSIS SUMMARY
================================================================================

Total Messages: 1,234
Total Sessions: 45
Date Range: 2024-01-01 10:30 - 2024-01-15 18:45

Token Usage:              Input:           123,456
                          Cache Creation:   12,345
                          Cache Read:       23,456
                          Output:           34,567
                          Total:           193,824

Total Cost:              $5.67
Average Cost per Message: $0.0046
Average Cost per Session: $0.1260

COST BY MODEL
--------------------------------------------------------------------------------
claude-3-5-sonnet-20241022                         $4.20 (856 messages)
claude-3-opus-20240229                             $1.47 (378 messages)
```

### Session Details
```
--------------------------------------------------------------------------------
TOP 10 MOST EXPENSIVE SESSIONS
--------------------------------------------------------------------------------
Session                             Date        Start      End      Cost  Messages   In Tokens  Out Tokens
my-project/abc123def456...          2024-01-15     14:30    15:45    $0.89       15       8,543       2,156
another-project/789ghi012...        2024-01-14     09:15    10:30    $0.67       12       6,234       1,890
```

## Configuration

### Timezone Settings
Both tools automatically detect and use your local timezone for display purposes while storing all data in UTC internally.

## Export Formats

### CSV Export
Includes session summaries and detailed message data suitable for spreadsheet analysis.

### JSON Export
Structured data export for programmatic analysis:
```json
{
  "generated": "2024-01-15T18:30:00Z",
  "summary": {
    "total_cost_usd": 5.67,
    "total_sessions": 45,
    "date_range": {
      "start": "2024-01-01T10:30:00Z",
      "end": "2024-01-15T18:45:00Z"
    }
  },
  "sessions": { ... },
  "messages": [ ... ]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Requirements

- Python 3.7+
- Standard library only (no external dependencies)
- Claude Code CLI installed and configured

## Troubleshooting

### No log files found
- Ensure Claude Code is installed and you've run at least one session
- Check that `~/.claude/projects/` exists and contains project directories
- Verify log files exist with `.jsonl` extension

### Incorrect timezone display
- Tools automatically detect local timezone
- If issues persist, check your system's timezone configuration

### Permission errors
- Ensure read access to `~/.claude/projects/` directory
- Check file permissions on log files

For more help, please open an issue on GitHub.