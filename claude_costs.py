#!/usr/bin/env python3
"""
Claude Code Cost Analyzer
A comprehensive tool to analyze costs from Claude Code session logs.
"""

import json
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import csv
import sys
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import time

@dataclass
class Message:
    """Represents a single message in the conversation"""
    timestamp: datetime
    role: str
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Decimal = Decimal('0')
    duration_ms: int = 0
    content: str = ""
    message_type: str = "message"  # message, tool_use, tool_result
    tool_name: Optional[str] = None
    session_id: str = ""
    project_name: str = ""
    file_path: str = ""

@dataclass
class SessionStats:
    """Statistics for a single session"""
    session_id: str
    project_name: str
    start_time: datetime
    end_time: datetime
    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    tool_uses: int = 0
    tool_results: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: Decimal = Decimal('0')
    total_duration_ms: int = 0
    models_used: set = field(default_factory=set)
    tools_used: Dict[str, int] = field(default_factory=dict)

class CostAnalyzer:
    def __init__(self):
        self.claude_dir = Path.home() / ".claude" / "projects"
        self.messages: List[Message] = []
        self.sessions: Dict[str, SessionStats] = {}
        
        # Get local timezone
        self.local_tz = datetime.now().astimezone().tzinfo
        
        # Model pricing (per million tokens)
        self.model_pricing = {
            "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
            "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
            "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
            "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            # Default pricing if model not found
            "default": {"input": 3.00, "output": 15.00}
        }
    
    def find_log_files(self, date_from: Optional[datetime] = None, 
                      date_to: Optional[datetime] = None) -> List[Path]:
        """Find all Claude Code log files within date range"""
        if not self.claude_dir.exists():
            return []
        
        log_files = []
        for jsonl_file in self.claude_dir.glob("*/*.jsonl"):
            # Check if file is within date range
            if date_from or date_to:
                file_mtime = datetime.fromtimestamp(jsonl_file.stat().st_mtime)
                if date_from and file_mtime < date_from:
                    continue
                if date_to and file_mtime > date_to:
                    continue
            log_files.append(jsonl_file)
        
        return sorted(log_files, key=lambda p: p.stat().st_mtime)
    
    def parse_message(self, data: Dict[str, Any], file_path: Path) -> Optional[Message]:
        """Parse a single message from log data"""
        try:
            # Extract basic info
            timestamp_str = data.get('timestamp', '')
            if not timestamp_str:
                return None
            
            # Handle timezone-aware timestamps - parse as UTC
            try:
                # Parse ISO format with Z as UTC
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                timestamp = datetime.fromisoformat(timestamp_str)
            
            # Ensure we have a timezone-aware datetime in UTC
            if timestamp.tzinfo is None:
                # Assume UTC if no timezone info
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                # Convert to UTC if not already
                timestamp = timestamp.astimezone(timezone.utc)
            
            # Get session and project info from file path
            project_name = file_path.parent.name
            session_id = file_path.stem
            
            # Handle different message structures
            if 'message' in data:
                inner_msg = data['message']
                role = inner_msg.get('role', 'unknown')
                content = inner_msg.get('content', '')
                
                # Extract model info from inner message for assistant messages
                model = inner_msg.get('model') if role == 'assistant' else None
                
                # For assistant messages, get cost info from root level
                if role == 'assistant' and data.get('type') == 'assistant':
                    # Get token counts from usage object
                    usage = inner_msg.get('usage', {})
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    
                    return Message(
                        timestamp=timestamp,
                        role=role,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=Decimal(str(data.get('costUSD', 0))),  # Cost is at root level
                        duration_ms=data.get('durationMs', 0),  # Duration is at root level
                        content=self._extract_text_content(content),
                        message_type="message",
                        session_id=session_id,
                        project_name=project_name,
                        file_path=str(file_path)
                    )
                
                # Handle tool uses and results in content
                if isinstance(content, list) and role == 'assistant':
                    messages = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get('type') == 'tool_use':
                                messages.append(Message(
                                    timestamp=timestamp,
                                    role=role,
                                    model=model,
                                    message_type="tool_use",
                                    tool_name=part.get('name', 'unknown'),
                                    session_id=session_id,
                                    project_name=project_name,
                                    file_path=str(file_path)
                                ))
                    return messages if messages else None
                
                # Regular message
                return Message(
                    timestamp=timestamp,
                    role=role,
                    model=model,
                    content=self._extract_text_content(content),
                    message_type="message",
                    session_id=session_id,
                    project_name=project_name,
                    file_path=str(file_path)
                )
            
            return None
            
        except Exception as e:
            print(f"Error parsing message: {e}", file=sys.stderr)
            return None
    
    def _extract_text_content(self, content: Any) -> str:
        """Extract text content from various content formats"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text_parts.append(part.get('text', ''))
            return '\n'.join(text_parts)
        return str(content)
    
    def load_logs(self, files: List[Path], date_from: Optional[datetime] = None,
                  date_to: Optional[datetime] = None):
        """Load and parse log files"""
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        
                        try:
                            data = json.loads(line)
                            result = self.parse_message(data, file_path)
                            
                            if result:
                                if isinstance(result, list):
                                    for msg in result:
                                        if self._is_within_date_range(msg, date_from, date_to):
                                            self.messages.append(msg)
                                else:
                                    if self._is_within_date_range(result, date_from, date_to):
                                        self.messages.append(result)
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                print(f"Error reading file {file_path}: {e}", file=sys.stderr)
    
    def _is_within_date_range(self, msg: Message, date_from: Optional[datetime],
                             date_to: Optional[datetime]) -> bool:
        """Check if message is within specified date range"""
        # Work with timezone-aware datetimes in UTC
        msg_time = msg.timestamp
        
        if date_from:
            # Ensure date_from is timezone-aware
            if date_from.tzinfo is None:
                from_time = date_from.replace(tzinfo=self.local_tz).astimezone(timezone.utc)
            else:
                from_time = date_from.astimezone(timezone.utc)
            if msg_time < from_time:
                return False
        if date_to:
            # Ensure date_to is timezone-aware
            if date_to.tzinfo is None:
                to_time = date_to.replace(tzinfo=self.local_tz).astimezone(timezone.utc)
            else:
                to_time = date_to.astimezone(timezone.utc)
            if msg_time > to_time:
                return False
        return True
    
    def calculate_session_stats(self):
        """Calculate statistics for each session"""
        self.sessions.clear()
        
        for msg in self.messages:
            session_key = f"{msg.project_name}/{msg.session_id}"
            
            if session_key not in self.sessions:
                self.sessions[session_key] = SessionStats(
                    session_id=msg.session_id,
                    project_name=msg.project_name,
                    start_time=msg.timestamp,
                    end_time=msg.timestamp
                )
            
            stats = self.sessions[session_key]
            
            # Update time range
            if msg.timestamp < stats.start_time:
                stats.start_time = msg.timestamp
            if msg.timestamp > stats.end_time:
                stats.end_time = msg.timestamp
            
            # Update counts
            stats.total_messages += 1
            
            if msg.message_type == "message":
                if msg.role == "user":
                    stats.user_messages += 1
                elif msg.role == "assistant":
                    stats.assistant_messages += 1
            elif msg.message_type == "tool_use":
                stats.tool_uses += 1
                if msg.tool_name:
                    stats.tools_used[msg.tool_name] = stats.tools_used.get(msg.tool_name, 0) + 1
            elif msg.message_type == "tool_result":
                stats.tool_results += 1
            
            # Update tokens and costs
            stats.total_input_tokens += msg.input_tokens
            stats.total_output_tokens += msg.output_tokens
            stats.total_cost_usd += msg.cost_usd
            stats.total_duration_ms += msg.duration_ms
            
            # Track models
            if msg.model:
                stats.models_used.add(msg.model)
    
    def get_time_period_stats(self, period: str) -> Tuple[datetime, datetime]:
        """Get date range for specified time period"""
        # Get current time in local timezone
        now = datetime.now(self.local_tz)
        
        if period == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
        elif period == "yesterday":
            yesterday = now - timedelta(days=1)
            start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif period == "week":
            start = now - timedelta(days=7)
            end = now
        elif period == "month":
            start = now - timedelta(days=30)
            end = now
        else:
            raise ValueError(f"Unknown time period: {period}")
        
        return start, end
    
    def format_currency(self, amount: Decimal, currency: str = "USD") -> str:
        """Format currency amount"""
        if currency == "USD":
            return f"${amount:.4f}"
        elif currency == "EUR":
            return f"€{amount:.4f}"
        elif currency == "GBP":
            return f"£{amount:.4f}"
        else:
            return f"{amount:.4f} {currency}"
    
    def print_summary(self, currency: str = "USD"):
        """Print summary statistics"""
        if not self.messages:
            print("No messages found in the specified date range.")
            return
        
        total_cost = sum(msg.cost_usd for msg in self.messages)
        total_input_tokens = sum(msg.input_tokens for msg in self.messages)
        total_output_tokens = sum(msg.output_tokens for msg in self.messages)
        total_duration = sum(msg.duration_ms for msg in self.messages)
        
        print("\n" + "="*80)
        print("CLAUDE CODE COST ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nTotal Messages: {len(self.messages)}")
        print(f"Total Sessions: {len(self.sessions)}")
        # Convert timestamps to local timezone for display
        min_time = min(m.timestamp for m in self.messages).astimezone(self.local_tz)
        max_time = max(m.timestamp for m in self.messages).astimezone(self.local_tz)
        print(f"Date Range: {min_time.strftime('%Y-%m-%d %H:%M')} - "
              f"{max_time.strftime('%Y-%m-%d %H:%M')}")
        
        print(f"\n{'Token Usage:':<30} {'Input:':<15} {total_input_tokens:,}")
        print(f"{'':<30} {'Output:':<15} {total_output_tokens:,}")
        print(f"{'':<30} {'Total:':<15} {total_input_tokens + total_output_tokens:,}")
        
        print(f"\n{'Total Cost:':<30} {self.format_currency(total_cost, currency)}")
        print(f"{'Average Cost per Message:':<30} {self.format_currency(total_cost / len(self.messages) if self.messages else Decimal('0'), currency)}")
        print(f"{'Average Cost per Session:':<30} {self.format_currency(total_cost / len(self.sessions) if self.sessions else Decimal('0'), currency)}")
        
        if total_duration > 0:
            print(f"\n{'Total Duration:':<30} {total_duration / 1000:.2f} seconds")
            print(f"{'Average Duration per Message:':<30} {total_duration / len(self.messages) / 1000:.2f} seconds")
        
        # Model breakdown
        model_costs = defaultdict(Decimal)
        model_counts = defaultdict(int)
        for msg in self.messages:
            if msg.model and msg.cost_usd > 0:
                model_costs[msg.model] += msg.cost_usd
                model_counts[msg.model] += 1
        
        if model_costs:
            print("\n" + "-"*80)
            print("COST BY MODEL")
            print("-"*80)
            for model, cost in sorted(model_costs.items(), key=lambda x: x[1], reverse=True):
                print(f"{model:<50} {self.format_currency(cost, currency):>15} ({model_counts[model]} messages)")
    
    def print_session_details(self, top_n: int = 10, currency: str = "USD", sort_by: str = "cost"):
        """Print detailed session information"""
        if not self.sessions:
            return
        
        # First, collect all unique models across all sessions
        all_models = set()
        for stats in self.sessions.values():
            all_models.update(stats.models_used)
        
        # Sort models for consistent column ordering
        sorted_models = sorted(all_models)
        
        # Determine sort method
        if sort_by == "date":
            sorted_sessions = sorted(self.sessions.items(), 
                                   key=lambda x: x[1].start_time, 
                                   reverse=True)[:top_n]
            title = f"TOP {top_n} MOST RECENT SESSIONS"
        elif sort_by == "duration":
            sorted_sessions = sorted(self.sessions.items(), 
                                   key=lambda x: (x[1].end_time - x[1].start_time).total_seconds(), 
                                   reverse=True)[:top_n]
            title = f"TOP {top_n} LONGEST SESSIONS"
        else:  # default to cost
            sorted_sessions = sorted(self.sessions.items(), 
                                   key=lambda x: x[1].total_cost_usd, 
                                   reverse=True)[:top_n]
            title = f"TOP {top_n} MOST EXPENSIVE SESSIONS"
        
        # Build the header with model columns
        header_width = 120 + (len(sorted_models) * 15)
        print("\n" + "-"*header_width)
        print(title)
        print("-"*header_width)
        
        # Build header line
        header = f"{'Session':<40} {'Date':>12} {'Start':>8} {'End':>8} {'Cost':>10} {'Messages':>10} {'Duration':>15}"
        
        # Add model columns to header
        for model in sorted_models:
            # Shorten model names for column headers
            if 'claude' in model:
                parts = model.split('-')
                if len(parts) >= 3:
                    model_short = parts[1] + '-' + parts[2]  # e.g., "opus-4" or "sonnet-4"
                else:
                    model_short = model[:14]
            else:
                model_short = model[:14]
            header += f" {model_short:>14}"
        
        print(header)
        print("-"*header_width)
        
        session_total_cost = Decimal('0')
        for session_key, stats in sorted_sessions:
            duration = (stats.end_time - stats.start_time).total_seconds()
            # Convert to local timezone for display
            local_start = stats.start_time.astimezone(self.local_tz)
            local_end = stats.end_time.astimezone(self.local_tz)
            date = local_start.strftime('%Y-%m-%d')
            start_time = local_start.strftime('%H:%M')
            end_time = local_end.strftime('%H:%M')
            session_total_cost += stats.total_cost_usd
            
            # Count messages per model for this session
            model_counts = self._get_model_message_counts(stats.project_name, stats.session_id)
            
            line = f"{session_key:<40} {date:>12} {start_time:>8} {end_time:>8} {self.format_currency(stats.total_cost_usd, currency):>10} " \
                   f"{stats.total_messages:>10} {f'{duration:.1f}s':>15}"
            
            # Add model message counts
            for model in sorted_models:
                count = model_counts.get(model, 0)
                line += f" {count:>14}"
            
            print(line)
        
        # Add total line
        print("-"*header_width)
        sort_label = "sorted by " + sort_by
        print(f"{'TOTAL for ' + str(len(sorted_sessions)) + ' sessions (' + sort_label + '):':<68} {self.format_currency(session_total_cost, currency):>10}")
    
    def _get_model_message_counts(self, project_name: str, session_id: str) -> Dict[str, int]:
        """Get message counts per model for a specific session"""
        model_counts = defaultdict(int)
        
        for msg in self.messages:
            if msg.project_name == project_name and msg.session_id == session_id:
                if msg.model and msg.role == "assistant":
                    model_counts[msg.model] += 1
        
        return model_counts
    
    def print_tool_usage(self):
        """Print tool usage statistics"""
        all_tools = defaultdict(int)
        tool_costs = defaultdict(Decimal)
        
        for session in self.sessions.values():
            for tool, count in session.tools_used.items():
                all_tools[tool] += count
        
        # Calculate tool costs from tool results
        for msg in self.messages:
            if msg.message_type == "tool_result" and msg.tool_name:
                tool_costs[msg.tool_name] += msg.cost_usd
        
        if all_tools:
            print("\n" + "-"*80)
            print("TOOL USAGE STATISTICS")
            print("-"*80)
            print(f"{'Tool Name':<30} {'Uses':>10} {'Cost':>15}")
            print("-"*80)
            
            for tool, count in sorted(all_tools.items(), key=lambda x: x[1], reverse=True):
                cost = tool_costs.get(tool, Decimal('0'))
                print(f"{tool:<30} {count:>10} {self.format_currency(cost):>15}")
    
    def export_csv(self, filename: str):
        """Export analysis to CSV file"""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write summary
            writer.writerow(['CLAUDE CODE COST ANALYSIS REPORT'])
            writer.writerow(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow([])
            
            # Write session summary
            writer.writerow(['Session Summary'])
            writer.writerow(['Project', 'Session ID', 'Start Time', 'End Time', 
                           'Messages', 'Cost (USD)', 'Input Tokens', 'Output Tokens', 
                           'Duration (ms)', 'Models Used'])
            
            for session_key, stats in sorted(self.sessions.items()):
                writer.writerow([
                    stats.project_name,
                    stats.session_id,
                    stats.start_time.astimezone(self.local_tz).strftime('%Y-%m-%d %H:%M:%S'),
                    stats.end_time.astimezone(self.local_tz).strftime('%Y-%m-%d %H:%M:%S'),
                    stats.total_messages,
                    float(stats.total_cost_usd),
                    stats.total_input_tokens,
                    stats.total_output_tokens,
                    stats.total_duration_ms,
                    ', '.join(stats.models_used)
                ])
            
            writer.writerow([])
            
            # Write message details
            writer.writerow(['Message Details'])
            writer.writerow(['Timestamp', 'Project', 'Session', 'Role', 'Type', 
                           'Model', 'Cost (USD)', 'Input Tokens', 'Output Tokens', 
                           'Duration (ms)', 'Tool Name'])
            
            for msg in sorted(self.messages, key=lambda m: m.timestamp):
                writer.writerow([
                    msg.timestamp.astimezone(self.local_tz).strftime('%Y-%m-%d %H:%M:%S'),
                    msg.project_name,
                    msg.session_id,
                    msg.role,
                    msg.message_type,
                    msg.model or '',
                    float(msg.cost_usd),
                    msg.input_tokens,
                    msg.output_tokens,
                    msg.duration_ms,
                    msg.tool_name or ''
                ])
        
        print(f"\nData exported to {filename}")
    
    def export_json(self, filename: str):
        """Export analysis to JSON file"""
        data = {
            'generated': datetime.now().isoformat(),
            'summary': {
                'total_messages': len(self.messages),
                'total_sessions': len(self.sessions),
                'total_cost_usd': float(sum(msg.cost_usd for msg in self.messages)),
                'total_input_tokens': sum(msg.input_tokens for msg in self.messages),
                'total_output_tokens': sum(msg.output_tokens for msg in self.messages),
                'total_duration_ms': sum(msg.duration_ms for msg in self.messages),
                'date_range': {
                    'start': min(m.timestamp for m in self.messages).isoformat() if self.messages else None,
                    'end': max(m.timestamp for m in self.messages).isoformat() if self.messages else None
                }
            },
            'sessions': {},
            'messages': []
        }
        
        # Add session data
        for session_key, stats in self.sessions.items():
            data['sessions'][session_key] = {
                'project_name': stats.project_name,
                'session_id': stats.session_id,
                'start_time': stats.start_time.isoformat(),
                'end_time': stats.end_time.isoformat(),
                'total_messages': stats.total_messages,
                'user_messages': stats.user_messages,
                'assistant_messages': stats.assistant_messages,
                'tool_uses': stats.tool_uses,
                'tool_results': stats.tool_results,
                'total_cost_usd': float(stats.total_cost_usd),
                'total_input_tokens': stats.total_input_tokens,
                'total_output_tokens': stats.total_output_tokens,
                'total_duration_ms': stats.total_duration_ms,
                'models_used': list(stats.models_used),
                'tools_used': stats.tools_used
            }
        
        # Add message data
        for msg in self.messages:
            data['messages'].append({
                'timestamp': msg.timestamp.isoformat(),
                'project_name': msg.project_name,
                'session_id': msg.session_id,
                'role': msg.role,
                'message_type': msg.message_type,
                'model': msg.model,
                'cost_usd': float(msg.cost_usd),
                'input_tokens': msg.input_tokens,
                'output_tokens': msg.output_tokens,
                'duration_ms': msg.duration_ms,
                'tool_name': msg.tool_name
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nData exported to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Claude Code session costs")
    
    # Date filtering
    parser.add_argument("--from", dest="date_from", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="date_to", help="End date (YYYY-MM-DD)")
    parser.add_argument("--period", choices=["today", "yesterday", "week", "month"],
                       help="Predefined time period")
    
    # Display options
    parser.add_argument("--sessions", type=int, default=10,
                       help="Number of top sessions to show (default: 10)")
    parser.add_argument("--sort", choices=["cost", "date", "duration"], default="cost",
                       help="Sort sessions by: cost, date, or duration (default: cost)")
    parser.add_argument("--currency", choices=["USD", "EUR", "GBP"], default="USD",
                       help="Currency for display (default: USD)")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed message-level analysis")
    parser.add_argument("--tools", action="store_true",
                       help="Show tool usage statistics")
    
    # Export options
    parser.add_argument("--csv", help="Export results to CSV file")
    parser.add_argument("--json", help="Export results to JSON file")
    
    # Session filtering
    parser.add_argument("--project", help="Filter by project name")
    parser.add_argument("--session", help="Filter by session ID")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CostAnalyzer()
    
    # Determine date range
    date_from = None
    date_to = None
    
    if args.period:
        date_from, date_to = analyzer.get_time_period_stats(args.period)
    else:
        if args.date_from:
            # Parse as local date and add timezone info
            date_from = datetime.strptime(args.date_from, "%Y-%m-%d")
            date_from = date_from.replace(tzinfo=analyzer.local_tz)
        if args.date_to:
            # Parse as local date, set to end of day, and add timezone info
            date_to = datetime.strptime(args.date_to, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, microsecond=999999)
            date_to = date_to.replace(tzinfo=analyzer.local_tz)
    
    # Find and load log files
    print("Searching for Claude Code log files...")
    log_files = analyzer.find_log_files(date_from, date_to)
    
    if not log_files:
        print("No log files found.")
        return
    
    print(f"Found {len(log_files)} log file(s)")
    
    # Load and parse logs
    print("Loading log data...")
    analyzer.load_logs(log_files, date_from, date_to)
    
    # Filter by project/session if specified
    if args.project:
        analyzer.messages = [m for m in analyzer.messages if args.project in m.project_name]
    if args.session:
        analyzer.messages = [m for m in analyzer.messages if args.session in m.session_id]
    
    # Calculate statistics
    analyzer.calculate_session_stats()
    
    # Display results
    analyzer.print_summary(args.currency)
    analyzer.print_session_details(args.sessions, args.currency, args.sort)
    
    if args.tools:
        analyzer.print_tool_usage()
    
    if args.detailed:
        print("\n" + "-"*80)
        print("DETAILED MESSAGE COSTS")
        print("-"*80)
        print(f"{'Timestamp':<20} {'Project/Session':<30} {'Type':<15} {'Cost':>10}")
        print("-"*80)
        
        for msg in sorted(analyzer.messages, key=lambda m: m.timestamp):
            if msg.cost_usd > 0:
                session_key = f"{msg.project_name}/{msg.session_id}"
                print(f"{msg.timestamp.astimezone(analyzer.local_tz).strftime('%Y-%m-%d %H:%M:%S'):<20} "
                      f"{session_key[:30]:<30} {msg.message_type:<15} "
                      f"{analyzer.format_currency(msg.cost_usd, args.currency):>10}")
    
    # Export if requested
    if args.csv:
        analyzer.export_csv(args.csv)
    if args.json:
        analyzer.export_json(args.json)

if __name__ == "__main__":
    main()