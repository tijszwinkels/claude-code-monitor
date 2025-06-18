#!/usr/bin/env python3
"""
Claude Code Cost Analyzer
A comprehensive tool to analyze costs from Claude Code session logs.
"""

import json
import argparse
from datetime import datetime, timedelta, timezone
from calendar import monthrange
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
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
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
    total_cache_creation_input_tokens: int = 0
    total_cache_read_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: Decimal = Decimal('0')
    total_duration_ms: int = 0
    models_used: set = field(default_factory=set)
    tools_used: Dict[str, int] = field(default_factory=dict)

@dataclass
class AnthropicSession:
    """Represents a 5-hour Anthropic session window"""
    start_time: datetime
    end_time: datetime
    first_message_time: datetime
    last_message_time: datetime
    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    total_cost_usd: Decimal = Decimal('0')
    total_input_tokens: int = 0
    total_cache_creation_input_tokens: int = 0
    total_cache_read_input_tokens: int = 0
    total_output_tokens: int = 0
    claude_sessions: set = field(default_factory=set)  # Set of Claude Code session IDs in this window
    projects: set = field(default_factory=set)
    models_used: set = field(default_factory=set)

class CostAnalyzer:
    def __init__(self):
        self.claude_dir = Path.home() / ".claude" / "projects"
        self.messages: List[Message] = []
        self.sessions: Dict[str, SessionStats] = {}
        self.anthropic_sessions: List[AnthropicSession] = []
        
        # Get local timezone
        self.local_tz = datetime.now().astimezone().tzinfo
        
        # Claude pricing (per million tokens, as of 2024)
        self.pricing = {
            'claude-sonnet-4-20250514': {
                'input': Decimal('3.00'),  # $3 per million input tokens
                'output': Decimal('15.00'),  # $15 per million output tokens
                'cache_write': Decimal('3.75'),  # $3.75 per million cache write tokens
                'cache_read': Decimal('0.30')   # $0.30 per million cache read tokens
            },
            'claude-opus-4-20250514': {
                'input': Decimal('15.00'),  # $15 per million input tokens
                'output': Decimal('75.00'),  # $75 per million output tokens
                'cache_write': Decimal('18.75'),  # $18.75 per million cache write tokens
                'cache_read': Decimal('1.50')   # $1.50 per million cache read tokens
            },
            # Legacy model names
            'claude-3-5-sonnet-20241022': {
                'input': Decimal('3.00'),
                'output': Decimal('15.00'),
                'cache_write': Decimal('3.75'),
                'cache_read': Decimal('0.30')
            },
            'claude-3-opus-20240229': {
                'input': Decimal('15.00'),
                'output': Decimal('75.00'),
                'cache_write': Decimal('18.75'),
                'cache_read': Decimal('1.50')
            }
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
                file_mtime = datetime.fromtimestamp(jsonl_file.stat().st_mtime, tz=self.local_tz)
                if date_from and file_mtime < date_from:
                    continue
                if date_to and file_mtime > date_to:
                    continue
            log_files.append(jsonl_file)
        
        return sorted(log_files, key=lambda p: p.stat().st_mtime)
    
    def calculate_cost(self, model: str, input_tokens: int, cache_creation_tokens: int, 
                      cache_read_tokens: int, output_tokens: int) -> Decimal:
        """Calculate cost based on model and token usage"""
        if not model or model not in self.pricing:
            # Default to Sonnet pricing if model unknown
            pricing = self.pricing.get('claude-sonnet-4-20250514', {
                'input': Decimal('3.00'),
                'output': Decimal('15.00'),
                'cache_write': Decimal('3.75'),
                'cache_read': Decimal('0.30')
            })
        else:
            pricing = self.pricing[model]
        
        # Calculate cost per token type (prices are per million tokens)
        input_cost = (Decimal(input_tokens) / Decimal('1000000')) * pricing['input']
        cache_write_cost = (Decimal(cache_creation_tokens) / Decimal('1000000')) * pricing['cache_write']
        cache_read_cost = (Decimal(cache_read_tokens) / Decimal('1000000')) * pricing['cache_read']
        output_cost = (Decimal(output_tokens) / Decimal('1000000')) * pricing['output']
        
        return input_cost + cache_write_cost + cache_read_cost + output_cost
    
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
                    cache_creation_input_tokens = usage.get('cache_creation_input_tokens', 0)
                    cache_read_input_tokens = usage.get('cache_read_input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    
                    # Calculate cost if not present (for newer log formats)
                    cost_usd = data.get('costUSD')
                    if cost_usd is None:
                        cost_usd = self.calculate_cost(model, input_tokens, cache_creation_input_tokens, 
                                                     cache_read_input_tokens, output_tokens)
                    else:
                        cost_usd = Decimal(str(cost_usd))
                    
                    return Message(
                        timestamp=timestamp,
                        role=role,
                        model=model,
                        input_tokens=input_tokens,
                        cache_creation_input_tokens=cache_creation_input_tokens,
                        cache_read_input_tokens=cache_read_input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=cost_usd,
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
            stats.total_cache_creation_input_tokens += msg.cache_creation_input_tokens
            stats.total_cache_read_input_tokens += msg.cache_read_input_tokens
            stats.total_output_tokens += msg.output_tokens
            stats.total_cost_usd += msg.cost_usd
            stats.total_duration_ms += msg.duration_ms
            
            # Track models
            if msg.model:
                stats.models_used.add(msg.model)
    
    def calculate_anthropic_sessions(self):
        """Calculate 5-hour Anthropic session windows from messages"""
        if not self.messages:
            return
        
        self.anthropic_sessions.clear()
        
        # Sort messages by timestamp
        sorted_messages = sorted(self.messages, key=lambda m: m.timestamp)
        
        current_session = None
        
        for msg in sorted_messages:
            # Skip messages that don't represent actual user interactions
            if msg.role not in ['user', 'assistant']:
                continue
                
            # Only include messages that involve LLM calls (consume tokens)
            # This filters out messages without token usage (non-LLM interactions)
            if (msg.input_tokens == 0 and msg.cache_creation_input_tokens == 0 and 
                msg.cache_read_input_tokens == 0 and msg.output_tokens == 0):
                continue
                
            # If no current session or message is more than 5 hours after current session start
            if (current_session is None or 
                msg.timestamp > current_session.start_time + timedelta(hours=5)):
                
                # Start new session
                current_session = AnthropicSession(
                    start_time=msg.timestamp,
                    end_time=msg.timestamp + timedelta(hours=5),
                    first_message_time=msg.timestamp,
                    last_message_time=msg.timestamp
                )
                self.anthropic_sessions.append(current_session)
            
            # Update session stats
            current_session.last_message_time = msg.timestamp
            current_session.total_messages += 1
            
            if msg.role == 'user':
                current_session.user_messages += 1
            elif msg.role == 'assistant':
                current_session.assistant_messages += 1
                
            current_session.total_cost_usd += msg.cost_usd
            current_session.total_input_tokens += msg.input_tokens
            current_session.total_cache_creation_input_tokens += msg.cache_creation_input_tokens
            current_session.total_cache_read_input_tokens += msg.cache_read_input_tokens
            current_session.total_output_tokens += msg.output_tokens
            
            # Track which Claude Code sessions and projects are in this Anthropic session
            session_key = f"{msg.project_name}/{msg.session_id}"
            current_session.claude_sessions.add(session_key)
            current_session.projects.add(msg.project_name)
            
            if msg.model:
                current_session.models_used.add(msg.model)
    
    def get_billing_cycles(self, billing_day: int) -> List[Tuple[datetime, datetime, str]]:
        """Get list of billing cycles based on billing day and data range"""
        if not self.messages:
            return []
        
        # Get data range in local timezone
        min_time = min(m.timestamp for m in self.messages).astimezone(self.local_tz)
        max_time = max(m.timestamp for m in self.messages).astimezone(self.local_tz)
        
        cycles = []
        
        # Start from the billing cycle that contains the earliest message
        current_date = min_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Adjust to find the correct billing cycle start
        if billing_day <= min_time.day:
            # Billing cycle already started this month
            cycle_start = current_date.replace(day=billing_day)
        else:
            # Billing cycle starts in previous month
            if current_date.month == 1:
                prev_month = current_date.replace(year=current_date.year - 1, month=12)
            else:
                prev_month = current_date.replace(month=current_date.month - 1)
            
            # Handle case where previous month doesn't have the billing day
            max_day_prev = monthrange(prev_month.year, prev_month.month)[1]
            actual_billing_day = min(billing_day, max_day_prev)
            cycle_start = prev_month.replace(day=actual_billing_day)
        
        # Generate cycles until we cover all data
        while cycle_start <= max_time:
            # Calculate next cycle start
            if cycle_start.month == 12:
                next_month = cycle_start.replace(year=cycle_start.year + 1, month=1)
            else:
                next_month = cycle_start.replace(month=cycle_start.month + 1)
            
            # Handle case where next month doesn't have the billing day
            max_day_next = monthrange(next_month.year, next_month.month)[1]
            actual_billing_day = min(billing_day, max_day_next)
            cycle_end = next_month.replace(day=actual_billing_day) - timedelta(microseconds=1)
            
            # Create cycle label
            cycle_label = f"{cycle_start.strftime('%Y-%m-%d')} to {cycle_end.strftime('%Y-%m-%d')}"
            
            cycles.append((cycle_start, cycle_end, cycle_label))
            
            # Move to next cycle
            cycle_start = cycle_end + timedelta(microseconds=1)
            cycle_start = cycle_start.replace(hour=0, minute=0, second=0, microsecond=0)
        
        return cycles
    
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
    
    def format_currency(self, amount: Decimal, currency: str = "USD", precision: int = 2) -> str:
        """Format currency amount"""
        if currency == "USD":
            return f"${amount:.{precision}f}"
        elif currency == "EUR":
            return f"€{amount:.{precision}f}"
        elif currency == "GBP":
            return f"£{amount:.{precision}f}"
        else:
            return f"{amount:.{precision}f} {currency}"
    
    def print_summary(self, currency: str = "USD"):
        """Print summary statistics"""
        if not self.messages:
            print("No messages found in the specified date range.")
            return
        
        total_cost = sum(msg.cost_usd for msg in self.messages)
        total_input_tokens = sum(msg.input_tokens for msg in self.messages)
        total_cache_creation_tokens = sum(msg.cache_creation_input_tokens for msg in self.messages)
        total_cache_read_tokens = sum(msg.cache_read_input_tokens for msg in self.messages)
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
        
        total_all_input_tokens = total_input_tokens + total_cache_creation_tokens + total_cache_read_tokens
        print(f"\n{'Token Usage:':<30} {'Input:':<15} {total_input_tokens:,}")
        print(f"{'':<30} {'Cache Creation:':<15} {total_cache_creation_tokens:,}")
        print(f"{'':<30} {'Cache Read:':<15} {total_cache_read_tokens:,}")
        print(f"{'':<30} {'Output:':<15} {total_output_tokens:,}")
        print(f"{'':<30} {'Total:':<15} {total_all_input_tokens + total_output_tokens:,}")
        
        print(f"\n{'Total Cost:':<30} {self.format_currency(total_cost, currency, 2)}")
        print(f"{'Average Cost per Message:':<30} {self.format_currency(total_cost / len(self.messages) if self.messages else Decimal('0'), currency, 2)}")
        print(f"{'Average Cost per Session:':<30} {self.format_currency(total_cost / len(self.sessions) if self.sessions else Decimal('0'), currency, 2)}")
        
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
                print(f"{model:<50} {self.format_currency(cost, currency, 2):>15} ({model_counts[model]} messages)")
    
    def print_session_details(self, top_n: int = 10, currency: str = "USD", sort_by: str = "cost", gpu_hours: bool = False):
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
                                   key=lambda x: x[1].end_time, 
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
        gpu_hours_col_width = 12 if gpu_hours else 0
        header_width = 119 + (len(sorted_models) * 15) + gpu_hours_col_width
        print("\n" + "-"*header_width)
        print(title)
        print("-"*header_width)
        
        # Build header line
        header = f"{'Session':<35} {'Date':>12} {'Start':>8} {'End':>8} {'Cost':>10}"
        if gpu_hours:
            header += f" {'GPU Hours':>10}"
        header += f" {'Messages':>10} {'In Tokens':>12} {'Out Tokens':>12}"
        
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
            # Convert to local timezone for display
            local_start = stats.start_time.astimezone(self.local_tz)
            local_end = stats.end_time.astimezone(self.local_tz)
            date = local_start.strftime('%Y-%m-%d')
            start_time = local_start.strftime('%H:%M')
            end_time = local_end.strftime('%H:%M')
            session_total_cost += stats.total_cost_usd
            
            # Count messages per model for this session
            model_counts = self._get_model_message_counts(stats.project_name, stats.session_id)
            
            # Calculate total input tokens (including cache)
            total_input_tokens = stats.total_input_tokens + stats.total_cache_creation_input_tokens + stats.total_cache_read_input_tokens
            
            # Truncate session key to fit the column width
            truncated_session = session_key[:34] + "…" if len(session_key) > 35 else session_key
            
            line = f"{truncated_session:<35} {date:>12} {start_time:>8} {end_time:>8} {self.format_currency(stats.total_cost_usd, currency, 2):>10}"
            if gpu_hours:
                gpu_hours_value = float(stats.total_cost_usd) / 8
                line += f" {gpu_hours_value:>10.4f}"
            line += f" {stats.total_messages:>10} {total_input_tokens:>12} {stats.total_output_tokens:>12}"
            
            # Add model message counts
            for model in sorted_models:
                count = model_counts.get(model, 0)
                line += f" {count:>14}"
            
            print(line)
        
        # Add total line
        print("-"*header_width)
        sort_label = "sorted by " + sort_by
        total_line = f"{'TOTAL for ' + str(len(sorted_sessions)) + ' sessions (' + sort_label + '):':<68} {self.format_currency(session_total_cost, currency, 2):>10}"
        if gpu_hours:
            total_gpu_hours = float(session_total_cost) / 8
            total_line += f" {total_gpu_hours:>10.4f}"
        print(total_line)
    
    def _get_model_message_counts(self, project_name: str, session_id: str) -> Dict[str, int]:
        """Get message counts per model for a specific session"""
        model_counts = defaultdict(int)
        
        for msg in self.messages:
            if msg.project_name == project_name and msg.session_id == session_id:
                if msg.model and msg.role == "assistant":
                    model_counts[msg.model] += 1
        
        return model_counts
    
    def print_anthropic_sessions(self, top_n: int = 10, currency: str = "USD", sort_by: str = "cost", gpu_hours: bool = False, billing_day: int = 1):
        """Print Anthropic 5-hour session windows with billing cycle breakdown"""
        if not self.anthropic_sessions:
            print("No Anthropic sessions found.")
            return
        
        # Get billing cycles
        billing_cycles = self.get_billing_cycles(billing_day)
        
        # Group sessions by billing cycle
        cycle_sessions = defaultdict(list)
        for session in self.anthropic_sessions:
            session_time = session.first_message_time.astimezone(self.local_tz)
            
            # Find which billing cycle this session belongs to
            for cycle_start, cycle_end, cycle_label in billing_cycles:
                if cycle_start <= session_time <= cycle_end:
                    cycle_sessions[cycle_label].append(session)
                    break
        
        # Determine sort method
        if sort_by == "date":
            sorted_sessions = sorted(self.anthropic_sessions, 
                                   key=lambda x: x.first_message_time, 
                                   reverse=True)[:top_n]
            title = f"TOP {top_n} MOST RECENT ANTHROPIC SESSIONS (5-hour windows)"
        elif sort_by == "duration":
            sorted_sessions = sorted(self.anthropic_sessions, 
                                   key=lambda x: (x.last_message_time - x.first_message_time).total_seconds(), 
                                   reverse=True)[:top_n]
            title = f"TOP {top_n} LONGEST ANTHROPIC SESSIONS (5-hour windows)"
        else:  # default to cost
            sorted_sessions = sorted(self.anthropic_sessions, 
                                   key=lambda x: x.total_cost_usd, 
                                   reverse=True)[:top_n]
            title = f"TOP {top_n} MOST EXPENSIVE ANTHROPIC SESSIONS (5-hour windows)"
        
        # Calculate header width
        gpu_hours_col_width = 12 if gpu_hours else 0
        header_width = 165 + gpu_hours_col_width
        
        print("\n" + "-"*header_width)
        print(title)
        print("-"*header_width)
        
        # Build header line
        header = f"{'Session Window':<35} {'Date':>12} {'First':>8} {'Last':>8} {'Window Cost':>12}"
        if gpu_hours:
            header += f" {'GPU Hours':>10}"
        header += f" {'Messages':>10} {'Projects':>10} {'Claude Sessions':>15} {'In Tokens':>12} {'Out Tokens':>12}"
        
        print(header)
        print("-"*header_width)
        
        total_cost = Decimal('0')
        for i, session in enumerate(sorted_sessions, 1):
            # Convert to local timezone for display
            local_first = session.first_message_time.astimezone(self.local_tz)
            local_last = session.last_message_time.astimezone(self.local_tz)
            date = local_first.strftime('%Y-%m-%d')
            first_time = local_first.strftime('%H:%M')
            last_time = local_last.strftime('%H:%M')
            total_cost += session.total_cost_usd
            
            # Calculate total input tokens (including cache)
            total_input_tokens = (session.total_input_tokens + 
                                session.total_cache_creation_input_tokens + 
                                session.total_cache_read_input_tokens)
            
            # Create session identifier
            session_id = f"Anthropic-{i:02d}"
            
            line = f"{session_id:<35} {date:>12} {first_time:>8} {last_time:>8} {self.format_currency(session.total_cost_usd, currency, 2):>12}"
            if gpu_hours:
                gpu_hours_value = float(session.total_cost_usd) / 8
                line += f" {gpu_hours_value:>10.4f}"
            line += f" {session.total_messages:>10} {len(session.projects):>10} {len(session.claude_sessions):>15} {total_input_tokens:>12} {session.total_output_tokens:>12}"
            
            print(line)
            
            # Show projects and models used
            if session.projects:
                projects_str = ", ".join(sorted(session.projects))
                if len(projects_str) > 100:
                    projects_str = projects_str[:97] + "..."
                print(f"  {'Projects:':<15} {projects_str}")
            
            if session.models_used:
                models_str = ", ".join(sorted(session.models_used))
                print(f"  {'Models:':<15} {models_str}")
            
            print()  # Empty line between sessions
        
        # Add total line
        print("-"*header_width)
        sort_label = "sorted by " + sort_by
        total_line = f"{'TOTAL for ' + str(len(sorted_sessions)) + ' Anthropic sessions (' + sort_label + '):':<68} {self.format_currency(total_cost, currency, 2):>12}"
        if gpu_hours:
            total_gpu_hours = float(total_cost) / 8
            total_line += f" {total_gpu_hours:>10.4f}"
        print(total_line)
        
        # Show billing cycle breakdown
        print(f"\n" + "="*80)
        print(f"BILLING CYCLE BREAKDOWN (Billing day: {billing_day})")
        print("="*80)
        
        total_sessions_all_cycles = 0
        cycles_over_limit = []
        
        for cycle_start, cycle_end, cycle_label in sorted(billing_cycles, reverse=True):
            sessions_in_cycle = cycle_sessions.get(cycle_label, [])
            session_count = len(sessions_in_cycle)
            total_sessions_all_cycles += session_count
            
            cycle_cost = sum(s.total_cost_usd for s in sessions_in_cycle)
            
            status = "✅"
            if session_count > 50:
                status = "⚠️  OVER LIMIT"
                cycles_over_limit.append((cycle_label, session_count))
            elif session_count > 40:
                status = "⚠️  APPROACHING"
            
            print(f"{cycle_label:<35} {session_count:>3} sessions  {self.format_currency(cycle_cost, currency, 2):>10}  {status}")
        
        print("-"*80)
        print(f"{'TOTAL across all cycles:':<35} {total_sessions_all_cycles:>3} sessions  {self.format_currency(sum(s.total_cost_usd for s in self.anthropic_sessions), currency, 2):>10}")
        
        # Show warnings for cycles over limit
        if cycles_over_limit:
            print(f"\n⚠️  BILLING CYCLES OVER 50 SESSION LIMIT:")
            for cycle_label, count in cycles_over_limit:
                print(f"   {cycle_label}: {count} sessions ({count - 50} over limit)")
        else:
            print(f"\n✅ All billing cycles within 50 session limit")
    
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
                print(f"{tool:<30} {count:>10} {self.format_currency(cost, 'USD', 2):>15}")
    
    def print_daily_stats(self, num_days: int, currency: str = "USD", gpu_hours: bool = False):
        """Print daily aggregated statistics"""
        if not self.messages:
            print("No messages found to aggregate by day.")
            return
        
        # Group messages by day
        daily_stats = defaultdict(lambda: {
            'cost': Decimal('0'),
            'messages': 0,
            'input_tokens': 0,
            'cache_creation_tokens': 0,
            'cache_read_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'sessions': set(),
            'models': set(),
            'sonnet_messages': 0,
            'opus_messages': 0,
            'sonnet_tokens': 0,
            'opus_tokens': 0,
            'sonnet_cost': Decimal('0'),
            'opus_cost': Decimal('0')
        })
        
        for msg in self.messages:
            # Convert to local timezone and get date
            local_time = msg.timestamp.astimezone(self.local_tz)
            date_key = local_time.date()
            
            day_stats = daily_stats[date_key]
            day_stats['cost'] += msg.cost_usd
            day_stats['messages'] += 1
            day_stats['input_tokens'] += msg.input_tokens
            day_stats['cache_creation_tokens'] += msg.cache_creation_input_tokens
            day_stats['cache_read_tokens'] += msg.cache_read_input_tokens
            day_stats['output_tokens'] += msg.output_tokens
            
            msg_total_tokens = (msg.input_tokens + msg.cache_creation_input_tokens + 
                              msg.cache_read_input_tokens + msg.output_tokens)
            day_stats['total_tokens'] += msg_total_tokens
            
            day_stats['sessions'].add(f"{msg.project_name}/{msg.session_id}")
            if msg.model:
                day_stats['models'].add(msg.model)
                
                # Track model-specific stats
                if 'sonnet' in msg.model.lower():
                    day_stats['sonnet_messages'] += 1
                    day_stats['sonnet_tokens'] += msg_total_tokens
                    day_stats['sonnet_cost'] += msg.cost_usd
                elif 'opus' in msg.model.lower():
                    day_stats['opus_messages'] += 1
                    day_stats['opus_tokens'] += msg_total_tokens
                    day_stats['opus_cost'] += msg.cost_usd
        
        # Sort by date (most recent first) and limit to num_days
        sorted_days = sorted(daily_stats.items(), key=lambda x: x[0], reverse=True)[:num_days]
        
        # Print header
        header_width = 220
        if gpu_hours:
            header_width += 12
        
        print("\n" + "-"*header_width)
        print(f"DAILY STATISTICS (Last {num_days} days)")
        print("-"*header_width)
        
        header = f"{'Date':<12} {'Sessions':>10} {'Cost':>10}"
        if gpu_hours:
            header += f" {'GPU Hours':>10}"
        header += f" {'Total Msgs':>12} {'Total Tokens':>14} │ {'Sonnet Msgs':>12} {'Sonnet Cost':>12} {'Sonnet Tokens':>14} │ {'Opus Msgs':>10} {'Opus Cost':>10} {'Opus Tokens':>12}"
        print(header)
        print("-"*header_width)
        
        total_cost = Decimal('0')
        total_messages = 0
        total_sessions = set()
        total_all_tokens = 0
        total_sonnet_messages = 0
        total_sonnet_tokens = 0
        total_sonnet_cost = Decimal('0')
        total_opus_messages = 0
        total_opus_tokens = 0
        total_opus_cost = Decimal('0')
        
        for date, stats in sorted_days:
            total_cost += stats['cost']
            total_messages += stats['messages']
            total_sessions.update(stats['sessions'])
            total_all_tokens += stats['total_tokens']
            total_sonnet_messages += stats['sonnet_messages']
            total_sonnet_tokens += stats['sonnet_tokens']
            total_sonnet_cost += stats['sonnet_cost']
            total_opus_messages += stats['opus_messages']
            total_opus_tokens += stats['opus_tokens']
            total_opus_cost += stats['opus_cost']
            
            line = f"{date.strftime('%Y-%m-%d'):<12} {len(stats['sessions']):>10} {self.format_currency(stats['cost'], currency, 2):>10}"
            if gpu_hours:
                gpu_hours_value = float(stats['cost']) / 8
                line += f" {gpu_hours_value:>10.4f}"
            line += f" {stats['messages']:>12,} {stats['total_tokens']:>14,} │ {stats['sonnet_messages']:>12,} {self.format_currency(stats['sonnet_cost'], currency, 2):>12} {stats['sonnet_tokens']:>14,} │ {stats['opus_messages']:>10,} {self.format_currency(stats['opus_cost'], currency, 2):>10} {stats['opus_tokens']:>12,}"
            print(line)
        
        # Print totals
        print("-"*header_width)
        total_line = f"{'TOTAL':<12} {len(total_sessions):>10} {self.format_currency(total_cost, currency, 2):>10}"
        if gpu_hours:
            total_gpu_hours = float(total_cost) / 8
            total_line += f" {total_gpu_hours:>10.4f}"
        total_line += f" {total_messages:>12,} {total_all_tokens:>14,} │ {total_sonnet_messages:>12,} {self.format_currency(total_sonnet_cost, currency, 2):>12} {total_sonnet_tokens:>14,} │ {total_opus_messages:>10,} {self.format_currency(total_opus_cost, currency, 2):>10} {total_opus_tokens:>12,}"
        print(total_line)
    
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
                           'Messages', 'Cost (USD)', 'Input Tokens', 'Cache Creation Tokens', 
                           'Cache Read Tokens', 'Output Tokens', 'Models Used'])
            
            for session_key, stats in sorted(self.sessions.items()):
                writer.writerow([
                    stats.project_name,
                    stats.session_id,
                    stats.start_time.astimezone(self.local_tz).strftime('%Y-%m-%d %H:%M:%S'),
                    stats.end_time.astimezone(self.local_tz).strftime('%Y-%m-%d %H:%M:%S'),
                    stats.total_messages,
                    float(stats.total_cost_usd),
                    stats.total_input_tokens,
                    stats.total_cache_creation_input_tokens,
                    stats.total_cache_read_input_tokens,
                    stats.total_output_tokens,
                    ', '.join(stats.models_used)
                ])
            
            writer.writerow([])
            
            # Write message details
            writer.writerow(['Message Details'])
            writer.writerow(['Timestamp', 'Project', 'Session', 'Role', 'Type', 
                           'Model', 'Cost (USD)', 'Input Tokens', 'Cache Creation Tokens',
                           'Cache Read Tokens', 'Output Tokens', 'Duration (ms)', 'Tool Name'])
            
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
                    msg.cache_creation_input_tokens,
                    msg.cache_read_input_tokens,
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
                'total_cache_creation_input_tokens': sum(msg.cache_creation_input_tokens for msg in self.messages),
                'total_cache_read_input_tokens': sum(msg.cache_read_input_tokens for msg in self.messages),
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
                'total_cache_creation_input_tokens': stats.total_cache_creation_input_tokens,
                'total_cache_read_input_tokens': stats.total_cache_read_input_tokens,
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
                'cache_creation_input_tokens': msg.cache_creation_input_tokens,
                'cache_read_input_tokens': msg.cache_read_input_tokens,
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
    parser.add_argument("--sort", choices=["cost", "date", "duration"], default="date",
                       help="Sort sessions by: cost, date, or duration (default: date)")
    parser.add_argument("--currency", choices=["USD", "EUR", "GBP"], default="USD",
                       help="Currency for display (default: USD)")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed message-level analysis")
    parser.add_argument("--tools", action="store_true",
                       help="Show tool usage statistics")
    parser.add_argument("--days", type=int, metavar="NUM",
                       help="Show daily aggregated statistics for the last NUM days")
    
    # Export options
    parser.add_argument("--csv", help="Export results to CSV file")
    parser.add_argument("--json", help="Export results to JSON file")
    
    # Session filtering
    parser.add_argument("--project", help="Filter by project name")
    parser.add_argument("--session", help="Filter by session ID")
    
    # Anthropic sessions
    parser.add_argument("--anthropic-sessions", action="store_true",
                       help="Show Anthropic 5-hour session windows instead of Claude Code sessions")
    parser.add_argument("--billing-day", type=int, metavar="DAY",
                       help="Day of month when your Anthropic billing cycle starts (1-31). Required when using --anthropic-sessions.")
    
    # GPU hours
    parser.add_argument("--gpu-hours", action="store_true",
                       help="Add GPU hours column (cost divided by 8)")
    
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
    
    # Calculate Anthropic sessions if requested
    if args.anthropic_sessions:
        if not args.billing_day:
            print("Error: --billing-day is required when using --anthropic-sessions")
            print("Please specify the day of the month when your Anthropic billing cycle starts (1-31)")
            print("Example: --billing-day 25")
            return
        
        if args.billing_day < 1 or args.billing_day > 31:
            print("Error: --billing-day must be between 1 and 31")
            return
            
        analyzer.calculate_anthropic_sessions()
    
    # Display results
    analyzer.print_summary(args.currency)
    
    if args.days:
        analyzer.print_daily_stats(args.days, args.currency, args.gpu_hours)
    elif args.anthropic_sessions:
        analyzer.print_anthropic_sessions(args.sessions, args.currency, args.sort, args.gpu_hours, args.billing_day)
    else:
        analyzer.print_session_details(args.sessions, args.currency, args.sort, args.gpu_hours)
    
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
                      f"{analyzer.format_currency(msg.cost_usd, args.currency, 2):>10}")
    
    # Export if requested
    if args.csv:
        analyzer.export_csv(args.csv)
    if args.json:
        analyzer.export_json(args.json)

if __name__ == "__main__":
    main()