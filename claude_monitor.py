#!/usr/bin/env python3
"""
Claude Code Message Flow Monitor
A terminal-based utility to watch message flows to/from the LLM in real-time.
Supports monitoring multiple sessions simultaneously.
"""

import json
import time
import sys
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
import argparse
from typing import Optional, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import signal

class MessageMonitor:
    def __init__(self, log_file: Optional[str] = None, follow: bool = True, show_full: bool = False, show_tools: bool = False):
        self.log_file = log_file
        self.follow = follow
        self.show_full = show_full
        self.show_tools = show_tools or show_full  # --full implies --tools
        
        # Get local timezone
        self.local_tz = datetime.now().astimezone().tzinfo
        self.colors = {
            'user': '\033[94m',      # Blue
            'assistant': '\033[92m', # Green
            'system': '\033[93m',    # Yellow
            'error': '\033[91m',     # Red
            'reset': '\033[0m',      # Reset
            'bold': '\033[1m',       # Bold
            'dim': '\033[2m',        # Dim
            'cyan': '\033[96m',      # Cyan
            'magenta': '\033[95m'    # Magenta
        }
        self.message_queue = Queue()
        self.active_files: Dict[str, threading.Thread] = {}
        self.file_positions: Dict[str, int] = {}
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.running = True
        self.lock = threading.Lock()
        self.claude_dir = Path.home() / ".claude" / "projects"
    
    def _find_claude_logs(self) -> Optional[str]:
        """Try to find Claude Code log files in common locations"""
        claude_dir = Path.home() / ".claude" / "projects"
        
        if claude_dir.exists():
            # Find the most recent .jsonl file
            jsonl_files = list(claude_dir.glob("*/*.jsonl"))
            if jsonl_files:
                # Return the most recently modified file
                return str(max(jsonl_files, key=lambda p: p.stat().st_mtime))
        
        # Fallback to other locations
        possible_paths = [
            Path.home() / ".claude" / "logs" / "messages.log",
            Path.home() / ".config" / "claude" / "logs" / "messages.log",
            Path("/tmp/claude_messages.log"),
            Path("./claude_messages.log")
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def colorize(self, text: str, color: str) -> str:
        """Add color to text if terminal supports it"""
        if sys.stdout.isatty():
            return f"{self.colors.get(color, '')}{text}{self.colors['reset']}"
        return text
    
    def format_timestamp(self, timestamp: Optional[str] = None) -> str:
        """Format timestamp for display in local timezone"""
        if not timestamp:
            timestamp = datetime.now().isoformat()
        
        try:
            # Parse ISO format with Z as UTC
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Ensure we have a timezone-aware datetime in UTC
            if dt.tzinfo is None:
                # Assume UTC if no timezone info
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                # Convert to UTC if not already
                dt = dt.astimezone(timezone.utc)
            
            # Convert to local timezone for display
            local_dt = dt.astimezone(self.local_tz)
            return local_dt.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        except:
            return timestamp[:12] if len(timestamp) > 12 else timestamp
    
    def format_message(self, message: Dict[str, Any]) -> str:
        """Format a message for display"""
        msg_type = message.get('type', 'unknown')
        timestamp = self.format_timestamp(message.get('timestamp'))
        
        # Handle Claude Code message structure
        if 'message' in message:
            inner_msg = message['message']
            role = inner_msg.get('role', 'unknown')
            content = inner_msg.get('content', '')
            
            # Extract text content from Claude messages
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get('type') == 'text':
                            text_parts.append(part.get('text', ''))
                        elif part.get('type') == 'tool_use':
                            tool_name = part.get('name', 'unknown_tool')
                            tool_input = part.get('input', {})
                            tool_id = part.get('id', '')
                            
                            # Show tool data if requested
                            if self.show_tools:
                                input_str = json.dumps(tool_input, indent=2) if tool_input else 'No input'
                                text_parts.append(f"[Tool Use: {tool_name}] (ID: {tool_id[:8]}...)\n                Input: {input_str}")
                            else:
                                text_parts.append(f"[Tool Use: {tool_name}]")
                                
                        elif part.get('type') == 'tool_result':
                            tool_name = part.get('name', 'unknown_tool')
                            tool_id = part.get('tool_use_id', '')
                            result = part.get('content', '')
                            cost = part.get('costUSD', 0)
                            duration = part.get('durationMs', 0)
                            is_error = part.get('is_error', False)
                            cost_info = f", Cost: ${cost:.4f}, Duration: {duration}ms" if cost > 0 else ""
                            error_info = " [ERROR]" if is_error else ""
                            
                            # Show tool results if requested
                            if self.show_tools:
                                result_str = str(result)
                                if not self.show_full and len(result_str) > 500:
                                    result_str = result_str[:500] + "..."
                                text_parts.append(f"[Tool Result: {tool_name}{error_info}] (ID: {tool_id[:8]}...{cost_info})\n                {result_str}")
                            else:
                                text_parts.append(f"[Tool Result: {tool_name}{error_info}{cost_info}]")
                content = '\n'.join(text_parts)
            
            # Add cost and duration info for assistant messages
            if role == 'assistant' and msg_type == 'assistant':
                cost = message.get('costUSD', 0)
                duration = message.get('durationMs', 0)
                model = inner_msg.get('model', 'unknown')
                if cost > 0:
                    content += f"\n[Cost: ${cost:.4f}, Duration: {duration}ms, Model: {model}]"
        else:
            # Fallback for other message formats
            role = message.get('role', msg_type)
            content = str(message.get('content', message))
        
        # Truncate long content only if not showing full
        if not self.show_full and len(content) > 300:
            content = content[:297] + "..."
        
        # Check if this is actually a tool result disguised as a user message
        is_tool_result = False
        if role == 'user' and isinstance(content, str):
            # Check for patterns that indicate this is a tool result
            tool_patterns = [
                "Based on the Trustpilot reviews",
                "Based on the search results",
                "I found the following",
                "Here are the search results",
                "According to the documentation"
            ]
            if any(pattern in content for pattern in tool_patterns):
                is_tool_result = True
        
        # Format based on role/type
        if is_tool_result:
            header = self.colorize(f"[{timestamp}] SUB-AGENT RESULT", 'system')
        elif role == 'user':
            header = self.colorize(f"[{timestamp}] USER", 'user')
        elif role == 'assistant':
            header = self.colorize(f"[{timestamp}] ASSISTANT", 'assistant')
        elif msg_type == 'system':
            header = self.colorize(f"[{timestamp}] SYSTEM", 'system')
        else:
            header = self.colorize(f"[{timestamp}] {role.upper()}", 'dim')
        
        # Format content with proper line breaks
        content_lines = content.split('\n')
        formatted_content = []
        max_lines = None if self.show_full else 8
        
        for i, line in enumerate(content_lines):
            if max_lines and i >= max_lines:
                break
            if i == 0:
                formatted_content.append(f"{header}: {line}")
            else:
                formatted_content.append(f"{'':>15} {line}")
        
        if max_lines and len(content_lines) > max_lines:
            formatted_content.append(f"{'':>15} ... ({len(content_lines)-max_lines} more lines)")
        
        return '\n'.join(formatted_content)
    
    def get_session_name(self, file_path: str) -> str:
        """Extract a readable session name from the file path"""
        path = Path(file_path)
        # Get the parent directory name (project name) and filename
        project_name = path.parent.name
        file_name = path.stem  # filename without extension
        return f"{project_name}/{file_name}"
    
    def monitor_single_file(self, file_path: str):
        """Monitor a single log file for new messages"""
        session_name = self.get_session_name(file_path)
        
        try:
            # Get initial file position
            with self.lock:
                if file_path not in self.file_positions:
                    self.file_positions[file_path] = 0
                    if self.follow:
                        # For new files in follow mode, seek to end
                        try:
                            self.file_positions[file_path] = Path(file_path).stat().st_size
                        except:
                            pass
            
            with open(file_path, 'r') as f:
                # Seek to the stored position
                f.seek(self.file_positions[file_path])
                
                while self.running and file_path in self.active_files:
                    line = f.readline()
                    if line:
                        # Update position
                        with self.lock:
                            self.file_positions[file_path] = f.tell()
                        
                        try:
                            message = json.loads(line.strip())
                            self.message_queue.put({
                                'session': session_name,
                                'message': message,
                                'file_path': file_path
                            })
                        except json.JSONDecodeError:
                            # Handle non-JSON lines
                            self.message_queue.put({
                                'session': session_name,
                                'raw': line.strip(),
                                'file_path': file_path
                            })
                    else:
                        if not self.follow:
                            break
                        time.sleep(0.1)
        
        except FileNotFoundError:
            # File was deleted
            with self.lock:
                if file_path in self.active_files:
                    self.message_queue.put({
                        'session': session_name,
                        'event': 'file_deleted',
                        'file_path': file_path
                    })
        except Exception as e:
            self.message_queue.put({
                'session': session_name,
                'error': str(e),
                'file_path': file_path
            })
        finally:
            # Clean up
            with self.lock:
                if file_path in self.active_files:
                    del self.active_files[file_path]
    
    def scan_for_files(self):
        """Continuously scan for new .jsonl files in the Claude projects directory"""
        while self.running:
            try:
                if self.claude_dir.exists():
                    # Find all .jsonl files
                    current_files = set()
                    for jsonl_file in self.claude_dir.glob("*/*.jsonl"):
                        current_files.add(str(jsonl_file))
                    
                    # Start monitoring new files
                    with self.lock:
                        for file_path in current_files:
                            if file_path not in self.active_files and Path(file_path).exists():
                                # Start a new thread to monitor this file
                                thread = threading.Thread(
                                    target=self.monitor_single_file,
                                    args=(file_path,),
                                    daemon=True
                                )
                                self.active_files[file_path] = thread
                                thread.start()
                                
                                session_name = self.get_session_name(file_path)
                                self.message_queue.put({
                                    'session': session_name,
                                    'event': 'new_session',
                                    'file_path': file_path
                                })
                        
                        # Check for deleted files
                        deleted_files = set(self.active_files.keys()) - current_files
                        for file_path in deleted_files:
                            if file_path in self.active_files:
                                # Thread will detect the deletion and clean up
                                pass
                
                time.sleep(2)  # Check for new files every 2 seconds
            except Exception as e:
                print(self.colorize(f"Scanner error: {e}", 'error'))
                time.sleep(5)
    
    def display_messages(self):
        """Display messages from the queue"""
        while self.running:
            try:
                item = self.message_queue.get(timeout=0.1)
                
                if 'event' in item:
                    # Handle special events
                    if item['event'] == 'new_session':
                        print(self.colorize(f"\n📁 New session detected: {item['session']}", 'cyan'))
                        print(self.colorize("-" * 80, 'dim'))
                    elif item['event'] == 'file_deleted':
                        print(self.colorize(f"\n❌ Session ended: {item['session']}", 'error'))
                        print(self.colorize("-" * 80, 'dim'))
                
                elif 'error' in item:
                    print(self.colorize(f"\n⚠️  Error in {item['session']}: {item['error']}", 'error'))
                
                elif 'raw' in item:
                    # Raw non-JSON line
                    timestamp = self.format_timestamp()
                    session_prefix = self.colorize(f"[{item['session']}]", 'magenta')
                    print(f"{session_prefix} {self.colorize(f'[{timestamp}] RAW', 'dim')}: {item['raw']}")
                
                elif 'message' in item:
                    # Regular message
                    formatted = self.format_message(item['message'])
                    session_prefix = self.colorize(f"[{item['session']}]", 'magenta')
                    
                    # Add session prefix to each line
                    lines = formatted.split('\n')
                    prefixed_lines = []
                    for i, line in enumerate(lines):
                        if i == 0:
                            prefixed_lines.append(f"{session_prefix} {line}")
                        else:
                            # Align subsequent lines
                            padding = ' ' * (len(item['session']) + 3)
                            prefixed_lines.append(f"{padding}{line}")
                    
                    print('\n'.join(prefixed_lines))
                    print()
                
            except Empty:
                continue
            except Exception as e:
                print(self.colorize(f"Display error: {e}", 'error'))
    
    def monitor_multiple(self):
        """Monitor multiple log files simultaneously"""
        print(self.colorize("Claude Code Multi-Session Monitor", 'bold'))
        print(self.colorize(f"Watching directory: {self.claude_dir}", 'dim'))
        print(self.colorize("Press Ctrl+C to stop", 'dim'))
        print(self.colorize("=" * 80, 'dim'))
        
        # Set up signal handler for clean shutdown
        def signal_handler(sig, frame):
            self.running = False
            print(self.colorize("\n\nShutting down monitor...", 'dim'))
            self.executor.shutdown(wait=False)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Start the directory scanner
            scanner_thread = threading.Thread(target=self.scan_for_files, daemon=True)
            scanner_thread.start()
            
            # Start the display thread
            self.display_messages()
            
        except Exception as e:
            print(self.colorize(f"Monitor error: {e}", 'error'))
        finally:
            self.running = False
            self.executor.shutdown(wait=True)
    
    def monitor_file(self):
        """Monitor log file(s) - single file mode for backward compatibility"""
        if self.log_file:
            # Single file mode
            if not Path(self.log_file).exists():
                print(self.colorize("Log file not found. Creating mock data stream...", 'error'))
                self.create_mock_stream()
                return
            
            print(self.colorize(f"Monitoring: {self.log_file}", 'bold'))
            print(self.colorize("Press Ctrl+C to stop", 'dim'))
            print("-" * 80)
            
            # Use the multi-file infrastructure for single file
            with self.lock:
                thread = threading.Thread(
                    target=self.monitor_single_file,
                    args=(self.log_file,),
                    daemon=True
                )
                self.active_files[self.log_file] = thread
                thread.start()
            
            try:
                self.display_messages()
            except KeyboardInterrupt:
                print(self.colorize("\nMonitoring stopped.", 'dim'))
            finally:
                self.running = False
        else:
            # Multi-file mode
            self.monitor_multiple()
    
    def create_mock_stream(self):
        """Create a mock message stream for demonstration"""
        mock_messages = [
            {"role": "user", "content": "Hello Claude, can you help me with a Python script?", "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": "I'd be happy to help you with a Python script! What would you like to create?", "timestamp": datetime.now().isoformat()},
            {"role": "user", "content": "I need a script to monitor log files in real-time", "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": "Great! I can help you create a log file monitor. Let me create a Python script that can tail log files and display new entries as they appear...", "timestamp": datetime.now().isoformat()},
        ]
        
        print(self.colorize("Demo Mode - Showing sample message flow", 'bold'))
        print("-" * 80)
        
        try:
            for message in mock_messages:
                print(self.format_message(message))
                print()
                time.sleep(2)
            
            print(self.colorize("Demo complete. Use with actual log file for real monitoring.", 'dim'))
            
        except KeyboardInterrupt:
            print(self.colorize("\nDemo stopped.", 'dim'))

def main():
    parser = argparse.ArgumentParser(description="Monitor Claude Code message flows")
    parser.add_argument("-f", "--file", help="Log file to monitor (single file mode)")
    parser.add_argument("-m", "--multi", action="store_true", help="Monitor multiple sessions simultaneously")
    parser.add_argument("-n", "--no-follow", action="store_true", help="Don't follow file, just read existing content")
    parser.add_argument("-d", "--demo", action="store_true", help="Run in demo mode with mock data")
    parser.add_argument("--full", action="store_true", help="Show full content without truncation (implies --tools)")
    parser.add_argument("--tools", action="store_true", help="Show detailed tool inputs and outputs")
    
    args = parser.parse_args()
    
    if args.demo:
        monitor = MessageMonitor(show_full=args.full, show_tools=args.tools)
        monitor.create_mock_stream()
    elif args.multi or (not args.file and not args.demo):
        # Multi-session mode (default if no file specified)
        monitor = MessageMonitor(follow=not args.no_follow, show_full=args.full, show_tools=args.tools)
        monitor.monitor_file()  # This will automatically use multi-mode when log_file is None
    else:
        # Single file mode
        monitor = MessageMonitor(log_file=args.file, follow=not args.no_follow, show_full=args.full, show_tools=args.tools)
        monitor.monitor_file()

if __name__ == "__main__":
    main()