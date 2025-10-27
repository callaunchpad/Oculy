#!/usr/bin/env python3
"""
Keypress Data Recorder
Records keyboard input at 1000Hz (1ms intervals) for eye tracking label generation.

Maps:
- No keypress -> "stare"
- A -> "left"
- W -> "up"
- S -> "down"
- D -> "right"
- X -> "unknown"
- Space -> "blink"

Press ESC to stop recording.
"""

import time
import threading
from datetime import datetime
from pathlib import Path
from pynput import keyboard


class KeypressRecorder:
    def __init__(self, output_file=None):
        """Initialize the keypress recorder."""
        self.current_state = "stare"
        self.is_recording = True
        self.lock = threading.Lock()
        
        # Set up output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_file = f"keypress_labels_{timestamp}.txt"
        
        self.output_file = Path(output_file)
        self.start_time = None
        self.sample_count = 0
        
        # Key mapping
        self.key_map = {
            'a': 'left',
            'w': 'up',
            's': 'down',
            'd': 'right',
            'x': 'unknown',
            ' ': 'blink'  # space
        }
    
    def on_press(self, key):
        """Handle key press events."""
        # Handle ESC key first
        if key == keyboard.Key.esc:
            self.is_recording = False
            return False
        
        # Handle space key
        if key == keyboard.Key.space:
            with self.lock:
                self.current_state = 'blink'
            return
        
        # Handle character keys
        try:
            if hasattr(key, 'char') and key.char:
                char = key.char.lower()
                if char in self.key_map:
                    with self.lock:
                        self.current_state = self.key_map[char]
        except AttributeError:
            pass
    
    def on_release(self, key):
        """Handle key release events - return to stare state."""
        # Handle ESC key
        if key == keyboard.Key.esc:
            return False
        
        # Handle space key release
        if key == keyboard.Key.space:
            with self.lock:
                self.current_state = 'stare'
            return
        
        # Handle character key releases
        try:
            if hasattr(key, 'char') and key.char:
                char = key.char.lower()
                if char in self.key_map:
                    with self.lock:
                        self.current_state = 'stare'
        except AttributeError:
            pass
    
    def write_header(self, file_handle):
        """Write file header with metadata."""
        header = f"""# Eye Tracking Keypress Labels
# Recording started: {self.start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}
# Sampling rate: 1000 Hz (1 sample per millisecond)
# Columns: sample_number, timestamp_ms, elapsed_ms, label
# Labels: stare, left, up, down, right, unknown, blink
# Press ESC to stop recording
# EndOfHeader
"""
        file_handle.write(header)
    
    def record(self, duration=None):
        """
        Start recording keypress data.
        
        Args:
            duration: Maximum recording duration in seconds (None for unlimited)
        """
        self.start_time = datetime.now()
        
        print("=" * 60)
        print("KEYPRESS RECORDER - Eye Tracking Label Generation")
        print("=" * 60)
        print(f"Output file: {self.output_file}")
        print(f"Sampling rate: 1000 Hz (1 sample per millisecond)")
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Key mappings:")
        print("  A = left")
        print("  W = up")
        print("  S = down")
        print("  D = right")
        print("  X = unknown")
        print("  SPACE = blink")
        print("  (no key) = stare")
        print()
        print("Press ESC to stop recording")
        print("=" * 60)
        print()
        
        # Start keyboard listener in background
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        listener.start()
        
        # Open output file
        with open(self.output_file, 'w') as f:
            self.write_header(f)
            
            # Recording loop at 1000Hz
            start_timestamp = time.time()
            next_sample_time = start_timestamp
            
            try:
                while self.is_recording:
                    current_time = time.time()
                    
                    # Check if it's time for the next sample
                    if current_time >= next_sample_time:
                        # Get current state (thread-safe)
                        with self.lock:
                            state = self.current_state
                        
                        # Calculate timestamps
                        elapsed_ms = (current_time - start_timestamp) * 1000
                        timestamp_ms = current_time * 1000
                        
                        # Write data row
                        f.write(f"{self.sample_count}\t{timestamp_ms:.3f}\t{elapsed_ms:.3f}\t{state}\n")
                        
                        self.sample_count += 1
                        
                        # Schedule next sample (1ms = 0.001s)
                        next_sample_time += 0.001
                        
                        # Check duration limit
                        if duration and elapsed_ms >= duration * 1000:
                            break
                        
                        # Periodic status update
                        if self.sample_count % 1000 == 0:
                            print(f"Samples recorded: {self.sample_count} ({elapsed_ms/1000:.1f}s) - Current: {state}")
                    
                    # Small sleep to prevent CPU spinning
                    time.sleep(0.0001)
                    
            except KeyboardInterrupt:
                print("\n\nRecording interrupted by user (Ctrl+C)")
            
            finally:
                self.is_recording = False
                listener.stop()
        
        # Print summary
        end_time = datetime.now()
        duration_seconds = (end_time - self.start_time).total_seconds()
        
        print()
        print("=" * 60)
        print("Recording Complete")
        print("=" * 60)
        print(f"Total samples: {self.sample_count}")
        print(f"Duration: {duration_seconds:.2f} seconds")
        print(f"Average sampling rate: {self.sample_count/duration_seconds:.1f} Hz")
        print(f"Output file: {self.output_file}")
        print(f"File size: {self.output_file.stat().st_size / 1024:.1f} KB")
        print("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Record keyboard input at 1000Hz for eye tracking labels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record indefinitely (stop with ESC)
  python record_keypresses.py
  
  # Record for 30 seconds
  python record_keypresses.py --duration 30
  
  # Specify custom output file
  python record_keypresses.py --output my_recording.txt
  
Key mappings:
  A = left
  W = up
  S = down
  D = right
  X = unknown
  SPACE = blink
  (no key) = stare
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path (default: keypress_labels_TIMESTAMP.txt)'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=None,
        help='Recording duration in seconds (default: unlimited, stop with ESC)'
    )
    
    args = parser.parse_args()
    
    # Create recorder and start recording
    recorder = KeypressRecorder(output_file=args.output)
    recorder.record(duration=args.duration)


if __name__ == '__main__':
    main()


