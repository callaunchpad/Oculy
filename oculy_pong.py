import pygame
import sys
import os
import time
from pathlib import Path

# ------------------------------
# CONFIG
# ------------------------------
WIDTH, HEIGHT = 800, 600
# Paddle is wide and short for the bottom position
PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
BALL_SIZE = 20
PADDLE_SPEED = 14
BALL_SPEED_X, BALL_SPEED_Y = 2, 2

# File to read eye action labels from (written by live classification script)
EYE_ACTION_FILE = "/Users/neharavikumar/Oculy/game.txt"  # Placeholder filename - update with your actual file path

# This variable will be updated by reading from the file
last_eye_action = None  # "left", "right", "blink", etc.
last_file_mod_time = 0

# ------------------------------
# POSITION-BASED (INTEGRAL) CONTROL
# ------------------------------
# eye_position accumulates movements over time
# This helps because brief "return to center" movements 
# don't fully cancel out intentional sustained looks
eye_position = 0.0
EYE_SENSITIVITY = 8  # How much each frame of left/right affects position
DECAY_RATE = 0.98  # Position slowly decays toward center (1.0 = no decay)
MAX_EYE_OFFSET = None  # Will be set based on screen width


def read_eye_action_from_file():
    """Read the latest eye action label from the prediction file."""
    global last_eye_action, last_file_mod_time
    
    file_path = Path(EYE_ACTION_FILE)
    
    # Check if file exists
    if not file_path.exists():
        return None
    
    try:
        # Check if file has been modified
        current_mod_time = file_path.stat().st_mtime
        if current_mod_time <= last_file_mod_time:
            # File hasn't changed, return current action
            return last_eye_action
        
        # File has been updated, read the latest label
        last_file_mod_time = current_mod_time
        
        with open(file_path, 'r') as f:
            # Read the last line (most recent prediction)
            lines = f.readlines()
            print(lines)
            if lines:
                # # Get the last non-empty line
                # for line in reversed(lines):
                #     line = line.strip()
                #     if line:
                #         # Expected format: label or "label,confidence" or "[timestamp] label (confidence)"
                #         # Extract just the label part
                #         parts = line.split()
                #         if parts:
                #             # Try to find a label word (not numbers, not timestamps)
                #             for part in parts:
                #                 part_clean = part.strip('[],():').lower()
                #                 # Check if it's a known action
                #                 if part_clean in ['left', 'right', 'up', 'down', 'blink', 'stare', 'neutral']:
                #                     last_eye_action = part_clean
                #                     return last_eye_action
                #             # If no known action found, use the first word as fallback
                #             first_word = parts[0].strip('[],():').lower()
                #             print(first_word)
                            last_eye_action = lines[0]
                            # print(last_eye_action)
                            return last_eye_action
    except (IOError, OSError, PermissionError) as e:
        # File might be locked or inaccessible, silently ignore
        pass
    except Exception as e:
        # Other errors - print once to avoid spam
        if not hasattr(read_eye_action_from_file, '_error_printed'):
            print(f"Warning: Error reading {EYE_ACTION_FILE}: {e}")
            read_eye_action_from_file._error_printed = True
    
    return last_eye_action


def run_game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Oculy Pong (Bottom Paddle)")

    clock = pygame.time.Clock()

    # --- Paddle (Horizontal at the bottom) ---
    # The paddle's X position will change, Y position is fixed at the bottom
    paddle_x = WIDTH // 2 - PADDLE_WIDTH // 2
    paddle_y = HEIGHT - PADDLE_HEIGHT

    # --- Ball ---
    ball_x = WIDTH // 2 - BALL_SIZE // 2
    ball_y = HEIGHT // 2 - BALL_SIZE // 2

    global last_eye_action
    global BALL_SPEED_X, BALL_SPEED_Y
    global eye_position, MAX_EYE_OFFSET
    
    # Set max offset based on screen width (paddle can go edge to edge)
    MAX_EYE_OFFSET = (WIDTH - PADDLE_WIDTH) / 2

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # ------------------------------
        # EYE CONTROL → Paddle movement (Horizontal)
        # Using POSITION-BASED (INTEGRAL) control
        # ------------------------------
        current_action = read_eye_action_from_file()
        print(current_action)
        
        # Apply decay - eye_position slowly returns toward center
        # This helps filter out brief "return to center" movements
        eye_position *= DECAY_RATE
        
        # Accumulate eye position based on detected movement
        # Strip whitespace for clean comparison
        action = current_action.strip().lower() if current_action else ""
        
        if action == "left":
            eye_position -= EYE_SENSITIVITY
        elif action == "right":
            eye_position += EYE_SENSITIVITY
        elif action == "blink":
            # Optional: reset position to center on blink
            # eye_position = 0
            pass
        
        # Clamp eye_position to valid range
        eye_position = max(-MAX_EYE_OFFSET, min(MAX_EYE_OFFSET, eye_position))
        
        # Map eye_position to paddle_x (center of screen + offset)
        center_x = (WIDTH - PADDLE_WIDTH) / 2
        paddle_x = center_x + eye_position
        
        # Clamp paddle to the screen edges (redundant but safe)
        paddle_x = max(0, min(WIDTH - PADDLE_WIDTH, paddle_x))

        # ------------------------------
        # UPDATE BALL
        # ------------------------------
        ball_x += BALL_SPEED_X
        ball_y += BALL_SPEED_Y

        # Bounce on top and sides
        if ball_x <= 0 or ball_x >= WIDTH - BALL_SIZE:
            BALL_SPEED_X *= -1
        if ball_y <= 0: # Only bounce on the top wall
            BALL_SPEED_Y *= -1

        # Bounce on the bottom paddle
        # Check if ball hits the paddle's Y position AND is within the paddle's X range
        if (ball_y + BALL_SIZE >= paddle_y and # Ball reaches the paddle's Y position
            paddle_x <= ball_x + BALL_SIZE and # Ball's right edge is past paddle's left edge
            ball_x <= paddle_x + PADDLE_WIDTH): # Ball's left edge is before paddle's right edge

            # Reverse the vertical speed
            BALL_SPEED_Y *= -1

        # Reset if ball goes out (passes the bottom paddle)
        if ball_y > HEIGHT:
            # Reset ball position and speed
            ball_x, ball_y = WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2
            BALL_SPEED_X, BALL_SPEED_Y = 4, 4 # Reset speed for a new round
        

        # DRAW
        screen.fill((0, 0, 0)) # Black background
        
        # Draw the paddle (at the bottom)
        pygame.draw.rect(screen, (255,255,255), (paddle_x, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        
        # Draw the ball
        pygame.draw.rect(screen, (255,255,255), (ball_x, ball_y, BALL_SIZE, BALL_SIZE))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    print(f"Starting Oculy Pong...")
    print(f"Reading eye actions from: {EYE_ACTION_FILE}")
    print(f"Make sure your live classification script writes predictions to this file.")
    run_game()