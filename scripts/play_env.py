#!/usr/bin/env python3
"""Play a YAML2Gym environment using keyboard controls."""
import argparse
import sys
from pathlib import Path

import pygame

# Add the parent directory to the path so we can import yaml2gym
sys.path.insert(0, str(Path(__file__).parent.parent))

from yaml2gym.env import GenericGridEnv


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Play a YAML2Gym environment")
    parser.add_argument(
        "spec_file",
        type=str,
        help="Path to the YAML environment specification file",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        nargs=2,
        default=[600, 600],
        metavar=("WIDTH", "HEIGHT"),
        help="Window size for rendering (default: 600 600)",
    )
    args = parser.parse_args()
    
    # Create the environment
    env = GenericGridEnv(
        spec_path=args.spec_file,
        render_mode="human",
        render_size=tuple(args.render_size),
    )
    
    # Reset the environment
    obs, info = env.reset()
    
    # Action mapping (key: action_index)
    action_mapping = {
        pygame.K_UP: 0,    # up
        pygame.K_DOWN: 1,  # down
        pygame.K_LEFT: 2,  # left
        pygame.K_RIGHT: 3, # right
        pygame.K_SPACE: 4, # stay/noop
    }
    
    # Game loop
    running = True
    total_reward = 0.0
    
    print("\n=== YAML2Gym Environment ===")
    print(f"Environment: {env.spec_dict['name']}")
    print(f"Grid size: {env.grid_size[0]}x{env.grid_size[1]}")
    print("\nControls:")
    print("  ↑/↓/←/→ : Move")
    print("  SPACE   : Stay")
    print("  R       : Reset")
    print("  Q/ESC   : Quit")
    print("\nCollect coins (yellow), avoid lava (red), reach the goal (green).")
    print("=" * 30)
    
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Check for quit keys
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                # Check for reset key
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
                    print("\nEnvironment reset!")
                # Check for action keys
                elif event.key in action_mapping:
                    action = action_mapping[event.key]
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    
                    print(f"Step: {info['step']}, "
                          f"Reward: {reward:.2f}, "
                          f"Total: {total_reward:.2f}, "
                          f"Position: {info['agent_pos']}")
                    
                    if terminated or truncated:
                        if terminated:
                            print("\nEpisode finished!")
                        else:
                            print("\nEpisode truncated (max steps reached)!")
                        
                        print(f"Total reward: {total_reward:.2f}")
                        print("Press R to reset or Q to quit.")
        
        # Render the environment
        env.render()
        
        # Cap the frame rate
        pygame.time.Clock().tick(10)
    
    # Clean up
    env.close()
    pygame.quit()
    print("\nThanks for playing!")


if __name__ == "__main__":
    main()
