"""Core environment class implementing the Gymnasium interface."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pygame
from gymnasium import Env, spaces
from gymnasium.core import ActType, ObsType

from yaml2gym.spec import EnvSpec, load_spec


@dataclass
class ObjectState:
    """Represents the state of an object in the environment."""
    type: str
    positions: List[Tuple[int, int]]
    properties: Dict[str, Any] = field(default_factory=dict)


class GenericGridEnv(Env):
    """A generic grid environment defined by a YAML specification."""
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }
    
    def __init__(
        self, 
        spec_path: str,
        render_mode: Optional[str] = None,
        render_size: Tuple[int, int] = (600, 600),
    ):
        """Initialize the environment from a YAML specification.
        
        Args:
            spec_path: Path to the YAML file containing the environment specification.
            render_mode: The render mode to use ('human' or 'rgb_array').
            render_size: The size of the rendering window in pixels (width, height).
        """
        super().__init__()
        
        # Load and validate the environment specification
        self.spec_dict = load_spec(spec_path)
        self.spec = self.spec_dict  # For compatibility with gym.Env
        
        # Environment parameters
        self.grid_size = self.spec_dict["grid_size"]
        self.max_steps = self.spec_dict["max_steps"]
        self.observation_type = self.spec_dict["observation_type"]
        
        # Initialize object states
        self.objects: Dict[str, ObjectState] = {}
        for obj in self.spec_dict["objects"]:
            self.objects[obj["type"]] = ObjectState(
                type=obj["type"],
                positions=[(p["x"], p["y"]) for p in obj["positions"]],
                properties=obj.get("properties", {}),
            )
        
        # Agent state
        self.agent_pos = (
            self.spec_dict["agent_start"]["x"],
            self.spec_dict["agent_start"]["y"]
        )
        self.step_count = 0
        
        # Action space
        self.action_space = spaces.Discrete(len(self.spec_dict["actions"]))
        self.action_names = [a["name"] for a in self.spec_dict["actions"]]
        
        # Observation space
        if self.observation_type == "grid":
            # One-hot encoding of object types + agent position
            num_object_types = len(self.objects) + 1  # +1 for the agent
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(self.grid_size[1], self.grid_size[0], num_object_types),
                dtype=np.float32,
            )
        elif self.observation_type == "dict":
            # Dictionary with agent position and object positions
            self.observation_space = spaces.Dict({
                "agent": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.grid_size[0]-1, self.grid_size[1]-1]),
                    dtype=np.int32,
                ),
                **{
                    obj_type: spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.grid_size[1], self.grid_size[0]),
                        dtype=np.float32,
                    )
                    for obj_type in self.objects
                }
            })
        else:  # flat
            # Flattened grid with one-hot encoding
            num_object_types = len(self.objects) + 1  # +1 for the agent
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(self.grid_size[0] * self.grid_size[1] * num_object_types,),
                dtype=np.float32,
            )
        
        # Rendering
        self.render_mode = render_mode
        self.render_size = render_size
        self.window_surface = None
        self.clock = None
        self.cell_size = (
            render_size[0] // self.grid_size[0],
            render_size[1] // self.grid_size[1],
        )
        
        # Colors for rendering
        self.colors = {
            "agent": (41, 128, 185),    # Blue
            "wall": (44, 62, 80),       # Dark gray
            "coin": (241, 196, 15),     # Yellow
            "lava": (231, 76, 60),      # Red
            "goal": (39, 174, 96),      # Green
            "empty": (236, 240, 241),   # Light gray
            "grid": (189, 195, 199),    # Medium gray
        }
        
        # Initialize Pygame if rendering
        if self.render_mode == "human":
            self._init_render()
    
    def _init_render(self):
        """Initialize Pygame for rendering."""
        pygame.init()
        pygame.display.set_caption(f"YAML2Gym - {self.spec_dict['name']}")
        self.window_surface = pygame.display.set_mode(self.render_size)
        self.clock = pygame.time.Clock()
    
    def _get_obs(self) -> ObsType:
        """Return the current observation."""
        if self.observation_type == "grid":
            # Create a 3D array with one channel per object type + 1 for the agent
            obs = np.zeros((self.grid_size[1], self.grid_size[0], len(self.objects) + 1), dtype=np.float32)
            
            # Mark agent position
            obs[self.agent_pos[1], self.agent_pos[0], 0] = 1.0
            
            # Mark object positions
            for i, (obj_type, obj) in enumerate(self.objects.items(), 1):
                for x, y in obj.positions:
                    obs[y, x, i] = 1.0
            
            return obs
            
        elif self.observation_type == "dict":
            obs = {
                "agent": np.array(self.agent_pos, dtype=np.int32),
            }
            
            # Create a grid for each object type
            for obj_type, obj in self.objects.items():
                grid = np.zeros((self.grid_size[1], self.grid_size[0]), dtype=np.float32)
                for x, y in obj.positions:
                    grid[y, x] = 1.0
                obs[obj_type] = grid
            
            return obs
            
        else:  # flat
            # Create a flat one-hot encoded observation
            num_object_types = len(self.objects) + 1
            obs = np.zeros(self.grid_size[0] * self.grid_size[1] * num_object_types, dtype=np.float32)
            
            # Mark agent position
            idx = (self.agent_pos[1] * self.grid_size[0] + self.agent_pos[0]) * num_object_types
            obs[idx] = 1.0
            
            # Mark object positions
            for i, (obj_type, obj) in enumerate(self.objects.items(), 1):
                for x, y in obj.positions:
                    idx = (y * self.grid_size[0] + x) * num_object_types + i
                    obs[idx] = 1.0
            
            return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Return additional information about the environment state."""
        return {
            "step": self.step_count,
            "agent_pos": self.agent_pos,
            "objects": {
                obj_type: {
                    "positions": obj.positions,
                    "properties": obj.properties,
                }
                for obj_type, obj in self.objects.items()
            },
        }
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[ObsType, Dict]:
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        
        # Reset object states
        for obj in self.spec_dict["objects"]:
            self.objects[obj["type"]] = ObjectState(
                type=obj["type"],
                positions=[(p["x"], p["y"]) for p in obj["positions"]],
                properties=obj.get("properties", {}).copy(),
            )
        
        # Reset agent state
        self.agent_pos = (
            self.spec_dict["agent_start"]["x"],
            self.spec_dict["agent_start"]["y"]
        )
        self.step_count = 0
        
        # Reset rendering if needed
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict]:
        """Run one timestep of the environment's dynamics."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Get action effect
        action_spec = self.spec_dict["actions"][action]
        effect = action_spec["effect"]
        
        # Parse and apply action effect from YAML
        dx, dy = 0, 0
        
        # Support both simple string effects and complex effect objects
        if isinstance(effect, dict):
            # Complex effect with parameters
            effect_type = effect.get('type', '')
            effect_params = effect.get('params', {})
            
            if effect_type == 'move':
                # Move effect with relative coordinates
                dx = effect_params.get('dx', 0)
                dy = effect_params.get('dy', 0)
            elif effect_type == 'teleport':
                # Teleport to absolute coordinates
                if 'x' in effect_params and 'y' in effect_params:
                    self.agent_pos = (effect_params['x'], effect_params['y'])
                    return  # Skip movement logic since we teleported
            elif effect_type == 'modify_property':
                # Modify an object's property
                obj_type = effect_params.get('object')
                prop_name = effect_params.get('property')
                value = effect_params.get('value')
                op = effect_params.get('op', 'set')  # 'set', 'add', 'multiply', etc.
                
                if obj_type in self.objects and prop_name is not None:
                    obj = self.objects[obj_type]
                    if op == 'set':
                        obj.properties[prop_name] = value
                    elif op == 'add':
                        obj.properties[prop_name] = obj.properties.get(prop_name, 0) + value
                    elif op == 'multiply':
                        obj.properties[prop_name] = obj.properties.get(prop_name, 1) * value
        else:
            # Simple string effect (backward compatibility)
            if effect == "up":
                dy = -1
            elif effect == "down":
                dy = 1
            elif effect == "left":
                dx = -1
            elif effect == "right":
                dx = 1
        
        # Calculate new position
        new_x = max(0, min(self.grid_size[0] - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.grid_size[1] - 1, self.agent_pos[1] + dy))
        
        # Check for collisions with walls
        can_move = True
        if "wall" in self.objects:
            if (new_x, new_y) in self.objects["wall"].positions:
                can_move = False
        
        # Update agent position if movement is allowed
        if can_move:
            self.agent_pos = (new_x, new_y)
        
        # Process object interactions
        reward = 0.0
        terminated = False
        truncated = False
        
        # Get objects at current position (except walls)
        objects_here = []
        for obj_type, obj in self.objects.items():
            if obj_type == "wall":
                continue
                
            if self.agent_pos in obj.positions:
                objects_here.append((obj_type, obj))
        
        # Process interactions based on object properties
        for obj_type, obj in objects_here:
            props = obj.properties
            
            # Handle collectable objects
            if props.get('collectable', False):
                obj.positions.remove(self.agent_pos)
                reward += float(props.get('reward', 1.0))
            
            # Handle damage/health effects
            if 'damage' in props:
                reward -= float(props['damage'])
            
            # Handle termination conditions
            if props.get('terminate_on_contact', False):
                terminated = True
            
            # Handle property modifications
            for prop, value in props.items():
                if prop.startswith('modify_') and ':' in prop:
                    # Format: modify_<property>:<target_object>:<operation>
                    _, target_prop, target_obj, op = prop.split(':')
                    if target_obj in self.objects:
                        target = self.objects[target_obj]
                        if op == 'add':
                            target.properties[target_prop] = target.properties.get(target_prop, 0) + value
                        elif op == 'set':
                            target.properties[target_prop] = value
        
        # Increment step count and check for truncation
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _render_frame(self):
        """Render a single frame of the environment."""
        if self.window_surface is None:
            self._init_render()
        
        # Clear the screen
        self.window_surface.fill(self.colors["empty"])
        
        # Draw grid lines
        for x in range(0, self.render_size[0], self.cell_size[0]):
            pygame.draw.line(
                self.window_surface,
                self.colors["grid"],
                (x, 0),
                (x, self.render_size[1]),
            )
        for y in range(0, self.render_size[1], self.cell_size[1]):
            pygame.draw.line(
                self.window_surface,
                self.colors["grid"],
                (0, y),
                (self.render_size[0], y),
            )
        
        # Draw objects
        for obj_type, obj in self.objects.items():
            color = self.colors.get(obj_type, (200, 200, 200))  # Default to gray
            for x, y in obj.positions:
                rect = pygame.Rect(
                    x * self.cell_size[0],
                    y * self.cell_size[1],
                    self.cell_size[0],
                    self.cell_size[1],
                )
                pygame.draw.rect(self.window_surface, color, rect)
                pygame.draw.rect(self.window_surface, (0, 0, 0), rect, 1)  # Border
        
        # Draw agent
        agent_rect = pygame.Rect(
            self.agent_pos[0] * self.cell_size[0],
            self.agent_pos[1] * self.cell_size[1],
            self.cell_size[0],
            self.cell_size[1],
        )
        pygame.draw.rect(self.window_surface, self.colors["agent"], agent_rect)
        pygame.draw.rect(self.window_surface, (0, 0, 0), agent_rect, 2)  # Border
        
        # Update the display
        pygame.display.update()
        
        # Maintain frame rate
        self.clock.tick(self.metadata["render_fps"])
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # Create a surface for rendering
            surface = pygame.Surface(self.render_size)
            
            # Save the current window surface and replace with our surface
            old_surface = self.window_surface
            self.window_surface = surface
            
            # Render the frame
            self._render_frame()
            
            # Restore the original window surface
            self.window_surface = old_surface
            
            # Return the rendered frame as a numpy array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
            )
        elif self.render_mode == "human":
            self._render_frame()
    
    def close(self):
        """Close the environment and release resources."""
        if self.window_surface is not None:
            pygame.quit()
            self.window_surface = None
            self.clock = None
