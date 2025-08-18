"""YAML schema and parser for environment specifications."""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import yaml
from typing_extensions import TypedDict


class Position(TypedDict):
    x: int
    y: int


class ObjectSpec(TypedDict):
    type: str
    positions: List[Position]
    properties: Optional[Dict[str, Union[bool, int, float, str]]]


class ActionSpec(TypedDict):
    name: str
    effect: str


class RewardRule(TypedDict):
    condition: str
    value: float


class TerminationRule(TypedDict):
    condition: str
    done: bool
    success: Optional[bool]


class EnvSpec(TypedDict):
    name: str
    grid_size: Tuple[int, int]
    objects: List[ObjectSpec]
    agent_start: Position
    actions: List[ActionSpec]
    rewards: List[RewardRule]
    terminations: List[TerminationRule]
    max_steps: int
    observation_type: Literal["grid", "dict", "flat"]


class EnvSpecError(ValueError):
    """Raised when there is an error in the environment specification."""
    pass


def load_spec(yaml_path: Union[str, Path]) -> EnvSpec:
    """Load and validate an environment specification from a YAML file.
    
    Args:
        yaml_path: Path to the YAML file containing the environment specification.
        
    Returns:
        A validated environment specification.
        
    Raises:
        EnvSpecError: If the specification is invalid.
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path, 'r') as f:
        spec = yaml.safe_load(f)
    
    # Basic validation
    required_fields = [
        'name', 'grid_size', 'objects', 'agent_start', 
        'actions', 'rewards', 'terminations', 'max_steps',
        'observation_type'
    ]
    
    for field in required_fields:
        if field not in spec:
            raise EnvSpecError(f"Missing required field: {field}")
    
    # Validate grid_size
    try:
        grid_size = tuple(spec['grid_size'])
        if len(grid_size) != 2 or not all(isinstance(x, int) and x > 0 for x in grid_size):
            raise EnvSpecError("grid_size must be a list of two positive integers [width, height]")
        spec['grid_size'] = grid_size
    except (TypeError, ValueError) as e:
        raise EnvSpecError(f"Invalid grid_size: {e}")
    
    # Validate agent_start
    try:
        agent_start = spec['agent_start']
        if not all(k in agent_start for k in ('x', 'y')):
            raise EnvSpecError("agent_start must have 'x' and 'y' keys")
        x, y = agent_start['x'], agent_start['y']
        if not (0 <= x < grid_size[0] and 0 <= y < grid_size[1]):
            raise EnvSpecError("agent_start must be within grid bounds")
    except (TypeError, KeyError) as e:
        raise EnvSpecError(f"Invalid agent_start: {e}")
    
    # Validate objects
    if not isinstance(spec['objects'], list):
        raise EnvSpecError("'objects' must be a list")
    
    for obj in spec['objects']:
        if 'type' not in obj:
            raise EnvSpecError("Object is missing 'type' field")
        if 'positions' not in obj or not isinstance(obj['positions'], list):
            raise EnvSpecError(f"Object {obj['type']} is missing or has invalid 'positions'")
        
        for pos in obj['positions']:
            if not all(k in pos for k in ('x', 'y')):
                raise EnvSpecError(f"Position in object {obj['type']} is missing 'x' or 'y'")
            x, y = pos['x'], pos['y']
            if not (0 <= x < grid_size[0] and 0 <= y < grid_size[1]):
                raise EnvSpecError(f"Position {x}, {y} in object {obj['type']} is out of bounds")
    
    # Validate actions
    if not isinstance(spec['actions'], list) or not spec['actions']:
        raise EnvSpecError("'actions' must be a non-empty list")
    
    for action in spec['actions']:
        if 'name' not in action or 'effect' not in action:
            raise EnvSpecError("Action is missing 'name' or 'effect' field")
    
    # Validate observation_type
    if spec['observation_type'] not in ('grid', 'dict', 'flat'):
        raise EnvSpecError("observation_type must be one of: 'grid', 'dict', 'flat'")
    
    # Validate max_steps
    if not isinstance(spec['max_steps'], int) or spec['max_steps'] <= 0:
        raise EnvSpecError("max_steps must be a positive integer")
    
    # Set default properties if not specified
    for obj in spec['objects']:
        if 'properties' not in obj:
            obj['properties'] = {}
    
    # Validate rewards and terminations (syntax validation happens in the environment)
    if not isinstance(spec['rewards'], list):
        raise EnvSpecError("'rewards' must be a list")
    
    for reward in spec['rewards']:
        if 'condition' not in reward or 'value' not in reward:
            raise EnvSpecError("Reward rule is missing 'condition' or 'value'")
    
    if not isinstance(spec['terminations'], list):
        raise EnvSpecError("'terminations' must be a list")
    
    for term in spec['terminations']:
        if 'condition' not in term or 'done' not in term:
            raise EnvSpecError("Termination rule is missing 'condition' or 'done'")
    
    return spec
