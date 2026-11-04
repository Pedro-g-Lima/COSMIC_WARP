"""
Shared simulation context.

Bundles the per-entity device arrays that the pipeline, integrators and
physics kernels operate on, so subsystem interfaces stay stable as new
fields are added.
"""
from dataclasses import dataclass

import warp as wp


@dataclass
class SimContext:
    """Device arrays for one simulator instance (all length num_entities,
    except flags and termination_conditions which are single-element)."""

    states: wp.array                  # State structs
    vehicles: wp.array                # VehicleConfig structs
    actuator_commands: wp.array       # ActuatorCommand structs
    actuator_states: wp.array         # ActuatorState structs
    thruster_configs: wp.array        # ThrusterConfig structs
    rw_configs: wp.array              # ReactionWheelConfig structs
    flags: wp.array                   # SimulationFlags (single element)
    termination_conditions: wp.array  # TerminationConditions (single element)
    termination_status: wp.array      # TerminationStatus structs
    episode_steps: wp.array           # int32 steps since last reset per entity
