from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)

_REGISTERED = False


def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering multiworld pygame gym environments")
    register(
        id='Point2DLargeEnv-offscreen-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        tags={
            'git-commit-hash': '166f0f3',
            'author': 'Vitchyr'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 1,
            'render_onscreen': False,
        },
    )
    register(
        id='Point2DLargeEnv-onscreen-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        tags={
            'git-commit-hash': '166f0f3',
            'author': 'Vitchyr'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 1,
            'render_onscreen': True,
        },
    )
    register(
        id='Point2DEnv-debug-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        tags={
            'author': 'Kyle'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 1,
            'render_onscreen': True,
            'fixed_goal': True,
            'randomize_position_on_reset': True,
            'steps_per_episode': 50
        },
    )
    register(
        id='Point2DEnv-goals-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        tags={
            'author': 'Kyle'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 1,
            'render_onscreen': True,
            'fixed_goal': False,
            'randomize_position_on_reset': True,
        },
    )
    register(
        id='Point2DEnv-traj-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DTrajectoryEnv',
        tags={
            'author': 'Kyle'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 0.5,
            'render_onscreen': True,
            'fixed_goal': False,
            'randomize_position_on_reset': True,
            'render_size': 256,
            'boundary_dist': 10
        },
    )
    register(
        id='Point2DEnv-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DEnv',
        tags={
            'author': 'Kyle'
        },
        kwargs={
            'initial_position': (0, 0),
            'images_are_rgb': True,
            'target_radius': 0,
            'ball_radius': 0.25,
            'render_onscreen': False,
            'fixed_goal': False,
            'randomize_position_on_reset': False,
            'render_size': 84,
            'boundary_dist': 10,
            'action_limit': 1.0,
        },
    )
    register(
        id='Point2DWalls-corner-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        tags={
            'author': 'Kyle'
        },
        kwargs={
            'wall_shape': 'maze',
            'initial_position': (-8, -8),
            'images_are_rgb': True,
            'target_radius': 0,
            'ball_radius': 0.25,
            'render_onscreen': False,
            'fixed_goal': False,
            'randomize_position_on_reset': False,
            'render_size': 84,
            'boundary_dist': 10,
            'action_limit': 1.0,
            'show_goal': False
        }
    )
    register(
        id='Point2DWalls-center-v0',
        entry_point='multiworld.envs.pygame.point2d:Point2DWallEnv',
        tags={
            'author': 'Kyle'
        },
        kwargs={
            'wall_shape': 'maze',
            'initial_position': (0, 0),
            'images_are_rgb': True,
            'target_radius': 0,
            'ball_radius': 0.25,
            'render_onscreen': False,
            'fixed_goal': False,
            'randomize_position_on_reset': False,
            'render_size': 84,
            'boundary_dist': 10,
            'action_limit': 1.0,
            'show_goal': False
        }
    )
register_custom_envs()
