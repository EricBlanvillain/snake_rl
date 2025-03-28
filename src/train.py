# train.py
import os
import time
import json
from datetime import datetime
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import numpy as np

from snake_env import SnakeEnv
from constants import *

class LearningRateScheduler(BaseCallback):
    """Custom callback for dynamic learning rate adjustment"""
    def __init__(self, initial_lr, min_lr=1e-5, decay_factor=0.5, decay_steps=100000, total_timesteps=1000000, verbose=0):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.total_timesteps = total_timesteps

    def _on_step(self):
        # Calculate new learning rate
        progress = self.num_timesteps / self.total_timesteps
        decay_progress = min(1.0, self.num_timesteps / self.decay_steps)
        new_lr = max(
            self.min_lr,
            self.initial_lr * (1.0 - decay_progress * self.decay_factor)
        )

        # Update the learning rate
        self.model.learning_rate = new_lr
        return True

class MazeCurriculumCallback(BaseCallback):
    """Implements curriculum learning for maze environments"""
    def __init__(self, eval_freq=10000, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.current_level = 0
        self.maze_levels = [
            "maze_training.txt",    # Level 0: Easiest
            "maze_natural.txt",     # Level 1: Medium
            "maze_symmetric.txt",   # Level 2: Medium
            "maze_arena.txt",       # Level 3: Medium-Hard
            "maze_rooms.txt",       # Level 4: Hard
            "maze_spiral.txt"       # Level 5: Hardest - requires precise long-term planning
        ]
        self.required_reward = -8.0  # Threshold to progress to next level
        self.evaluation_rewards = []
        self.current_maze = self.maze_levels[0]
        self.last_mean_reward = -float('inf')

    def _on_step(self):
        # Every eval_freq steps, check if we should progress to a harder maze
        if self.num_timesteps % self.eval_freq == 0:
            # Get the mean reward from the model's episode info buffer
            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
                mean_reward = np.mean(rewards[-100:])
                self.last_mean_reward = mean_reward
            else:
                mean_reward = self.last_mean_reward

            # If performance is good enough and we haven't reached the hardest maze
            if mean_reward > self.required_reward and self.current_level < len(self.maze_levels) - 1:
                self.current_level += 1
                self.current_maze = self.maze_levels[self.current_level]
                print(f"\n--- Curriculum Progress ---")
                print(f"Mean reward: {mean_reward:.2f}")
                print(f"Progressing to maze level {self.current_level}: {self.current_maze}")
                print(f"Required reward for next level: {self.required_reward}")

                # Update the environment with the new maze
                self.training_env.env_method("update_maze", f"mazes/{self.current_maze}")

                # Make it slightly harder to progress to the next level
                self.required_reward += 1.0

        return True

class ExperimentConfig:
    def __init__(self,
                 run_name,
                 algorithm='DQN',
                 approach_num=2,
                 total_timesteps=1_500_000,
                 n_envs=1,
                 opponent_policy='basic_follow',
                 learning_rate=3e-4,
                 min_learning_rate=1e-5,
                 lr_decay_factor=0.5,
                 lr_decay_steps=400000,
                 batch_size=128,
                 net_arch=[256, 256, 128],
                 buffer_size=100_000,
                 learning_starts=10_000,
                 exploration_fraction=0.2,
                 exploration_final_eps=0.05,
                 use_maze_rotation=True,
                 use_distance_features=True,
                 use_danger_features=True,
                 use_food_direction=True):

        self.run_name = run_name
        self.algorithm = algorithm
        self.approach_num = approach_num
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.opponent_policy = opponent_policy
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_steps = lr_decay_steps
        self.batch_size = batch_size
        self.net_arch = net_arch
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps
        self.use_maze_rotation = use_maze_rotation
        self.use_distance_features = use_distance_features
        self.use_danger_features = use_danger_features
        self.use_food_direction = use_food_direction

        # Add timestamp to make each run unique
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_run_dir(self):
        """Get the directory for this run."""
        return os.path.join("runs", f"{self.run_name}_{self.timestamp}")

    def save_config(self):
        """Save the configuration to a JSON file."""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        run_dir = self.get_run_dir()
        os.makedirs(run_dir, exist_ok=True)

        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load_config(cls, config_path):
        """Load a configuration from a JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = cls(run_name=config_dict['run_name'])
        for k, v in config_dict.items():
            setattr(config, k, v)
        return config

def train_agent(config):
    """Trains a snake agent using the specified configuration."""

    print(f"\n--- Training Run: {config.run_name} ---")
    print(f"Algorithm: {config.algorithm}")
    print(f"Approach: {config.approach_num}")
    start_time = time.time()

    # Create run directory
    run_dir = config.get_run_dir()
    log_dir = os.path.join(run_dir, "logs")
    model_dir = os.path.join(run_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save configuration
    config.save_config()

    # Environment kwargs - start with the easiest maze
    env_kwargs = {
        'reward_approach': config.approach_num,
        'opponent_policy': config.opponent_policy,
        'maze_file': 'mazes/maze_training.txt'  # Start with the training maze
    }

    # Create vectorized environment
    vec_env = make_vec_env(
        lambda: SnakeEnv(**env_kwargs),
        n_envs=config.n_envs,
        vec_env_cls=DummyVecEnv
    )

    # Algorithm specific configuration
    policy_kwargs = dict(
        net_arch=config.net_arch
    )

    # Initialize the algorithm
    if config.algorithm == 'DQN':
        model = DQN(
            'MlpPolicy',
            vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            device=DEVICE,
            buffer_size=config.buffer_size,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            learning_starts=config.learning_starts,
            exploration_fraction=config.exploration_fraction,
            exploration_final_eps=config.exploration_final_eps,
            policy_kwargs=policy_kwargs
        )
    elif config.algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            device=DEVICE,
            policy_kwargs=policy_kwargs
        )
    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // config.n_envs, 1000),
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="model"
    )
    callbacks.append(checkpoint_callback)

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(
        initial_lr=config.learning_rate,
        min_lr=config.min_learning_rate,
        decay_factor=config.lr_decay_factor,
        decay_steps=config.lr_decay_steps,
        total_timesteps=config.total_timesteps
    )
    callbacks.append(lr_scheduler)

    # Add curriculum learning callback
    curriculum_callback = MazeCurriculumCallback(eval_freq=10000)
    callbacks.append(curriculum_callback)

    # Train the model
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save the final model
        model_path = os.path.join(model_dir, "final_model.zip")
        print(f"Saving final model to {model_path}")
        model.save(model_path)
        vec_env.close()

    end_time = time.time()
    duration = (end_time - start_time) / 60

    # Save training results
    results = {
        "duration_minutes": duration,
        "completed": True,
        "final_model_path": model_path
    }

    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nTraining completed in {duration:.2f} minutes")
    print(f"Run directory: {run_dir}")
    print(f"Use TensorBoard to view logs: tensorboard --logdir {log_dir}")

if __name__ == '__main__':
    # Configuration for training with diverse mazes
    config = ExperimentConfig(
        run_name="diverse_mazes_dqn",
        algorithm="DQN",
        approach_num=2,  # Using the sophisticated reward structure
        total_timesteps=2_000_000,  # Increased timesteps for better maze exploration
        n_envs=1,
        opponent_policy='basic_follow',
        learning_rate=3e-4,
        min_learning_rate=5e-5,
        lr_decay_factor=0.5,
        lr_decay_steps=500000,  # Slower decay for better adaptation
        batch_size=256,  # Larger batch size for diverse experiences
        net_arch=[512, 256, 128],  # Deeper network for complex maze patterns
        buffer_size=200_000,  # Larger buffer for diverse experiences
        learning_starts=20_000,  # More initial exploration
        exploration_fraction=0.3,  # More exploration
        exploration_final_eps=0.05,
        use_maze_rotation=True,  # Enable maze rotation for better generalization
        use_distance_features=True,
        use_danger_features=True,
        use_food_direction=True
    )

    # Train with the new configuration
    train_agent(config)
