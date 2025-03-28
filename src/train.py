# train.py
import os
import time
import json
from datetime import datetime
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from snake_env import SnakeEnv
from constants import *

class LearningRateScheduler(BaseCallback):
    """Custom callback for dynamic learning rate adjustment"""
    def __init__(self, initial_lr, min_lr=1e-5, decay_factor=0.5, decay_steps=100000, verbose=0):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps

    def _on_step(self):
        # Calculate new learning rate
        progress = self.num_timesteps / self.model.total_timesteps
        decay_progress = min(1.0, self.num_timesteps / self.decay_steps)
        new_lr = max(
            self.min_lr,
            self.initial_lr * (1.0 - decay_progress * self.decay_factor)
        )

        # Update the learning rate
        self.model.learning_rate = new_lr
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

    # Environment kwargs
    env_kwargs = {
        'reward_approach': config.approach_num,
        'opponent_policy': config.opponent_policy,
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
        decay_steps=config.lr_decay_steps
    )
    callbacks.append(lr_scheduler)

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
    # Example configurations for different experiments
    configs = [
        # Baseline configuration with learning rate decay
        ExperimentConfig(
            run_name="dqn_with_lr_decay",
            algorithm="DQN",
            approach_num=2,
            total_timesteps=1_500_000,
            learning_rate=3e-4,
            min_learning_rate=5e-5,
            lr_decay_factor=0.5,
            lr_decay_steps=400000  # Start decay after this many steps
        ),

        # Faster learning configuration
        ExperimentConfig(
            run_name="fast_learning_dqn",
            algorithm="DQN",
            approach_num=2,
            total_timesteps=1_500_000,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            lr_decay_factor=0.6,
            lr_decay_steps=300000,
            batch_size=256,
            exploration_fraction=0.3
        ),

        # PPO configuration
        ExperimentConfig(
            run_name="ppo_with_lr_decay",
            algorithm="PPO",
            approach_num=2,
            total_timesteps=1_000_000,
            learning_rate=3e-4,
            min_learning_rate=5e-5,
            lr_decay_factor=0.5,
            lr_decay_steps=300000,
            net_arch=[128, 128]
        )
    ]

    # Train with the first configuration (comment/uncomment as needed)
    train_agent(configs[0])
