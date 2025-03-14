import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

class DrivingEnv(gym.Env):
    def __init__(self):
        super(DrivingEnv, self).__init__()
        
        # Define action space (steering, acceleration, braking)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space (simplified for demonstration)
        # [car_position_x, car_position_y, car_velocity, car_heading, 
        #  distance_to_obstacle, obstacle_angle]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, 0.0, -np.pi, 0.0, -np.pi]),
            high=np.array([np.inf, np.inf, 30.0, np.pi, 100.0, np.pi]),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Initialize car state
        self.car_position = np.array([0.0, 0.0])
        self.car_velocity = 0.0
        self.car_heading = 0.0
        self.distance_to_obstacle = 50.0
        self.obstacle_angle = 0.0
        
        return self._get_observation(), {}
    
    def step(self, action):
        # Unpack actions
        steering, acceleration, braking = action
        
        # Update car state (simplified physics)
        self.car_heading += steering * 0.1
        self.car_velocity += (acceleration - braking) * 0.1
        self.car_velocity = np.clip(self.car_velocity, 0, 30)
        
        # Update position
        self.car_position[0] += np.cos(self.car_heading) * self.car_velocity
        self.car_position[1] += np.sin(self.car_heading) * self.car_velocity
        
        # Update obstacle position (simplified)
        self.distance_to_obstacle -= self.car_velocity * 0.1
        self.obstacle_angle += 0.01
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self):
        return np.array([
            self.car_position[0],
            self.car_position[1],
            self.car_velocity,
            self.car_heading,
            self.distance_to_obstacle,
            self.obstacle_angle
        ])
    
    def _calculate_reward(self):
        # Reward for maintaining speed
        speed_reward = 0.1 * self.car_velocity
        
        # Penalty for getting too close to obstacle
        distance_penalty = -0.5 * (1.0 / (self.distance_to_obstacle + 1e-6))
        
        # Penalty for going off track
        track_penalty = -0.1 * (abs(self.car_position[0]) + abs(self.car_position[1]))
        
        return speed_reward + distance_penalty + track_penalty
    
    def _is_done(self):
        # Episode ends if car crashes or goes too far off track
        return (self.distance_to_obstacle < 1.0 or
                abs(self.car_position[0]) > 100 or
                abs(self.car_position[1]) > 100)

def train_rl_agent():
    # Create and wrap the environment
    env = DummyVecEnv([lambda: DrivingEnv()])
    
    # Initialize the agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    # Train the agent
    total_timesteps = 100000  # Adjust based on your needs
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model
    model.save("driving_agent")
    
    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Close the environment
    env.close()

def main():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Train the RL agent
    train_rl_agent()

if __name__ == "__main__":
    main() 