from Utilities.Data_Structures.Config import Config
from Agents.DQN_Agents.DDQN_Agent import DDQN_Agent
from Agents.DQN_Agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from Agents.DQN_Agents.DQN_Agent import DQN_Agent
from Agents.DQN_Agents.DQN_Agent_With_Fixed_Q_Targets import DQN_Agent_With_Fixed_Q_Targets
from Environments.Open_AI_Gym_Environments.Cart_Pole_Environment import Cart_Pole_Environment
from Environments.Game_enviroment import Game_enviroment
from Environments.Open_AI_Gym_Environments.Lunar_Lander_Continuous import Lunar_Lander_Continuous
from Environments.Open_AI_Gym_Environments.Cart_Pole_Environment import Cart_Pole_Environment
from Utilities.Utility_Functions import run_games_for_agents
from Agents.Actor_Critic_Agents.DDPG_Agent import DDPG_Agent

config = Config()
config.seed = 1
config.environment = Game_enviroment()
#config.environment = Cart_Pole_Environment ()
#config.environment = Lunar_Lander_Continuous()
config.max_episodes_to_run = 1000
config.file_to_save_data_results = "Results_Data.pkl"
config.file_to_save_data_results_graph = "Results_Graph.png"
config.visualise_individual_results = True
config.visualise_overall_results = True
config.runs_per_agent = 10
config.use_GPU = False

config.hyperparameters = {
    "Policy_Gradient_Agents": {
            "learning_rate": 0.02,
            "nn_layers": 2,
            "nn_start_units": 20,
            "nn_unit_decay": 1.0,
            "final_layer_activation": "TANH",
            "learning_iterations_per_round": 10,
            "discount_rate": 0.99,
            "batch_norm": False,
            "clip_epsilon": 0.2,
            "episodes_per_learning_round": 7,
            "normalise_rewards": True,
            "gradient_clipping_norm": 5,
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.25,
            "noise_decay_denominator": 1
        },

    "Actor_Critic_Agents": {
        "Actor": {
            "learning_rate": 0.001,
            "nn_layers": 2,
            "nn_start_units": 20,
            "nn_unit_decay": 1.0,
            "final_layer_activation": "TANH",
            "batch_norm": False,
            "tau": 0.001,
            "gradient_clipping_norm": 1
        },

        "Critic": {
            "learning_rate": 0.01,
            "nn_layers": 2,
            "nn_start_units": 20,
            "nn_unit_decay": 1.0,
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 100000,
            "tau": 0.001,
            "gradient_clipping_norm": 1
        },

        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,
        "theta": 0.15,
        "sigma": 0.25, #0.22 did well before
        "noise_decay_denominator": 5,
        "update_every_n_steps": 10,
        "learning_updates_per_learning_session": 5


    }
}

AGENTS = [DDPG_Agent]

run_games_for_agents(config, AGENTS)