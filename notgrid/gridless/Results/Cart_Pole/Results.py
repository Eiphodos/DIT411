from Utilities.Data_Structures.Config import Config
from Agents.DQN_Agents.DDQN_Agent import DDQN_Agent
from Agents.DQN_Agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from Agents.DQN_Agents.DQN_Agent import DQN_Agent
from Agents.DQN_Agents.DQN_Agent_With_Fixed_Q_Targets import DQN_Agent_With_Fixed_Q_Targets
from Environments.Open_AI_Gym_Environments.Cart_Pole_Environment import Cart_Pole_Environment
from Environments.Game_enviroment import Game_enviroment
from Environments.Open_AI_Gym_Environments.Cart_Pole_Environment import Cart_Pole_Environment
from Utilities.Utility_Functions import run_games_for_agents

config = Config()
config.seed = 1
config.environment = Game_enviroment()
#config.environment = Cart_Pole_Environment ()
config.max_episodes_to_run = 1000
config.file_to_save_data_results = "Results_Data.pkl"
config.file_to_save_data_results_graph = "Results_Graph.png"
config.visualise_individual_results = True
config.visualise_overall_results = True
config.runs_per_agent = 1
config.use_GPU = True

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.005,
        "batch_size": 256,
        "buffer_size": 40000,
        "epsilon": 0.1,
        "epsilon_decay_rate_denominator": 200,
        "discount_rate": 0.99,
        "tau": 0.1,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.4,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "nn_layers": 3,
        "nn_start_units": 20,
        "nn_unit_decay": 1.0,
        "final_layer_activation": None,
        "batch_norm": False,
        "gradient_clipping_norm": 5
    }
}

if __name__ == "__main__":
    AGENTS = [DDQN_Agent, DQN_Agent_With_Fixed_Q_Targets, DDQN_With_Prioritised_Experience_Replay, DQN_Agent]
    #AGENTS = [PPO_Agent, DDQN_Agent, DQN_Agent_With_Fixed_Q_Targets, DDQN_With_Prioritised_Experience_Replay, DQN_Agent,\
             #Genetic_Agent, Hill_Climbing_Agent]
    print("_______config : " + str(config))
    run_games_for_agents(config, AGENTS)
