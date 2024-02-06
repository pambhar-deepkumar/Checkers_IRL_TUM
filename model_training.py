import yaml
from models import *
def read_config(config_path='config.yaml'):
    """
    Reads a YAML configuration file and returns a dictionary.

    Parameters:
    - config_path: str, path to the configuration file.

    Returns:
    - config: dict, the configuration variables.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config = read_config()
    boards_list, metrics, winning = generate_game_states(10000)
    metrics_model, history = train_metrics_model(metrics = metrics[:10000], winning = winning[:10000], save_path=config["paths"]['metrics_model_save_path'])
    plot_history_metrics_model(history=history)    
    # board_model = train_board_model(boards_list, metrics, winning, metrics_model, save_path=config["paths"]['board_model_save_path'])
    # winrates = reinforce_model(board_model=board_model,num_generations=5, save_path=config["paths"]['reinforced_model_save_path'])
    # print(winrates)
main()