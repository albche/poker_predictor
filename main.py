import torch
from train import *
import json
import argparse
import gc
from model import Poker_Model
from util import *
from train import train
from dataloader import load_batches

# TODO determine which device to use (cuda or cpu)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)

if __name__ == "__main__":
    #python3 main.py --config config.json  -> To Run the code
    
    #Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Specify the config file')
    args = parser.parse_args()

    # Load the configuration from the specified config file
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    # Extract configuration parameters
    TEMPERATURE = config["temperature"]
    SHOW_HEATMAP = config["show_heatmap"]
    loss_plot_file_name = config["loss_plot_file_name"]
    evaluate_model_only = config["evaluate_model_only"]
    model_path = config["model_path"]

    print('==> Building model..')

    # 17 spaces for each card, max of 5 cards
    num_board_features = 10

    # 17 spaces for each card, max of 2 cards
    pocket_cards = 4

    # pre-flop, flop, turn, river
    num_rounds = 4

    # fold, call, raise
    num_actions = 10
    
    # only considering 2 player games
    num_players = 2

    # each player has a bankroll and active bet
    money_per_player = 3

    # pot is common knowledge among players
    pot = 1

    in_size = (
        (num_board_features) + 
        (pocket_cards) + 
        (pot) + 
        (num_rounds) + 
        (num_actions) + 
        (money_per_player * num_players)
    )
    print("in size", in_size)
    out_size = 104 # number of predictions i think?
    model = Poker_Model(in_size, out_size)

    data_train, targets_train, data_val, targets_val, data_test, targets_test = load_batches(batch_size=128)

    # Train the model and get the training and validation losses
    losses, v_losses, accs, v_accs = train(model, data_train, data_val, targets_train, targets_val, config, device)

    # Plot the training and validation losses
    plot_losses(losses, v_losses)
    # plot_accs(accs, v_accs)

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
