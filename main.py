import torch
from train import *
import json
import argparse
import gc
from model import Poker_Model
from util import plot_losses
from train import train
from dataloader import load_data

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
    num_board_features = 52*5

    # 17 spaces for each card, max of 2 cards
    pocket_cards = 52*2

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
        (num_actions * num_players) + 
        (money_per_player * num_players)
    )
    out_size = 104 # number of predictions i think?
    model = Poker_Model(in_size, out_size)

    data_p = 0.05 #percentage of data loaded because running everything takes too long (min, max) = (0.05, 1.0)
    data, targets = load_data(p=data_p)
    data_val, targets_val = load_data(mode='validation', p=data_p)

    # If evaluating model only and trained model path is provided:
    if(evaluate_model_only and model_path != ""):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # Load the model's state dictionary from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        print('==> Model loaded from checkpoint..')
    else:
        # Train the model and get the training and validation losses
        losses, v_losses = train(model, data, data_val, targets, targets_val, config, device)

        # Plot the training and validation losses
        plot_losses(losses, v_losses)

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
