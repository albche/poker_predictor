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
	num_board_features = 85

	# 17 spaces for each card, max of 2 cards
	pocket_cards = 34

	# pre-flop, flop, turn, river
	num_rounds = 4

	# fold, call, raise
	num_actions = 3
	
	# only considering 2 player games
	num_players = 2

	# each player has a bankroll and active bet
	money_per_player = 2

	# pot is common knowledge among players
	pot = 1

	# 4 rounds, each player can take an action per round: 8 representations of the game
	max_num_rounds = num_players * num_rounds

	in_size = (
		(num_board_features * max_num_rounds) + 
		(pocket_cards * num_players) + 
		(pot * max_num_rounds) + 
		(num_rounds * max_num_rounds) + 
		(num_actions * max_num_rounds * num_players) + 
		(money_per_player * num_players * max_num_rounds)
	)
	out_size = 10 # number of predictions i think?
	model = Poker_Model(in_size, out_size, config)

	data = load_data()
	data_val = load_data()
	targets = None
	targets_val = None

	# If evaluating model only and trained model path is provided:
	if(evaluate_model_only and model_path != ""):
		# Load the checkpoint from the specified model path
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
