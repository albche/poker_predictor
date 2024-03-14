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
	MAX_GENERATION_LENGTH = config["max_generation_length"]
	TEMPERATURE = config["temperature"]
	SHOW_HEATMAP = config["show_heatmap"]
	generated_song_file_path = config["generated_song_file_path"]
	loss_plot_file_name = config["loss_plot_file_name"]
	evaluate_model_only = config["evaluate_model_only"]
	model_path = config["model_path"]

	print('==> Building model..')

	out_size = 10 # number of predictions i think?
	model = Poker_Model(out_size, config)

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
