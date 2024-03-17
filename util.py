import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

current_time = datetime.now().time()
hours = current_time.hour
seconds = current_time.second

sns.set_theme()

def plot_losses(training_losses, validation_losses):
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.savefig(f"outputs/loss_{hours}_{seconds}.png")
    plt.close()

def plot_accs(training_accs, validation_accs):
    plt.plot(training_accs, label='Training Accuracy')
    plt.plot(validation_accs, label='Validation Accuracy')
    plt.axhline(0.000735, color='red', label='Random Guess')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.savefig(f'outputs/accuracy_{hours}_{seconds}.png')
    plt.close()