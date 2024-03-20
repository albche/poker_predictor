from pokerLoss import *
import torch.optim as optim
import torch.nn as nn
import sys
import os

###NOTES:
# - work out the Loss class with a string input that gives the type of loss being used

def train(model, data, data_val, targets, targets_val, config, device, loss_type="loss 1"):
    # print(model.parameters().shape)
    print(sum(p.numel() for p in model.parameters()))

    # Extracting configuration parameters
    N_EPOCHS = 20
    LR = 0.001
    SAVE_EVERY = 5
    HIDDEN_SIZE = 50
    DROPOUT_P = 0.2
    CHECKPOINT = 'ckpt_ep_{}_hsize_{}_dout_{}'.format(N_EPOCHS, HIDDEN_SIZE, DROPOUT_P)
    DATA_P = 0.01

    model = model.to(device) # TODO: Move model to the specified device

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR) # TODO: Initialize optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(data))

    loss = fourLoss(loss_type) # TODO: Initialize loss function
    # loss = torch.nn.CrossEntropyLoss()

    # Lists to store training and validation losses over the epochs
    train_losses, validation_losses = [], []
    train_accs, validation_accs = [], []

    # Training over epochs
    for epoch in range(N_EPOCHS):
        # TRAIN: Train model over training data
        avg_loss_per_sequence = 0
        total_accuracy = 0
        for i, (input, target) in enumerate(zip(data[:round(len(data)*DATA_P)], targets[:round(len(targets)*DATA_P)])):
            model.init_hidden(input.shape[0], device) # Zero out the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.)
            model.zero_grad()   # Zero out the gradient
            input, target = input.float().to(device), target.float().to(device)

            seq_loss = 0
            out = None
            for t in range(input.shape[1]):
                out = model(input[:,t,:].reshape(input.shape[0], 1, input.shape[2]))
                seq_loss += loss(out.view(out.shape[0], out.shape[2]), target)
            seq_loss.backward() # Backprop the total losses across each timestep (maybe we only do the last?)
            optimizer.step() # optimizer
            scheduler.step()
            total_accuracy += accuracy(out.view(out.shape[0], out.shape[2]), target)
            avg_loss_per_sequence += seq_loss.item()/input.shape[1] 

            # Display progress
            msg = '\rTraining Epoch: {}, {:.2f}% iter: {} Loss: {:.4}'.format(epoch, (i+1)/round(len(data)*DATA_P)*100, i, seq_loss.item()/input.shape[1])
            sys.stdout.write(msg)
            sys.stdout.flush()

        # TODO: Append the avg loss on the training dataset to train_losses list
        train_losses.append(avg_loss_per_sequence/round(len(data)*DATA_P))
        train_accs.append(total_accuracy / round(len(data)*DATA_P))

        print(f'\nEpoch Average Loss: {train_losses[-1]:.4}, Epoch Average Accuracy: {train_accs[-1]:.4}')
        
        # VAL: Evaluate Model on Validation dataset
        model.eval() # Put in eval mode (disables batchnorm/dropout) !
        with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
            avg_loss_per_sequence = 0
            total_accuracy = 0
            # Iterate over validation data
            for i, (input, target) in enumerate(zip(data_val[:round(len(data_val)*DATA_P)], targets_val[:round(len(targets_val)*DATA_P)])):
                model.init_hidden(input.shape[0], device) # Zero out the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.)
                input, target = input.float().to(device), target.float().to(device)

                seq_loss = 0
                out = None
                for t in range(input.shape[1]):
                    out = model(input[:,t,:].reshape(input.shape[0], 1, input.shape[2]))
                    seq_loss += loss(out.view(out.shape[0], out.shape[2]), target)

                total_accuracy += accuracy(out.view(out.shape[0], out.shape[2]), target)
                avg_loss_per_sequence += seq_loss.item()/input.shape[1] 

                # Display progress
                msg = '\rValidation Epoch: {}, {:.2f}% iter: {} Loss: {:.4}'.format(epoch, (i+1)/round(len(data_val)*DATA_P)*100, i, seq_loss.item()/input.shape[1])
                sys.stdout.write(msg)
                sys.stdout.flush()
        
        # TODO: Append the avg loss on the validation dataset to validation_losses list
        validation_losses.append(avg_loss_per_sequence/round(len(data_val)*DATA_P))
        validation_accs.append(total_accuracy / round(len(data)*DATA_P))

        print(f'\nVal Average Loss: {validation_losses[-1]:.4}, Val Average Accuracy: {validation_accs[-1]:.4}')

        model.train() #TURNING THE TRAIN MODE BACK ON !
        
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        # Save checkpoint.
        if (epoch % SAVE_EVERY == 0 and epoch != 0)  or epoch == N_EPOCHS - 1:
            print('=======>Saving..')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, './checkpoint/' + CHECKPOINT + '.t%s' % epoch)

    return train_losses, validation_losses, train_accs, validation_accs