from pokerLoss import *
import torch.optim as optim
import torch.nn as nn
import sys
import os

###NOTES:
# - work out the Loss class with a string input that gives the type of loss being used

def train(model, data, data_val, targets, targets_val, config, device, loss_type="loss 1"):
    # Extracting configuration parameters
    N_EPOCHS = 10
    LR = 0.01
    SAVE_EVERY = 5
    HIDDEN_SIZE = 150
    DROPOUT_P = 0.2
    CHECKPOINT = 'ckpt_ep_{}_hsize_{}_dout_{}'.format(N_EPOCHS, HIDDEN_SIZE, DROPOUT_P)
    DATA_P = 0.1

    model = model.to(device) # TODO: Move model to the specified device

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR) # TODO: Initialize optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(data))

    loss = fourLoss(loss_type) # TODO: Initialize loss function
    # loss = torch.nn.CrossEntropyLoss()

    # Lists to store training and validation losses over the epochs
    train_losses, validation_losses = [], []

    # Training over epochs
    for epoch in range(N_EPOCHS):
        # TRAIN: Train model over training data
        avg_loss_per_sequence = 0
        for i in range(round(len(data)*DATA_P)):
            model.init_hidden(device) # Zero out the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.)
            model.zero_grad()   # Zero out the gradient

            seq = torch.t(data[i]).t().float() # Needs a transpose. float() turns float64 -> float32, which we need
            target = targets[i].view(34).float() # Fixing dimensionality with view (maybe get rid of magic num. l8r)
            seq = seq.to(device)
            target = target.to(device)

            seq_loss = 0
            output = []
            for c in range(seq.shape[0]): # For each column...
                if device == "cuda":
                    inp = seq[c].cuda()
                    tar = target.cuda()
                else:
                    inp = seq[c]
                    tar = target

                output_card = model.forward(inp)
                output.append(output_card) 
                seq_loss += loss(output_card, tar)
            seq_loss.backward() # Backprop the total losses across each timestep (maybe we only do the last?)
            optimizer.step() # optimizer
            scheduler.step()
            avg_loss_per_sequence += seq_loss.item()/len(seq) 

            # Display progress
            msg = '\rTraining Epoch: {}, {:.2f}% iter: {} Loss: {:.4}'.format(epoch, (i+1)/round(len(data)*DATA_P)*100, i, seq_loss.item()/len(seq))
            sys.stdout.write(msg)
            sys.stdout.flush()

        # TODO: Append the avg loss on the training dataset to train_losses list
        train_losses.append(avg_loss_per_sequence/round(len(data)*DATA_P))

        print("\nepoch avg loss: ", train_losses[-1])
        
        # VAL: Evaluate Model on Validation dataset
        model.eval() # Put in eval mode (disables batchnorm/dropout) !
        with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
            avg_loss_per_sequence = 0
            # Iterate over validation data
            for i in range(round(len(data_val)*DATA_P)):
                '''
                TODO: 
                    - For each song:
                        - Zero out/Re-initialise the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.) (Done for you)
                        - Get a random sequence of length: SEQ_SIZE from each song- Get a random sequence of length: SEQ_SIZE from each song (check util.py)
                        - Iterate over sequence characters : 
                            - Transfer the input and the corresponding ground truth to the same device as the model's
                            - Do a forward pass through the model
                            - Calculate loss per character of sequence
                        - Calculate avg loss for the sequence
                    - Calculate avg loss for the validation dataset 
                '''

                model.init_hidden(device) # Zero out the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.)

                #TODO: Finish next steps here
                seq = torch.t(data[i]).t().float()
                target = targets[i].view(34).float()
                seq.to(device)
                target.to(device)

                seq_loss = 0
                for c in range(len(seq)):
                    if device == "cuda":
                        inp = seq[c].cuda()
                        tar = target.cuda()
                    else:
                        inp = seq[c]
                        tar = target
                    output = model.forward(inp)
                    seq_loss += loss(output, tar)

                avg_loss_per_sequence += seq_loss.item()/len(seq)

                # Display progress
                msg = '\rValidation Epoch: {}, {:.2f}% iter: {} Loss: {:.4}'.format(epoch, (i+1)/round(len(data_val)*DATA_P)*100, i, seq_loss.item()/len(seq))
                sys.stdout.write(msg)
                sys.stdout.flush()
        
        # TODO: Append the avg loss on the validation dataset to validation_losses list
        validation_losses.append(avg_loss_per_sequence/round(len(data_val)*DATA_P))

        print("\nval avg loss: ", validation_losses[-1])

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

    return train_losses, validation_losses

