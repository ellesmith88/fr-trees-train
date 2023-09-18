from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR, TRAIN_TEST_VALID_DIR
from config import BATCH_SIZE, RESIZE_TO, CLASSES
from model import create_model
from utils import Averager, collate_fn, get_train_transform, get_valid_transform
from tqdm.auto import tqdm
from datasets import TreeDataset
import torch
import time
import argparse
import os
from torch.utils.data import DataLoader


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', help='path to train and valid directories')

    return parser.parse_args()

# function for running training iterations
def train(train_data_loader, optimizer, train_itr, train_loss_list, train_loss_hist, model):

    print('Training')

     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for _, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    
    return train_loss_hist, train_itr, train_loss_list


# function for running validation iterations
def validate(valid_data_loader, val_itr, val_loss_list, val_loss_hist, model):
    print('Validating')
    
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for _, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    
    val_loss_hist, val_itr, val_loss_list


def train():

    args = arg_parse()

    if args.path:
        path = args.path
    
    else:
        path = TRAIN_TEST_VALID_DIR

    TRAIN_DIR = os.path.join(path, 'train')
    VALID_DIR = os.path.join(path, 'valid')

        # prepare the final datasets and data loader
    train_dataset = TreeDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
    valid_dataset = TreeDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")


    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]

    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1

    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []
    best_val_loss = 1000

    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()

        # start timer and carry out training and validation
        start = time.time()
        train_loss_hist, train_itr, train_loss_list = train(train_loader, optimizer, train_itr, train_loss_list, train_loss_hist, model)
        val_loss_hist, val_itr, val_loss_list = validate(valid_loader, val_itr, val_loss_list, val_loss_hist, model)

        if val_loss_hist.value < best_val_loss:
            best_val_loss = val_loss_hist.value
            torch.save(model.state_dict(), f"{OUT_DIR}/best.pth")
            print('Updated saved model')

        if epoch == NUM_EPOCHS-1:
            torch.save(model.state_dict(), f"{OUT_DIR}/last.pth")
            print('Saved last model')

        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")
        end = time.time()

        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")


if __name__ == '__main__':
    train()
    