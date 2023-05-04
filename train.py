import torch
import random
import time
import torch.nn as nn
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from config import get_config
import Dataset.BaseLoader
import Dataset.PURELoader
import Dataset.UBFCLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from Models.CardioVTnet import CardioVTnet
from Models.PhysNet import PhysNet
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm
from torch.optim import Adam
from torch.optim import SGD
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from Utils.Neg_pearson_Loss import Neg_Pearson
from Evaluator.Metrics import calculate_metrics
import argparse

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def train(model, train_loader, val_loader, optimizer, loss_function, twriter, vwriter, num_epochs=25):
    # Set the model to train mode
    model.train()
    
    # Initialize the best loss to a large value
    best_loss = float('inf')
    
    # Initialize the early stopping counter
    early_stopping_counter = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_error = 0.0
        
        # Train loop
        for idx, batch in enumerate(tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{num_epochs}')):
            frames = batch[0]
            frames = frames.to(torch.float32).to(device)
            labels = batch[1].to(torch.float32).to(device)
            outputs,_ = model(frames)
            rPPG = (outputs - torch.mean(outputs)) / torch.std(outputs)  # normalize
            BVP_label = (labels - torch.mean(labels)) / torch.std(labels)  # normalize
            loss = loss_function(rPPG, BVP_label)
            error = torch.sqrt(nn.MSELoss()(rPPG, BVP_label))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.float()
            train_error += error.item()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_error /= len(train_loader)
        # Log the loss
        twriter.add_scalar('Loss/train', train_loss, epoch+1)
        twriter.add_scalar('RMSE/train', train_error, epoch+1)
        print("Epoch: {} Loss: {:.4f} RMSE: {:.4f}".format(epoch+1, train_loss, train_error))
        print('-' * 20)
        
        # Validation loop
        val_loss = 0.0
        val_error = 0.0
        
        model.eval()
        
        for idx, batch in enumerate(tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{num_epochs}')):
            frames = batch[0]
            frames = frames.to(torch.float32).to(device)
            labels = batch[1].to(torch.float32).to(device)
            with torch.no_grad():
                outputs,_ = model(frames)
                rPPG = (outputs - torch.mean(outputs)) / torch.std(outputs)  # normalize
                BVP_label = (labels - torch.mean(labels)) / torch.std(labels)  # normalize
                loss = loss_function(rPPG, BVP_label)
                error = torch.sqrt(nn.MSELoss()(rPPG, BVP_label))
            val_loss += loss.item()
            val_error += error.item()
        
        # Calculate the mean of the validation metrics
        val_loss /= len(val_loader)
        val_error /= len(val_loader)
        
        # Log the validation metrics
        vwriter.add_scalar('Loss/val', val_loss, epoch+1)
        vwriter.add_scalar('RMSE/val', val_error, epoch+1)
        print("Epoch: {} Loss: {:.4f} RMSE: {:.4f}".format(epoch+1, val_loss, val_error))
        print('=' * 20)
        
        # Check if this is the best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'Exp3/best_trained_timesformer_none.pth')
        
        scheduler.step(val_loss)
        model.train() 
    
    model.eval()


def save_feature_map_to_png(feature_map, output_dir, filename):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert the feature map to a NumPy array and squeeze it
    feature_map_np = feature_map.numpy().squeeze()

    # Normalize the feature map to the range [0, 1]
    normalized_feature_map = (feature_map_np - feature_map_np.min()) / (feature_map_np.max() - feature_map_np.min())

    # Save the feature map as a PNG image
    plt.imsave(os.path.join(output_dir, filename), normalized_feature_map, cmap='viridis')

def find_available_index(ximage_features):
    for subj_index, subject_data in ximage_features.items():
        for sort_index, feature_map in subject_data.items():
            return subj_index, sort_index, feature_map
    return None, None, None

def find_n_feature_maps(ximage_features, n):
    feature_maps = []
    counter = 0
    for subj_index, subject_data in ximage_features.items():
        for sort_index, feature_map in subject_data.items():
            feature_maps.append((subj_index, sort_index, feature_map))
            counter += 1
            if counter >= n:
                return feature_maps
    return feature_maps
'''
def test(model, test_loader, config):
    model.eval()
    predictions = dict()
    labels = dict()
    ximage_features = dict()
    # Convert data to torch tensor
    with torch.no_grad():
        for _, test_batch in enumerate(test_loader):
            batch_size = test_batch[0].shape[0]
            data, label = test_batch[0].to(
                config.DEVICE), test_batch[1].to(config.DEVICE)
            pred_ppg_test,_ = model(data)
            for idx in range(batch_size):
                subj_index = test_batch[2][idx]
                sort_index = int(test_batch[3][idx])
                if subj_index not in predictions.keys():
                    predictions[subj_index] = dict()
                    labels[subj_index] = dict()
                predictions[subj_index][sort_index] = pred_ppg_test[idx]
                labels[subj_index][sort_index] = label[idx]
        calculate_metrics(predictions, labels, config)

'''
def test(model, test_loader, config):
    model.eval()
    predictions = dict()
    labels = dict()
    ximage_features = dict()
    # Convert data to torch tensor
    with torch.no_grad():
        for _, test_batch in enumerate(test_loader):
            batch_size = test_batch[0].shape[0]
            data, label = test_batch[0].to(config.DEVICE), test_batch[1].to(config.DEVICE)
            pred_ppg_test, ximage = model(data)
            #red_ppg_test = model(data)
            for idx in range(batch_size):
                subj_index = test_batch[2][idx]
                sort_index = int(test_batch[3][idx])
                if subj_index not in predictions.keys():
                    predictions[subj_index] = dict()
                    labels[subj_index] = dict()
                    #ximage_features[subj_index] = dict()  # Add this line to create a dictionary for feature maps
                predictions[subj_index][sort_index] = pred_ppg_test[idx].detach().cpu()
                labels[subj_index][sort_index] = label[idx].detach().cpu()
                #ximage_features[subj_index][sort_index] = ximage[idx].detach().cpu()  # Save feature map for each subject and index
    '''
    # Example usage
    n = 10
    feature_maps = find_n_feature_maps(ximage_features, n)

    output_dir = 'Exp1/output/feature_maps'

    for idx, (subj_index, sort_index, feature_map) in enumerate(feature_maps):
        filename = f'subj_{subj_index}_idx_{sort_index}.png'
        save_feature_map_to_png(feature_map, output_dir, filename)
        print(f"Saved feature map {idx + 1} for subj_index {subj_index} and sort_index {sort_index}")
    '''
    calculate_metrics(predictions, labels, config)

    


if __name__ == '__main__':
    # Define the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Print the device being used
    print("Using device:", device)
    
    model = CardioVTnet(
        dim = 512,
        image_size = 64,
        patch_size = 8,
        num_frames = 32,
        num_classes = 32,
        depth = 32,
        heads = 8,
        dim_head =  64,
        attn_dropout = 0.2,
        ff_dropout = 0.2
        )

    #model = PhysNet()
    model = model.to(device)
    #loss_function = Neg_Pearson()#
    loss_function = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    loss_function = loss_function.to(device)

    pretrained_weights = torch.load('/notebooks/rPPG-Toolbox/neural_methods/model/TimeSformer_divST_8x32_224_K600.pyth')
    if pretrained_weights:
        model.load_state_dict(pretrained_weights, strict=False)
        print("Pretrained Loaded")

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last few layers
    for param in model.layers.parameters():
        param.requires_grad = True
    for param in model.Stem0.parameters():
        param.requires_grad = True
    for param in model.ConvBlockLast.parameters():
        param.requires_grad = True
    for param in model.to_patch_embedding.parameters():
        param.requires_grad = True
    for param in model.to_unpatch_embedding.parameters():
        param.requires_grad = True

    # Define learning rates for different parameter groups
    learning_rates = [
        {'params': model.layers.parameters(), 'lr': 0.005},
        {'params': model.Stem0.parameters(), 'lr': 0.01},
        {'params': model.ConvBlockLast.parameters(), 'lr': 0.01},
        {'params': model.to_patch_embedding.parameters(), 'lr': 0.01},
        {'params': model.to_unpatch_embedding.parameters(), 'lr': 0.01},
    ]

    # Define optimizer and scheduler
    momentum = 0.9
    weight_decay = 0.00001
    optimizer = SGD(learning_rates, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=0.0000000009, verbose=True)

    parser = argparse.ArgumentParser()
    #parser = add_args(parser)
    parser.add_argument('--config_file', required=False,
                        default="/notebooks/HR-VViT-Contactless-Heart-Rate-Estimation-Based-on-VViTs/config.yaml", type=str, help="The name of the model.")
    parser = Dataset.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()
    # configurations.
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')
    
    train_set = Dataset.PURELoader.PURELoader
    valid_set = Dataset.PURELoader.PURELoader
    test_set  = Dataset.PURELoader.PURELoader
    train_data_loader = train_set(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA)
    train_loader = DataLoader(
        dataset=train_data_loader,
        num_workers=8,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=train_generator
    )
    valid_data = valid_set(
                name="valid",
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA)
    val_loader = DataLoader(
        dataset=valid_data,
        num_workers=8,
        batch_size=config.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=general_generator
    )
    
    test_data = test_set(
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA)
    test_loader = DataLoader(
        dataset=test_data,
        num_workers=8,
        batch_size=8,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=general_generator)

    twriter = SummaryWriter(log_dir='/notebooks/HR-VViT-Contactless-Heart-Rate-Estimation-Based-on-VViTs/Exp3/logs/best_trained_timesformer_train_none')
    vwriter = SummaryWriter(log_dir='/notebooks/HR-VViT-Contactless-Heart-Rate-Estimation-Based-on-VViTs/Exp3/logs/best_trained_timesformer_val_none')
    # Define the device on which the code should run
    
    # Define the early stopping patience

    # Train the model
    train(model, train_loader, val_loader, optimizer, loss_function, twriter,vwriter)
    print("Training Complete !!!!")
    
    
    # Load the best model from the training process
    model.load_state_dict(torch.load('/notebooks/HR-VViT-Contactless-Heart-Rate-Estimation-Based-on-VViTs/Exp3/best_trained_timesformer_none.pth'))
    test(model, test_loader, config)
    

    # Close the TensorBoard writer
    twriter.close()
    vwriter.close()