# %%
import os
import sys
sys.path.append('/home/lawrence/Meningioma/code')
from preprocessing.utils import explore_3D_array_with_mask_contour
from deeplearning.transforms import CenterOnTumor, Normalize
from deeplearning.prep_data import *
from deeplearning.models import CalabreseModel
from deeplearning.metrics import *
from sklearn.metrics import average_precision_score, roc_auc_score, auc, roc_curve
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
while not os.getcwd().endswith('Meningioma'): os.chdir('..')
OUTPUT_DIR = 'results/deeplearning/chr1p'

# %%
# Set up directory structures and GPU/CPU/MPS device
while not os.getcwd().endswith('Meningioma'): os.chdir('..')
DEVICE = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')
SEED = 0
torch.manual_seed(SEED)  # Set the seed for CPU random number generators
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  # Set the seed for GPU random number generators
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def evaluate(model, criterion, dataloader):
    # Setup for evaluation
    model.eval()
    with torch.no_grad():
        loss = 0.
        y_preds, y_trues, sub_IDs = torch.tensor([]).to(DEVICE), torch.tensor([]).to(DEVICE), torch.tensor([])
        for batch in dataloader:
            # Grab the batch data
            X_batch = stack_volumes(batch['mris']).to(DEVICE)
            y_batch = batch['label'].to(DEVICE)
            # Run inference
            outputs = model(X_batch)
            # Keep track of predictions and true labels
            y_preds = torch.cat((y_preds, outputs.squeeze(1)))
            y_trues = torch.cat((y_trues, y_batch))
            sub_IDs = torch.cat((sub_IDs, batch['sub_id']))
            # Backward pass
            loss += criterion(outputs.squeeze(1), y_batch.float()).item()
    
    # Calculate evaluation metrics and return
    loss /= len(dataloader)
    metrics = {
        'loss': loss,
        'balancedacc': balanced_accuracy(y_trues, y_preds).item(),
        'aucpr': average_precision_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy()),
        'auroc': roc_auc_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy()),
        'tpr': true_positive_rate(y_trues, y_preds).item(),
        'fpr': false_positive_rate(y_trues, y_preds).item(),
        'fdr': false_discovery_rate(y_trues, y_preds).item()
    }
    preds = pd.DataFrame({
        'SubjectID': sub_IDs.cpu().numpy(),
        'y': y_trues.cpu().numpy(),
        'y_pred': y_preds.cpu().squeeze().numpy()
    })
    return metrics, preds

def test(model, dataloaders, criterion, output_dir):
    eval_dict = {}
    for weights in ['best_val_loss', 'best_val_balancedacc']:
        weights_path = f'{output_dir}/model_weights/{weights}.pt'
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, weights_only=True))
            eval_dict[weights] = {}
            for k, dataloader in dataloaders.items():
                eval_dict[weights][k], preds_df = evaluate(model, criterion, dataloader)
                eval_dict[weights][k]['num_samples_in_split'] = len(preds_df)
                preds_dir = f'{output_dir}/predictions/{weights}'
                if not os.path.exists(preds_dir): os.makedirs(preds_dir)
                preds_df.to_csv(f'{preds_dir}/{k}_preds.csv', index=False)

    eval_df = pd.json_normalize(eval_dict, sep='?').T
    eval_df.index = eval_df.index.str.split('?', n=3, expand=True)
    eval_df = eval_df.reset_index()
    eval_df.columns = ['model_chosen_by', 'split', 'metric', 'value']
    return eval_df

def train(model, optimizer, criterion, data, output_dir, epochs=40):
    # Set up logging and metrics
    tensorboard_writer = SummaryWriter(log_dir=f'{output_dir}/tensorboard_logs')
    train_loss = 0.
    best_val_balanced_acc = 0.
    best_val_loss = float('inf')
    # Loop thru all epochs
    for epoch in tqdm(range(epochs), desc='Epoch', total=epochs, position=1, leave=False):
        # Setup for the epoch
        model.train()
        y_preds, y_trues = torch.tensor([]).to(DEVICE), torch.tensor([]).to(DEVICE)
        # Loop thru all batches
        for batch in tqdm(data['train'], desc='Batch', total=len(data['train']), position=2, leave=False):
            # Grab the batch data
            X_batch = stack_volumes(batch['mris']).to(DEVICE)
            y_batch = batch['label'].to(DEVICE)
            # Zero out the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(X_batch)
            # Keep track of predictions and true labels
            y_preds = torch.cat((y_preds, outputs.squeeze(1)))
            y_trues = torch.cat((y_trues, y_batch))
            # Backward pass
            loss = criterion(outputs.squeeze(1), y_batch.float())
            loss.backward()
            # Take an optimization step
            optimizer.step()
            # Keep track of training loss
            train_loss += loss.item()
        
        # Training metrics
        train_loss /= len(data['train'])
        train_metrics = {
            'loss': train_loss,
            'balancedacc': balanced_accuracy(y_trues, y_preds).item(),
            'aucpr': average_precision_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy()),
            'auroc': roc_auc_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy()),
            'tpr': true_positive_rate(y_trues, y_preds).item(),
            'fpr': false_positive_rate(y_trues, y_preds).item(),
            'fdr': false_discovery_rate(y_trues, y_preds).item()
        }

        # Validation metrics
        val_metrics, _ = evaluate(model, criterion, data['val'])

        # Log metrics
        tensorboard_writer.add_scalars('Train', train_metrics, epoch)
        tensorboard_writer.add_scalars('Val', val_metrics, epoch)

        # Save best performing models
        if not os.path.exists(f'{output_dir}/model_weights'): os.makedirs(f'{output_dir}/model_weights')
        if val_metrics['loss'] < best_val_loss:
            torch.save(model.state_dict(), f'{output_dir}/model_weights/best_val_loss.pt')
            best_val_loss = val_metrics['loss']
        if val_metrics['balancedacc'] > best_val_balanced_acc:
            torch.save(model.state_dict(), f'{output_dir}/model_weights/best_val_balancedacc.pt')
            best_val_loss = val_metrics['balancedacc']
        
    # Close logging
    tensorboard_writer.flush()
    tensorboard_writer.close()

def run_kfold_exp(ds, k=5, epochs=40):
    # Set up dataset
    ds.precache()
    ds.plot_data_split()

    # Set up crossval
    ds.create_kfold_xval_splits(k=k)
    for fold in tqdm(range(k), desc='Fold', total=k, position=0):
        # Define output dir for the fold
        output_dir = f'{OUTPUT_DIR}/fold_{fold}'

        # Construct dataloaders for this fold
        dataloaders = construct_foldk_dataloaders(ds, fold, bs=4, seed=SEED)
        
        # Initialize model, optimizer, and loss fn
        model = CalabreseModel(input_channels=3).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
        criterion = nn.BCELoss()

        # Train
        train(model, optimizer, criterion, dataloaders, output_dir, epochs)

        # Test
        eval_df = test(model, dataloaders, criterion, output_dir)
        eval_df['fold'] = fold
        eval_df.to_csv(f'{output_dir}/eval_stats.csv', index=False)

# Create dataset, and then dataloaders
ds = MeningiomaDataset(
    task_name='Chr1p',
    pulse_sequences=['t1_post', 'flair', 'adc'],
    seg_rois=[22],
    transforms=transforms.Compose([
        Normalize(mean=[0], std=[1]),
        CenterOnTumor(cube_size=96, margin=5, pad_size=60),
    ])
)

# %%
run_kfold_exp(ds, epochs=40)

# %%
fprs = []
tprs = []
aucs_list = []
for d in os.listdir(OUTPUT_DIR):
    preds_df = pd.read_csv(f'{OUTPUT_DIR}/{d}/predictions/best_val_balancedacc/val_preds.csv')
    fpr, tpr, _ = roc_curve(preds_df['y'], preds_df['y_pred'])
    auc_score = auc(fpr, tpr)
    fprs.append(fpr)
    tprs.append(tpr)
    aucs_list.append(auc_score)

# Calculate mean and standard deviation of ROCs
# mean_fpr = np.linspace(0, 1, 100)
# mean_tpr = np.mean([i[1] for i in tprs], axis=0)
# std_tpr = np.std([i[1] for i in tprs], axis=0)
# mean_auc = 0# auc(np.mean([i[0] for i in fprs], axis=0), mean_tpr)
# std_auc = 0# np.sqrt(np.mean([(i - mean_auc)**2 for i in aucs_list]))

# Plotting
plt.figure(figsize=(10, 6))

# Plot individual fold ROCs for reference
for i in range(len(tprs)):
    plt.plot(fprs[i], tprs[i], label=f'Fold {i+1} (AUC = {aucs_list[i]:.3f})')

# Plot mean ROC with error bar
# plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.3f ± %0.3f)' % (mean_auc, std_auc))
# plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='b', alpha=.2)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC with ±1 Std Dev')
plt.legend(loc='lower right')
plt.show()
# %%
