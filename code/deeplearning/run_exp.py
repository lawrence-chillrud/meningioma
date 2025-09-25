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
task = 'MethylationSubgroup' # 'MethylationSubgroup'
OUTPUT_DIR = f'results/deeplearning/{task}'

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

def make_full_dict(input_dict, class_names):
    output_dict = {}
    for key, array in input_dict.items():
        for i in range(len(array)):
            output_dict[f'{class_names[i]}_{key}'] = array[i]
    return output_dict

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
            # Adjust for multiclass labels
            if len(y_batch.shape) == 1 and criterion.__class__.__name__ == "CrossEntropyLoss":
                y_batch = y_batch.long()
            # Run inference
            outputs = model(X_batch)
            # Keep track of predictions and true labels
            y_preds = torch.cat((y_preds, outputs.squeeze(1)))
            y_trues = torch.cat((y_trues, y_batch))
            sub_IDs = torch.cat((sub_IDs, batch['sub_id']))
            # Backward pass
            if criterion.__class__.__name__ == "CrossEntropyLoss":
                loss += criterion(outputs, y_batch).item()
            else:
                loss += criterion(outputs.squeeze(1), y_batch.float()).item()
    
        # Calculate evaluation metrics and return
        loss /= len(dataloader)
        metrics = all_metrics(y_trues, y_preds)
        # if criterion.__class__.__name__ == "CrossEntropyLoss":
        #     metrics = make_full_dict(metrics, ['Merlin Intact', 'Immune Enriched', 'Hypermetabolic'])
        
        metrics["AUCPR"] = average_precision_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy(), average='macro')
        metrics["AUROC"] = roc_auc_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy(), multi_class='ovr')
        metrics["LOSS"] = loss
        if criterion.__class__.__name__ == "CrossEntropyLoss":
            preds = pd.DataFrame({
                'SubjectID': sub_IDs.cpu().numpy(),
                'y': y_trues.cpu().numpy(),
                'y_pred': y_preds.argmax(dim=1).cpu().squeeze().numpy()
            })
        else:
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
            # Adjust for multiclass labels
            if len(y_batch.shape) == 1 and criterion.__class__.__name__ == "CrossEntropyLoss":
                y_batch = y_batch.long()
            # Zero out the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(X_batch)
            # Keep track of predictions and true labels
            y_preds = torch.cat((y_preds, outputs.squeeze(1)))
            y_trues = torch.cat((y_trues, y_batch))
            # Backward pass
            if criterion.__class__.__name__ == "CrossEntropyLoss":
                loss = criterion(outputs, y_batch)
            else:
                loss = criterion(outputs.squeeze(1), y_batch.float())
            loss.backward()
            # Take an optimization step
            optimizer.step()
            # Keep track of training loss
            train_loss += loss.item()
        
        # Training metrics
        train_loss /= len(data['train'])
        train_metrics = all_metrics(y_trues, y_preds)
        # if criterion.__class__.__name__ == "CrossEntropyLoss":
        #     train_metrics = make_full_dict(train_metrics, ['Merlin Intact', 'Immune Enriched', 'Hypermetabolic'])

        train_metrics["AUCPR"] = average_precision_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy(), average='macro')
        train_metrics["AUROC"] = roc_auc_score(y_trues.cpu().numpy(), y_preds.cpu().detach().numpy(), multi_class='ovr')
        train_metrics["LOSS"] = train_loss

        # Validation metrics
        val_metrics, _ = evaluate(model, criterion, data['val'])

        # Log metrics
        tensorboard_writer.add_scalars('Train', train_metrics, epoch)
        tensorboard_writer.add_scalars('Val', val_metrics, epoch)

        # Save best performing models
        if not os.path.exists(f'{output_dir}/model_weights'): os.makedirs(f'{output_dir}/model_weights')
        if val_metrics['LOSS'] < best_val_loss:
            torch.save(model.state_dict(), f'{output_dir}/model_weights/best_val_loss.pt')
            best_val_loss = val_metrics['LOSS']
        if val_metrics['BACC'] > best_val_balanced_acc:
            torch.save(model.state_dict(), f'{output_dir}/model_weights/best_val_balancedacc.pt')
            best_val_loss = val_metrics['BACC']
        
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
        out_feats = 1 if ds.num_classes == 2 else ds.num_classes
        final_layer = "sigmoid" if ds.num_classes == 2 else "softmax"
        criterion = nn.BCELoss() if ds.num_classes == 2 else nn.CrossEntropyLoss()
        learning_rate = 0.0001 if ds.num_classes == 2 else 0.00001
        model = CalabreseModel(input_channels=len(ds.pulse_sequences), output_features=out_feats, final_layer=final_layer).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)

        # Train
        train(model, optimizer, criterion, dataloaders, output_dir, epochs)

        # Test
        eval_df = test(model, dataloaders, criterion, output_dir)
        eval_df['fold'] = fold
        eval_df.to_csv(f'{output_dir}/eval_stats.csv', index=False)

# Create dataset, and then dataloaders
ds = MeningiomaDataset(
    task_name=task,
    pulse_sequences=['t1_post', 'flair', 'adc'],
    seg_rois=[22],
    transforms=transforms.Compose([
        Normalize(mean=[0], std=[1]),
        CenterOnTumor(cube_size=96, margin=5, pad_size=60),
    ])
)

# %%
run_kfold_exp(ds, epochs=100)

# %%
fprs = []
tprs = []
aucs_list = []
is_multiclass = ds.num_classes > 2

for d in os.listdir(OUTPUT_DIR):
    preds_df = pd.read_csv(f'{OUTPUT_DIR}/{d}/predictions/best_val_balancedacc/val_preds.csv')
    if is_multiclass:
        y_true = pd.get_dummies(preds_df['y']).values
        y_pred = preds_df.iloc[:, 2:].values  # Assuming predictions start from the 3rd column
        for i in range(ds.num_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            auc_score = auc(fpr, tpr)
            fprs.append(fpr)
            tprs.append(tpr)
            aucs_list.append((i, auc_score))
    else:
        fpr, tpr, _ = roc_curve(preds_df['y'], preds_df['y_pred'])
        auc_score = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs_list.append(auc_score)

# Plotting
plt.figure(figsize=(10, 6))

if is_multiclass:
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        plt.plot(fpr, tpr, label=f'Class {aucs_list[i][0]} (AUC = {aucs_list[i][1]:.3f})')
else:
    for i in range(len(tprs)):
        plt.plot(fprs[i], tprs[i], label=f'Fold {i+1} (AUC = {aucs_list[i]:.3f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()
