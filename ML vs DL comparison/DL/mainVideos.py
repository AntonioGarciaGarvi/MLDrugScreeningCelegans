import pandas as pd
import torch
from torchvision import transforms
from utils import EarlyStopping, make_subvideos_dataset,divide_into_time_intervals_idx,divide_videos_into_subvideos
import dataloader_utils as datasets
import models
import time
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import random
from utils import EarlyStopping
from tqdm import tqdm
import os
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import math
import config


if __name__ == "__main__":
    # In order to ensure reproducible experiments, we must set random seeds.
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Read the metadata    
    metadata_train = pd.read_csv('/home/jovyan/dataset/train.csv', dtype=None)
    metadata_val = pd.read_csv('/home/jovyan/dataset/val.csv', dtype=None)
    
    # Define video sampling parameters
    sample_freq = config.CAPTURED_FPS / (config.SKIPPED_FRAMES + 1)
    seq_length = int(sample_freq * config.DURATION)  # total number of frames
    frames_per_int = int(config.CAPTURED_FPS * config.DURATION)  # frames at each interval of the duration of the subvideos
    print('spliting dataset')

    dataset_list_train, idx_list_train = divide_videos_into_subvideos(metadata_train, config.SKIPPED_FRAMES,
                                                          seq_length, frames_per_int,
                                                          video_cap_fps=config.CAPTURED_FPS)

    dataset_list_val, idx_list_val = divide_videos_into_subvideos(metadata_val, config.SKIPPED_FRAMES,
                                                          seq_length, frames_per_int,
                                                          video_cap_fps=config.CAPTURED_FPS)

    train_data_full = make_subvideos_dataset(idx_list_train, dataset_list_train, sampled_subvideos=6)
    val_data_full = make_subvideos_dataset(idx_list_val, dataset_list_val, sampled_subvideos=6)
    
    print('dataset splited')
    print('Train dataset len: ' + str(len(train_data_full)))
    print('Validation dataset len: ' + str(len(val_data_full)))
    

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])


    data_train = datasets.TrajectoryVideosDataset(videos_folder=config.VIDEOS_ROOT_DIR, data_list=train_data_full,
                                            seq_length=seq_length,
                                            transform=data_transform, augmentation=config.AUGMENTATION,
                                            img_size=config.DIM_RESIZE, gaussian_filter=False)

    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=config.BATCH_SIZE, shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=config.N_WORKERS)
    data_val = datasets.TrajectoryVideosDataset(videos_folder=config.VIDEOS_ROOT_DIR, data_list=val_data_full,
                                          seq_length=seq_length,
                                          transform=data_transform, augmentation=False, img_size=config.DIM_RESIZE, gaussian_filter=False)
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=config.BATCH_SIZE, shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=config.N_WORKERS)

    # Load Model
    model_name = "CNNTrUni" + str(config.DIM_RESIZE[0]) + "skf_" + str(config.SKIPPED_FRAMES) + "d_" + str(
        config.DURATION)
    saving_folder = config.RESULTS_FOLDER + "W" + str(config.DURATION) + "s/seed" + str(seed) + "/" + model_name + "/"
    print('Saving folder: ' + saving_folder)
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    
    nw = models.CNN_Transformer(backbone_name=config.BACKBONE, seq_length=seq_length, n_classes=config.N_CLASSES, dim=config.DIM, depth=config.DEPTH,
                            heads=config.HEADS,
                            mlp_dim=config.MLP_DIM, dim_head=config.DIM_HEAD, dropout=config.DROPOUT,
                            emb_dropout=config.EMB_DROPOUT)
        
    print(nw)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    nw.to(device)
    pytorch_total_params = sum(p.numel() for p in nw.parameters())
    print('Model parameters: ' + str(pytorch_total_params))

    # Loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nw.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    if config.USING_EARLY_STOP:
        early_stopping = EarlyStopping(patience=config.E_STOP_PATIENCE, min_delta=0)

    # learning_rate scheduler, each patience epochs without improvement varies lr by factor
    if config.USING_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config.PATIENCE,
                                                               factor=config.FACTOR, verbose=True)
        print('using lr scheduler')


    # Training and validation loop
    start_train = time.time()
    tmstart = time.strftime("%Y-%d-%b-%Hh-%Mm", time.localtime(time.time()))

    print("__________Training started at: " + str(datetime.datetime.now()) + " ________________")
    print("Initial Learning rate: " + str(config.LEARNING_RATE))

    loss_train = []
    loss_val = []
    best_loss = 100000.0

    for epoch in range(config.NUM_EPOCHS):
        print("___________New Epoch______________")
        start_epoch = time.time()

        loss_list = []
        loss_list_val = []
        nw.train()


        for batch in tqdm(dataloader_train):
            optimizer.zero_grad()
            imgs_batch = batch[0].to(device)
            labels = batch[1].to(device)
            pre = nw(imgs_batch)

            train_loss = loss_func(pre, labels)
            train_loss.backward()  # Calculate all gradients by backpropagation
            optimizer.step()  # Optimize

            loss_list.append(train_loss.item())

        print("Calculating cross entropy loss for eval set....")
        nw.eval()

        with torch.no_grad():
            # val data
            y_true = []
            y_pred = []
            for batch in tqdm(dataloader_val):
                imgs_batch = batch[0].to(device)
                labels = batch[1].to(device)
                
                y_true.extend(labels.cpu().numpy())
                pre = nw(imgs_batch)

                _, predicted = torch.max(pre, 1)
                y_pred.extend(predicted.cpu().numpy())
                loss = loss_func(pre, labels)
                loss_list_val.append(loss.item())

        cf_matrix_val = confusion_matrix(y_true, y_pred)
        class_names = ('N2', 'others')

        # Create Confusion Matrix and calculate metrics
        print('----------------------------------------')
        print("Confusion Matrix  for validation data")
        print('----------------------------------------')
        dataframe = pd.DataFrame(cf_matrix_val, index=class_names, columns=class_names)
        print(dataframe)
        print('----------------------------------------')
        print("Metrics  for validation data")
        print(classification_report(y_true, y_pred, target_names=class_names))

        loss_ep_train = stat.mean(loss_list)
        loss_ep_val = stat.mean(loss_list_val)

        loss_train.append(loss_ep_train)
        loss_val.append(loss_ep_val)

        if config.USING_SCHEDULER:
            scheduler.step(loss_ep_val)
            # current_lr = scheduler.get_last_lr()
            # print(f"Epoch {epoch}, Learning Rate: {current_lr}")

        if loss_ep_val < best_loss:
            print('This is the best cross entropy loss obtained: ' + str(loss_ep_val))
            best_loss = loss_ep_val

            torch.save({
                'epoch': epoch,
                'model_state_dict': nw.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'train_losses': loss_train,
                'val_losses': loss_val,
                'early_stopping_counter': early_stopping.counter,
                'early_stopping_best_loss': early_stopping.best_loss,
            }, saving_folder + model_name + "_" + tmstart + ".pth")
            

            

            plt.figure(figsize=(8, 8))

            # Create heatmap
            sns.heatmap(dataframe, annot=True, cbar=None, cmap="YlGnBu", fmt="d")
            plt.title("Confusion Matrix")
            plt.ylabel("True Class")
            plt.xlabel("Predicted Class")
            plt.tight_layout()
            plt.savefig(saving_folder + 'confmat' + model_name + "_" + tmstart + ".jpg", dpi=96)
            plt.close()

        print("Epoch: " + str(epoch) + "\nLoss in this epoch on train data: " + str(
            loss_ep_train) + "\nLoss in this epoch on test data: " + str(
            loss_ep_val) + "\nbest test loss obtained: " + str(
            best_loss))
        print("Epoch took " + str(int((time.time() - start_epoch) / 60)) + " mins to run.\n")

        if config.USING_EARLY_STOP:
            early_stopping(loss_ep_val)
            if early_stopping.early_stop:
                break

    train_time = int(math.floor((time.time() - start_train) / 60))
    print("\n\n--------Training finished-----------\nExecution time in minutes: " + str(train_time))

    ts = time.strftime("%Y-%d-%b-%Hh-%Mm", time.localtime(time.time()))

    # Generate training curve
    txt = "Batch size=" + str(config.BATCH_SIZE) + "; Sequence length=" + str(seq_length) + "; Skipped_frames=" + str(
        config.SKIPPED_FRAMES) + "; Initial learning rate=" + str(
        config.LEARNING_RATE) + " ; Best cross entropy on val set: " + str(
        best_loss) + "; Total training time=" + str(train_time) + "mins"
    txt_params = "Resnet pretrained" + "; weight decay: " + str(config.WEIGHT_DECAY) + "; TRdim: " + str(
        config.DIM) + ";TRdepth: " + str(
        config.DEPTH) + ";TRheads: " + str(config.HEADS) + ";TRdim_head: " + str(
        config.DIM_HEAD) + ";TRmlp_dim: " + str(
        config.MLP_DIM) + "; Dropout:" + str(config.DROPOUT)

    plt.figure(figsize=(1855 / 96, 986 / 96), dpi=96)
    plt.plot(loss_train, "ro-", label="Train data")
    plt.plot(loss_val, "bx-", label="Val data")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cross Entropy loss for the train and val data over all training epochs, trained Model: " + model_name)
    plt.figtext(0.5, 0.03, txt, wrap=True, horizontalalignment='center', fontsize=10)
    plt.figtext(0.5, 0.01, txt_params, wrap=True, horizontalalignment='center', fontsize=8)
    plt.grid(True, axis="y")
    plt.ylim((0, int(max(np.amax(loss_train), np.amax(loss_val))) + 1))
    plt.legend()
    plt.savefig(saving_folder + model_name + "_" + ts + ".jpg", dpi=96)
    plt.close()