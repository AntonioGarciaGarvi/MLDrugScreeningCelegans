import random
import ast

def make_subvideos_dataset(videos_list, dataset_list, sampled_subvideos):
    subvideos_list = []
    for video in videos_list:
        # get subvideos indices
        indices = list(range(video[2][0], video[2][1] + 1))
        indices = random.sample(indices, sampled_subvideos)
        for ind in indices:
            subvideos_list.append(dataset_list[ind])

    return subvideos_list

def divide_into_time_intervals_idx(frames_indices, framesxint):
    nframes = len(frames_indices)
    nintervals = int(nframes / framesxint)
    value = frames_indices[0]
    intervals = []
    for interval in range(0, nintervals):
        ll = value
        value += framesxint
        upl = value - 1
        intervals.append([ll, upl])
    return intervals


def divide_videos_into_subvideos(metadata, skipped_frames, seq_length, frames_per_int, video_cap_fps):
    dataset_list = []  
    idx_list = []  
    idx = 0 

    for index, video in metadata.iterrows():
        video_path = video['File']
        well_name = video['Well']
        worm_gene = video['Gene']
        coordX = ast.literal_eval(video['Coord_x'])
        coordY = ast.literal_eval(video['Coord_y'])
        nframes_video = video['Frames']
        frames_ind = ast.literal_eval(video['frame_number'])
        roi = ast.literal_eval(video['roi'])
        traj_idx = video['Traj_idx']

        idx_ini = idx
        sample_freq = video_cap_fps / (skipped_frames + 1)  

        time_intervals = divide_into_time_intervals_idx(frames_ind, frames_per_int)

        for interv in time_intervals:
            start, end = interv
            frame_list = []  
            frames_sampled = 0
            cx = []
            cy = []

            for i in range(start, end + 1, skipped_frames + 1):
                cx.append(coordX[frames_ind.index(i)])
                cy.append(coordY[frames_ind.index(i)])

                frame_list.append(i)
                frames_sampled += 1
                if frames_sampled == seq_length: 
                    dataset_list.append([video_path, traj_idx, well_name, worm_gene, cx, cy, frame_list, roi])
                    idx += 1
                    break

        idx_end = idx - 1
        idx_list.append([video_path, traj_idx, [idx_ini, idx_end], well_name, worm_gene, sample_freq])
    return dataset_list, idx_list



class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
