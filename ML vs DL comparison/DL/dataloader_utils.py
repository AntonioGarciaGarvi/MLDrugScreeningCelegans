import torch
import cv2
from PIL import Image
import numpy as np
import random
from glob import glob
import re
import os

## Functions to sort folder, files in the "natural" way:
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def center_crop(image, crop_width, crop_height):
    """
    Center crops the given image to the specified width and height.

    Parameters:
    image (numpy.ndarray): The input image to be cropped.
    crop_width (int): The width of the crop.
    crop_height (int): The height of the crop.

    Returns:
    numpy.ndarray: The center-cropped image.
    """
    # Get the dimensions of the image
    height, width, _ = image.shape

    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2

    # Calculate the cropping box
    crop_x1 = max(center_x - crop_width // 2, 0)
    crop_y1 = max(center_y - crop_height // 2, 0)
    crop_x2 = min(center_x + crop_width // 2, width)
    crop_y2 = min(center_y + crop_height // 2, height)

    # Crop the image
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    return cropped_image



def colorjitter(img, cj_type, value, brightness, contrast):
    '''
    ### Different Color Jitter ###
    img: image
    cj_type: {b: brightness, s: saturation, c: constast}
    '''
    if cj_type == "b":
        # value = random.randint(-50, 50)
        # value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    elif cj_type == "s":
        # value = random.randint(-50, 50)
        # value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    elif cj_type == "c":
        # brightness = 10
        # contrast = random.randint(40, 100)
        dummy = np.int16(img)
        dummy = dummy * (contrast / 127 + 1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        return img


def get_worm_square_window(im, square_size, cx, cy):
    height, width = im.shape[0:2]

    x1 = int(cx - square_size / 2)
    y1 = int(cy - square_size / 2)

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0

    x2 = x1 + square_size
    y2 = y1 + square_size

    if x2 > width:
        x1 = x1 - (x2 - width)
        x2 = width

    if y2 > height:
        y1 = y1 - (y2 - height)
        y2 = height

    im = im[y1:y2, x1:x2]
    return im


def rotate_frame(frame, angle):
    rows = frame.shape[0]
    cols = frame.shape[1]
    center = (cols / 2, rows / 2)
    R = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_img = cv2.warpAffine(frame, R, (cols, rows))

    return rotated_img


def get_traj_frames_windows(filename, frame_list, cx, cy, transform, seq_length, dim_resize, augmentation, gaussian_filter):
    
    if gaussian_filter:
        min_kernel=3
        max_kernel=7
        ksize = random.choice([k for k in range(min_kernel, max_kernel + 1) if k % 2 == 1])
        sigma = random.uniform(0.1, 2.0)  # Random sigma for variability
        
    apply_augment = 0
    if augmentation:
        apply_augment = random.randint(0, 1)
        if apply_augment:
            value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
            brightness = 10
            contrast = random.randint(40, 100)
            cj_type = random.choice(['b', 's', 'c'])

            possible_degrees = list(range(0, 360, 15))
            degrees = random.choice(possible_degrees)

    frames = []

    v_cap = cv2.VideoCapture(filename + '/000000.mp4')
    
    if not v_cap.isOpened():
        print(f"Error: Cannot open video file {filename + '/000000.mp4'}")
        return -1  # Return an empty list or some default value
    
    
    start_frame_number = frame_list[0]
    last_frame_number = frame_list[-1]

    v_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    
    
    square_size = 80  # worm roi size
    for fn in range(start_frame_number, last_frame_number + 1):
        success, frame = v_cap.read()
        if success is False:
            continue
        try:
            if fn in frame_list:
                augmented_square_size = int(square_size * 2**(1/2)) + 3 # for rotations
                frame = get_worm_square_window(frame, augmented_square_size, int(cx[frame_list.index(fn)]), int(cy[frame_list.index(fn)]))

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if apply_augment:
                    frame = colorjitter(frame, cj_type, value, brightness, contrast)
                    frame = rotate_frame(frame, degrees)
                    
                    
                frame = center_crop(frame, square_size, square_size)
                
                if gaussian_filter:
                    frame = cv2.GaussianBlur(frame, (ksize, ksize), sigma)### blur
                    
                frame = cv2.resize(frame, dim_resize, interpolation=cv2.INTER_AREA)
                frame = Image.fromarray(frame)
                img_tens = transform(frame)
                frames.append(img_tens)

        except Exception as e:
            print(e)
            print(filename)

    v_cap.release()

    # Check that the video has the correct length
    if len(frames) > seq_length:
        frames = frames[:seq_length]

    if len(frames) < seq_length:
        for rep in range(0, (seq_length - len(frames))):
            try:
                frames.append(img_tens)
            except:
                print(filename)

    return frames


class TrajectoryCoordVideosDataset(torch.utils.data.Dataset):
    def __init__(self, videos_folder, data_list, seq_length,  transform=None, augmentation=None, img_size=(224, 224), gaussian_filter=False):
        self.videos_folder = videos_folder
        self.all_videos = data_list
        self.dataset_size = len(self.all_videos)
        self.seq_length = seq_length
        self.transform = transform
        self.augmentation = augmentation
        self.img_size = img_size
        self.gaussian_filter = gaussian_filter

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        video_folder, traj_idx, well_name, worm_gene, cx, cy, frames_list, roi = self.all_videos[idx]
        # Get Label
        if worm_gene == 'N2':
            label = 0
        else:
            label = 1

        video_path = self.videos_folder + video_folder 
        frames = get_traj_frames_windows(video_path, frames_list, cx, cy, self.transform, self.seq_length, self.img_size, self.augmentation, self.gaussian_filter)

        # Create centroid displacement tensor from cx and cy

        vx = [(cx[i+1] - cx[i]) for i in range(len(cx)-1)]
        vy = [(cy[i+1] - cy[i]) for i in range(len(cy)-1)]
        
        lastVx = vx[-1]
        lastVy = vy[-1]
        
        vx.append(lastVx)
        vy.append(lastVy)

        vx = torch.tensor(vx, dtype=torch.float32)
        vy = torch.tensor(vy, dtype=torch.float32)

        speeds = torch.stack((vx, vy), dim=-1)

        label = torch.tensor(label)
        stacked_set = torch.stack(frames)
        composed_sample = [stacked_set, speeds, label]

        return composed_sample

 
class TrajectoryVideosDataset(torch.utils.data.Dataset):
    def __init__(self, videos_folder, data_list, seq_length,  transform=None, augmentation=None, img_size=(224, 224), gaussian_filter=False):
        self.videos_folder = videos_folder
        self.all_videos = data_list
        self.dataset_size = len(self.all_videos)
        self.seq_length = seq_length
        self.transform = transform
        self.augmentation = augmentation
        self.img_size = img_size
        self.gaussian_filter = gaussian_filter

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        video_folder, traj_idx, well_name, worm_gene, cx, cy, frames_list, roi = self.all_videos[idx]
        # Get Label
        if worm_gene == 'N2':
            label = 0
        else:
            label = 1

        video_path = self.videos_folder + video_folder 
        frames = get_traj_frames_windows(video_path, frames_list, cx, cy, self.transform, self.seq_length, self.img_size, self.augmentation, self.gaussian_filter)
        label = torch.tensor(label)
        stacked_set = torch.stack(frames)
        composed_sample = [stacked_set, label]
        
        return composed_sample  