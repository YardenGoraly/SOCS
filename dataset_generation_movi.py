import argparse
import os
import numpy as np
import pickle

from PIL import Image
from tqdm import tqdm

from dataset_generation_const import *

def save_train_seq(seq, seq_num, out_dir):
    num_cam = 1
    seq_len = 8
    viewpoint_transform = np.zeros((seq_len, 1, 4, 4))
    timestamps = np.linspace(0, 1, seq_len)
    # seq = seq.astype('float') / 255  may or may not be necessary

    print('saving')
    print('out_dir', out_dir)
    with open(os.path.join(out_dir, f'{seq_num}.npz'), 'wb') as f:
        np.savez_compressed(f, rgb=seq,
                            viewpoint_transform=viewpoint_transform,
                            time=timestamps,
                            bc_waypoints=None,
                            bc_mask=None)
    
    return

def make_train_seqs(first_seq_num, unique_start_ids, in_dir, out_dir):
    # seq_num = first_seq_num
    # train_files = os.path.join(in_dir, '*.tfrecord')
    # filenames = tf.io.matching_files(train_files)
    # dataset = tf.data.TFRecordDataset(filenames)
    # current_seq = []

    # train_files = os.path.join(in_dir, '*.png')
    train_files = os.listdir(in_dir)
    imgs_arr = []
    counter = 0
    for file in train_files:
        counter += 1
        img = Image.open(os.path.join(in_dir,file))
        arr = np.asarray(img, dtype='uint8')[..., :-1]
        imgs_arr += [arr]
        if counter % 8 == 0:
            imgs_arr = np.array(imgs_arr)
            save_train_seq(imgs_arr, counter//8, out_dir)
            imgs_arr = []
    # print('imgs_arr:', imgs_arr.shape)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=['train', 'val'])
    parser.add_argument('--in_dir', default='data_raw')
    parser.add_argument('--out_dir', default='output')
    parser.add_argument('--load_seq_ids', default=None,
        help='To resume generating training sequences, load previously generated IDs from file')
    parser.add_argument('--save_seq_ids', default=None,
        help='To resume generating training sequences later, save newly generated IDs to file')
    
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    in_dir = os.path.join(args.in_dir, args.split)
    out_dir = os.path.join(args.out_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)

    if args.split == 'train':

        if args.load_seq_ids is not None:
            with open(args.load_seq_ids, 'rb') as f:
                unique_start_ids = pickle.load(f)
        else:
            unique_start_ids = {}

        prev_seq_num = len(unique_start_ids)
        new_seq_num = make_train_seqs(prev_seq_num, unique_start_ids, in_dir, out_dir)