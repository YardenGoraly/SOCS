import argparse
import os
import numpy as np
from tqdm import tqdm

def filter_out_missing(student_dir, teacher_dir, out_dir):
    student_sequences = sorted(os.listdir(student_dir))
    teacher_sequences = sorted(os.listdir(teacher_dir))
    index = 0
    for seq in tqdm(student_sequences):
        if seq in teacher_sequences:
            student_seq_path = os.path.join(student_dir, seq)
            teacher_seq_path = os.path.join(teacher_dir, seq)
            student_loaded_npz = np.load(student_seq_path)
            teacher_loaded_npz = np.load(teacher_seq_path)
            np.savez_compressed(os.path.join(out_dir, str(index) + '.npz'),
                                 rgb=student_loaded_npz['rgb'],
                                viewpoint_transform=student_loaded_npz['viewpoint_transform'],
                                time=student_loaded_npz['time'],
                                bc_waypoints=None,
                                bc_mask=None,
                                teacher_masks=teacher_loaded_npz['masks']
                                )
            index += 1


if __name__ == '__main__':
    #example run: python match_npz_numbers.py train --student_dir=200x200-12500-gen --teacher_dir=teacher_masks --out_dir=test_fixed

    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=['train', 'val'])
    parser.add_argument('--student_dir', default='data_raw')
    parser.add_argument('--teacher_dir', default='data_raw')
    parser.add_argument('--out_dir', default='data_gen')

    args = parser.parse_args()

    student_dir = os.path.join(args.student_dir, args.split)
    teacher_dir = os.path.join(args.teacher_dir, args.split)
    out_dir = os.path.join(args.out_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)

    filter_out_missing(student_dir, teacher_dir, out_dir)


