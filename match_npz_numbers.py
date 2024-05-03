import argparse
import os
import numpy as np
from tqdm import tqdm

def filter_out_missing(student_dir, teacher_dir, out_dir):
    student_sequences = sorted(os.listdir(student_dir))
    teacher_sequences = sorted(os.listdir(teacher_dir))
    for seq in tqdm(student_sequences):
        if seq in teacher_sequences:
            seq_path = os.path.join(student_dir, seq)
            loaded_npz = np.load(seq_path)
            np.savez_compressed(os.path.join(out_dir, seq))


if __name__ == '__main__':
    #example run: python match_npz_numbers.py train --student_dir=200x200_12500_gen --teacher_dir=teacher_masks --out_dir=fixed_200x200_12500_gen

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


