import os
import numpy as np
import argparse

def extract_time_windows(input_file, output_file, window_size=100, step_size=100):
    """
    Convert segmented EEG (7,40,5,62,400) into time windows (7,40,5,F,62,100).
    - window_size: number of time points per window (default=100 -> 0.5s at 200Hz).
    - step_size: stride for windows (default=100 -> non-overlapping).
    """

    data = np.load(input_file)  # shape: (7,40,5,62,400)
    B, C, I, Ch, T = data.shape
    assert T >= window_size, f"Clip length {T} < window size {window_size}"

    # Number of windows per clip
    F = (T - window_size) // step_size + 1
    print(f"Processing {input_file} -> {F} windows per clip")

    out = np.zeros((B, C, I, F, Ch, window_size), dtype=np.float32)

    for b in range(B):
        for c in range(C):
            for i in range(I):
                for f in range(F):
                    start = f * step_size
                    end = start + window_size
                    out[b, c, i, f] = data[b, c, i, :, start:end]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, out)
    print(f"Saved time-window EEG: {out.shape} -> {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing segmented EEG .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save time-windowed EEG .npy files")
    parser.add_argument("--window_size", type=int, default=100, help="Window length (samples)")
    parser.add_argument("--step_size", type=int, default=100, help="Step size (samples)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Process all subject files in input_dir
    for fname in sorted(os.listdir(args.input_dir)):
        if fname.endswith(".npy") and fname.startswith("sub"):
            input_file = os.path.join(args.input_dir, fname)
            output_file = os.path.join(args.output_dir, fname)
            extract_time_windows(input_file, output_file, args.window_size, args.step_size)
