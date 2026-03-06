import os
import numpy as np
import mne
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


path_raw = './data/sleep-edf-database-expanded-1.0.0/sleep-cassette'  # Path to the directory containing .edf files
path_output = './data/SleepEDF_78/'
target_fs = 100
epoch_sec = 30
sample_per_epoch = target_fs * epoch_sec

stage_dict = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}


def preprocess_subject(psg_file, label_file):
    # 1. Read raw data
    raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)

    # 2. Check and select channels
    available_channels = raw.ch_names
    target_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']

    select_ch = [ch for ch in target_channels if ch in available_channels]
    if len(select_ch) < 4:
        print(f"Warning: {psg_file} only has {len(select_ch)} target channels.")

    raw.pick(select_ch)

    # 3. Preprocessing: Bandpass filter (0.3-35Hz)
    raw.filter(0.3, 35, fir_design='firwin', verbose=False)

    # 4. Resampling
    if raw.info['sfreq'] != target_fs:
        raw.resample(target_fs, npad="auto")

    data = raw.get_data()  # (Channels, Points)

    # 5. Read labels
    annot = mne.read_annotations(label_file)
    n_epochs = int(raw.n_times / sample_per_epoch)
    labels = np.full(n_epochs, 5)

    for a in annot:
        onset_sec = a['onset']
        duration_sec = a['duration']
        stage_str = a['description']

        if stage_str in stage_dict:
            start_epoch = int(round(onset_sec / epoch_sec))
            n_stage_epochs = int(round(duration_sec / epoch_sec))
            end_epoch = start_epoch + n_stage_epochs
            if end_epoch > n_epochs: end_epoch = n_epochs
            labels[start_epoch:end_epoch] = stage_dict[stage_str]

    # 6. Clip epochs (only keep non-Wake epochs with context)
    non_wake_idx = np.where((labels != 0) & (labels <= 4))[0]
    if len(non_wake_idx) == 0: return None, None

    start_idx = max(0, non_wake_idx[0] - 60)
    end_idx = min(n_epochs, non_wake_idx[-1] + 60)

    psg_clipped = []
    label_clipped = []

    for i in range(start_idx, end_idx):
        if labels[i] <= 4:
            snippet = data[:, i * sample_per_epoch: (i + 1) * sample_per_epoch]
            if snippet.shape[1] == sample_per_epoch:
                snippet = (snippet - np.mean(snippet, axis=1, keepdims=True)) / (
                            np.std(snippet, axis=1, keepdims=True) + 1e-8)
                psg_clipped.append(snippet)
                label_clipped.append(labels[i])

    return np.array(psg_clipped), np.array(label_clipped)


if __name__ == "__main__":
    if not os.path.exists(path_output): os.makedirs(path_output)

    files = os.listdir(path_raw)
    psg_files = sorted([f for f in files if 'PSG' in f and f.endswith('.edf')])
    hyp_files = sorted([f for f in files if 'Hypnogram' in f and f.endswith('.edf')])

    fold_psg, fold_label, fold_len = [], [], []

    for p_f, h_f in zip(psg_files, hyp_files):
        print(f"Processing {p_f}...")
        p_data, l_data = preprocess_subject(
            os.path.join(path_raw, p_f),
            os.path.join(path_raw, h_f)
        )

        if p_data is not None:
            l_one_hot = np.eye(5)[l_data.astype(int)]
            fold_psg.append(p_data.astype(np.float32))
            fold_label.append(l_one_hot.astype(np.float32))
            fold_len.append(len(p_data))

    # Save to .npz
    np.savez(os.path.join(path_output, 'SleepEDF_78.npz'),
             Fold_data=np.array(fold_psg, dtype=object),
             Fold_label=np.array(fold_label, dtype=object),
             Fold_len=np.array(fold_len))
    print(f"Successfully saved to {path_output}, Total subjects: {len(fold_psg)}")