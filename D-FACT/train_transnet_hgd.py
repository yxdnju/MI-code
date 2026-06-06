import argparse
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from copmare_model.tcfoormmer.model3_sensitivity import Model3
from data.dataset import eegDataset
from model.baseModel import baseModel
from utils import count_parameters


HGD_CHANNELS = [
    "EEG FC5", "EEG FC1", "EEG FC2", "EEG FC6", "EEG C3", "EEG C4",
    "EEG CP5", "EEG CP1", "EEG CP2", "EEG CP6", "EEG FC3", "EEG FCz",
    "EEG FC4", "EEG C5", "EEG C1", "EEG C2", "EEG C6", "EEG CP3",
    "EEG CPz", "EEG CP4", "EEG FFC5h", "EEG FFC3h", "EEG FFC4h",
    "EEG FFC6h", "EEG FCC5h", "EEG FCC3h", "EEG FCC4h", "EEG FCC6h",
    "EEG CCP5h", "EEG CCP3h", "EEG CCP4h", "EEG CCP6h", "EEG CPP5h",
    "EEG CPP3h", "EEG CPP4h", "EEG CPP6h", "EEG FFC1h", "EEG FFC2h",
    "EEG FCC1h", "EEG FCC2h", "EEG CCP1h", "EEG CCP2h", "EEG CPP1h",
    "EEG CPP2h",
]

CLASS_ORDER = ("right_hand", "left_hand", "rest", "feet")
CLASS_ALIASES = {
    "right_hand": {"right_hand", "right hand", "righthand", "right"},
    "left_hand": {"left_hand", "left hand", "lefthand", "left"},
    "rest": {"rest"},
    "feet": {"feet", "foot"},
}


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def dict_to_yaml(file_path, dict_to_write):
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(dict_to_write, f, allow_unicode=True, sort_keys=False)


def normalize_label_text(text):
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def make_hgd_event_id(raw):
    descriptions = list(dict.fromkeys(raw.annotations.description))
    normalized = {normalize_label_text(desc): desc for desc in descriptions}
    event_id = {}

    for class_index, class_name in enumerate(CLASS_ORDER, start=1):
        actual_description = None
        for alias in CLASS_ALIASES[class_name]:
            alias_norm = normalize_label_text(alias)
            if alias_norm in normalized:
                actual_description = normalized[alias_norm]
                break
        if actual_description is None:
            raise ValueError(
                f"Cannot find HGD annotation for class '{class_name}'. "
                f"Available annotations: {descriptions}"
            )
        event_id[actual_description] = class_index

    return event_id


def exponential_running_standardize(data, factor_new=1e-3, init_block_size=1000, eps=1e-4):
    data = np.asarray(data, dtype=np.float32)
    standardized = np.empty_like(data, dtype=np.float32)
    n_channels, n_times = data.shape

    init_stop = min(init_block_size, n_times)
    mean = np.mean(data[:, :init_stop], axis=1)
    var = np.var(data[:, :init_stop], axis=1)

    for i_time in range(n_times):
        sample = data[:, i_time]
        mean = (1.0 - factor_new) * mean + factor_new * sample
        var = (1.0 - factor_new) * var + factor_new * np.square(sample - mean)
        standardized[:, i_time] = (sample - mean) / np.maximum(np.sqrt(var), eps)

    if init_stop > 0:
        init_mean = np.mean(data[:, :init_stop], axis=1, keepdims=True)
        init_std = np.std(data[:, :init_stop], axis=1, keepdims=True)
        standardized[:, :init_stop] = (data[:, :init_stop] - init_mean) / np.maximum(init_std, eps)

    return standardized


def sanitize_cache_token(value):
    if value is None:
        return "none"
    return str(value).replace(".", "p").replace("-", "m")


def cache_path_for(config, split, subject_id):
    preprocess = config["preprocess"]
    cache_dir = Path(config.get("cache_dir", "dataset/hgd_model3_cache"))
    token = (
        f"sub{subject_id:02d}_{split}"
        f"_sf{sanitize_cache_token(preprocess['sfreq'])}"
        f"_n{preprocess['n_samples']}"
        f"_lc{sanitize_cache_token(preprocess.get('low_cut'))}"
        f"_hc{sanitize_cache_token(preprocess.get('high_cut'))}"
        f"_std{int(preprocess.get('standardize', True))}"
    )
    return cache_dir / f"{token}.npz"


def normalize_labels(labels, n_classes):
    labels = np.asarray(labels).squeeze().astype(np.int64)
    unique = sorted(np.unique(labels).tolist())

    if unique and min(unique) == 1 and max(unique) <= n_classes:
        labels = labels - 1
        unique = sorted(np.unique(labels).tolist())

    expected = list(range(n_classes))
    if set(unique).issubset(set(expected)):
        return labels

    if unique != expected:
        if len(unique) == n_classes:
            remap = {old: new for new, old in enumerate(unique)}
            labels = np.asarray([remap[item] for item in labels], dtype=np.int64)
        else:
            raise ValueError(f"Unexpected labels {unique}; expected {expected}.")

    return labels


def shuffle_data(data, labels):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    return data[indices], labels[indices]


def load_preprocessed_npy(split_dir, subject_id, n_classes):
    data_path = split_dir / f"{subject_id}_data.npy"
    label_path = split_dir / f"{subject_id}_label.npy"
    if not data_path.exists() or not label_path.exists():
        return None

    data = np.load(data_path).astype(np.float32)
    labels = normalize_labels(np.load(label_path), n_classes)
    return shuffle_data(data, labels)


def make_epochs(raw, events, event_id, n_samples):
    import mne

    sfreq = float(raw.info["sfreq"])
    tmax = (n_samples - 1) / sfreq
    return mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        preload=True,
        picks="eeg",
        event_repeated="drop",
        verbose=False,
    )


def preprocess_edf_split(config, split, subject_id):
    import mne

    preprocess = config["preprocess"]
    dataset_path = Path(config["data_path"])
    edf_path = dataset_path / split / f"{subject_id}.edf"
    if not edf_path.exists():
        raise FileNotFoundError(f"HGD EDF file not found: {edf_path}")

    print(f"Loading {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    channels = config["network_args"].get("channels", HGD_CHANNELS)
    available_channels = [channel for channel in channels if channel in raw.ch_names]
    missing_channels = sorted(set(channels) - set(available_channels))
    if missing_channels:
        raise ValueError(f"Missing HGD channels in {edf_path}: {missing_channels}")
    if hasattr(raw, "pick"):
        raw.pick(available_channels)
    else:
        raw.pick_channels(available_channels)

    sfreq = float(preprocess.get("sfreq", 250.0))
    if float(raw.info["sfreq"]) != sfreq:
        raw.resample(sfreq, verbose=False)

    low_cut = preprocess.get("low_cut")
    high_cut = preprocess.get("high_cut")
    if low_cut is not None or high_cut is not None:
        raw.filter(
            l_freq=low_cut,
            h_freq=high_cut,
            method="iir",
            iir_params={"order": 3, "ftype": "butter"},
            verbose=False,
        )

    event_id = make_hgd_event_id(raw)
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    n_samples = int(preprocess.get("n_samples", 1000))

    clean_trial_mask = None
    if preprocess.get("remove_artifacts", True):
        artifact_epochs = make_epochs(raw, events, event_id, n_samples)
        artifact_data_uv = artifact_epochs.get_data() * 1e6
        threshold = float(preprocess.get("artifact_threshold_uv", 800.0))
        clean_trial_mask = np.max(np.abs(artifact_data_uv), axis=(1, 2)) < threshold
        print(
            f"Subject {subject_id} {split}: clean trials "
            f"{int(np.sum(clean_trial_mask))}/{len(clean_trial_mask)}"
        )

    if preprocess.get("standardize", True):
        raw._data = exponential_running_standardize(
            raw.get_data(),
            factor_new=float(preprocess.get("standardize_factor", 1e-3)),
            init_block_size=int(preprocess.get("standardize_init_block", 1000)),
            eps=float(preprocess.get("standardize_eps", 1e-4)),
        )

    epochs = make_epochs(raw, events, event_id, n_samples)
    data = epochs.get_data().astype(np.float32)
    labels = normalize_labels(epochs.events[:, -1], config["num_classes"])

    if clean_trial_mask is not None:
        data = data[clean_trial_mask]
        labels = labels[clean_trial_mask]

    return shuffle_data(data, labels)


def load_hgd_split(config, split, subject_id):
    split_dir = Path(config["data_path"]) / split
    preprocessed = load_preprocessed_npy(split_dir, subject_id, config["num_classes"])
    if preprocessed is not None:
        data, labels = preprocessed
        print(f"Loaded cached npy subject {subject_id} {split}: {data.shape}, {labels.shape}")
        return data, labels

    cache_path = cache_path_for(config, split, subject_id)
    if cache_path.exists():
        cached = np.load(cache_path)
        data = cached["data"].astype(np.float32)
        labels = normalize_labels(cached["labels"], config["num_classes"])
        print(f"Loaded cached npz subject {subject_id} {split}: {data.shape}, {labels.shape}")
        return shuffle_data(data, labels)

    data, labels = preprocess_edf_split(config, split, subject_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, data=data, labels=labels)
    print(f"Saved cache: {cache_path}")
    return data, labels


def parse_subjects(subjects_value):
    if isinstance(subjects_value, int):
        return [subjects_value]
    if isinstance(subjects_value, (list, tuple)):
        return [int(item) for item in subjects_value]

    subjects = []
    for part in str(subjects_value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(item) for item in part.split("-", 1)]
            subjects.extend(range(start, end + 1))
        else:
            subjects.append(int(part))
    return subjects


def write_summary_header(summary_path, config, subjects):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Yunxingxing HGD Model3 Training Summary\n")
        f.write("=" * 72 + "\n")
        f.write(f"Data path: {config['data_path']}\n")
        f.write(f"Subjects: {subjects}\n")
        f.write(f"Epochs: {config['epochs']}\n")
        f.write(f"Batch size: {config['batch_size']}\n")
        f.write(f"Learning rate: {config['lr']}\n")
        f.write("-" * 72 + "\n")
        f.write("Subject | Accuracy | F1-Score | Kappa\n")


def append_summary_row(summary_path, subject_id, acc, f1_score, kappa):
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"Subject {subject_id:02d} | {acc:.6f} | {f1_score:.6f} | {kappa:.6f}\n")


def append_summary_average(summary_path, rows):
    if rows:
        avg_acc = float(np.mean([row["acc"] for row in rows]))
        avg_f1 = float(np.mean([row["f1"] for row in rows]))
        avg_kappa = float(np.mean([row["kappa"] for row in rows]))
    else:
        avg_acc = avg_f1 = avg_kappa = 0.0

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("-" * 72 + "\n")
        f.write(f"Average | {avg_acc:.6f} | {avg_f1:.6f} | {avg_kappa:.6f}\n")

    return avg_acc, avg_f1, avg_kappa


def train_subject(config, subject_id, run_folder):
    out_path = Path(config["out_folder"]) / config["network"] / f"sub{subject_id}" / run_folder
    out_path.mkdir(parents=True, exist_ok=True)
    dict_to_yaml(out_path / "config.yaml", config)

    set_random(int(config["random_seed"]))

    train_data, train_labels = load_hgd_split(config, "train", subject_id)
    test_data, test_labels = load_hgd_split(config, "test", subject_id)

    train_dataset = eegDataset(train_data, train_labels)
    test_dataset = eegDataset(test_data, test_labels)

    net_args = deepcopy(config["network_args"])
    net_args.pop("channels", None)
    net = Model3(**net_args)
    print("Trainable Parameters in the network are: " + str(count_parameters(net)))

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 1e-3)),
    )

    model = baseModel(net, config, optimizer, loss_func, result_savepath=str(out_path))
    acc, kappa, f1_score = model.train_test(train_dataset, test_dataset, 1, subject_id)
    return float(acc), float(f1_score), float(kappa)


def main(config):
    subjects = parse_subjects(config.get("subjects", "1-14"))
    run_folder = str(time.strftime("%Y-%m-%d--%H-%M", time.localtime()))
    summary_path = Path(config["summary_path"])
    rows = []

    config["network_args"]["n_channels"] = int(config["network_args"].get("n_channels", len(HGD_CHANNELS)))
    config["network_args"]["n_classes"] = int(config["network_args"].get("n_classes", config["num_classes"]))

    write_summary_header(summary_path, config, subjects)

    for subject_id in subjects:
        print(f"\n========== Subject {subject_id} ==========")
        acc, f1_score, kappa = train_subject(config, subject_id, run_folder)
        rows.append({"subject": subject_id, "acc": acc, "f1": f1_score, "kappa": kappa})
        append_summary_row(summary_path, subject_id, acc, f1_score, kappa)
        print(
            f"Subject {subject_id} summary: "
            f"acc={acc:.6f}, f1={f1_score:.6f}, kappa={kappa:.6f}"
        )

    avg_acc, avg_f1, avg_kappa = append_summary_average(summary_path, rows)
    print("\nFinal HGD Model3 Results")
    print(f"Average Accuracy: {avg_acc:.6f}")
    print(f"Average F1-Score: {avg_f1:.6f}")
    print(f"Average Kappa: {avg_kappa:.6f}")
    print(f"Summary saved to: {summary_path}")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model3 on Yunxingxing HGD data.")
    parser.add_argument("--config", default="config/hgd_model3.yaml", help="Path to yaml config.")
    parser.add_argument("--subjects", default=None, help="Override subjects, for example 1 or 1-14.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--data-path", default=None, help="Override HGD dataset path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.subjects is not None:
        cfg["subjects"] = args.subjects
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.data_path is not None:
        cfg["data_path"] = args.data_path

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    torch.set_num_threads(int(cfg.get("torch_num_threads", 10)))
    main(cfg)
