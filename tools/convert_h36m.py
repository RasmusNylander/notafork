import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, os.getcwd())
from lib.data.datareader_h36m import DataReaderH36M
from tqdm import tqdm


def save_clips(subset_name, root_path, data, labels):
    save_path = root_path / subset_name
    if not save_path.exists():
        save_path.mkdir(parents=True)
    i = 0
    for performer, performer_label in tqdm(zip(data, labels), desc=f"Saving {subset_name} data", unit="performer"):
        for activity, activity_label in tqdm(zip(performer, performer_label), leave=False, unit="activity"):
            activity, activity_label = activity.swapaxes(0, 1), activity_label.swapaxes(0, 1)
            for batch, label in zip(activity, activity_label):
                data_dict = {
                    "data_input": batch,
                    "data_label": label
                }
                with open(save_path / f"{i:08d}.pkl", "wb") as file:
                    pickle.dump(data_dict, file)
                i += 1

data_root = Path(__file__).parent.parent / "data" / "motion3d"
datareader = DataReaderH36M(n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243, dt_file = 'h36m_sh_conf_cam_source_final.pkl', dt_root=data_root.as_posix())
train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()
# print(train_data.shape, test_data.shape)
# assert len(train_data) == len(train_labels)
# assert len(test_data) == len(test_labels)

root_path = data_root / "MB3D_f243s81" / "H36M-SH"
if not root_path.exists():
    root_path.mkdir(parents=True)

save_clips("train", root_path, train_data, train_labels)
save_clips("test", root_path, test_data, test_labels)

