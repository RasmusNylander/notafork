import os
import sys
import pickle
from pathlib import Path

from numpy import ndarray

sys.path.insert(0, os.getcwd())
from lib.data.datareader_h36m import DataReaderH36M
from tqdm import tqdm


def save_clips(
        subset_name: str,
        root_path: Path,
        pose_2d: list[list[ndarray]],
        target: list[list[ndarray]],
        performers: list[int],
):
    save_path = root_path / subset_name
    if not save_path.exists():
        save_path.mkdir(parents=True)
    i = 0
    for performer_2d, performer_target, performer in tqdm(list(zip(pose_2d, target, performers)), desc=f"Saving {subset_name} data", unit="performer"):
        for activity_2d, activity_target in zip(performer_2d, performer_target):
            activity_2d, activity_target = activity_2d.swapaxes(0, 1), activity_target.swapaxes(0, 1)
            for input, target in zip(activity_2d, activity_target):
                data_dict = {
                    "data_input": input,
                    "data_label": target,
                    "meta": {
                        "performer": performer,
                    }
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

save_clips("train", root_path, train_data, train_labels, [1, 5, 6, 7, 8])
save_clips("test", root_path, test_data, test_labels, [9, 11])

