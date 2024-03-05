import os
import sys
import pickle
from itertools import product
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
        resolution: list[list[ndarray]],
        action: list[list[str]] | None = None,
        factor: list[list[int]] | None = None,
        gt: list[list[ndarray]] | None = None
):
    save_path = root_path / subset_name
    if not save_path.exists():
        save_path.mkdir(parents=True)
    i = 0
    actions = ['Direction', 'Discuss', 'Eating', 'Greet', 'Phone', 'Pose', 'Purchase', 'Sitting', 'SittingDown', 'Smoke', 'Photo',
     'Wait', 'Walk', 'WalkDog', 'WalkTwo']
    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']
    with tqdm(total=sum(map(lambda x: sum(map(lambda z: z.shape[1], x)), pose_2d)), desc=f"Saving {subset_name} data", unit="files") as pbar:
        for performer_index in range(len(performers)):
            for activity_index in range(len(pose_2d[performer_index])):
                activity_2d = pose_2d[performer_index][activity_index].swapaxes(0, 1)
                activity_target = target[performer_index][activity_index].swapaxes(0, 1)
                res = resolution[performer_index][activity_index].swapaxes(0, 1)
                for j in range(len(activity_2d)):
                    data_dict = {
                        "data_input": activity_2d[j],
                        "data_label": activity_target[j],
                        "meta": {
                            "performer": performers[performer_index],
                            "resolution": res[j],
                            "factor": factor[performer_index][activity_index][:, j] if factor else None,
                            "gt": gt[performer_index][activity_index][:, j] if gt else None,
                            "action": actions.index(action[performer_index][activity_index][j]) if action else None
                        }
                    }
                    with open(save_path / f"{i:08d}.pkl", "wb") as file:
                        pickle.dump(data_dict, file)
                    pbar.update(1); i += 1

data_root = Path(__file__).parent.parent / "data" / "motion3d"
datareader = DataReaderH36M(n_frames=243, sample_stride=1, data_stride_train=81, data_stride_test=243, dt_file = 'h36m_sh_conf_cam_source_final.pkl', dt_root=data_root.as_posix())
train_data, test_data, train_labels, test_labels, resolution_train, resolution_test, factor, actions, gt = datareader.get_sliced_data()
# print(train_data.shape, test_data.shape)
# assert len(train_data) == len(train_labels)
# assert len(test_data) == len(test_labels)

root_path = data_root / "MB3D_f243s81" / "H36M-SH"
if not root_path.exists():
    root_path.mkdir(parents=True)

save_clips("train", root_path, train_data, train_labels, [1, 5, 6, 7, 8], resolution_train)
save_clips("test", root_path, test_data, test_labels, [9, 11], resolution_test, actions, factor, gt)

