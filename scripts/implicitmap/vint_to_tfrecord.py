"""
Converts data from the BridgeData raw format to TFRecord format.

Consider the following directory structure for the input data:

    sacson_raw/
        month-day-year-location-run/
            0.jpg
            ...
            n.jpg
            traj_data.pkl
        

The --depth parameter controls how much of the data to process at the
--input_path; for example, if --depth=1, then --input_path should be
"sacson", and all data will be processed.

Can write directly to Google Cloud Storage, but not read from it.
"""

import glob
import logging
import os
import pickle
import random
import yaml
from datetime import datetime
from functools import partial
from multiprocessing import Pool
import torch

import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from tqdm_multiprocess import TqdmMultiProcessPool

import dlimp as dl
from dlimp.utils import read_resize_encode_image, tensor_feature
from vint_train.data.vint_dataset import ViNT_Dataset
from torch.utils.data import DataLoader, ConcatDataset, Subset


FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_integer(
    "depth",
    5,
    "Number of directories deep to traverse to the dated directory. Looks for"
    "{input_path}/dir_1/dir_2/.../dir_{depth-1}/2022-01-01_00-00-00/...",
)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_float(
    "train_proportion", 0.9, "Proportion of data to use for training (rather than val)"
)
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")
flags.DEFINE_integer("shard_size", 200, "Maximum number of trajectories per shard")
flags.DEFINE_string('text_annots', None, 'text annotations path', required=False)
flags.DEFINE_string('config', "sacson.yaml", 'config path', required=False)

IMAGE_SIZE = (128, 128)

def process_images(path):  # processes images at a trajectory level
    image_dirs = set(os.listdir(str(path)))
    image_paths = [
        sorted(
            glob.glob(os.path.join(path, image_dir, "*.jpg")),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        for image_dir in image_dirs
    ]

    filenames = [[path.split("/")[-1] for path in x] for x in image_paths]
    assert all(x == filenames[0] for x in filenames)

    d = {
        image_dir: [read_resize_encode_image(path, IMAGE_SIZE) for path in p]
        for image_dir, p in zip(image_dirs, image_paths)
    }

    return d


def process_state(path):
    with open(os.path.join(path, "traj_data.pkl"), "rb") as f:
        traj_data = pickle.load(f)
    
    start_index = curr_time
    end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
    return 

def _compute_actions(self, traj_data, curr_time, goal_time):
    start_index = curr_time
    end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
    yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
    positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]
    goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

    if len(yaw.shape) == 2:
        yaw = yaw.squeeze(1)

    if yaw.shape != (self.len_traj_pred + 1,):
        const_len = self.len_traj_pred + 1 - yaw.shape[0]
        yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
        positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

    assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
    assert positions.shape == (self.len_traj_pred + 1, 2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

    waypoints = to_local_coords(positions, positions[0], yaw[0])
    goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])

    assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

    if self.learn_angle:
        yaw = yaw[1:] - yaw[0]
        actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
    else:
        actions = waypoints[1:]
    
    if self.normalize:
        actions[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
        goal_pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing

    assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

    return actions, goal_pos

def process_actions(path):
    with open(os.path.join(path, "traj_data.pkl"), "rb") as f:
        traj_data = pickle.load(f)
    actions, goal_pos = _compute_actions(0, 1)
    return act_list


def process_lang(path):
    fp = os.path.join(path, "lang.json")
    text = ""  # empty string is a placeholder for missing text
    if os.path.exists(fp):
        with open(fp, "r") as f:
            text = f.readline().strip()
    return text


# create a tfrecord for a group of trajectories
def create_tfrecord(items, output_path, tqdm_func, global_tqdm):
    writer = tf.io.TFRecordWriter(output_path)
    print(items)
    for idx, item in enumerate(iter(items)):
        try: 
            (obs_image,
            goal_image,
            action_label,
            dist_label,
            goal_pos,
            dataset_index,
            action_mask,) = item 

            out = dict()

            out["obs_image"] = obs_image.numpy()
            out["goal_image"] = goal_image.numpy()
            out["actions"] = action_label.numpy()
            out["goal_pos"] = goal_pos.numpy()
            out["dataset_index"] = dataset_index.numpy()
            out["action_mask"] = action_mask.numpy()
            out["lang"] = ""

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        k: tensor_feature(v)
                        for k, v in dl.transforms.flatten_dict(out).items()
                    }
                )
            )
            writer.write(example.SerializeToString())
        except Exception as e:
            import sys
            import traceback

            traceback.print_exc()
            logging.error(f"Error processing {idx}")
            sys.exit(1)

        global_tqdm.update(1)

    writer.close()
    global_tqdm.write(f"Finished {output_path}")


def get_traj_paths(path, train_proportion):
    train_traj = []
    val_traj = []

    all_traj = glob.glob(path)
    if not all_traj:
        logging.info(f"no trajs found in {path}")

    random.shuffle(all_traj)
    train_traj += all_traj[: int(len(all_traj) * train_proportion)]
    val_traj += all_traj[int(len(all_traj) * train_proportion) :]

    return train_traj, val_traj


def main(_):
    assert FLAGS.depth >= 1

    if tf.io.gfile.exists(FLAGS.output_path):
        if FLAGS.overwrite:
            logging.info(f"Deleting {FLAGS.output_path}")
            tf.io.gfile.rmtree(FLAGS.output_path)
        else:
            logging.info(f"{FLAGS.output_path} exists, exiting")
            return

    # each path is a directory that contains dated directories
    paths = glob.glob(os.path.join(FLAGS.input_path, *("*" * (FLAGS.depth - 1))))
    # Should only be sacson 
    print(paths)

    ## Load configs
    with open("defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(FLAGS.config, "r") as f:
        user_config = yaml.safe_load(f)
    
    config.update(user_config)

    dataset_name = FLAGS.input_path.split("/")[-1]

    data_config = config["datasets"][dataset_name]
    if "negative_mining" not in data_config:
        data_config["negative_mining"] = True
    if "goals_per_obs" not in data_config:
        data_config["goals_per_obs"] = 1
    if "end_slack" not in data_config:
        data_config["end_slack"] = 0
    if "waypoint_spacing" not in data_config:
        data_config["waypoint_spacing"] = 1

    train_dataset = []
    test_dataset = []
    dataset_name = paths[0].split("/")[-1]
    for data_split_type in ["train", "test"]:
        if data_split_type in data_config:
            dataset = ViNT_Dataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config[data_split_type],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                waypoint_spacing=data_config["waypoint_spacing"],
                min_dist_cat=config["distance"]["min_dist_cat"],
                max_dist_cat=config["distance"]["max_dist_cat"],
                min_action_distance=config["action"]["min_dist_cat"],
                max_action_distance=config["action"]["max_dist_cat"],
                negative_mining=data_config["negative_mining"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                context_size=config["context_size"],
                context_type=config["context_type"],
                end_slack=data_config["end_slack"],
                goals_per_obs=data_config["goals_per_obs"],
                normalize=config["normalize"],
                goal_type=config["goal_type"],
            )
            if data_split_type == "train":
                train_dataset = dataset
            else:
                test_dataset = dataset
    # shard paths
    train_shards = [Subset(train_dataset, np.arange(int((i-1)*FLAGS.shard_size), int(i*FLAGS.shard_size))) for i in range(int(np.ceil(len(train_dataset) / FLAGS.shard_size)))]
    val_shards = [Subset(test_dataset, np.arange(int((i-1)*FLAGS.shard_size), int(i*FLAGS.shard_size))) for i in range(int(np.ceil(len(test_dataset) / FLAGS.shard_size)))]
    # create output paths
    tf.io.gfile.makedirs(os.path.join(FLAGS.output_path, "train"))
    tf.io.gfile.makedirs(os.path.join(FLAGS.output_path, "val"))
    train_output_paths = [
        os.path.join(FLAGS.output_path, "train", f"{i}.tfrecord")
        for i in range(len(train_shards))
    ]
    val_output_paths = [
        os.path.join(FLAGS.output_path, "val", f"{i}.tfrecord")
        for i in range(len(val_shards))
    ]
    print("Starting create tfrecord tasks")
    # create tasks (see tqdm_multiprocess documenation)
    tasks = [
        (create_tfrecord, (train_shards[i], train_output_paths[i]))
        for i in range(len(train_shards))
    ] + [
        (create_tfrecord, (val_shards[i], val_output_paths[i]))
        for i in range(len(val_shards))
    ]

    # run tasks
    pool = TqdmMultiProcessPool(FLAGS.num_workers)
    with tqdm.tqdm(
        total=len(train_dataset) + len(test_dataset),
        dynamic_ncols=True,
        position=0,
        desc="Total progress",
    ) as pbar:
        pool.map(pbar, tasks, lambda _: None, lambda _: None)


if __name__ == "__main__":
    app.run(main)
