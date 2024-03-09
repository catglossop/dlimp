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
from typing import Tuple

import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from tqdm_multiprocess import TqdmMultiProcessPool

import dlimp as dl
from dlimp.utils import read_resize_encode_image, tensor_feature, resize_image
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
flags.DEFINE_integer('traj_len', 20, "num steps per traj", required=False)

IMAGE_SIZE = (128, 128)

def resize_encode_image(tensor: torch.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    """Reads, decodes, resizes, and then re-encodes an image."""
    image = tf.convert_to_tensor(np.transpose((tensor.numpy()*255).astype(np.uint8), axes=(1,2,0)))
    image = resize_image(image, size)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return tf.io.encode_jpeg(image, quality=95)

# create a tfrecord for a group of trajectories
def create_tfrecord(items, output_path, tqdm_func=None, global_tqdm=None):
    writer = tf.io.TFRecordWriter(output_path)
    for idx, item in enumerate(iter(items)):
        try: 
            (obs_image,
            goal_image,
            action_label,
            dist_label,
            goal_pos,
            dataset_index,
            action_mask,
            lang,) = item 

            out = dict()

            obs_image = torch.cat((obs_image, goal_image), dim=0) 

            obs = [resize_encode_image(obs_image[idx:idx+3,:,:], IMAGE_SIZE) for idx in range(int(np.ceil(obs_image.shape[0]/3)))]

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "obs": tensor_feature(obs),
                        "lang": tensor_feature(lang),
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

        # global_tqdm.update(1)

    writer.close()
    # global_tqdm.write(f"Finished {output_path}")


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
    for (shard, tf_path) in tqdm.tqdm(zip(train_shards, train_output_paths), total=len(train_shards)):
        create_tfrecord(shard, tf_path)
    
    for (shard, tf_path) in tqdm.tqdm(zip(val_shards, val_output_paths), total=len(val_shards)):
        create_tfrecord(shard, tf_path)
    # tasks = []
    #     (create_tfrecord, (train_shards[i], train_output_paths[i]))
    #     for i in range(len(train_shards))
    # ] + [
    #     (create_tfrecord, (val_shards[i], val_output_paths[i]))
    #     for i in range(len(val_shards))
    # ]

    # run tasks
    # pool = TqdmMultiProcessPool(FLAGS.num_workers)
    # with tqdm.tqdm(
    #     total=len(train_dataset) + len(test_dataset),
    #     dynamic_ncols=True,
    #     position=0,
    #     desc="Total progress",
    # ) as pbar:
    #     pool.map(pbar, tasks, lambda _: None, lambda _: None)


if __name__ == "__main__":
    app.run(main)
