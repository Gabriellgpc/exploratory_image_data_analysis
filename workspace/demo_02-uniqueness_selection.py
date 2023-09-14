# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-09-09 18:46:06
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-09-10 12:05:06

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

import click
from tqdm import tqdm
import os 

import logging

logging.basicConfig(level=logging.INFO)

def create_dataset_from_dir(images_dir, name=None, persistent=False):
    dataset = fo.Dataset.from_images_dir(images_dir=images_dir,
                                         name=name,
                                         persistent=persistent,
                                         recursive=True)

    return dataset


def main():
    images_dir = '/home/lcondados/workspace/data/bdd100k/bdd100k/bdd100k/images/10k/train'
    name = 'my_images'
    max_num_images = 1000



    # dataset = foz.load_zoo_dataset("mnist", split="test")

    if fo.dataset_exists(name):
        logging.info('Dataset {} already exists.'.format(name))
        dataset = fo.load_dataset(name)
    else:
        dataset = create_dataset_from_dir(images_dir, name, persistent=True)

    # Launch the App
    session = fo.launch_app(dataset)

    # (Perform any additional operations here)

    # Compute embeddings
    # You will likely want to run this on a machine with GPU, as this requires
    # running inference on 10,000 images

    #TIP: run foz.list_zoo_models() to see the whole list of models

    # model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
    model = foz.load_zoo_model('clip-vit-base32-torch')
    embeddings = dataset.compute_embeddings(model)
 
    # Image embeddings
    ## "umap": fiftyone.brain.visualization.UMAPVisualizationConfig
    ## "tsne": fiftyone.brain.visualization.TSNEVisualizationConfig
    ## "pca": fiftyone.brain.visualization.PCAVisualizationConfig
    ## "manual": fiftyone.brain.visualization.ManualVisualizationConfig
    fob.compute_visualization(dataset,
                              brain_key="latent_space_tsne",
                              embeddings=embeddings,
                              method='tsne'
                              )

    # duplicates = fob.compute_exact_duplicates(dataset)
    # print(duplicates)

    # Index images by similarity
    res = fob.compute_similarity(
        dataset,
        embeddings=embeddings,
    )

    res.find_unique(max_num_images)
    uniq_view = dataset.select(res.unique_ids)
    # fo.launch_app(view=uniq_view)
    # fo.launch_app(view=uniq_view)
    session = fo.launch_app(view=uniq_view)

    fob.compute_uniqueness(dataset, embeddings=embeddings)

    # Blocks execution until the App is closed
    session.wait()

if __name__ == "__main__":
    main()