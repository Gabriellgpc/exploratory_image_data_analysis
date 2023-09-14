# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-09-09 18:46:06
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-09-10 12:01:26

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

    dataset = create_dataset_from_dir(images_dir, name, persistent=True)

    #TIP: run foz.list_zoo_models() to see the whole list of models

    # Image embeddings
    model = foz.load_zoo_model('clip-vit-base32-torch')
    embeddings = dataset.compute_embeddings(model)
 
    # latent space visualization on 2D
    fob.compute_visualization(dataset,
                              brain_key="latent_space_tsne",
                              embeddings=embeddings
                              )

    # Index images by similarity
    fob.compute_similarity(
        dataset,
        embeddings=embeddings,
    )

    # Launch the App
    session = fo.launch_app(dataset)

    # Blocks execution until the App is closed
    session.wait()

if __name__ == "__main__":
    main()