# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-09-09 18:46:06
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-09-10 18:09:38

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
from sklearn.cluster import KMeans

import click

import logging

logging.basicConfig(level=logging.INFO)

def create_dataset_from_dir(images_dir, name=None, persistent=False):
    dataset = fo.Dataset.from_images_dir(images_dir=images_dir,
                                         name=name,
                                         persistent=persistent,
                                         recursive=True)

    return dataset

@click.command()
@click.option('--images_dir', '-i')
@click.option('--dataset_name', '--name', '-n')
@click.option('--persistent', '-p', type=bool, default=True, is_flag=True)
@click.option('--n_clusters', default=None)
def main(images_dir, dataset_name, persistent, n_clusters):

    if fo.dataset_exists(dataset_name):
        logging.info('Dataset {} already exists.'.format(dataset_name))
        dataset = fo.load_dataset(dataset_name)
    else:
        dataset = create_dataset_from_dir(images_dir, dataset_name)
        dataset.persistent = persistent

    ####################
    # Compute embeddings
    ####################

    logging.info('Computing embedding ...')
    #TIP: run foz.list_zoo_models() to see the whole list of models
    # model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
    model = foz.load_zoo_model('clip-vit-base32-torch')
    embeddings = dataset.compute_embeddings(model)
 
    logging.info('Working on the 2D for visualization ...')
    # Image embeddings
    fob.compute_visualization(dataset,
                              brain_key="latent_space",
                              embeddings=embeddings,
                              )

    ####################
    # K-means Clustering
    ####################
    if n_clusters != None:
        logging.info('Computing k-means clustering ...')
        # TODO: change the "n_clusters" variable
        k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10)
        k_means.fit(embeddings)

        for i, sample in enumerate(dataset):
            cluster_id = k_means.labels_[i]
            sample['cluster_id'] = cluster_id
            sample.save()

    ################
    # Launch the App
    ################
    session = fo.launch_app(dataset)

    # Blocks execution until the App is closed
    session.wait()

if __name__ == "__main__":
    main()