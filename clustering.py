import os
import yaml
import argparse
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms as transforms

import umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, Birch, DBSCAN
from sklearn.preprocessing import StandardScaler


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


### Approach ###
# Following cohesive approach is adopted from the paper:  https://arxiv.org/abs/2311.10093
# The Chosen One: Consistent Characters in Text-to-Image Diffusion Models

def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
      '-p',
      '--data',
      help='path to the data',
    )

    parser.add_argument(
      '-n',
      '--num_clusters',
      help='num of clusters to be detected',
    )

    args = parser.parse_args()

    return args


def config_2_args(path):
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    parser = argparse.ArgumentParser(description="Generate args from config")
    for key, value in yaml_data.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args = parser.parse_args([])
        
    return args



def load_data(dir_path):
    paths = [os.path.join(dir_path, path) for path in os.listdir(dir_path)]

    return paths



def load_image_emeddings(data_paths):
    def load_dinov2():
        dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda()
        dinov2_vitl14.eval()
        return dinov2_vitl14
    
    def load_image(image_path):
        image = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(image).unsqueeze(0).to('cuda')

    model = load_dinov2()
    
    # - loop through each image and get the embeddings
    for i, data_path in enumerate(data_paths):
        img_arr = load_image(data_path)
        img_embeddings = model.forward(img_arr).flatten(start_dim=1).cpu().detach()
        embeddings_stack = img_embeddings if i == 0 else torch.vstack((embeddings_stack, img_embeddings))

    # clean cuda-memory
    del model
    torch.cuda.empty_cache()
    
    return embeddings_stack



def get_embeddings(data_type, data_paths):
    if data_type == 'image':
        return load_image_emeddings(data_paths=data_paths)        
    else:
        raise NotImplementedError('Support for other data types is not implemented yet!!!')



def kmeans_clustering(no_of_clusters, dmin_c, data_points, images = None):
    """Apply k-means clustering to the data
    """

    kmeans = KMeans(n_clusters=no_of_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(data_points)
    labels = kmeans.labels_

    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))

    # - filter out clusters where count dmin_c
    selected_clusters = [cluster for cluster, count in cluster_counts.items() if count > dmin_c]
    selected_centers = kmeans.cluster_centers_[selected_clusters]
    selected_labels = []
    
    selected_labels = [label for label in labels if label in selected_clusters]
    selected_labels = make_continuous(selected_labels)
    selected_labels = np.array(selected_labels)
    
    selected_elements = np.array([data_points[i] for i, label in enumerate(labels) if label in selected_clusters])
    if images:
        selected_images = [images[i] for i, label in enumerate(labels) if label in selected_clusters]
    else:
        selected_images = None

    return selected_centers, selected_labels, selected_elements, selected_images



def dbscan_clustering(data_points, radius, dmin_c, images = None):
    """Apply DBSCAN clustering to the data"""

    dbscan = DBSCAN(eps=radius, min_samples=dmin_c)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_points)

    # Applying umap
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(scaled_features)

    dbscan.fit(embedding)
    labels = dbscan.labels_
    for i, label in enumerate(labels): # In-place, rearrange labels: [-1, 0, 1, 2, 3] to [0, 1, 2, 3, 4], now cluster 0: noise for this case!
        if label == -1:
            labels[i] = 0
        else:
            labels[i] = label + 1

    # Calculating centroids
    # For more info refer to the dbscan.ipynb notebook..
    cluster_centroids = []
    for i, label in enumerate(np.unique(labels).tolist()):
            centroid = np.mean(embedding[labels==label, :], axis=0) # taking the mean value as the cluster centroid
            cluster_centroids.append(centroid)
    cluster_centroids = np.array(cluster_centroids)

    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    # - filter out clusters where count dmin_c
    selected_clusters = [cluster for cluster, count in cluster_counts.items() if count > dmin_c] 
    selected_centers = cluster_centroids[selected_clusters]
    selected_labels = []
    
    selected_labels = [label for label in labels if label in selected_clusters]
    selected_labels = make_continuous(selected_labels)
    selected_labels = np.array(selected_labels)
    
    selected_elements = np.array([data_points[i] for i, label in enumerate(labels) if label in selected_clusters])
    scaled_features_selected = scaler.fit_transform(selected_elements)
    selected_elements = reducer.fit_transform(scaled_features_selected)

    if images:
        selected_images = [images[i] for i, label in enumerate(labels) if label in selected_clusters]
    else:
        selected_images = None

    return selected_centers, selected_labels, selected_elements, selected_images, len(np.unique(selected_labels).tolist())



def make_continuous(lst):
    """Make the mapping
    
    Returns: list of indexes
    """
    unique_elements = sorted(set(lst))
    mapping = {elem: i for i, elem in enumerate(unique_elements)}
    return [mapping[elem] for elem in lst]



def compare_features(features, cluster_centroid):
    # Calculate the Euclidean distance between the two feature vectors
    distance = np.linalg.norm(features - cluster_centroid)
    return distance



def find_cohesive_clusters(centers, elements, labels):
    """Find the most cohesive cluster given set of clusters"""

    # each data point subtract its coresponding center
    center_norms = np.linalg.norm(centers[labels] - elements, axis=-1, keepdims=True) 
    unique_labels = np.unique(labels) # unique labels for formed clusters
    cohesions = np.zeros(len(unique_labels))
    for label_id in range(len(np.unique(labels))):
        cohesions[label_id] = sum(center_norms[labels == label_id]) / sum(labels == label_id)
    
    # find the most cohesive cluster, and save the corresponding sample
    print(cohesions, unique_labels)
    min_cohesion_idx = np.argmin(cohesions)

    return unique_labels[min_cohesion_idx].item()



def visualize_2D(algo, no_of_clusters, data, labels):
    """Saving the 2D vizualizations of the resulted clusters
       using TSNE dimensionality reduction technique.
    """

    # visualize 2D t-SNE results
    plt.figure(figsize=(20, 16))
    tsne = TSNE(n_components=2, random_state=42, perplexity=len(data) - 1)
    embeddings_2d = tsne.fit_transform(data)
    
    for i in range(no_of_clusters):
        cluster_points = np.array(embeddings_2d[labels==i])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}", s=100)
        plt.legend()
    
    # saving the viz result
    save_path = f'./output/{algo}_resutls'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/tsne_viz.png")



def create_dataframe(image_paths, cluster_ids):
    """Store low-dimensional embeddings with metadata: image filenames"""
    df = pd.DataFrame({'image_path': image_paths,
                      'cluster_id': cluster_ids})

    return df



def make_clusters(df, algo, cohesive_cluster_ids):
    """Create cluster directories for resulted clusters for the algo"""

    path = f'./output/resutled_clusters/{algo}'

    if os.path.exists(path):
        shutil.rmtree(path)

    # creating clusters
    for cl in df['cluster_id'].to_list():
        cluster_dir = os.path.join(path, f'cluster_{cl}')
        os.makedirs(cluster_dir, exist_ok=True)

        for i in range(len(df)):
            if df['cluster_id'][i] == cl:
                shutil.copy(df['image_path'][i], cluster_dir)

    # copying most cohesive clusters
    if isinstance(cohesive_cluster_ids, int):
        cohesive_cluster_ids = [cohesive_cluster_ids]
    for cl in cohesive_cluster_ids:
        cluster_dir = os.path.join(path, f'most_cohesive_{cl}')
        os.makedirs(cluster_dir, exist_ok=True)

        for i in range(len(df)):
            if df['cluster_id'][i] == cl:
                shutil.copy(df['image_path'][i], cluster_dir) 



if __name__ == '__main__':
    # args = argument_parser()
    main_args = config_2_args('./config/main_config.yaml')

    # -load data
    data_paths = load_data(main_args.data)
    # print(data_paths)

    # get the embeddings
    embeddings = get_embeddings(data_type='image', data_paths=data_paths)    

    # Based on the algorithm, find the clusters...
    if main_args.algo_name == 'kmeans++':
        args = config_2_args('./config/kmeans++_config.yaml') 
        # -clustering
        centers, labels, elements, selected_images = kmeans_clustering(args.num_clusters, args.dim_c, embeddings, images=data_paths)
        # print(centers.shape, labels, elements.shape)

        # -visualize
        if args.vis:
            visualize_2D(main_args.algo_name, args.num_clusters, elements, labels)

        # -find the most cohesive cluster
        cohesive_cluster_id = find_cohesive_clusters(centers=centers, elements=elements, labels=labels)
        print(cohesive_cluster_id)
        print('Most cohesive clusters: ', cohesive_cluster_id) 

        # create folders for each resulted cluster
        df = create_dataframe(image_paths=selected_images, cluster_ids=labels)
        make_clusters(df=df, algo='kmeans++', cohesive_cluster_ids=cohesive_cluster_id)

    elif main_args.algo_name == 'DBSCAN':
        args = config_2_args('./config/dbscan_config.yaml') 
        # -clustering
        centers, labels, elements, selected_images, num_clusters = dbscan_clustering(data_points=embeddings, radius=args.radius, dmin_c=args.dim_c, images=data_paths)
        # print(centers.shape, labels, elements.shape)

        # -visualize
        if args.vis:
            visualize_2D(main_args.algo_name, num_clusters, elements, labels)

        # -find the most cohesive cluster
        cohesive_cluster_id = find_cohesive_clusters(centers=centers, elements=elements, labels=labels)
        print('Most cohesive clusters: ', cohesive_cluster_id) 

        # create folders for each resulted cluster
        df = create_dataframe(image_paths=selected_images, cluster_ids=labels)
        make_clusters(df=df, algo='DBSCAN', cohesive_cluster_ids=cohesive_cluster_id)

    else:
        raise ValueError('Error in config yaml file!!!')
        


