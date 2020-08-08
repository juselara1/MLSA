# Importing libraries
import h5py, glob, os, gc
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

# Dataset paths
partition_paths = {"ims_train": "/tf/TCGA/train/",
                   "ims_val": "/tf/TCGA/val/",
                   "ims_test": "/tf/TCGA/test/"}

subdirs = ["G67/Gleason_6/", "G67/Gleason_7/", "G8910/Gleason_8/",
           "G8910/Gleason_9/", "G8910/Gleason_10/"]

# method to get all images in a partition
def get_number_of_ims(path, subdirs):
    count = 0
    for subdir in subdirs:
        for patient_id in os.listdir(path+subdir):
            count += len(os.listdir(path+subdir+patient_id)) 
    return count

# Dictionary with the total number of images per partition
partition_lengths = {key:get_number_of_ims(path, subdirs) for key, path in partition_paths.items()}

# Defining an special type to store images paths
dt = h5py.special_dtype(vlen=str)

# We create a hdf5 file to store the dataset
with h5py.File("/tf/TCGA/images.h5", "w") as df:
    for partition in partition_paths:
        # Creating empty tensor to store all the images of current partition
        df.create_dataset(partition, shape=(partition_lengths[partition], 256, 256, 3), dtype=np.uint8)
        # Name of the labels of current partition
        labels_name = "y_"+partition.split("_")[1]
        df.create_dataset(labels_name, shape=(partition_lengths[partition], ), dtype=np.uint8)
        # Creating empty array to store patients id
        id_name = "id_"+partition.split("_")[1]
        df.create_dataset(id_name, (partition_lengths[partition], ), dtype=dt)
        
        print("="*30)
        print(f"Saving partition {partition} with labels {labels_name}:")
        cont = 0
        for gleason, gl_path in enumerate(subdirs):
            for patient_id in os.listdir(os.path.join(partition_paths[partition], gl_path)):
                for im_path in os.listdir(os.path.join(partition_paths[partition], gl_path, patient_id)):
                    # Loading and saving image
                    df[partition][cont] = np.array(load_img(os.path.join(partition_paths[partition], gl_path, patient_id, im_path)))
                    # Saving label
                    df[labels_name][cont] = gleason
                    # Saving patient id
                    df[id_name][cont] = patient_id
                    cont +=1
                    print(f"{cont}/{partition_lengths[partition]}\t{cur_id}", end="\r")