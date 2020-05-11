# Importing libraries
import h5py, gc
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# Image generator from a hdf5 file with class weights and data augmentation
class ImageAugmentationSequenceH5(tf.keras.utils.Sequence):
    def __init__(self, path, X_name, y_name=None, num_classes=2, shuffle=True, batch_size=32,
                 labels_transform=lambda i:i, class_weighted=False, categorical=False, return_y=True,
                 augmentation = {"preprocessing_function": lambda i:i, "apply":False,
                                 "random_brightness": {"max_delta": 0.5},
                                 "random_contrast": {"lower":0.6, "upper":1},
                                 "random_hue": {"max_delta": 0.1},
                                 "random_saturation": {"lower": 0.6, "upper":1},
                                 "random_rotation": {"minval": 0, "maxval": 2*np.pi},
                                 "horizontal_flip": True, "vertical_flip": True}):
        # Number of classes
        self.__num_classes = num_classes
        # Parameters for data augmentation
        self.__augmentation = augmentation
        # Path of h5 file
        self.__path = path
        # Number of images loaded per batch
        self.__batch_size = batch_size
        # Names in the h5 dataset of the images
        self.__X_name = X_name
        # Names in the h5 dataset of the labels
        self.__y_name = y_name
        # Specify if class weights must be considered
        self.__class_weighted = class_weighted
        # Specify if the dataset must be randomized after each epoch
        self.__shuffle = shuffle
        # Function to transform the labels
        self.__labels_transform = labels_transform        
        # Specify if labels must be one-hot encoded
        self.__categorical = categorical
        # Specify if labels must be returned
        self.__return_y = return_y
        # Get indexes
        self.__order_idx()
        
    def __order_idx(self):
        with h5py.File(self.__path, 'r') as df:
            # Get indexes for all the images
            self.__idx = np.arange(df[self.__X_name].shape[0])
            if self.__class_weighted:
                # Load labels
                labels = self.__labels_transform(df[self.__y_name][:])
                # Get counts per category
                categories, counts = np.unique(labels, return_counts=True)
                # Get class weights
                self.__class_weights = np.max(counts)-counts
                self.__idx_category = []
                for cat in categories:
                    if self.__class_weights[cat]==0:
                        idx_temp = self.__idx[labels==cat]
                        if self.__shuffle:
                            np.random.shuffle(idx_temp)
                        self.__idx_category.append(idx_temp)
                    else:
                        idx_temp = self.__idx[labels==cat]
                        if self.__shuffle:
                            np.random.shuffle(idx_temp)
                        self.__idx_category.append(np.concatenate([idx_temp,
                                                                   np.random.choice(self.__idx[labels==cat],
                                                                                    size=self.__class_weights[cat])]))
                # Total number of batches per epoch
                self.__n_batches = sum(map(lambda i:i.size, self.__idx_category))//self.__batch_size
            else:
                if self.__shuffle:
                    np.random.shuffle(self.__idx)
                # Total number of batches per epoch
                self.__n_batches = self.__idx.size//self.__batch_size
    def __len__(self):
        return self.__n_batches
    
    def __getitem__(self, idx):
        # Choosing the indexes of the current batch
        if self.__class_weighted:
            vals = []; acum = 0
            for i, cat_idx in enumerate(self.__idx_category):
                size = self.__batch_size-acum if i==self.__class_weights.size-1 else self.__batch_size//self.__class_weights.size
                acum += size
                vals.append(self.__idx_category[i][idx*size:(idx+1)*size])
            vals = np.sort(np.concatenate(vals))
        else:
            vals = np.sort(self.__idx[idx*self.__batch_size:(idx+1)*self.__batch_size])
        # loading the images and the labels of the batch
        with h5py.File(self.__path, 'r') as df:
            # Batch of images
            if not self.__class_weighted:
                vals = vals.tolist()
                X_batch = np.array(df[self.__X_name][vals]).astype(np.float32)
                y_batch = np.array(self.__labels_transform(df[self.__y_name][vals]))
            else:
                unique_vals, unique_idxs = np.unique(vals, return_index=True)
                
                unique_vals = unique_vals.tolist()
                X_batch1 = np.array(df[self.__X_name][unique_vals])
                y_batch1 = np.array(self.__labels_transform(df[self.__y_name][unique_vals]))
                
                bool_idx = np.array([1 for i in range(vals.size)], dtype=np.bool)
                bool_idx[unique_idxs] = False
                repeated_vals = vals[bool_idx].tolist()
                if not repeated_vals:
                    X_batch = X_batch1.astype(np.float32)
                    y_batch = y_batch1
                else:
                    X_batch2 = np.array(df[self.__X_name][repeated_vals])
                    y_batch2 = np.array(self.__labels_transform(df[self.__y_name][repeated_vals]))

                    X_batch = np.concatenate([X_batch1, X_batch2], axis=0).astype(np.float32)
                    y_batch = np.concatenate([y_batch1, y_batch2], axis=0)
            if self.__augmentation["apply"]:
                X_batch = self.__transformations(X_batch)
            else:
                X_batch = self.__augmentation["preprocessing_function"](X_batch)

            if self.__categorical:
                y_batch = tf.keras.utils.to_categorical(y_batch, num_classes=self.__num_classes)
                
        gc.collect()
        if idx==self.__n_batches-1:
            self.__order_idx()
        if self.__return_y:
            return (X_batch, y_batch)
        else:
            return X_batch
        
    def __transformations(self, batch):
        batch = tf.image.random_brightness(batch, **self.__augmentation["random_brightness"])
        batch = tf.image.random_contrast(batch, **self.__augmentation["random_contrast"])
        batch = tf.image.random_hue(batch, **self.__augmentation["random_hue"])
        batch = tf.image.random_saturation(batch, **self.__augmentation["random_saturation"])
        
        random_angles = tf.random.uniform(shape = (self.__batch_size, ), **self.__augmentation["random_rotation"])
        batch = tfa.image.transform(batch,
                                             tfa.image.transform_ops.angles_to_projective_transforms(
                                                 random_angles, tf.cast(tf.shape(batch)[1], tf.float32),
                                                 tf.cast(tf.shape(batch)[2], tf.float32)))
        if self.__augmentation["horizontal_flip"]:
            batch = tf.image.random_flip_left_right(batch)
        if self.__augmentation["vertical_flip"]:
            batch = tf.image.random_flip_up_down(batch)
        return self.__augmentation["preprocessing_function"](batch)