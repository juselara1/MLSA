# Multimodal Latent Semantic Alignment for Automated Prostate Tissue Classification and Retrieval

![mlsa](https://raw.githubusercontent.com/larajuse/Resources/f6fef253e70798db0901108b730b33036e64af30/mlsa/mlsa.svg)

## Abstract

This paper presents an information fusion method for the automatic classification and retrieval of prostate histopathology whole-slide images (WSIs). The approach employs a weakly-supervised machine learning model that combines a bag-of-features representation, kernel methods, and deep learning. The primary purpose of the method is to incorporate text information during the model training to enrich the representation of the images. It automatically learns an alignment of the visual and textual space since each modality has different statistical properties. This alignment enriches the visual representation with complementary semantic information extracted from the text modality. The method was evaluated in both classification and retrieval tasks over a dataset of 235 prostate WSIs with their pathology report from the TCGA-PRAD dataset. The results show that the multimodal-enhanced model outperforms unimodal models in both classification and retrieval. It outperforms state--of--the--art baselines by an improvement in WSI cancer detection of 4.74% achieving 77.01% in accuracy, and an improvement of 19.35% for the task of retrieving similar cases, obtaining 64.50% in mean average precision.

![fusion](https://raw.githubusercontent.com/larajuse/Resources/f6fef253e70798db0901108b730b33036e64af30/mlsa/fusion.svg)

## Requirements

If you have [anaconda](https://www.anaconda.com/) installed, you can create the same environment that we used for the experiments using the following command:

```
conda env create -f mlsa_env.yml
```

Then, you must activate the environment:

```
source activate mlsa
```

or 

```
conda activate mlsa
```

## Dataset

You can download the preprocessed TCGA-PRAD dataset (images, texts and labels) in the following link:

* https://drive.google.com/drive/folders/14pbie6QsN64i0ArpfOnpiyEtFDX8LWqS?usp=sharing

You can download the pretrained models in the following link:

* https://drive.google.com/drive/folders/1WPkJ_aCGHnA4SQf_9lmIB_GhBMP92NHH?usp=sharing

## Usage 

This implementation is based on `tf.keras.layers` and `tf.keras.Model`, therefore, both M-LSA and V-LSE can be easily used in other deep learning models as an intermediate layers or models. The replication of the experiments is in the folder `experiments`. There are unimodal `experiments/image`, `experiments/text` and multimodal `experiments/multimodal` experiments.

### Image

#### Weak Supervision

The weak supervision experiments can be found in the folder `experiments/image`.

You can run the following command for information about the training arguments:

```sh
python train_CNN.py --help
```

If you want to train an InceptionV3:

```sh
python train_CNN.py --model inceptionv3 --save_path inception.h5 --data_path images.h5
```

You can run the following command for information about the test arguments:

```sh
python test_CNN.py --help
```

For instance, to test the InceptionV3:

```sh
python test_CNN.py --model inceptionv3 --save_path inception.h5 --data_path images.h5
```

#### Summarization

The summarization experiments can be found in the folder `experiments/image`.

First, you need to extract a feature vector representation of each image patch, for more information you can run:

```sh
python extract_feature_vectors.py --help
```

For instance, to extract feature vectors using the InceptionV3:

```sh
python extract_feature_vectors.py --model inceptionv3 --weights inception.h5 --data_path images.h5 --save_path feature_vectors.h5
```

For information about the BoVW learning, you can execute:

```sh
python train_bovw.py --help
```

The following example would store different K-Means models in the folder 'km_models/' and the BoVW representations as 'bovw.h5'

```sh
python train_bovw.py --feature_vectors feature_vectors.h5 --save_path bovw.h5 --model_path km_models
```

You can execute the following command for information about the evaluation of the BoVW representations:

```sh
python test_bovw.py --help
```

For instance, to evaluate the representations with TF-IDF you can do:

```sh
python test_bovw.py --repr tfidf --bovw_path bovw.h5 --trials 10
```

### Text

The text experiments can be found in the folder `experiments/text`.

For the N-gram learning information, you can run

```sh
python train_embeddings.py --help
```

For instance, to train the N-gram representation you can do:

```sh
python train_embeddings.py --repr ngram --images_path images.h5 --texts_path reports_txt/ --save_path bow.h5
```

For information about the evaluation of the representations you can do:

```sh
python test_embeddings.py --help
```

For example, if you want to evaluate the representations with TF-IDF, then:

```sh
python test_embeddings.py --repr tfidf --bow_path bow.h5 --trials 10
```

### Weak Multimodal Supervision

The weak multimodal supervision experiments can be found in the folder `experiments/multimodal`

For information about the hyperparameter combinations that are generated, you can run:

```sh
python random_params_generator.py --help
```

For example, to generate 100 combinations using a random seed of 0:

```sh
python random_params_generator.py --seed 0 --trials 100 --save_path params.csv
```

For information about the hyperparameter exploration, use:

```sh
python random_search.py --help
```

For instance, if you want to explore V-LSE with a visual RBF kernel:

```sh
python random_search.py --codebook 1700 --ngram 2 --visual_kernel rbf --hyperparameters params.csv --training_type unimodal  --classification_type binary --bovw_path bovw.h5 --bow_path bow.h5 --save_path best_params/rbf_unimodal.csv
```

To explore M-LSA with a visual RBF kernel and a text linear kernel:

```sh
python random_search.py --codebook 1700 --ngram 2 --visual_kernel rbf --text_kernel linear --hyperparameters params.csv --training_type multimodal  --classification_type binary --bovw_path bovw.h5 --bow_path bow.h5 --save_path best_params/rbf_linear.csv
```

For information about the classification evaluation, you can run:

```sh
python classification_test.py --help
```

For instance, to evaluate V-LSE you can:

```sh
python classification_test.py --hyperparameters best_params/rbf_unimodal.csv --training_type unimodal --classification_type binary --bovw_path bovw.h5 --trials 10
```

To evaluate M-LSA:

```sh
python classification_test.py --hyperparameters best_params/rbf_linear.csv --training_type multimodal --classification_type binary --bovw_path bovw.h5 --bow_path bow.h5 --trials 10
```

For the retrieval evaluation, you must install [TREC-EVAL](https://github.com/usnistgov/trec_eval). You can execute the following command for more information about the evaluation:

```sh
python retrieval_test.py --help
```

To evaluate V-LSE in retrieval:

```sh
python retrieval_test.py --hyperparameters best_params5/rbf_unimodal.csv --training_type unimodal --retrieval_type Y --bovw_path bovw.h5 --trials 10
```

To evaluate M-LSA in retrieval:

```sh
python retrieval_test.py --hyperparameters best_params5/rbf_linear.csv --training_type multimodal --retrieval_type Y --bovw_path bovw.h5 --bow_path bow.h5 --trials 10
```




















