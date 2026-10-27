# SC4002: Samuel's joint CNN experiment

This branch adds a shared dilated-CNN model for intent classification and slot filling. The comparison uses a single seed-42 model as Stage 1 and a three-seed ensemble as Stage 2. Stage 1 has validation results; its separate test evaluation is pending. Stage 2 has completed validation and test evaluation.

- [Implementation, results, and reproduction instructions](cnn_improvement/README.md)
- [Executed results notebook](cnn_improvement/results.ipynb)

Source code and recorded evidence are included. Downloaded assets and trained model binaries are excluded from Git; the linked instructions explain how to reproduce them.

## Original dataset setup

Amazon MASSIVE Dataset Setup

Follow these steps to download, extract, and inspect the Amazon MASSIVE NLU dataset.

1. Download and Extract

Run the following commands in your terminal to download the tarball and extract its contents:

curl https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz --output amazon-massive-dataset-1.0.tar.gz
tar -xzvf amazon-massive-dataset-1.0.tar.gz


2. Directory Structure

Once extracted, the 1.0 directory will have the following structure:

1.0
├── LICENSE
└── data
    ├── af-ZA.jsonl
    ├── am-ET.jsonl
    ├── ar-SA.jsonl
    ...

