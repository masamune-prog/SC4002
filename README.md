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

