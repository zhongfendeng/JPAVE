# JPAVE
The code for reviewing paper "JPAVE: A Generation and Classification-based Model for Joint Product Attribute Prediction and Value Extraction".

## Requirements
+ Python >= 3.6
+ torch >= 0.4.1
+ numpy >= 1.17.4
+ transformers

## Preparation before train the model
### Data preprocess
+ Please get the entire MEPAVE dataset [here](https://github.com/jd-aig/JAVE).
+ use data.data_process.py to preprocess the MEPAVE dataset to obtain train.json, valid.json and test.json for model training and testing, and also to generate a "tagmaster.json" file which stores all the attributes and their corresponding values in the dataset.
+ use data.generate_mepave_attribute_value_embeddings.py to generate pre-trained attribute and value embeddings by using pre-trained BERT model (we use the pre-trained "[bert-base-chinese](https://huggingface.co/bert-base-chinese)" from huggingface).
+ move the generated "tagmaster.json", "mepave_attribute_embeddings.json" and "mepave_value_embeddings.json" to the root of this project.

## Train
Run the train.py file to train the model as follows:
```bash
python train.py
```

