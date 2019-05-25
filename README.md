# Multi-Relation Script Learning for Discourse Relations
This repository contains code, data, and pre-trained models for the following papers. 

```
I-Ta Lee, and Dan Goldwasser, "Multi-Relational Script Learning for Discourse Relations", ACL 2019
```

bibtex
```
@inproceedings{lee2019multi,
  title={Multi-Relational Script Learning for Discourse Relations},
  author={Lee, I-Ta and Goldwasser, Dan},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  year={2019}
}
```

If you use any resources within this repository, please cite the paper.

# Dependencies

```
pip install -r requirements.txt
```

# Data

Our core experiments (MCNC, MCNS, MCNE) use data splits of NYT section of English Gigawords, following the data splits given by Granroth-Wilding (https://mark.granroth-wilding.co.uk/papers/what_happens_next/) [1].

As our models need entity mention spans rather than entity head words only, which are not provided in Granroth-Wilding's code outputs, considering the stochastic factors in the pre-processing, we release a newly pre-processsed train/dev/test data for MCNC, MCNS, MCNE (check the experiments in the paper). Unlike Granroth-Wilding's code, we use Stanford CoreNLP as the text pipeline tool and follow the heuristic mentioned in the paper for retrieving all the events.

## Download

  - test data and misc: url
  - training data: url
  - pretrained models: url
  - GloVe Word Embeddings: you can get glove.6B.300d.txt from https://nlp.stanford.edu/projects/glove/. 
  - ELMo: you can download medium size ELMo model from https://allennlp.org/elmo (elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5 and elmo_2x2048_256_2048cnn_1xhighway_options.hdf5). Put them in ./data folder.

# Pre-trained models

You can download the models from the link in the Download section

- EventTransE: 
    - model: pretrained/out_transe_v0.2.10_long9_tmp
    - config: train_config_transe_v0.2.10_long9.json
- EventTransR: pretrained/out_transr_v0.2.10_long9_tmp
    - model: pretrained/out_transr_v0.2.10_long9_tmp
    - config: train_config_transr_v0.2.10_long9.json

# Run experiments with pre-trained models

To begin with, download the test data and pretrain models in the Download section, and decompress them in the repo folder. You need two configuation files: train_config_{xxx}.json and relation_{xxx}.json. Check all the file paths in the configuration file are correct.

Note that for all the executable .py, you can use **-h** argument to see the help and **-g <gpu_id>** to specify running on a specific gpu.

## MCNC

For EventTransE
```
python bin/evaluations/eval_mcnc.py -v pretrained/out_transe_v02.10_long9_tmp/model_2_3_2591.pt pretrained/out_transe_v0.2.10_long9_tmp/argw_enc_2_3_2591.pt data/mcnc_test_v0.3.4/mcnc_coref_next.pkl train_config_transe_v0.2.10_long9.json relation_9disc.json
```

For EventTransR, you simply need to replace the model file, argument encoder file, and config file.


## MCNS and MCNE

Two experiments run in one command.

For EventTransE:
```
python bin/evaluations/eval_mcns.py -v pretrained/out_transe_v0.2.10_long9_tmp/model_2_3_2591.pt pretrained/out_transe_v0.2.0_long9_tmp/argw_enc_2_3_2591.pt glove.6B.300d.txt data/mcnc_test_v0.3.4/mcns_coref_next.pkl train_config_transe_v0.2.10_long9.json relation_9disc.json Viterbi
```

For EventTransR, you simply need to replace the model file, argument encoder file, and config file.


## Intrinsic Discourse Relations

The evaluations here take long time to run. GPUs recommended.

For EventTransE
```
python bin/evaluations/eval_disc.py pretrained/out_transe_v0.2.10_long9_tmp/model_2_3_2591.pt pretrained/out_transe_v0.2.10_ong9_tmp/argw_enc_2_3_2591.pt data/disc_test_v0.2.0.pkl train_config_transe_v0.2.10_long9.json relation_9disc.json -v
```

For EventTransR, you simply need to replace the model file, argument encoder file, and config file.

For testing ELMo, download ELMo models from the Download section. Python3 is required by ELMo.
```
python3 bin/evaluations/eval_disc_elmo.py -v data/disc_test_v0.2.0.pkl relation_9disc.json
```


## Implicit Discourse Sense Classifications

place holder


# Train from scratch

Download the training data from the Download section. Make sure the training file path in the config file is correct. Then run
```
python bin/train.py -v -r train_config_transe_v0.2.10_long9.json output_model
```
Again, for EventTransR, simply replace the config file.

# References

[1] Granroth-Wilding, Mark, and Stephen Clark. "What happens next? event prediction using a compositional neural network model." Thirtieth AAAI Conference on Artificial Intelligence. 2016.
