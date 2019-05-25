# mutlirelational_script
This repository contains code, data, and models for the following papers. 

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

# Models

EventTransE
EventTransR


# Data

Our core experiments (MCNC, MCNS, MCNE) use data splits of NYT section of English Gigawords, following the data splits given by Granroth-Wilding (https://mark.granroth-wilding.co.uk/papers/what_happens_next/) [1].

As our models need entity mention spans rather than entity head words only, which are not provided in Granroth-Wilding's code outputs, considering the stochastic factors in the pre-processing, we release a newly pre-processsed train/dev/test data for MCNC, MCNS, MCNE (check the experiments in the paper). Unlike Granroth-Wilding's code, we use Stanford CoreNLP as the text pipeline tool and follow the heuristic mentioned in the paper for retrieving all the events.


# pre-trained models

EventTransE
EventTransR

# running experiments with pre-trained models

Two configuation files: train_config_{xxx}.json and relation_{xxx}.json.

## MCNC

place holder

## MCNS

place holder

## MCNE

place holder

## Intrinsic Discourse Relations

place holder

## Implicit Discourse Sense Classifications

place holder

# train from scratch

place holder


# References

[1] Granroth-Wilding, Mark, and Stephen Clark. "What happens next? event prediction using a compositional neural network model." Thirtieth AAAI Conference on Artificial Intelligence. 2016.
