# MuLX-QA
This repository contains codes for the model introduced in the paper titled ***"MuLX-QA: Classifying Multi-Labels and Extracting Rationale Spans in Social Media Posts"***
accepted for the Transactions on the Web (TWeb) Journal: [Link to paper](https://dl.acm.org/doi/10.1145/3653303)

## Requirements
Get the dataset here: [CAVES Data](https://github.com/sohampoddar26/caves-data)

- python==3.9
- pytorch==2.0.1
- transformers==4.31.0
- scikit-learn==1.3.0


## Example Usage 
First set the different parameters on the top of the prep_data and main.py; then run the following:

```
python prep_data.py
python main.py
```


## Citation
If you find our work useful, please cite the following paper:
```
@article{poddar2024MuLX-QA,
author = {Poddar, Soham and Mukherjee, Rajdeep and Samad, Azlaan Mustafa and Ganguly, Niloy and Ghosh, Saptarshi},
title = {MuLX-QA: Classifying Multi-Labels and Extracting Rationale Spans in Social Media Posts},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1559-1131},
url = {https://doi.org/10.1145/3653303},
doi = {10.1145/3653303},
journal = {ACM Transactions on the Web}
}
```

If you use the CAVES dataset, cite the following paper:
```
@inproceedings{poddar2022caves,
  title={CAVES: A dataset to facilitate Explainable Classification and Summarization of Concerns towards COVID Vaccines},
  author={Poddar, Soham and Samad, Azlaan Mustafa and Mukherjee, Rajdeep and Ganguly, Niloy and Ghosh, Saptarshi},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2022}
}
```

