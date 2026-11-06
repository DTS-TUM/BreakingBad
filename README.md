# Breaking BAD: Heterogeneous Byzantine-Robust Federated Learning via Gather and Scatter Scores

Implementation for paper: [Heterogeneous Byzantine-Robust Federated Learning via Gather and Scatter Scores](https://neurips.cc/virtual/2026/loc/paris/poster/148208)


FEMNIST data is originally created by running the following command on leaf framework [TalwalkarLab/leaf](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist):  
`./preprocess.sh -s niid --sf 1.0 -k 350 -t sample`  
Shakespeare data is originally created by running the following command on leaf framework [TalwalkarLab/leaf](https://github.com/TalwalkarLab/leaf/tree/master/data/shakespeare):  
`./preprocess.sh -s niid –sf 1 -k 50 -t sample -tf 0.8`

## Important Notes
Due to storage constraints, only a fraction of the FEMNIST dataset is included in the femnist_data folder, limiting experiments to a maximum of 10 clients. For precise reproduction of the reported results, the full dataset should be used; however, results obtained with the partial dataset remain consistent with those presented in the paper.

Partial FEMNIST data here is sampled using: 
`./preprocess.sh -s niid --sf .022 -k 350 -t sample`.

Similarly, partial Shakespeare data here is sampled with the command below:
`./preprocess.sh -s niid –sf 0.2 -k 50 -t sample -tf 0.8`

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{ergisi2026breaking,
  title={Breaking BAD: Heterogeneous Byzantine-Robust Federated Learning via Gather and Scatter Scores},
  author={Ergisi, Sena and Massny, Luis and Bitar, Rawad},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=zYNKoIyYhj}
}
```

## Acknowledgements 
This repository benefited from the following resources, projects, and references:
- [moranant/attacking_distributed_learning](https://github.com/moranant/attacking_distributed_learning)
- [epfml/byzantine-robust-noniid-optimizer](https://github.com/epfml/byzantine-robust-noniid-optimizer)
- [Xtra-Computing/NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench)
- [LPD-EPFL/robust-collaborative-learning](https://github.com/LPD-EPFL/robust-collaborative-learning)
- [sacs-epfl/decentralized-learning-simulator](https://github.com/sacs-epfl/decentralized-learning-simulator)
- [LPD-EPFL/robust-collaborative-learning](https://github.com/LPD-EPFL/robust-collaborative-learning)
