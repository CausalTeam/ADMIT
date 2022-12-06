# Generalization Bounds for Estimating Causal Effects of Continuous Treatments

## Introduction

This source code is exploited to support the work in "Generalization Bounds for Estimating Causal Effects of Continuous Treatments", submitted to NeurIPS 2022. We provide a PyTorch implementation of the ADMIT model for estimating the causal effects of continuous treatments, i.e., the average dose-response function (ADRF).

## Train & Test

Please create a python project in your local workstation, and import these files contained in the project. 

For example, this command trains an ADMIT model on the Simulation dataset with GPU 0.

```python
python main.py --data sim  --learning_rate 0.0002 --batch_size 500 --log
```

## Citation

If you find our work useful for your research, please consider citing the following papers :)

```
@inproceedings{wang2022generalization,
  title={Generalization Bounds for Estimating Causal Effects of Continuous Treatments},
  author={Xin Wang and Shengfei Lyu and Xingyu Wu and Tianhao Wu and Huanhuan Chen},
  booktitle={NeurIPS},
  year={2022}
}
```

