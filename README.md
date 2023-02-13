# Versatile Control of Fluid-Directed Solid Objects Using Multi-Task Reinforcement Learning

## Abstract

## environment:

taichi:

jittor:

## train

```
python launch_experiment.py ./configs/scoop-jelly-train.json
```

## test and show:

```
python launch_experiment.py ./configs/scoop-jelly-eval.json
```
We provide an example of catching a ball. A trained model. After configuring the environment, you can get the following animation：

![image](https://github.com/lvsichan/jittor-METARL/blob/master/image/MPM%20SCOOP%20THE%20JELLY.gif)

## Citation

If you found this code useful please cite our work as:

```
@article{ren2022versatile,
  title={Versatile Control of Fluid-directed Solid Objects Using Multi-task Reinforcement Learning},
  author={Ren, Bo and Ye, Xiaohan and Pan, Zherong and Zhang, Taiyuan},
  journal={ACM Transactions on Graphics},
  volume={42},
  number={2},
  pages={1--14},
  year={2022},
  publisher={ACM New York, NY}
}
```
