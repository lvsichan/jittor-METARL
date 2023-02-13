# Versatile Control of Fluid-Directed Solid Objects Using Multi-Task Reinforcement Learning



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

<img src="image\MPM SCOOP THE JELLY.gif" width="300" height="200" alt="result" style="zoom:50%;" />

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
