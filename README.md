# unet-reconnection

Magnetic reconnection identification using image segmentation with U-Net

## Outline

![](reconnection_points.png)

## Environment

Build the Docker image

```bash
docker build . -t unet-reconnection
```

Start a container interactively with current directory mounted

```bash
./run_docker.sh
```

Re-enter stopped container

```bash
docker start -i <container id>
```

## Train

```
train.py -i data --epochs 5 --file-fraction 0.5 --gpus 0 1 --normalize -o results/5
```

## Plot

```
plot.py -d results/5 --epochs 5
```

## Model

Ref. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![](unet.png)