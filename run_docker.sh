#!/bin/bash

docker run -v $(pwd):/work/2d-reconnection -w /work/2d-reconnection --user $(id -u):$(id -g) --gpus all -it 2d-reconnection