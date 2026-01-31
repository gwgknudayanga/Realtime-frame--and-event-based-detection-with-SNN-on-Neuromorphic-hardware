# Realtime-frame--and-event-based-detection-with-SNN-on-Neuromorphic-hardware

This code repository is related to our submitted journal paper entitled as "".
It is about development of three lightweight SNN models which can be run on intel's Single Loihi 2 chip (referred as Oheo Gulch) which contain upto 1 million spiking neurons neurons. The models were evaluated across four datasets. Two frame-based datasets and two event-based (captured with DVS cameras) datasets.
- As the two event-based datasets we used Prophesee GEN1 and event-based data component from ev-CIVIL dataset.
- As the two frame-based datasets we used PASCAL VOC and frame-based data component from ev-CIVIL dataset.


* The code includes ANN models, SNN models (with intel Loihi 2 compatible LIF neurons), SNN model training (Quantization aware) with spikingjelly library, ANN to SNN knowledge distillation based training.

* It also includes files which export SNN models trained with Spikingjelly library to loihi 2 lava-dl compatible models.

* Further, this repository includes .ipynb files which we used for running and benchmarking on intel Loihi 2.

