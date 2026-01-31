# Realtime-frame--and-event-based-detection-with-SNN-on-Neuromorphic-hardware

This code repository is related to our submitted journal paper entitled as "".
It is about development of three lightweight SNN models which can be run on intel's Single Loihi 2 chip (referred as Oheo Gulch) which contain upto 1 million spiking neurons neurons. The models were evaluated across four datasets. Two frame-based datasets and two event-based (captured with DVS cameras) datasets.
As the two event-based datasets we used Prophesee GEN1 and event-based data component from ev-CIVIL dataset. As the two frame-based datasets we used PASCAL VOC and frame-based data component from ev-CIVIL dataset. We developed ANN versions of the corresponding SNN models. And then we compare/benchmark the detection performance, inference rate, per inference energy comparison and Energy-delay-product (EDP) of SNNs on Loihi 2 compared to ANNs on Jetson nano (edge-GPU) and Macbook M2 CPU.
We observed that SNNs on Loihi 2 demonstrated substantial energy efficiency, consuming 10–55× lessdynamic energy and 4–7× lower total power than ANNs on Jetson Nano and Mac-
Book CPU, while supporting real-time inference at 62–170 samples/s with a compromized detection performance. However, with our ANN to SNN distillation approach, the distilled SNN models could retain at least 88% of ANN detection performance interms of mAP and F1-score metrics. 

* The code includes ANN models, SNN models (with intel Loihi 2 compatible LIF neurons), SNN model training (Quantization aware) with spikingjelly library, ANN to SNN knowledge distillation based training.

* It also includes files which export SNN models trained with Spikingjelly library to loihi 2 lava-dl compatible models.

* Further, this repository includes .ipynb files which we used for running and benchmarking on intel Loihi 2.

