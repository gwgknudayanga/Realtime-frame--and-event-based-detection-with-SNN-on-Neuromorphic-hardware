# Realtime-frame--and-event-based-detection-with-SNN-on-Neuromorphic-hardware

This code repository accompanies our submitted journal paper entitled “Real-Time Frame- and Event-Based Object Detection with SNNs on Edge Neuromorphic Hardware: Design, Deployment, and Benchmarking.”

The repository focuses on the design, development, deployment, and benchmarking of three lightweight spiking neural network (SNN) models that run on Intel’s single-chip Loihi 2 neuromorphic platform (Oheo Gulch), which supports up to one million spiking neurons. The models were evaluated across four datasets: two frame-based datasets and two event-based datasets captured using dynamic vision sensor (DVS) cameras.

For event-based evaluation, we used the Prophesee GEN1 dataset and the event-based component of the ev-CIVIL dataset. For frame-based evaluation, we used PASCAL VOC and the frame-based component of the ev-CIVIL dataset. Corresponding artificial neural network (ANN) versions of each SNN model were also developed.

We benchmarked and compared the SNNs running on Loihi 2 against ANNs deployed on a Jetson Nano edge GPU and an Apple M2 MacBook CPU, evaluating detection accuracy, inference throughput, per-inference energy consumption, and energy–delay product (EDP).

Our results show that SNNs on Loihi 2 achieve substantial energy efficiency, consuming 10–55× less dynamic energy and 4–7× lower total power than ANNs on the Jetson Nano and MacBook CPU, while enabling real-time inference at 62–170 samples per second, albeit with some degradation in detection performance. However, using our ANN-to-SNN distillation approach, the distilled SNN models retain at least 88% of the ANN detection performance, as measured by mAP and F1-score metrics.

* The code includes ANN models, SNN models (with intel Loihi 2 compatible LIF neurons), SNN model training (Quantization aware) with spikingjelly library, ANN to SNN knowledge distillation based training.
  * ffff
* It also includes files which export SNN models trained with Spikingjelly library to loihi 2 lava-dl compatible models.

* Further, this repository includes .ipynb files which we used for running and benchmarking on intel Loihi 2.

