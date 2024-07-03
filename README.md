# Optimal-Constant-Solutions-in-DNNs 
Project for the course DD2412 reproducing and expanding the results from the paper "Deep Neural Networks Tend To Extrapolate Predictably".

Abstract:
To deploy neural networks for decision critical applications we need metrics or
protocols to map the reliance of an input-output pair for a neural network. This
report highlights a trend in the extrapolated outputs of the deep neural network
called Optimal Constant Solution. The chosen paper, "Deep Neural Networks
Tend To Extrapolate Predictably", aims to dissect the behavior of neural networks
when exposed to data originating from a distinct distribution, showcasing its
implications for classification and prediction certainty. The central premise revolves
around the observation that, under the influence of high-dimensional input, the
model’s output tends to converge towards an Optimal Constant Solution (OCS)
We replicate the results of Kang et al. with Resnet-20 backbone for CIFAR-10
dataset. Moreover, we study the effect of two more Vision-transformer based
models CoatNet and CaiT. We also share our findings when trained under
an adversarial setting, as well as explore the effect of weight decay on the reversion
to the OCS. In addition, we experiment with the multi margin loss function to see
how the choice of loss function affects the reversion to the OCS. We get stronger
results reversion to the OCS for CoatNet and Cait than for ResNet20, and find that
the choice of loss function and regularization strength clearly affect the reversion
to the OCS.
