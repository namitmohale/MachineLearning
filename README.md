				Few-Shot Learning using Data Augmentation and Transfer Learning

					  Namit Mohale, Yashodhan Joshi, Anshum Pal
					nm3191@nyu.edu, yj1400@nyu.edu, aap695@nyu.edu


Project Report - https://github.com/namitmohale/MachineLearning/blob/master/Project%20Report%20MachineLearning.pdf

Formulation, Approach, Evaluation, Design, Architecture are given in the Report.

1    <b> Introduction </b>

Adequate training of neural networks requires a lot of data. In the case of low data regime, which is the case with most of the problems today, the existing networks generalize poorly. To make progress on this foundational problem, we have decided to follow a low-shot learning technique by exploiting the method of aggressive data augmentation and transfer learning.

1.1    <b> Related work </b>

<b> One-shot and low-shot learning. </b> One class of approaches to one-shot learning uses generative models of appearance that tap into a global or a super-category level prior. Generative models based on strokes or parts have shown promise in restricted domains such as hand- written characters. They also work well in datasets without much intra-class variation or clutter.

<b> Zero-shot learning. </b> Zero-shot recognition uses textual or attribute-level descriptions of object classes to train classifiers. While this problem is different than ours, the motivation is the same: to reduce the amount of data required to learn classifiers. One line of work uses hand-designed attribute descriptions that are provided to the system for the novel categories. Transfer learning. The ability to learn novel classes quickly is one of the main motivations for multitask and transfer learning. Thrun’s classic paper convincingly argues that “learning the n-th task should be easier than learning the first”, with ease referring to sample complexity. However, recent transfer learning research has mostly focused on the scenario where large amounts of training data are available for novel classes. For that situation, the efficacy of pre-trained ConvNets for extracting features is well known. There is also some analysis on what aspects of ImageNet training aid this transfer.[1]

<b> Data Augmentation. </b> The field of data augmentation is not new, and in fact, various data augmentation techniques have been applied to specific problems. The main techniques fall under the category of data warping, which is an approach which seeks to directly augment the input data to the model in data space. The idea can be traced back to augmentation performed on the MNIST set in [2].

A very generic and accepted current practice for augmenting image data is to perform geometric and color augmentations, such as reflecting the image, cropping and translating the image, and changing the color palette of the image. All of the transformation are affine transformation of the original image that take the form:

y = W x + b

The idea has been carried further in [3], where an error rate of 0.35% was achieved by generating new training samples using data augmentation techniques at each layer of a deep network. Specifically, digit data was augmented with elastic deformations, in addition to the typical affine transformation. Furthermore, data augmentation has found applicability in areas outside simply creating more data.


1.2	<b> Problem, Goal, and Approach </b>

Current recognition systems require days or even weeks of training on expensive hardware to develop good feature representations. The trained recognition systems may then be deployed as a service to be used by downstream applications. These downstream applications may need the ability to recognize new categories, but they may have neither the training data required, nor the infrastructure needed to retrain the models. Thus, there are two natural phases: in the first phase, we have the data and resources to train sophisticated feature extractors on large labeled datasets, and in the second phase, we want to add additional categories to our repertoire at minimal computational and data cost.

We are following the second phase. Our goal is to predict with high accuracy on a new class for which we have very few examples. Instead of collecting new data which would be both costly and time-consuming. Our approach is to use data augmentation and transfer learning to make more data for novel class, as well, as saving time by not training the whole model and using what the model already learned to extract basic features.

We are using aggressive data augmentation which uses various transformations like Coarse Dropout, Noise Embedding, Blur, Perspective Transformation, etc. to generate new examples, feeding to a pre-trained image-to-image GAN built on a U-Net architecture to generate even more samples and VGG16 model pre-trained on the ImageNet1K dataset for transfer learning.


1.3	<b> Running the code </b>

This code is divided into 3 parts:
	1. Augmentation using Image Transformations
	2. Augmentation using Generative Adversarial Networks
	3. Transfer Learning on VGG-16 pre-trained on ImageNet1k
	
> First run the 'Augmentation-Copy1.ipynb' which will create Image Transformations and augments it to the original dataset. In this code, you will need to specify explicitly how many original samples of novel classes you wish to consider and how many transformations you wish to create.

> Then run the 'ML_Project_GAN.ipynb' which first takes original samples and trans the GAN, then transfers the learnt parameters to novel classes and learns to generate More images.

> Then run 'VGG-16-Transfer-Learning.ipynb' to train the pre-trained VGG-16 classifier on novel classes by unfreezing the last 2 layers, and you will see the final accuracy printed on the screen with each epoch.

The ML_Project_GAN.ipynb can be run only on a powerful GPU since the network is very deep.

3	<b> Conclusion/Future Work </b>

Data augmentation has been shown to produce promising ways to increase the accuracy of classification tasks. While traditional augmentation is very effective alone, other techniques enabled by CycleGAN and other similar networks are promising. We experimented with our own way of combining training images allowing a neural net to learn augmentations that best improve the ability to correctly classify images. If given more time, we would like to explore more complex architecture and more varied datasets. Finally, although GANs and neural augmentations do not perform much better than traditional augmentations and consume almost 3x the compute time or more, we can always combine data augmentation techniques. Perhaps a combination of traditional augmentation followed by neural augmentation further improves classification strength.

Given the plethora of data, we would expect that such data augmentation techniques might be used to benefit not only classification tasks lacking sufficient data, but also help improve the current state of the art algorithms for classification. Furthermore, the work can be applicable in more generic ways, as ”style” transfer can be used to augment data in situations were the available data set is unbalanced. For example, it would be interesting to see if reinforcement learning techniques could benefit from similar data augmentation approaches. We would also like to explore the applicability of this technique to videos. Specifically, it is a well known challenge to collect video data in different conditions (night, rain, fog) which can be used to train selfdriving vehicles. However, these are the exact situations under which safety is the most critical. Can our style transfer method be applied to daytime videos so we can generate night time driving conditions? Can this improve safety? If such methods are successful, then we can greatly reduce the difficulty of collecting sufficient data and replace them with augmentation techniques, which by comparison are much more simpler.
