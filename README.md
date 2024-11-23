# 3D Object Generation using Generative Adversarial Networks
## Introduction
Generative Adversarial Networks (GANs) are a powerful approach for generating realistic images from random noise or specific inputs. In this framework, a generator network creates synthetic images, while a discriminator network evaluates their authenticity against real images. For image generation projects, GANs are widely used in applications such as generating faces, creating art, and synthesizing objects or scenes. Popular GAN variations like DCGAN (Deep Convolutional GAN) are optimized for image generation, leveraging convolutional layers for better feature extraction.3D object GANs generate volumetric data, such as voxels, point clouds, or meshes, which define the structure of a 3D object. The generator creates synthetic 3D objects, and the discriminator evaluates their authenticity against real 3D data, such as datasets like ModelNet10 or ModelNet40. Over iterative adversarial training, the generator learns to produce increasingly realistic and coherent 3D shapes that match the dataset's characteristics.
### Dataset
The dataset used is in this project is the <a href ="https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset">ModelNet 10</a>. The ModelNet10 dataset,a subset of the larger ModelNet40 dataset, consists of 4,899 pre-aligned 3D shapes across 10 categories. It is divided into 3,991 shapes (80%) for training and 908 shapes (20%) for testing. The CAD models in this dataset are provided in Object File Format (OFF).

### Collaborators
<a href ="https://github.com/skandanagowda">Skandana Gowda</a> and <a href ="https://github.com/ShejalShankar">Shejal Shankar</a>







