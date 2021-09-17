## FCN+RL: A Fully Convolutional Network followed by Refinement Layers to Offline Handwritten Signature Segmentation
This model is an approach to locate and extract the pixels of handwritten signatures on identification documents, without any prior information on the location of the signatures.

![alt text](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/9200848/9206590/9206594/junio1-p7-junio-large.gif)\
Fig.1 *The FCN+RL proposed architecture. In this example, FCN performs segmentation by pixel classification, providing an image with the estimated signature pixels in the foreground. After the first stage, the concatenation between the input image and the image predicted by the FCN encoder-decoder layers is sent for input from the layer block for RL at stage two.*

### Informations and credit
To learn more about this approach, go to the paper for this project [HERE](https://ieeexplore.ieee.org/abstract/document/9206594?casa_token=X1kyTs6Vh5QAAAAA:t04bOqeyxy_sxodZ6dkmm-wV3gOukq4AFKm29ScSHw6Lcff4PxQ4PzqhqspZ8ITo8-3kRI3StDE). When using this material, we ask that you quote the authors of this project correctly.
_____________________________________________________________________________________________________________________________________________________________________
If you use this code, please consider citing the following paper:

* *Lopes Junior, Celso A. M. , Silva, Matheus H. M., Bezerra, Byron L. D., Fernandes, Bruno J. T. and Impedovo, Donato, "FCN+RL: A Fully Convolutional Network followed by Refinement Layers to Offline Handwritten Signature Segmentation," 2020 International Joint Conference on Neural Networks (IJCNN), 2020, pp. 1-7, https//doi:10.1109/IJCNN48605.2020.9206594.*
_______________________________________________________________________________________________________________________________________________________________________
## Database:

To use the Database you need to fill out a **License Agreement** as informed on the repository website.\
We provide a database of complete identification documents [HERE](http://tc11.cvc.uab.es/datasets/SBR-Doc%20Database_1).\
If you use this Database, please consider citing the following paper:

* *LOPES JUNIOR, C. A. M., NEVES JUNIOR, R. B., BEZERRA, B. L. D., TOSELLI, A. H., IMPEDOVO, D.: Competition on components segmentation task of document photos. In: International Conference on Document Analysis and Recognition (ICDAR), pp. 1–15. Springer Nature, (2021). https://doi.org/10.1007/978-3-030-86337-1_45.*
_______________________________________________________________________________________________________________________________________________________________________
## Dependencies:

- Python 3.6
- numpy 1.17.4
- keras 2.3.1
- tensorflow-gpu 1.14.0 or tensorflow 1.14.0
- opencv-contrib-python 4.1.2.30

### Preliminary settings.
Divide your dataset/images into two directories: `./train` and `./validation`. 

Configure your global settings in the ```utils/global_config.py``` file.

### Training stages

#### *Stage 1*
Train the layers of the encoder-decoder model with the following command at your prompt:
```
python train_encoder_decoder_model.py "-mencoder_decoder_model" "--train-folder=<path_local_diretorio_de_treino>/" "--validation-folder=<path_local_diretorio_de_validacao>/" "--gpu=1" "--bs=12" "--train-steps=10" "--valid-steps=10" "--no-aug" "--train-samples=800" "--valid-samples=200" "--lr=0.0001"
```
After completing the training of the first stage, your model will be saved in the directory you defined, here we define the directory ```./model_path_hdf```. The file saved in this directory will be invoked for the training of the second stage.

#### *Stage 2*
Train the layers of the FCN+RL (RL) model with the following command at your prompt:
```
python train_fcnrl_model.py "-mrefinement_layer_model" "--train-folder=<path_local_diretorio_de_treino>/" "--validation-folder=<path_local_diretorio_de_validacao>/" "--gpu=1" "--bs=12" "--train-steps=10" "--valid-steps=10" "--no-aug" "--train-samples=800" "--valid-samples=200" "--lr=0.0001"
```

#### Comments:
The code is not updated for TensorFlow version 2.x. You can find migration information or code compatibility with Tensorflow 2.x [HERE](https://www.tensorflow.org/guide/migrate).
