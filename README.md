## FCN+RL: A Fully Convolutional Network followed by Refinement Layers to Offline Handwritten Signature Segmentation



__Versões_dependencias__:

Python 3.6

numpy 1.17.4

keras 2.3.1

tensorflow-gpu 1.14.0

opencv-contrib-python 4.1.2.30

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

Tran the FCN+RL model.

### RUN WINDOWS:

python fcn_plus_rl.py "-mrefinement_model" "--train-folder=<path_local_diretorio_de_treino>/" "--validation-folder=<path_local_diretorio_de_validacao>/" "--gpu=1" "--bs=12" "--train-steps=10" "--valid-steps=10" "--no-aug" "--train-samples=800" "--valid-samples=200" "--lr=0.0001"

### RUN LINUX:

python fcn_plus_rl.py -mrefinement_model --train-folder=<path_local_diretorio_de_treino> --validation-folder=<path_local_diretorio_de_validacao> --gpu=1 --bs=12 --train-steps=10 --valid-steps=10 --no-aug --train-samples=800 --valid-samples=200 --lr=0.0001
