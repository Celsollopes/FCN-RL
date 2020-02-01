# FCN-RL
FCN+RL Handwritten Signature Segmentation model with refinement layers blocks.



__Versões_dependencias__:

Python 3.6\n
numpy 1.17.4
keras 2.3.1
tensorflow-gpu 1.14.0
opencv-contrib-python 4.1.2.30

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

Execute o script abaixo para iniciar o reinamento do modelo FCN+RL.

### RUN sistema operacional WINDOWS:

python fcn_plus_rl.py "-mrefinement_model" "--train-folder=<path_local_diretorio_de_treino>/" "--validation-folder=<path_local_diretorio_de_validacao>/" "--gpu=1" "--bs=2" "--train-steps=1" "--valid-steps=1" "--no-aug" "--train-samples=800" "--valid-samples=200" "--lr=0.0001"

### RUN sistema operacional LINUX:

python fcn_plus_rl.py -mrefinement_model --train-folder=./train/ --validation-folder=./validation/ --gpu=1 --bs=2 --train-steps=1 --valid-steps=1 --no-aug --train-samples=800 --valid-samples=200 --lr=0.0001
