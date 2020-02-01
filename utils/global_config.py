"""
Arquivo de configurações globais
Set os valores das constantes invocadas pelo modelo FCN+RL
"""

__DEF_HEIGHT = 512 # dimensão h
__DEF_WIDTH = 512 # dimensão w
__CHANNEL = 1

RESHAPE_IMAGE = (__DEF_HEIGHT, __DEF_WIDTH, __CHANNEL)

INIT_CHANNELS = 32 # valor default = 32. é incrementado ao longo das camdas de expansão.






