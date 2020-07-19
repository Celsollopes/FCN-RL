"""
Global settings file
Set the values of the constants invoked by the FCN + RL model
"""

__DEF_HEIGHT = 512 # dimension h
__DEF_WIDTH = 512 # dimension w
__CHANNEL = 1

RESHAPE_IMAGE = (__DEF_HEIGHT, __DEF_WIDTH, __CHANNEL)

INIT_CHANNELS = 32 # default = 32. is increased over the expansion layers.






