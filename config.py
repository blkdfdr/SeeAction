'''
model_name = 'verb_only'
model_name = 'noun_only'
model_name = 'adv_only'
model_name = 'lstm'
model_name = 'all'

channel_name = 'cropimage'
channel_name = 'imagediff'
channel_name = 'cropdiff'
channel_name = 'image'
channel_name = 'crop'
channel_name = 'diff'
channel_name = 'all'

val_name = '[0-9]'
val_name = 'word'
val_name = 'windows'
val_name = 'zoom'
val_name = 'photoshop'
val_name = 'all'
'''
import os

data_dir = os.environ.get("DATA_DIR",print("No DATA_DIR defined in env"))

epochs = 10

model_name = 'all'
channel_name = 'all'
val_name = 'all'
