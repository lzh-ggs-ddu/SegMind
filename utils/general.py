
import numpy as np


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


lda_label_color_np=np.array([[255,255,255],[255,0,0],[255,255,0],[0,0,255],[128,64,200],[0,255,0],[255,128,0],[0,0,0]])
lda_label_name_list=['background','building', 'road', 'water', 'barren', 'forest', 'agriculture','no-data']  # (0,1,2,3,4,5,6,7)

dland_label_color_np=np.array([[0,255,255],[255,255,0],[255,0,255],[0,255,0],[0,0,255],[255,255,255],[0,0,0]])
dland_label_name_list=['urban','agriculture','rangeland','forest','water','barren','unknown']  # (0,1,2,3,4,5,6)

po_label_color_np=np.array([[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,255,0],[255,0,0]])
po_label_name_list=['impervious surface','building','low vegetation','tree','car','clutter']  # (0,1,2,3,4,5)

all_label_color_dict={1:lda_label_color_np,2:dland_label_color_np,3:po_label_color_np}
all_label_name_dict={1:lda_label_name_list,2:dland_label_name_list,3:po_label_name_list}

feature_color_list=['red', 'green','orange','blue','purple','cyan','yellow','limegreen']





