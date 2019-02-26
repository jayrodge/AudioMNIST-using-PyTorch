
import os
import librosa
import warnings
import numpy as np
from tqdm import tqdm
import librosa.display
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)


for i in tqdm(sorted(os.listdir('data/'))[25:42]):
    for j in sorted(os.listdir('data/'+i)):
        output_path = 'Spectogram/'+i+'/'+j[:-4]+'.png'
        # output_path = 'test/'+j[:-4]+'.png'
        fig = plt.figure(frameon=False)
        plt.ioff()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
#         print('data/'+i+'/'+j)        
        y, sr = librosa.load('data/'+i+'/'+j)
        data = librosa.feature.melspectrogram(y=y, sr=sr)
        fig.show(librosa.display.specshow(librosa.power_to_db(data,ref=np.max), y_axis='mel', fmax=8000, x_axis='time'))
        fig.savefig(output_path, dpi = 200)   
        # print(output_path)
#             fig.clf()
        plt.close(fig)
#             plt.close(ax)
 