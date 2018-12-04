from Lib import *
root_path = "D:/Learnning/TensorFlow/program/model_base_Unet/"

import pandas as pd
data = pd.read_csv(root_path+"OUTPUT/Unet_main_classifier_vgg.csv")

fig, ax = plt.subplots(2,1)
ax[0].plot(data['loss'], color='b', label="training loss")
ax[0].plot(data['val_loss'], color='r', label="Validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(data['acc'], color='b', label="training acc")
ax[1].plot(data['val_acc'], color='r', label="Validation acc")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()