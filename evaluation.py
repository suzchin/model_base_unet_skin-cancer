from Lib import *
root_path = "D:/Learnning/TensorFlow/program/model_base_Unet/"
test_images = np.load(root_path+'OUTPUT/segmented_images_evaluate.npy')
test_labels = np.load(root_path+'OUTPUT/evaluate_labels_class_labels.npy')

model = UNetModel.get_unet_model_class((128, 128, 3))
model.load_weights(root_path+'OUTPUT/Unet_lr_e4_bs_10_classifier.hdf5')

# 0->162 : others, 163->99 melanoma

# for iex in range(len(test_images)):
#     sample_predictions = model.predict(test_images[iex].reshape((1, 128, 128, 3)))
#     print('i={} sample_predictions={}'.format(iex, sample_predictions))
#     predicted_class = sample_predictions < 0.4
#     #print(predicted_class)
#     predicted_class = 0 if predicted_class == True else 1
#
# #predicted_class
# #print(type(predicted_class))
#
#     class_labels = {0: 'melanoma', 1: 'others'}
#     print('class_labels:', class_labels[predicted_class])
#
# #plt.imshow(test_images[iex])
# #plt.show()


#test_images = np.squeeze(test_images, axis=1)
test_predictions = model.predict(test_images)
class_names = ['melanoma', 'others']
print('test_labels:', test_labels)

predicted_labels = np.zeros(test_predictions.shape[0])

for i in range(test_predictions.shape[0]):
    predicted_labels[i] = 0 if test_predictions[i]>0.5 else 1
print('predicted_labels.shape:', predicted_labels.shape)

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_labels, predicted_labels)
np.set_printoptions(precision=2)

#Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

#Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
#plt.show()
# model.compile(optimizer=Adam(lr=1e-25), loss='binary_crossentropy', metrics=['accuracy'])
# evalu = model.evaluate(test_images, test_labels,batch_size=4,verbose=1)
# print('test los:', evalu[0])
# print('test accuracy:', evalu[1])

import pandas as pd
data = pd.read_csv(root_path+"OUTPUT/Unet_main_classifier_vgg.csv")

fig, ax = plt.subplots(2,1)
ax[0].plot(data['loss'], color='b', label="training loss")
ax[0].plot(data['val_loss'], color='r', label="Validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)
plt.show()