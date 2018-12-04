from Lib import *
root_path = "D:/Learnning/TensorFlow/program/model_base_Unet/"
test_images = np.load(root_path+'OUTPUT/segmented_images_evaluate.npy')

#print(test_labels)


model = UNetModel.get_unet_model_class((128, 128, 3))
model.load_weights('Unet_lr_e4_bs_10_classifier.hdf5')

# 0->162 : others, 163->99 melanoma

for iex in range(len(test_images)):
    sample_predictions = model.predict(test_images[iex].reshape((1, 128, 128, 3)))
    print('i={} sample_predictions={}'.format(iex, sample_predictions))
    predicted_class = sample_predictions > 0.5

    #print(predicted_class)
    predicted_class = 0 if predicted_class == True else 1

#predicted_class
#print(type(predicted_class))

    class_labels = {0: 'melanoma', 1: 'others'}
    print('class_labels:', class_labels[predicted_class])

#plt.imshow(test_images[iex])
#plt.show()
