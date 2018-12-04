from Lib import *
import easygui
# for one image
root_path = "D:/Learnning/TensorFlow/program/model_base_Unet/"
file_name = 'test_image_resized.png'
# load image
#test_image = Image.open(root_path+'DATA/melanoma/ISIC_0000004_resized.jpg')
test_image = easygui.fileopenbox()
test_image = Image.open(test_image)

imResize = test_image.resize((128, 128), Image.ANTIALIAS)
misc.imsave(root_path+'DATA/data_test_resized/'+file_name, imResize)
test_images = ndimage.imread(root_path+'DATA/data_test_resized/'+file_name)
print(test_images.shape)

#----------------------------------------------------------------
#----------------------------------------------------------------
#load weights model seg
model = UNetModel.get_unet_model_seg((128, 128, 3))
model.load_weights(root_path+'OUTPUT/Unet_lr_e4_bs_4.hdf5')
sample_predictions = model.predict(test_images.reshape((1, 128, 128, 3)))
sample_predictions = sample_predictions.reshape((128, 128))
sample_predictions = sample_predictions > 0.5
sample_predictions = np.array(sample_predictions, dtype=np.uint8)
#----------------------------------------------------------------
#----------------------------------------------------------------
ground_truth_images = sample_predictions
segmented_images = np.copy(test_images)

for j in range(128):
    for k in range(128):
        for l in range(3):
            segmented_images[j][k][l] = test_images[j][k][l] if ground_truth_images[j][k] == 1 else 0
#----------------------------------------------------------------
#load weights model class
#----------------------------------------------------------------
plt.figure()
plt.imshow(segmented_images)
test_mean = np.mean(segmented_images, axis=(0, 1, 2))
test_std = np.std(segmented_images, axis=(0, 1, 2))
segmented_images = (segmented_images-test_mean)/(test_std+1e-7)
plt.figure()
plt.imshow(segmented_images)
plt.show()

input_shape = (128,128,3)
num_classes = 2
tmodel_base = VGG16(weights='imagenet',include_top=False,input_shape=input_shape)
tmodel = Sequential()
tmodel.add(tmodel_base)
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.50))
tmodel.add(Flatten())
tmodel.add(Dense(512,activation='relu'))
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.25))
tmodel.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel.summary()
tmodel.load_weights(root_path+'OUTPUT/Unet_main_classifier_vgg.hdf5')
sample_predictions = tmodel.predict(segmented_images.reshape((1, 128, 128, 3)))

print('sample_predictions:',sample_predictions)

# predicted_class = sample_predictions > 0.5
# predicted_class = 0 if predicted_class == True else 1
# class_labels = {0: 'melanoma', 1: 'others'}
# print('class_labels:', class_labels[predicted_class])
#
# plt.figure()
# plt.imshow(test_images)
# plt.figure()
# plt.imshow(segmented_images)
# plt.show()
