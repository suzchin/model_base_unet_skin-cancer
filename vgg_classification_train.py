from Lib import *
root_path = "D:/Learnning/TensorFlow/program/model_base_Unet/"

train_images = np.load(root_path+'OUTPUT/segmented_images_class.npy')
print(train_images.shape)
test_images = np.load(root_path+'OUTPUT/segmented_images_evaluate.npy')
print(test_images.shape)
train_labels = np.load(root_path+'OUTPUT/classification_train_labels.npy')
print(train_labels)
test_labels = np.load(root_path+'OUTPUT/evaluate_labels_class_labels.npy')
print(test_labels.shape)

train_mean = np.mean(train_images, axis=(0, 1, 2, 3))
train_std = np.std(train_images, axis=(0, 1, 2, 3))
train_images = (train_images - train_mean)/(train_std+1e-7)
# plt.figure()
# plt.imshow(train_images[0])
# plt.figure()
# plt.imshow(train_images[1])
# plt.figure()
# plt.imshow(train_images[2])
# plt.show()
test_mean = np.mean(test_images, axis=(0, 1, 2, 3))
test_std = np.std(test_images, axis=(0, 1, 2, 3))
test_images = (test_images-test_mean)/(test_std+1e-7)

train_labels = to_categorical(train_labels, num_classes=2)
test_labels = to_categorical(test_labels, num_classes=2)
print(train_labels)


augs = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs.fit(train_images)

#annealer
annealer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

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

tmodel.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=6, min_lr=0.5e-6)
csv_logger = CSVLogger(root_path+'OUTPUT/Unet_main_classifier_vgg.csv')
print('csv_logger:', type(csv_logger))
model_checkpoint = ModelCheckpoint("Unet_main_classifier_vgg.hdf5", monitor='val_loss', verbose=1, save_best_only=True)

batch_size = 4
epochs = 30

history = tmodel.fit(train_images,train_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(test_images,test_labels),
                    verbose=1, shuffle=True, callbacks=[lr_reducer, csv_logger, model_checkpoint])



