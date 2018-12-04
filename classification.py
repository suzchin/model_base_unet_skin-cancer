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

test_mean = np.mean(test_images, axis=(0, 1, 2, 3))
test_std = np.std(test_images, axis=(0, 1, 2, 3))
test_images = (test_images-test_mean)/(test_std+1e-7)

#Training
model = UNetModel.get_unet_model_class((128, 128, 3))
model.summary()
model.compile(optimizer=Adam(lr=1e-25), loss='binary_crossentropy', metrics=['accuracy'])
lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=6, min_lr=0.5e-6)
csv_logger = CSVLogger(root_path+'OUTPUT/Unet_main_classifier.csv')
print('csv_logger:', type(csv_logger))
model_checkpoint = ModelCheckpoint(root_path+"OUTPUT/Unet_main_classifier.hdf5", monitor='val_loss', verbose=1,save_best_only=True)
model.fit(train_images, train_labels, batch_size=4, epochs=30, verbose=1, validation_data=(test_images, test_labels), shuffle=True, callbacks=[lr_reducer, csv_logger, model_checkpoint])

# #def save_model_every_epoch(model):
#
#     # print the summary of the model
# #model.summary()
#
#     # compile model
# #model.compile(optimizer=Adam(lr=1e-25), loss='binary_crossentropy', metrics=['accuracy'])
# lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=6, min_lr=0.5e-6)
# csv_logger = CSVLogger(root_path+'OUTPUT/Unet_classifier.csv')
#     # create checkpoint
# checkpoint_fn = root_path+"OUTPUT/Class_Unet_{epoch:03d}_{val_acc:.3f}.hdf5"
# model_checkpoint = ModelCheckpoint(checkpoint_fn, monitor='val_loss', verbose=1, save_best_only=True, period=1)
# #model_checkpoint = ModelCheckpoint('Unet_lr_e4_bs_10_class.hdf5', monitor='val_loss', verbose=1, save_best_only=True, period=1)
# history = model.fit(train_images, train_labels, batch_size=4, epochs=30, verbose=1, validation_data=(test_images, test_labels), shuffle=True,
#               callbacks=[lr_reducer, csv_logger, model_checkpoint])
## Testing save_model_every_epoch()
#model = get_unet_model((128, 192, 3))



