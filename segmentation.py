from Lib import *
root_path = "D:/Learnning/TensorFlow/program/model_base_Unet/"

# # ----- load data -----------
train_images = np.load(root_path+'OUTPUT/segmentation_train_image.npy')
train_labels = np.load(root_path+'OUTPUT/segmentation_train_labels.npy')
evaluate_images = np.load(root_path+'OUTPUT/segmentation_evaluate_image.npy')
evaluate_labels = np.load(root_path+'OUTPUT/segmentation_evaluate_labels.npy')

train_mean = np.mean(train_images, axis=(0, 1, 2, 3))
train_std = np.std(train_images, axis=(0, 1, 2, 3))
train_images = (train_images - train_mean)/(train_std+1e-7)

train_labels = np.expand_dims(train_labels, axis=3)

evaluate_mean = np.mean(evaluate_images, axis=(0, 1, 2, 3))
evaluate_std = np.std(evaluate_images, axis=(0, 1, 2, 3))
evaluate_images = (evaluate_images - evaluate_mean)/(evaluate_std+1e-7)

evaluate_labels = np.expand_dims(evaluate_labels, axis=3)

# Training
model = UNetModel.get_unet_model_seg((128, 128, 3))
model.summary()

model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=6, min_lr=0.5e-6)

csv_logger = CSVLogger(root_path+'OUTPUT/Unet_lr_e4_bs_4.csv')

model_checkpoint = ModelCheckpoint(root_path+"OUTPUT/Unet_lr_e4_bs_4.hdf5", monitor='val_loss', verbose=1, save_best_only=True)

model.fit(train_images, train_labels, batch_size=4, epochs=10, verbose=1, validation_data=(evaluate_images, evaluate_labels), shuffle=True, callbacks=[lr_reducer, csv_logger, model_checkpoint])
# finish training
print('Congratulations on your successful training')