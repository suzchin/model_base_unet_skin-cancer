from Lib import *
root_path = "D:/Learnning/TensorFlow/program/model_base_Unet/"
# load image
train_images = np.load(root_path+'OUTPUT/segmentation_evaluate_image.npy')
print('test_images.shape', train_images.shape)
filenames_train = get_filenames(root_path+"DATA/data_evaluate/")
filenames_train.sort(key=natural_key)

# load weights model
model = UNetModel.get_unet_model_seg((128, 128, 3))
model.load_weights(root_path+'OUTPUT/Unet_lr_e4_bs_4.hdf5')
gt_new_seg = []
for ix in range(len(train_images)):
    sample_predictions = model.predict(train_images[ix].reshape((1, 128, 128, 3)))
    sample_predictions = sample_predictions.reshape((128, 128))
    sample_predictions = sample_predictions > 0.5
    sample_predictions = np.array(sample_predictions, dtype=np.uint8)
    gt_new_seg.append(sample_predictions)

np.save(root_path+"OUTPUT/eval_gt_new_seg.npy", gt_new_seg)
# plt.figure()
# plt.imshow(gt_new_seg[2], cmap="gray")
# plt.figure()
#gt_images = np.load(root_path+"OUTPUT/gt_new_seg.npy")
# plt.imshow(gt_new_seg[2], cmap="gray")
# plt.show()

#----------------------------------------------------------------
#----------------------------------------------------------------

ground_truth_images = np.load(root_path+'OUTPUT/eval_gt_new_seg.npy')

segmented_images = np.copy(train_images)
x, y, z = segmented_images[0].shape

for i in range(len(train_images)):
    for j in range(x):
        for k in range(y):
            for l in range(z):
                segmented_images[i][j][k][l] = train_images[i][j][k][l] if ground_truth_images[i][j][k] == 1 else 0
    misc.imsave(root_path+"DATA/segmented_images_eval/segmented_"+filenames_train[i], segmented_images[i])

np.save(root_path+'OUTPUT/segmented_images_evaluate.npy', segmented_images)
segmented_images = np.load(root_path+'OUTPUT/segmented_images_evaluate.npy')
plt.imshow(segmented_images[2], cmap="gray")
plt.show()
