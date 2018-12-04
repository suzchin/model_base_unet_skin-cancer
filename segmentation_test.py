from Lib import *
root_path = "D:/Learnning/TensorFlow/program/model_base_Unet/"
root_path1 = "D:/Learnning/TensorFlow/program/model_base_Vgg/"
# load image
test_images = np.load(root_path+'OUTPUT/segmentation_test_image.npy')
print('test_images.shape', test_images.shape)
filenames_test = get_filenames(root_path+"DATA/data_train/")
filenames_test.sort(key=natural_key)

# load weights model
model = UNetModel.get_unet_model_seg((128, 128, 3))
#model.load_weights(root_path+'OUTPUT/Unet_lr_e4_bs_4.hdf5')
model.load_weights(root_path1+'OUTPUT/Unet_weight_segmentation.hdf5')
gt_new_seg = []
#for ix in range(len(test_images)):
sample_predictions = model.predict(test_images[2].reshape((1, 128, 128, 3)))
sample_predictions = sample_predictions.reshape((128, 128))
sample_predictions = sample_predictions > 0.5
sample_predictions = np.array(sample_predictions, dtype=np.uint8)
#    gt_new_seg.append(sample_predictions)

#np.save(root_path+"OUTPUT/gt_new_seg.npy", gt_new_seg)
plt.figure()
plt.imshow(sample_predictions, cmap="gray")
# plt.figure()
#gt_images = np.load(root_path+"OUTPUT/gt_new_seg.npy")
# plt.imshow(gt_new_seg[2], cmap="gray")
plt.show()

#----------------------------------------------------------------
#----------------------------------------------------------------

# ground_truth_images = np.load(root_path+'OUTPUT/gt_new_seg.npy')
#
# segmented_images = np.copy(test_images)
# x, y, z = segmented_images[0].shape
#
# for i in range(len(test_images)):
#     for j in range(x):
#         for k in range(y):
#             for l in range(z):
#                 segmented_images[i][j][k][l] = test_images[i][j][k][l] if ground_truth_images[i][j][k] == 1 else 0
#     misc.imsave(root_path+"DATA/segmented_images_train/segmented_"+filenames_test[i], segmented_images[i])
#
# np.save(root_path+'OUTPUT/segmented_images_class.npy', segmented_images)
# segmented_images = np.load(root_path+'OUTPUT/segmented_images_class.npy')
# plt.imshow(segmented_images[2], cmap="gray")
# plt.show()
