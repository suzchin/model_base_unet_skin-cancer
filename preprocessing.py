from Lib import *

root_path = "D:/Learnning/TensorFlow/program/model_base_Unet/"

#---------------------------------------------
#--resize_data 128x128 -----------------------
# resize_image(root_path+"DATA/data_evaluate1/", root_path+"DATA/data_evaluate/")
# resize_image(root_path+"DATA/data_test1/", root_path+"DATA/data_test/")
# resize_image(root_path+"DATA/data_train1/", root_path+"DATA/data_train/")
# resize_image(root_path+"DATA/gt_label1/", root_path+"DATA/gt_label/")

#---------------------------------------------
#--data train segment -----------------------
filenames_train = get_filenames(root_path+"DATA/data_train/")
print('len(filenames_train)=', len(filenames_train))
filenames_train.sort(key=natural_key)
image_train = []
for file in filenames_train:
    image_train.append(ndimage.imread(root_path+"DATA/data_train/"+file))
image_train = np.array(image_train)
print('len_image_train', len(image_train))
np.save(root_path+'OUTPUT/segmentation_train_image.npy', image_train)
#print('train_image_segXXX:', image_train.shape)
#plt.imshow(image_train[0])
#plt.show()
#---------------------------------------------
#--data test segment ----------------------
filenames_test = get_filenames(root_path+"DATA/data_test/")
filenames_test.sort(key=natural_key)
image_test = []
for file in filenames_test:
    image_test.append(ndimage.imread(root_path+"DATA/data_test/"+file))
print('len(filenames_test_seg)=', len(filenames_test))
image_test = np.array(image_test)
np.save(root_path+'OUTPUT/segmentation_test_image.npy', image_test)
print('test_image_seg:', len(image_test))

#---------------------------------------------
#--data test evaluation ----------------------
filenames_evaluate = get_filenames(root_path+"DATA/data_evaluate/")
filenames_evaluate.sort(key=natural_key)
image_evaluate = []
for file in filenames_evaluate:
    image_evaluate.append(ndimage.imread(root_path+"DATA/data_evaluate/"+file))
print('len(filenames_evaluate_seg)=', len(filenames_evaluate))
image_evaluate = np.array(image_evaluate)
np.save(root_path+'OUTPUT/segmentation_evaluate_image.npy', image_evaluate)
print('evaluate_image_seg:', len(image_evaluate))

#---------------------------------------------
#--data _label_ segment ----------------------
filenames_gt = get_filenames(root_path+"DATA/gt_label/")
gt_images = []
j = 0
for fike in filenames_gt:
    filee = fike.replace("_segmentation_","_")
    if os.path.exists(root_path+"DATA/data_train/"+filee):
        gt_images.append(ndimage.imread(root_path + "DATA/gt_label/" + fike))
        #print('j={} fike={}'.format(j,fike))
        #plt.imshow(gt_images[0], cmap="gray")
        #plt.show()
    j+=1
#print('gt_images:', len(gt_images))
np.unique(gt_images[0])
gt_labels_binary = []
for gt_image in gt_images:
    ret, image = cv2.threshold(gt_image, 127, 255, cv2.THRESH_BINARY)
    gt_labels_binary.append(image)

gt_labels_binary = np.array(gt_labels_binary)
np.unique(gt_labels_binary[0])
gt_labels_binary = gt_labels_binary/255
np.unique(gt_labels_binary[0])
train_labels_seg = gt_labels_binary
np.save(root_path+'OUTPUT/segmentation_train_labels.npy', train_labels_seg)

plt.imshow(train_labels_seg[0])
plt.show()

#---------------------------------------------
#--data _label_ evaluate ---------------------
filenames_gt = get_filenames(root_path+"DATA/gt_label/")
gt_images = []
j = 0
for fike in filenames_gt:
    filee = fike.replace("_segmentation_","_")
    if os.path.exists(root_path+"DATA/data_evaluate/"+filee):
        gt_images.append(ndimage.imread(root_path + "DATA/gt_label/" + fike))
        #print('j={} fike={}'.format(j,fike))
        #plt.imshow(gt_images[0], cmap="gray")
        #plt.show()
    j+=1
print('gt_images_evl:', len(gt_images))
np.unique(gt_images[0])
gt_labels_binary = []
for gt_image in gt_images:
    ret, image = cv2.threshold(gt_image, 127, 255, cv2.THRESH_BINARY)
    gt_labels_binary.append(image)

gt_labels_binary = np.array(gt_labels_binary)
np.unique(gt_labels_binary[0])
gt_labels_binary = gt_labels_binary/255
np.unique(gt_labels_binary[0])
evaluate_labels_seg = gt_labels_binary
np.save(root_path+'OUTPUT/segmentation_evaluate_labels.npy', evaluate_labels_seg)

#--data train_label_ classification --------------------------
classification_labels = np.zeros((len(image_train)))
i = 0
for file in filenames_train:
    if os.path.exists(root_path+"DATA/melanoma/"+file):
        classification_labels[i]=0
    else:
        classification_labels[i]=1
    i+=1
print('classification_labels', classification_labels)
train_labels_class = classification_labels
np.save(root_path+'OUTPUT/classification_train_labels.npy', train_labels_class)

#--data train_label_ evaluation --------------------------
classification_labels_evaluate = np.zeros((len(image_evaluate)))
i = 0
for file in filenames_evaluate:
    if os.path.exists(root_path+"DATA/melanoma/"+file):
        classification_labels_evaluate[i]=0
    else:
        classification_labels_evaluate[i]=1
    i+=1
evaluate_labels_class = classification_labels_evaluate
np.save(root_path+'OUTPUT/evaluate_labels_class_labels.npy', evaluate_labels_class)