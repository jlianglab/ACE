class JSRTLungDataset(Dataset):

    def __init__(self, image_path_file , image_size=(448,448), mode="train"):

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode


        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + "/images", line+".IMG.png"))
                    self.img_label.append(
                        (join(pathImageDirectory+"/masks/left_lung_png", line+".png"),(join(pathImageDirectory+"/masks/right_lung_png", line+".png")))
                         )
                    line = fileDescriptor.readline().strip()


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]



        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255



        leftMaskData = cv2.resize(cv2.imread(maskPath[0],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
        rightMaskData = cv2.resize(cv2.imread(maskPath[1],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        maskData = leftMaskData + rightMaskData
        maskData[maskData>0] =255
        maskData = maskData/255

        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])

        return img, mask