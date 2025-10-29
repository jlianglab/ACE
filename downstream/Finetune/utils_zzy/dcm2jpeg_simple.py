# for SIIM-ACR dataset
import os
import cv2
import pydicom
import numpy as np
import skimage.transform as transform

def dcm2jpeg(file, dst_path):
    print('FIle:', file)
    ds = pydicom.dcmread(file, force=True)
    # ds.file_meta.TransferSyntaxUID =
    # ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ori_img = np.array(ds.pixel_array)

    sharp = ori_img.shape
    _h = sharp[0]
    _w = sharp[1]
    if len(sharp) == 3:
        ori_img = ori_img[:, :, 0]
    img = transform.resize(ori_img, (_h, _w))

    start = img.min()
    end = img.max()

    img[img < start] = start
    img[img > end] = end
    img = np.array((img - start) * 255.0 / (end - start))
    if hasattr(ds, 'PhotometricInterpretation'):
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            img = 255 - img

    # img_name = os.path.basename(file).lower()
    # jpeg_path = str(ds.PatientID) + '_' + str(ds.SeriesDate) + '_' + str(ds.PatientSex) + '_'+str(ds.PatientAge) + '%d' % idx + '.jpeg'
    jpeg_name = os.path.basename(file).replace('.dcm', '.png')
    save_path = os.path.join(dst_path, jpeg_name)
    print(save_path)

    img = img.astype(np.uint8)
    cv2.imwrite(save_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print('save ok')
    return jpeg_name


def do_convert(file_path, png_folder):
    try:
         jpeg_path = dcm2jpeg(file_path, png_folder)

    except Exception as e:
        print('main process has error:%s' % e)


def run():
    ini_folder = '/sda1/zhouziyu/ssl/dataset/siim-acr-pneumothorax-segmentation/pneumothorax/dicom-images-train'
    jpeg_folder = '/sda1/zhouziyu/ssl/dataset/siim-acr-pneumothorax-segmentation/pneumothorax/png-images-train'
    for root, dirs, files in os.walk(ini_folder):
        for file in files:
            print(file)
            file_path = os.path.join(root, file)

            print('_pro' in file)
            if '_pro' in file:
                continue
            if file.lower().endswith('dcm') or file.lower().endswith('dicom'):
                do_convert(file_path, jpeg_folder)
                print('ok')

if __name__ == '__main__':
    run()
