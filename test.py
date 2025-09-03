import os
import cv2
import torch
import numpy as np

from torch import nn
from tqdm import tqdm
from torchvision import transforms
import argparse
from networks.vit_seg_modeling import VisionTransformer as ViT
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from networks.vit_seg_modeling import VisionTransformer

#from networks.vit_seg_modeling_unext import UNext as UNext
import scipy
from scipy.spatial.distance import directed_hausdorff, cdist
import os
from datetime import datetime
from segment_anything import sam_model_registry
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def calmdsc(imgdir1, imgdir2):  # mdice
    miou = 0
    for img in os.listdir(imgdir1):
        img = img.replace('.jpg.jpg', '.jpg')
        imgpath1 = os.path.join(imgdir1, img)
        imgpath2 = os.path.join(imgdir2, img)
        if not os.path.exists(imgpath1) or not os.path.exists(imgpath2):
            continue  # 如果任何一个图像文件不存在，则跳过当前循环迭代
        img1 = cv2.imread(imgpath1, 0)
        img2 = cv2.imread(imgpath2, 0)
        img1 = cv2.resize(img1, (448, 448))
        img2 = cv2.resize(img2, (448, 448))
        img1[img1 <= 125] = 0
        img1[img1 > 125] = 1
        img2[img2 <= 125] = 0
        img2[img2 > 125] = 1
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        img3 = cv2.bitwise_and(img1, img2)
        img4 = cv2.bitwise_or(img1, img2)
        iou = 2 * img3.ravel().sum() / (img4.ravel().sum() + img3.ravel().sum()) if (img4.ravel().sum() + img3.ravel().sum()) != 0 else 0
        miou = miou + iou
    return miou / len(os.listdir(imgdir1))


def calmiou(imgdir1, imgdir2):  # miou
    miou = 0
    mdsc = 0
    for img in os.listdir(imgdir1):
        img = img.replace('.jpg.jpg', '.jpg')

        imgpath1 = os.path.join(imgdir1, img)
        imgpath2 = os.path.join(imgdir2, img)
        
        if not os.path.exists(imgpath1) or not os.path.exists(imgpath2):
            continue  # 如果任何一个图像文件不存在，则跳过当前循环迭代
        img1 = cv2.imread(imgpath1, 0)
        img2 = cv2.imread(imgpath2, 0)
        img1 = cv2.resize(img1, (448, 448))
        img2 = cv2.resize(img2, (448, 448))
        img1[img1 <= 125] = 0
        img1[img1 > 125] = 1
        img2[img2 <= 125] = 0
        img2[img2 > 125] = 1
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        img3 = cv2.bitwise_and(img1, img2)
        img4 = cv2.bitwise_or(img1, img2)
        iou = img3.ravel().sum() / img4.ravel().sum() if img4.ravel().sum() != 0 else 0
        miou = miou + iou

        dsc = 2 * img3.ravel().sum() / (img4.ravel().sum() + img3.ravel().sum()) if (img4.ravel().sum() + img3.ravel().sum()) != 0 else 0
        mdsc = mdsc + dsc

    return miou / len(os.listdir(imgdir1)), mdsc / len(os.listdir(imgdir1))

def calall(imgdir1, imgdir2):
    # 初始化各项指标
    miou = mdsc = accuracy = specificity = sensitivity = 0
    total_imgs = len(os.listdir(imgdir1))  # 计算图像总数
    for img in os.listdir(imgdir1):
        img = img.replace('.jpg.jpg', '.jpg')
        imgpath1 = os.path.join(imgdir1, img)
        imgpath2 = os.path.join(imgdir2, img)
        if not os.path.exists(imgpath1) or not os.path.exists(imgpath2):
            continue  # 如果任何一个图像文件不存在，则跳过当前循环迭代
        img1 = cv2.imread(imgpath1, 0)
        img2 = cv2.imread(imgpath2, 0)
        img1 = cv2.resize(img1, (448, 448))
        img2 = cv2.resize(img2, (448, 448))
        img1[img1 <= 125] = 0
        img1[img1 > 125] = 1
        img2[img2 <= 125] = 0
        img2[img2 > 125] = 1
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)

        # 计算交并比(IoU)和Dice相似系数(DSC)
        img3 = cv2.bitwise_and(img1, img2)
        img4 = cv2.bitwise_or(img1, img2)
        iou = img3.ravel().sum() / img4.ravel().sum() if img4.ravel().sum() != 0 else 0
        miou += iou
        dsc = 2 * img3.ravel().sum() / (img4.ravel().sum() + img3.ravel().sum()) if (img4.ravel().sum() + img3.ravel().sum()) != 0 else 0
        mdsc += dsc

        # 计算TP, TN, FP, FN
        tp = np.logical_and(img1 == 1, img2 == 1).sum()
        tn = np.logical_and(img1 == 0, img2 == 0).sum()
        fp = np.logical_and(img1 == 0, img2 == 1).sum()
        fn = np.logical_and(img1 == 1, img2 == 0).sum()

        # 计算准确性(Accuracy)，特异性(Specificity)，敏感性(Sensitivity)
        accuracy += (tp + tn) / (tp + tn + fp + fn)
        specificity += tn / (tn + fp) if (tn + fp) != 0 else 0
        sensitivity += tp / (tp + fn) if (tp + fn) != 0 else 0

    # 计算平均值并返回
    return miou / total_imgs, mdsc / total_imgs, accuracy / total_imgs, specificity / total_imgs, sensitivity / total_imgs

def calalll(imgdir1, imgdir2):
    total_imgs = len(os.listdir(imgdir1))
    if total_imgs == 0:
        return 0, 0, 0, 0, 0, 0, 0  # 如果没有图像，返回0

    miou = mdsc = accuracy = specificity = sensitivity = hausdorff_distance = assd = 0

    for img in os.listdir(imgdir1):
        imgpath1 = os.path.join(imgdir1, img)
        imgpath2 = os.path.join(imgdir2, img)
        img1 = cv2.imread(imgpath1, 0)
        img2 = cv2.imread(imgpath2, 0)
        img1 = cv2.resize(img1, (448, 448))
        img2 = cv2.resize(img2, (448, 448))
        img1[img1 <= 125] = 0
        img1[img1 > 125] = 1
        img2[img2 <= 125] = 0
        img2[img2 > 125] = 1
        img1 = img1.astype(np.bool_)
        img2 = img2.astype(np.bool_)

        # 计算交并比(IoU)和Dice相似系数(DSC)
        img3 = img1 & img2
        img4 = img1 | img2
        iou = img3.sum() / img4.sum() if img4.sum() != 0 else 0
        miou += iou
        dsc = 2 * img3.sum() / (img4.sum() + img3.sum()) if (img4.sum() + img3.sum()) != 0 else 0
        mdsc += dsc

        # 计算准确性，特异性，敏感性
        tp = (img1 & img2).sum()
        tn = (~img1 & ~img2).sum()
        fp = (~img1 & img2).sum()
        fn = (img1 & ~img2).sum()
        accuracy += (tp + tn) / (tp + tn + fp + fn)
        specificity += tn / (tn + fp) if (tn + fp) != 0 else 0
        sensitivity += tp / (tp + fn) if (tp + fn) != 0 else 0

        # 计算Hausdorff距离
        y_true, x_true = np.where(img1)
        y_pred, x_pred = np.where(img2)
        if y_true.size and y_pred.size:  # 确保图像不全是黑色的
            hausdorff_dist = max(directed_hausdorff(np.stack([y_true, x_true], axis=1), np.stack([y_pred, x_pred], axis=1))[0],
                                 directed_hausdorff(np.stack([y_pred, x_pred], axis=1), np.stack([y_true, x_true], axis=1))[0])
        else:
            hausdorff_dist = 0
        hausdorff_distance += hausdorff_dist

        # 计算平均对称表面距离(ASSD)
        if y_true.size and y_pred.size:  # 确保图像不全是黑色的
            true_points = np.stack([y_true, x_true], axis=1)
            pred_points = np.stack([y_pred, x_pred], axis=1)
            dist_matrix = scipy.spatial.distance.cdist(true_points, pred_points, 'euclidean')
            assd += np.mean(np.min(dist_matrix, axis=0)) + np.mean(np.min(dist_matrix, axis=1))
        else:
            assd += 0

    return miou / total_imgs, mdsc / total_imgs, accuracy / total_imgs, specificity / total_imgs, sensitivity / total_imgs,hausdorff_distance / total_imgs, assd / (2 * total_imgs)

if __name__ == "__main__":
    num_classes = 24
    backbone = 'resnet50'
    input_shape = (448, 448)
    
    data_path = '/20TB/fyh/data/yixue/truehard'
    image_dir = '/20TB/fyh/data/yixue/allimg'
    save_pdir = '/20TB/fyh/transunet-test2/predictSamImg'
    modelsort = "/best"
    model_path = /20TB/fyh/transunet-test2/results/model/ModelLabel_mad_best2025_07_11_10_06/best_epoch_weights.pth


    pred_save_path = save_pdir + modelsort + '/pred_image'
    miou_save = save_pdir + modelsort + '/pred_miou.txt'
    mdsc_save = save_pdir + modelsort + '/pred_mdice.txt'
    macc_save = save_pdir + modelsort + '/pred_macc.txt'
    mspec_save = save_pdir + modelsort + '/pred_mspec.txt'
    msen_save = save_pdir + modelsort + '/pred_msen.txt'
    mhd_save = save_pdir + modelsort + '/pred_mhd.txt'
    massd_save = save_pdir + modelsort + '/pred_assd.txt'

    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将图像变成PIL格式    输入为[H, W, C]输出为[H, W, C]
        transforms.ToTensor(),  # 将PIL图像转换为tensor    输入为[H, W, C]输出为[C, H, W]
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                default='../data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                default='Synapse', help='experiment_name') #突触
    parser.add_argument('--list_dir', type=str,
                default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                default=24, help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                default=1, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                default=448, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                default=1234, help='random seed')
    parser.add_argument('--n_skip', type=int,
                default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--vit_name', type=str,
                default='vit_h', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--ckpt', type=str, default='/20TB/fyh/transunet-test2/results/model/ModelLabel_mad_best2025_07_11_10_06/best_epoch_weights.pth', help='Pretrained checkpoint')
    parser.add_argument('--adapt_ckpt', type=str, default=None, help='Finetuned checkpoint')
    args = parser.parse_args()
    device = torch.device(args.device)
    args.nb_classes = 30


    net , img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0., 0., 0.],
                                                                pixel_std=[1., 1., 1.])
    checkpoint = torch.load('/20TB/fyh/transunet-test2/results/model/ModelLabel_mad_best2025_07_11_10_06/best_epoch_weights.pth', map_location='cpu')

    net.load_state_dict(checkpoint)

    net = net.eval()         # 测试模式
    #net = nn.DataParallel(net)
    net = net.cuda()
    print('==============================================Predicted Image Save!==============================================')
    with open(os.path.join(data_path, "test.txt"), "r") as f:
        image_files = f.read().splitlines()
    for img in tqdm(image_files):
        imgpath = os.path.join(image_dir, img + '.jpg')
        image = cv2.imread(imgpath, 0)
        image = cv2.resize(image, (448, 448))
        

        
        # 归一化使用
        image = np.expand_dims(image, -1).repeat(3, axis=-1)    # [448, 448, 3]
        image = transform(image)                                # [3, 448, 448]
        image = image.unsqueeze(0)                              # [b, 3, 448, 448]

        with torch.no_grad():
            pred = net(image.cuda(),True,448)      # [b, num_classes, h, w]
        pred = pred['masks']
        # 将列表中的每个预测结果应用 sigmoid 函数
        pred = [torch.sigmoid(p) for p in pred]
        pred = [p.detach().cpu().numpy() for p in pred]
        pred = pred[0]

        for i in range(num_classes):
            save_path = os.path.join(pred_save_path, str(i))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            savepath = os.path.join(save_path, img + '.jpg')      # 拼接存储路径
            
            pred_image = pred[i, :, :]
                #print('tensor',tensor.shape)
            pred_image = pred_image * 255
            pred_image[pred_image <= 127] = 0
            pred_image[pred_image > 127] = 255
            pred_image = pred_image.astype(np.uint8)
            cv2.imwrite(savepath, pred_image)

    print(
        '==================================================Compute  MIOU==================================================')
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
               '19','20', '21', '22', '23']
    iou_list = []
    iou_str_list = []
    dsc_list = []
    dsc_str_list = []
    macc_list = []
    macc_str_list = []
    msen_list = []
    msen_str_list = []
    mspec_list = []
    mspec_str_list = []
    mhd_list = []
    mhd_str_list = []
    massd_list = []
    massd_str_list = []
    for cls in tqdm(classes):
        imgpath = os.path.join(pred_save_path, cls)

        labelpath = os.path.join('/20TB/fyh/data/yixue/true_labels/', cls)

        #miou, mdsc, macc, msen, mspec = calall(imgpath, labelpath)
        miou, mdsc  = calmiou(imgpath, labelpath)
        iou_list.append(miou)
        miou_str = '第{}标签的miou: {}'.format(str(int(cls) + 1), miou)
        iou_str_list.append(miou_str)

        dsc_list.append(mdsc)
        mdsc_str = '第{}标签的mdsc: {}'.format(str(int(cls) + 1), mdsc)
        dsc_str_list.append(mdsc_str)

    # 计算不同范围的平均值
    mmiou_0_23 = sum(iou_list[0:24]) / 24
    mmiou_24_25 = sum(iou_list[24:26]) / 2
    mmiou_26_27 = sum(iou_list[26:28]) / 2
    mmiou_28_29 = sum(iou_list[28:30]) / 2
    mmiou_30 = sum(iou_list[30:31]) if len(iou_list) >= 31 else 0
    mmiou_31 = sum(iou_list[31:32]) if len(iou_list) >= 31 else 0

    # 计算总体平均值
    mmiou_str = sum(iou_list) / len(iou_list)

    # 构建结果字符串
    miou_str = '\n'.join(
        iou_str_list) + f'\n平均miou指标（0-23）：{mmiou_0_23}\n平均miou指标（24-25）：{mmiou_24_25}\n平均miou指标（26-27）：{mmiou_26_27}\n平均miou指标（28-29）：{mmiou_28_29}\n平均miou指标（30）：{mmiou_30}\n平均miou指标（31）：{mmiou_31}'

    # 打印平均指标
    print('平均指标：', mmiou_str)
    print('肋骨平均指标（0-23）：', mmiou_0_23)
    print('锁骨平均指标（24-25）：', mmiou_24_25)
    print('肩胛骨平均指标（26-27）：', mmiou_26_27)
    print('肺部平均指标（28-29）：', mmiou_28_29)
    print('气管平均指标（30）：', mmiou_30)
    print('纵隔平均指标（31）：', mmiou_31)
    with open(miou_save, 'w') as f:
        f.write(miou_str)

    mmdsc_0_23 = sum(dsc_list[0:24]) / 24
    mmdsc_24_25 = sum(dsc_list[24:26]) / 2
    mmdsc_26_27 = sum(dsc_list[26:28]) / 2
    mmdsc_28_29 = sum(dsc_list[28:30]) / 2
    mmdsc_30 = sum(dsc_list[30:31]) if len(dsc_list) >= 31 else 0  # 计算范围为 30 的平均值
    mmdsc_31 = sum(dsc_list[31:32]) if len(dsc_list) >= 31 else 0  # 计算范围为 30 的平均值

    mmdsc_str = sum(dsc_list) / len(dsc_list)
    mdsc_str = '\n'.join(
        dsc_str_list) + f'\n平均mdsc指标（0-23）：{mmdsc_0_23}\n平均mdsc指标（24-25）：{mmdsc_24_25}\n平均mdsc指标（26-27）：{mmdsc_26_27}\n平均mdsc指标（28-29）：{mmdsc_28_29}\n平均mdsc指标（30）：{mmdsc_30}\n平均mdsc指标（31）：{mmdsc_31}'
    print('平均指标：', mmdsc_str)
    print('肋骨平均指标（0-23）：', mmdsc_0_23)
    print('锁骨平均指标（24-25）：', mmdsc_24_25)
    print('肩胛骨平均指标（26-27）：', mmdsc_26_27)
    print('肺部平均指标（28-29）：', mmdsc_28_29)
    print('气管平均指标（30）：', mmdsc_30)
    print('纵隔平均指标（31）：', mmdsc_31)
    with open(mdsc_save, 'w') as d:
        d.write(mdsc_str)
 

