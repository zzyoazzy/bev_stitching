import time
import cv2
import numpy as np
import os
from argparse import ArgumentParser



def file_name(file_dir: str)->list:
    '''

    打开目录并返回目录下的文件名列表
    :param file_dir: 目标文件所在目录

    '''

    pics = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            pics.append(os.path.join(root, file))
    return pics


def match_feather_point(img1: np.ndarray, img2: np.ndarray)->tuple:
    '''

    SIFT匹配点对
    :param img1: 左图
    :param img2: 右图
    :return kp1: 左图特征点
    :return kp2: 右图特征点
    :return matches: 点对匹配

    '''

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return kp1, kp2, matches


def stitching(img1, img2, kp1:list, kp2:list, matches:list)->tuple:
    '''
    
    拼接图片
    :param img1: 左图
    :param img2: 右图
    :param kp1: 左图特征点
    :param kp2: 右图特征点
    :param matches: 点对匹配
    :return img3: 匹配点可视化结果
    :return dst: 拼接结果
    :return M: homography 矩阵

    '''
    matches_mask = [[0, 0] for _ in range(len(matches))]
    good_match = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_match.append(m)
            matches_mask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matches_mask,
                       flags=0)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    MIN_MATCH_COUNT = 10
    if len(good_match) < MIN_MATCH_COUNT:
        raise RuntimeError(f'good match points:{len(good_match)} < MIN_MATCH_COUNT:{MIN_MATCH_COUNT}')

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    img2 = cv2.warpPerspective(img2, np.array(M), (img2.shape[1], img2.shape[0]),
                              flags=cv2.WARP_INVERSE_MAP)

    w = img1.shape[1]
    mask = img2[:, 0:w]
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 25, 1, cv2.THRESH_BINARY)

    #==============边缘高斯模糊=================#
    blur = np.float32(mask)
    blur = np.stack([blur, blur, blur], axis=2)
    blur *= 255
    blur = cv2.GaussianBlur(blur, (9,9), 0.5)       
    blur /= 255

    mask = 1 - mask
    mask = np.float32(mask)
    mask_rgb = np.stack([mask,mask,mask], axis=2)
    mask_rgb = 1 - blur
    img1 = img1*mask_rgb + img2[:, 0:w]*(1-mask_rgb)
    dst = img2.copy()
    dst[:, 0:w] = img1

    return img3, dst, M



def bev_single(origin_path:str = './images')->None:
    '''
    生成单张bev图片
    :param origin_path: 原始图像路径(目录)
    '''
    pics = file_name(origin_path)
    homography = np.load('./camera_1_H.npy') #加载bev转换所需的homography矩阵
    index = 0
    print(pics)
    for pic in pics:
        img = cv2.imread(pic)
        warp = cv2.warpPerspective(img, homography, (2048, 2048))
        warp = warp[512:2048,:,:]
        cv2.imwrite('./bev_single/'+ str(index)+'.jpg', warp)
        print('save: ' + str(index))
        index+=1



def bev_stitching(output:str = './output.jpg', viz:bool = False)->None:
    '''
    拼接bev图片
    :param output: 输出结果路径(文件)
    :param viz: 是否生成点对匹配结果图像
    '''

    bev_paths = file_name('./bev_single')
    bevs = list()
    for bev_path in bev_paths:
        img = cv2.imread(bev_path)
        # 最外层轮廓像素（0，0，0）减轻拼接线明显程度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        edge = cv2.Canny(thresh, 50, 100, apertureSize=3, L2gradient=True)
        img1, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        draw_img = img.copy()
        res = cv2.drawContours(draw_img, contours, -1, (0, 0, 0), 10)
        # 填补图片以拼接
        top, bot, left, right = 1024, 3072, 0, 3072
        img = cv2.copyMakeBorder(res, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        bevs.append(img)
    index = -1
    left, right = None, None
    for bev in bevs:
        index += 1
        if index == 0:
            left = bev
            continue
        else:
            right = bev
            src_img, warp_img = left, right
            start = time.time()
            kp1, kp2, matches = match_feather_point(src_img, warp_img)
            end = time.time()
            print('特征点计算以及匹配的时间：', end - start)
            start = time.time()
            img, dst, M = stitching(src_img, warp_img, kp1, kp2, matches)
            end = time.time()
            print('去除误匹配点、计算变换矩阵并进行拼接的时间：', end - start)
            print("变换矩阵\n", M)
            # cv2.imwrite("splicing.jpg", dst)
            if viz:
                cv2.imwrite("sift_match.jpg", img)  #匹配结果输出
            left = dst
    cv2.imwrite(output, left)


def bev_stitching_mask(output:str = './output_mask.jpg', viz:bool = False)->None:
    '''
    拼接bev图片
    :param output: 输出结果路径(文件)
    :param viz: 是否生成点对匹配结果图像
    '''

    bev_paths = file_name('./bev_single')
    bevs = list()
    for bev_path in bev_paths:
        img = cv2.imread(bev_path)
        # 最外层轮廓像素（0，0，0）减轻拼接线明显程度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        img1, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        draw_img = img.copy()
        res = cv2.drawContours(draw_img, contours, -1, (0, 0, 10), 50)
        
        # 填补图片以拼接
        top, bot, left, right = 1024, 3072, 0, 3072
        img = cv2.copyMakeBorder(res, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask = cv2.drawContours(np.zeros(img.shape), contours, -1, (0, 0, 255), 50)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(mask, 200, 1, cv2.THRESH_BINARY)
        cv2.imwrite('mask.jpg', mask)
        bevs.append(img)
    index = -1
    left, right = None, None
    for bev in bevs:
        index += 1
        if index == 0:
            left = bev
            continue
        else:
            right = bev
            src_img, warp_img = left, right
            start = time.time()
            kp1, kp2, matches = match_feather_point(src_img, warp_img)
            end = time.time()
            print('特征点计算以及匹配的时间：', end - start)
            start = time.time()
            img, dst, M = stitching(src_img, warp_img, kp1, kp2, matches)
            end = time.time()
            print('去除误匹配点、计算变换矩阵并进行拼接的时间：', end - start)
            print("变换矩阵\n", M)
            # cv2.imwrite("splicing.jpg", dst)
            if viz:
                cv2.imwrite("sift_match.jpg", img)  #匹配结果输出
            left = dst
    cv2.imwrite(output, left)




if __name__=='__main__':

    parser = ArgumentParser(description='transform forward perspective to BEV, then stitch all')
    parser.add_argument('--origin_image', type=str, help='path of original images', default='./images')
    parser.add_argument('--output', type=str, help='output image path', default='./output.jpg')
    parser.add_argument('-v', action='store_true', help='output visualizition of point matches result')

    args = parser.parse_args()

    bev_single(args.origin_image)
    # bev_stitching(args.output, args.v)
    bev_stitching_mask()


