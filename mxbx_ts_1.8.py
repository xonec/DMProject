'''
/*
 * Copyright 2020 DMProject and contributors.
 *
 * 此源代码的使用受 GNU AFFERO GENERAL PUBLIC LICENSE version 3 许可证的约束, 可以在以下链接找到该许可证.
 * Use of this source code is governed by the GNU AGPLv3 license that can be found through the following link.
 *
 * https://github.com/super1207/DMProject/blob/master/LICENSE
 */
'''

'''
@Description: 大漠插件7.1904[图色]API部分的python实现
@FilePath: dmpic.py
'''
import time
import numpy as np
import pyautogui
import copy
import cv2
from sklearn import cluster


class TuSe:
    def __init__(self):
        print('欢迎使用')

    def GetCapture(self, stax, stay, endx, endy):
        w = endx - stax
        h = endy - stay
        im = pyautogui.screenshot(region=(stax, stay, w, h))
        # im = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
        return np.array(im)

    def FindPic(self, x1, y1, x2, y2, path, thd=0.9, type=1):
        '''
        找图
        :param x1: 起点X
        :param y1: 起点Y
        :param x2: 终点X
        :param y2: 终点Y
        :param path: 图片路径
        :param thd: 相似度
        :param type: 默认1为灰度化找图，其他为彩色找图
        :return: 图片中心坐标
        '''
        img = self.GetCapture(x1, y1, x2, y2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        template = cv2.imread(path)
        th, tw = template.shape[:2]
        if type == 1:
            # print('灰度化找图')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        rv = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(rv)
        # print(minVal, maxVal, minLoc, maxLoc)
        if maxVal < thd:
            return -1, -1
        else:
            return maxLoc[0] + tw / 2 + x1, maxLoc[1] + th / 2 + y1

    def FindPics(self, des):
        img = pyautogui.screenshot()
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = {}
        for key, value in des.items():
            template = cv2.imread(value[4])
            base = img[value[1]:value[3], value[0]:value[2]]
            th, tw = template.shape[:2]
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            rv = cv2.matchTemplate(base, template, cv2.TM_CCOEFF_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(rv)
            if maxVal < value[5]:
                res.setdefault(key, (-1, -1))
            else:
                res.setdefault(key, (maxLoc[0] + tw / 2 + value[0], maxLoc[1] + th / 2 + value[1]))
        return res

    def Hex_to_Rgb(self, hex):
        '''
        十六进制转RGB
        :param hex: 十六进制颜色值
        :return: RGB
        '''
        return np.array(tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4)))

    def CmpColor(self, x, y, color, sim: float):
        '''
        比色
        :param x: X坐标
        :param y: Y坐标
        :param color: 十六进制颜色，可以从大漠直接获取
        :param sim: 相似度(0-1对应二值化的255-0)，1为完全匹配
        :return: 真或假
        '''
        img = self.GetCapture(x - 1, y - 1, x + 1, y + 1)
        img = img[1][1]
        color = self.Hex_to_Rgb(color)
        res = np.absolute(color - img)
        sim = int((1 - sim) * 255)
        return True if np.amax(res) <= sim else False

    def FindColor(self, x1, y1, x2, y2, des, sim: float):
        '''
        找色
        :param x1: 起点X
        :param y1: 起点Y
        :param x2: 终点X
        :param y2: 终点Y
        :param des: 十六进制颜色，可以从大漠直接获取
        :param sim: 相似度(0-1对应二值化的255-0)，1为完全匹配
        :return:
        '''
        img = self.GetCapture(x1, y1, x2, y2)
        res = np.absolute(img - self.Hex_to_Rgb(des))
        sim = int((1 - sim) * 255)
        res = np.argwhere(np.all(res <= sim, axis=2))
        res = res + (y1, x1)
        return res[:, [1, 0]]

    def GetColorNum(self, x1, y1, x2, y2, des, sim: float):
        '''
        获取颜色数量
        :param x1: 起点X
        :param y1: 起点Y
        :param x2: 终点X
        :param y2: 终点Y
        :param des: 十六进制颜色，可以从大漠直接获取
        :param sim: 相似度(0-1对应二值化的255-0)，1为完全匹配
        :return:
        '''
        return len(self.FindColor(x1, y1, x2, y2, des, sim))

    def FindMultColor(self, stax, stay, endx, endy, des):
        '''
        多点找色
        :param stax:
        :param stay:
        :param endx:
        :param endy:
        :param des: 大漠获取到的多点找色数据，偏色必须写上
        :return:
        '''
        w = endx - stax
        h = endy - stay
        img = pyautogui.screenshot(region=(stax, stay, w, h))
        img = np.array(img)
        rgby = []
        ps = []
        a = 0
        firstXY = []
        res = np.empty([0, 2])
        for i in des.split(','):
            rgb_y = i[-13:]
            r = int(rgb_y[0:2], 16)
            g = int(rgb_y[2:4], 16)
            b = int(rgb_y[4:6], 16)
            y = int(rgb_y[-2:])
            rgby.append([r, g, b, y])
        for i in range(1, len(des.split(','))):
            ps.append([int(des.split(',')[i].split('|')[0]), int(des.split(',')[i].split('|')[1])])
        for i in rgby:
            result = np.logical_and(abs(img[:, :, 0:1] - i[0]) < i[3], abs(img[:, :, 1:2] - i[1]) < i[3],
                                    abs(img[:, :, 2:3] - i[2]) < i[3])
            results = np.argwhere(np.all(result == True, axis=2)).tolist()
            if a == 0:
                firstXY = copy.deepcopy(results)
            else:
                nextnextXY = copy.deepcopy(results)
                for index in nextnextXY:
                    index[0] = int(index[0]) - ps[a - 1][1]
                    index[1] = int(index[1]) - ps[a - 1][0]
                q = set([tuple(t) for t in firstXY])
                w = set([tuple(t) for t in nextnextXY])
                matched = np.array(list(q.intersection(w)))
                res = np.append(res, matched, axis=0)
            a += 1
        unique, counts = np.unique(res, return_counts=True, axis=0)
        index = np.argmax(counts)
        re = unique[index] + (stay, stax)
        if np.max(counts) == len(des.split(',')) - 1:
            return np.flipud(re)
        return np.array([-1, -1])

    def FindPicEx(self, x1, y1, x2, y2, path, thd=0.9, MIN_MATCH_COUNT=8):
        '''
        全分辨率找图
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param path:
        :param thd: 相似度
        :param MIN_MATCH_COUNT: 特征点数量
        :return:
        '''
        thd = thd - 0.2
        template = cv2.imread(path, 0)  # queryImage
        # target = cv2.imread('target.jpg', 0)  # trainImage
        target = self.GetCapture(x1, y1, x2, y2)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        # Initiate SIFT detector创建sift检测器
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(template, None)
        kp2, des2 = sift.detectAndCompute(target, None)
        # 创建设置FLANN匹配
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        # 舍弃大于0.7的匹配
        for m, n in matches:
            if m.distance < thd * n.distance:
                good.append(m)
        if len(good) > MIN_MATCH_COUNT:
            # 获取关键点的坐标
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # 计算变换矩阵和MASK
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = template.shape
            # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            res = (dst[0] + dst[2]) / 2  # [[[ 39.11337  147.11575 ]] [[135.06624  255.12143 ]]
            return int(res[0][0]) + x1, int(res[0][1]) + y1
        else:
            return -1, -1

    def _FilterRec(self, res, loc):
        """ 对同一对象的多个框按位置聚类后，按置信度选最大的一个进行保留。
        :param res: 是 cv2.matchTemplate 返回值
        :param loc: 是 cv2.np.argwhere(res>threshold) 返回值
        :return: 返回保留的点的列表 pts
        """
        model = cluster.AffinityPropagation(damping=0.5, max_iter=100, convergence_iter=10, preference=-50).fit(loc)
        y_pred = model.labels_
        pts = []
        for i in set(y_pred):
            argj = loc[y_pred == i]
            argi = argj.T
            pt = argj[np.argmax(res[tuple(argi)])]
            pts.append(pt[::-1])
        return np.array(pts)

    def FindMultPic(self, x1, y1, x2, y2, path, thd=0.8):
        '''
        多目标找图
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param path:
        :param thd: 相似度
        :return:
        '''
        target = self.GetCapture(x1, y1, x2, y2)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(path, 0)
        w, h = template.shape[:2]
        res = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
        loc = np.argwhere(res >= thd)
        if len(loc):
            resc = self._FilterRec(res, loc)
            return resc + (h / 2 + x1, w / 2 + y1)
        else:
            return [[-1, -1]]

    def FindPic_TM(self, x1, y1, x2, y2, path, thd=0.9):
        '''
        找透明图，透明色为黑色
        :param x1: 起点X
        :param y1: 起点Y
        :param x2: 终点X
        :param y2: 终点Y
        :param path: 图片路径
        :param thd: 相似度
        :return: 图片中心坐标
        '''
        img = self.GetCapture(x1, y1, x2, y2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        template = cv2.imread(path)
        template2 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(template2, 20, 255, cv2.THRESH_BINARY)
        th, tw = template.shape[:2]
        rv = cv2.matchTemplate(img, template, 1, mask=mask)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(rv)
        if 1 - minVal >= thd:
            return minLoc[0] + tw / 2 + x1, minLoc[1] + th / 2 + y1
        else:
            return -1, -1

    def GetCaptre_TM(self, x1, y1, x2, y2, path, times=5):
        '''
        动图变静态图片，改为静态图片后使用FindPic_TM进行找图
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param times:
        :return:
        '''
        w = x2 - x1
        h = y2 - y1
        captime = 0
        st = time.time()
        a = pyautogui.screenshot(region=(x1, y1, w, h))
        a = np.array(a)
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        # time.sleep(0.1)
        while 1:
            if captime > times:
                break
            b = pyautogui.screenshot(region=(x1, y1, w, h))
            b = np.array(b)
            b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
            b = np.where(np.all(a == b, axis=-1, keepdims=True), a, [0, 0, 0])
            b = cv2.convertScaleAbs(b)
            a = b.copy()
            captime = time.time() - st
        cv2.imwrite(path, a)
        return a

    def StressShow(self, stax, stay, endx, endy, des, type=0):
        '''
        保留选中颜色，其他为黑色，相似度根据偏色调整
        :param stax:
        :param stay:
        :param endx:
        :param endy:
        :param des: 大漠的色彩描述
        :param type: 0为原来颜色，1为白色
        :return:
        '''
        # des = 'e81010-101010|f9ad08-000000'
        dess = des.split('|')
        des = [i[0:6] for i in dess]
        des = [np.array(self.Hex_to_Rgb(d)) for d in des]
        pds = [i[-6:] for i in dess]
        pds = tuple(tuple(int(item[i:i + 2]) for i in range(0, len(item), 2)) for item in pds)
        img = self.GetCapture(stax, stay, endx, endy)
        mask = np.zeros(img.shape[:2], dtype=np.bool_)
        for i, color in enumerate(des):
            mask += np.all(np.abs(img - color) <= pds[i], axis=-1)
        new_img = np.where(mask[..., None], [255, 255, 255], [0, 0, 0]) if type else np.where(mask[..., None], img,
                                                                                              [0, 0,
                                                                                               0])  # 修改这里，将选中的颜色设为白色
        img_converted = cv2.convertScaleAbs(new_img)
        img_converted = cv2.cvtColor(np.array(img_converted), cv2.COLOR_BGR2RGB)
        return img_converted

    def SetDict(self, path):
        des = {}
        with open(path, 'r', encoding='GBK') as f:
            text = f.read()
            lines = text.splitlines()
            for line in lines:
                parts = line.split("$")
                # self.des.setdefault(parts[1],parts[0])
                bin_str = ''
                for c in parts[0]:
                    byte = int(c, 16)
                    byte_bin = bin(byte)[2:].zfill(4)
                    bin_str += byte_bin
                m = len(bin_str) // 11
                if (m % 4):
                    bin_str = bin_str[:-(m % 4)]
                arr = np.array([list(bin_str[i:i + 11]) for i in range(0, len(bin_str), 11)], dtype=np.float32)
                arr = arr.transpose()
                des.setdefault(parts[1], arr)
        # print(self.des)
        return des

    def FindString(self, x1, y1, x2, y2, strs, color, thd, DIict):
        if strs not in DIict:
            print('字符串不存在')
            return -1, -1
        else:
            arr = DIict[strs]
        img = self.StressShow(x1, y1, x2, y2, color, 1)
        img = (img != 0).any(axis=2).astype(int)
        thresh = np.array(img, dtype=np.float32)
        result = cv2.matchTemplate(arr, thresh, cv2.TM_CCOEFF_NORMED)
        minv, maxv, minl, maxl = cv2.minMaxLoc(result)
        # print(minv, maxv, minl, maxl)
        w, h = arr.shape
        if maxv < thd:
            return -1, -1
        else:
            return int(maxl[0] + h / 2 + x1), int(maxl[1] + w / 2 + y1)

    def Ocr(self, x1, y1, x2, y2, des, thd, DIict):
        dess = des.split('|')
        des = [i[0:6] for i in dess]
        des = [np.array(self.Hex_to_Rgb(d)) for d in des]
        pds = [i[-6:] for i in dess]
        pds = tuple(tuple(int(item[i:i + 2]) for i in range(0, len(item), 2)) for item in pds)
        img = self.GetCapture(x1, y1, x2, y2)
        mask = np.zeros(img.shape[:2], dtype=np.bool_)
        for i, color in enumerate(des):
            mask += np.all(np.abs(img - color) <= pds[i], axis=-1)
        new_img = np.where(mask[..., None], [255, 255, 255], [0, 0, 0]) if type else np.where(mask[..., None], img,
                                                                                              [0, 0,
                                                                                               0])  # 修改这里，将选中的颜色设为白色
        img_converted = cv2.convertScaleAbs(new_img)
        img_converted = cv2.cvtColor(np.array(img_converted), cv2.COLOR_BGR2RGB)
        img = (img_converted != 0).any(axis=2).astype(int)
        img = np.array(img, dtype=np.float32)
        res = {}
        for key, value in DIict.items():
            w, h = value.shape
            result = cv2.matchTemplate(value, img, cv2.TM_CCOEFF_NORMED)
            loc = np.argwhere(result >= thd)
            if len(loc):
                resc = self._FilterRec(result, loc)
                resc.astype(np.int64)
                resc += np.array((h / 2 + x1, w / 2 + y1)).astype(np.int64)
                resc = [(i[0], i[1]) for i in resc]
                res.setdefault(key, resc)
            else:
                res.setdefault(key, [(-1, -1)])
        return res

    def getstr(self, original_data):
        sorted_data = sorted(original_data.items(), key=lambda item: item[1][0][1])
        grouped_data = []
        for char, coord_list in sorted_data:
            if not grouped_data:
                grouped_data.append([(char, coord) for coord in coord_list])
            else:
                added = False
                for sublist in grouped_data:
                    if coord_list[0][1] == sublist[0][1][1]:
                        sublist.extend([(char, coord) for coord in coord_list])
                        added = True
                        break
                if not added:
                    grouped_data.append([(char, coord) for coord in coord_list])
        return grouped_data

    def OcrFix(self, input_dict, size=20):
        items = sorted(input_dict, key=lambda x: x[1][0])
        merged_dict = {}
        i = 0
        while i < len(items):
            curr_key = items[i][0]
            curr_value = items[i][1]
            i += 1
            if curr_value == (-1, -1):
                continue
            while i < len(items) and items[i][1][0] - curr_value[0] <= size:
                curr_key += items[i][0]
                curr_value = items[i][1]
                if curr_value == (-1, -1):
                    merged_dict.pop(curr_key, None)
                    break
                i += 1
            merged_dict[curr_key] = curr_value
        return merged_dict

    def GetOcr(self, data):
        res = {}
        dat = self.getstr(data)
        for i in dat:
            x = self.OcrFix(i)
            res.update(x)
        return res

    def FindPics_TM(self, des):
        '''
        找多图
        :param des:
        :return:
        '''
        img = pyautogui.screenshot()
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = {}
        for key, value in des.items():
            template = cv2.imread(value[4])
            base = img[value[1]:value[3], value[0]:value[2]]
            th, tw = template.shape[:2]
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(template, 20, 255, cv2.THRESH_BINARY)
            rv = cv2.matchTemplate(base, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(rv)
            if maxVal < value[5]:
                pass
            else:
                res = (maxLoc[0] + tw / 2 + value[0], maxLoc[1] + th / 2 + value[1])
                break
        return res


a = TuSe()
time.sleep(3)
node = {
    'sb1': [6, 25, 1020, 849, './image/image0.bmp', 0.9],
    'sb2': [6, 25, 1020, 849, './image/image1.bmp', 0.9],
    'sb3': [6, 25, 1020, 849, './image/image10.bmp', 0.9],
    'sb4': [6, 25, 1020, 849, './image/image11.bmp', 0.9],
    'sb5': [6, 25, 1020, 849, './image/image13.bmp', 0.9],
    'sb6': [6, 25, 1020, 849, './image/image15.bmp', 0.9],
    'sb7': [6, 25, 1020, 849, './image/image17.bmp', 0.9],
    'sb8': [6, 25, 1020, 849, './image/image19.bmp', 0.9],
    'sb9': [6, 25, 1020, 849, './image/image25.bmp', 0.9],
    'sb10': [6, 25, 1020, 849, './image/image27.bmp', 0.9],
    'sb11': [6, 25, 1020, 849, './image/image29.bmp', 0.9]
}

while 1:
    c = a.FindPics_TM(node)
    print(c)
    img = a.GetCapture(6, 25, 1020, 849)
    height, width, _ = img.shape
    center = (int(c[0]) - 6, int(c[1]) - 25)  # 圆心坐标
    radius = 10  # 半径
    color = (0, 0, 255)  # 颜色，这里是红色（BGR格式）
    thickness = 2  # 线条粗细，这里是2个像素
    cv2.circle(img, center, radius, color, thickness)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
