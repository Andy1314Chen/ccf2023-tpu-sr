import warnings
import time
import json
import math
import argparse
import glob
import os
from fix import *
from npuengine import EngineOV
from PIL import Image
import cv2
from metrics.niqe import calculate_niqe
import copy
import multiprocessing
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import gc
import numpy as np
import sys
sys.path.append(".")
sys.path.append("..")


global tile_runtime, tile_extension_dict


class CustomDatasetv1(Dataset):
    def __init__(self,
                 filenames,
                 batch_size=8,
                 tile_size=(64, 64),
                 tile_pad=4,
                 upscale=4):
        super(CustomDatasetv1).__init__()

        if type(filenames) is list:
            self.filenames = sorted(filenames)
        elif os.path.isfile(filenames):
            self.filenames = [filenames]
        else:
            self.filenames = sorted(glob.glob(os.path.join(filenames, "*")))
        self.batch_size = batch_size
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.upscale = upscale
        self.input_tile_list = []
        self.extension_dict = dict()
        self.input_size_list = []

        for idx, path in enumerate(self.filenames):
            img_name, extension = os.path.splitext(os.path.basename(path))
            img = cv2.imread(path)
            self.tile(img)
            self.extension_dict[int(img_name)] = extension

        self.data_size = len(self.input_tile_list)

        # fill batch
        if (self.data_size % self.batch_size) != 0:
            fill_size = (self.batch_size - (self.data_size % self.batch_size))
            fill_tile_list = [
                np.zeros((3, self.tile_size[1] + 2 * self.tile_pad, self.tile_size[0] + 2 * self.tile_pad))] * fill_size
            self.input_tile_list += fill_tile_list

        # update
        self.data_size = len(self.input_tile_list)

    def tile(self, image):
        height, width, channel = image.shape

        tiles_x = math.ceil(width / self.tile_size[0])
        tiles_y = math.ceil(height / self.tile_size[1])

        image = np.array(image).astype(np.float32)
        image = image / 255.0

        # (height, width, channel) -> (height', width', channel)
        image = np.pad(image,
                       ((self.tile_pad, tiles_y * self.tile_size[1] - height + self.tile_pad),
                        (self.tile_pad, tiles_x * self.tile_size[0] - width + self.tile_pad), (0, 0)),
                       mode='constant',
                       constant_values=0)
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        # (height, width, offset)
        self.input_size_list.append((height, width, len(self.input_tile_list)))

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size[0] + self.tile_pad
                ofs_y = y * self.tile_size[1] + self.tile_pad

                input_start_x = ofs_x - self.tile_pad
                input_end_x = ofs_x + self.tile_size[0] + self.tile_pad
                input_start_y = ofs_y - self.tile_pad
                input_end_y = ofs_y + self.tile_size[1] + self.tile_pad

                input_tile = image[:, input_start_y:input_end_y,
                                   input_start_x:input_end_x]

                # NCHW
                # input_tile = input_tile[np.newaxis, :, :, :]
                self.input_tile_list += [input_tile]

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        tile_img = self.input_tile_list[index]

        return tile_img


class UpscaleModel:

    def __init__(self, model=None, model_size=(200, 200), upscale_rate=4, tile_size=(196, 196), padding=4, device_id=0):
        self.tile_size = tile_size
        self.padding = padding
        self.upscale_rate = upscale_rate
        if model is None:
            print("use default upscaler model")
            model = "./models/other/resrgan4x.bmodel"
        # 导入bmodel
        self.model = EngineOV(model, device_id=device_id)
        self.model_size = model_size

    def calc_tile_position(self, width, height, col, row):
        # generate mask
        tile_left = col * self.tile_size[0]
        tile_top = row * self.tile_size[1]
        tile_right = (col + 1) * self.tile_size[0] + self.padding
        tile_bottom = (row + 1) * self.tile_size[1] + self.padding
        if tile_right > height:
            tile_right = height
            tile_left = height - self.tile_size[0] - self.padding * 1
        if tile_bottom > width:
            tile_bottom = width
            tile_top = width - self.tile_size[1] - self.padding * 1

        return tile_top, tile_left, tile_bottom, tile_right

    def calc_upscale_tile_position(self, tile_left, tile_top, tile_right, tile_bottom):
        return int(tile_left * self.upscale_rate), int(tile_top * self.upscale_rate), int(
            tile_right * self.upscale_rate), int(tile_bottom * self.upscale_rate)

    def modelprocess(self, tile):
        ntile = tile.resize(self.model_size)
        # preprocess
        ntile = np.array(ntile).astype(np.float32)
        ntile = ntile / 255
        # HWC -> CHW
        ntile = np.transpose(ntile, (2, 0, 1))
        ntile = ntile[np.newaxis, :, :, :]

        res = self.model([ntile])[0]
        # extract padding
        res = res[0]
        # CHW -> HWC
        res = np.transpose(res, (1, 2, 0))
        res = res * 255
        res[res > 255] = 255
        res[res < 0] = 0
        res = res.astype(np.uint8)
        res = Image.fromarray(res)
        res = res.resize(self.target_tile_size)
        return res

    def predict(self, image, upscale_ratio=4.0):
        height, width, channel = image.shape
        output_height = height * upscale_ratio
        output_width = width * upscale_ratio

        # 向上取整
        tiles_x = math.ceil(width / self.tile_size[0])
        tiles_y = math.ceil(height / self.tile_size[1])

        # image preprocess
        image = np.array(image).astype(np.float32)
        image = image / 255.0

        # (height, width, channel) -> (height', width', channel)
        image = np.pad(image,
                       ((0, tiles_y * self.tile_size[1] - height),
                        (0, tiles_x * self.tile_size[0] - width), (0, 0)),
                       'constant')

        # HWC
        output_shape = (tiles_y * self.tile_size[1] * self.upscale_rate,
                        tiles_x * self.tile_size[0] * self.upscale_rate, channel)

        # start with block image
        output = np.zeros(output_shape)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size[0]
                ofs_y = y * self.tile_size[1]

                input_start_x = ofs_x
                input_end_x = ofs_x + self.tile_size[0]
                input_start_y = ofs_y
                input_end_y = ofs_y + self.tile_size[1]

                input_tile = image[input_start_y:input_end_y,
                                   input_start_x:input_end_x, :]

                # HWC -> CHW
                input_tile = np.transpose(input_tile, (2, 0, 1))
                # NCHW
                input_tile = input_tile[np.newaxis, :, :, :]

                # inference
                output_tile = self.model([input_tile])[0]
                output_tile = output_tile[0]
                # CHW -> HWC
                output_tile = np.transpose(output_tile, (1, 2, 0))
                output_tile = output_tile * 255
                output_tile[output_tile > 255] = 255
                output_tile[output_tile < 0] = 0
                output_tile = output_tile.astype(np.uint8)

                # output tile area on total image
                output_start_x = input_start_x * self.upscale_rate
                output_end_x = input_end_x * self.upscale_rate
                output_start_y = input_start_y * self.upscale_rate
                output_end_y = input_end_y * self.upscale_rate

                # put tile into output image
                output[output_start_y: output_end_y,
                       output_start_x: output_end_x, :] = output_tile

        output = output[:output_height, :output_width, :]

        return output

    def batch_predict(self, image, upscale_ratio=4.0, batch_size=2):
        height, width, channel = image.shape
        output_height = height * upscale_ratio
        output_width = width * upscale_ratio

        # 向上取整
        tiles_x = math.ceil(width / self.tile_size[0])
        tiles_y = math.ceil(height / self.tile_size[1])

        # image preprocess
        image = np.array(image).astype(np.float32)
        image = image / 255.0

        # (height, width, channel) -> (height', width', channel)
        image = np.pad(image,
                       ((0, tiles_y * self.tile_size[1] - height),
                        (0, tiles_x * self.tile_size[0] - width), (0, 0)),
                       'constant')
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        # CHW
        output_shape = (channel, tiles_y * self.tile_size[1] * self.upscale_rate,
                        tiles_x * self.tile_size[0] * self.upscale_rate)

        # start with block image
        output = np.zeros(output_shape)
        input_tile_list = []

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size[0]
                ofs_y = y * self.tile_size[1]

                input_start_x = ofs_x
                input_end_x = ofs_x + self.tile_size[0]
                input_start_y = ofs_y
                input_end_y = ofs_y + self.tile_size[1]

                input_tile = image[:, input_start_y:input_end_y,
                                   input_start_x:input_end_x]

                # NCHW
                input_tile = input_tile[np.newaxis, :, :, :]
                input_tile_list += [input_tile]

        # 结果应该有 tiles_y * tiles_x 个
        output_tile_list = []
        is_filled = 0
        for _idx in range(0, len(input_tile_list), batch_size):
            inputs = input_tile_list[_idx: _idx + batch_size]

            # 要保证 tiles_y * tiles_x 能被 batch_size 整除
            if len(inputs) != batch_size:
                fill_tiles = [
                    np.zeros((1, channel, self.tile_size[1], self.tile_size[0]))] * (batch_size - len(inputs))
                is_filled = batch_size - len(inputs)
                inputs += fill_tiles
            inputs = np.concatenate(inputs, axis=0)
            # inference (n, c, h, w)
            output_tiles = self.model([inputs])[0]
            # CHW
            output_tile_list += [output_tiles[i]
                                 for i in range(batch_size - is_filled)]

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * self.tile_size[0]
                ofs_y = y * self.tile_size[1]

                output_start_x = ofs_x * self.upscale_rate
                output_end_x = (ofs_x + self.tile_size[0]) * self.upscale_rate
                output_start_y = ofs_y * self.upscale_rate
                output_end_y = (ofs_y + self.tile_size[1]) * self.upscale_rate

                # CHW
                output_tile = output_tile_list[y * tiles_x + x]
                # output tile area on total image
                output[:, output_start_y: output_end_y,
                       output_start_x: output_end_x] = output_tile

        output = output[:, :output_height, :output_width]
        # CHW -> HWC
        output = np.transpose(output, (1, 2, 0))
        output = output * 255
        output[output > 255] = 255
        output[output < 0] = 0
        output = output.astype(np.uint8)

        return output

    def extract_and_enhance_tiles(self, image, upscale_ratio=2.0):
        if image.mode != "RGB":
            image = image.convert("RGB")
        # 获取图像的宽度和高度
        width, height = image.size
        self.upscale_rate = upscale_ratio
        self.target_tile_size = (int((self.tile_size[0] + self.padding * 1) * upscale_ratio),
                                 int((self.tile_size[1] + self.padding * 1) * upscale_ratio))
        target_width, target_height = int(
            width * upscale_ratio), int(height * upscale_ratio)
        # 计算瓦片的列数和行数
        num_cols = math.ceil((width - self.padding) / self.tile_size[0])
        num_rows = math.ceil((height - self.padding) / self.tile_size[1])

        # 遍历每个瓦片的行和列索引
        img_tiles = []
        for row in range(num_rows):
            img_h_tiles = []
            for col in range(num_cols):
                # 计算瓦片的左上角和右下角坐标
                tile_left, tile_top, tile_right, tile_bottom = self.calc_tile_position(
                    width, height, row, col)
                # 裁剪瓦片
                tile = image.crop(
                    (tile_left, tile_top, tile_right, tile_bottom))
                # 使用超分辨率模型放大瓦片
                upscaled_tile = self.modelprocess(tile)
                # 将放大后的瓦片粘贴到输出图像上
                # overlap
                ntile = np.array(upscaled_tile).astype(np.float32)
                # HWC -> CHW
                ntile = np.transpose(ntile, (2, 0, 1))
                img_h_tiles.append(ntile)

            img_tiles.append(img_h_tiles)
        res = imgFusion(img_list=img_tiles, overlap=int(self.padding * upscale_ratio), res_w=target_width,
                        res_h=target_height)
        res = Image.fromarray(np.transpose(res, (1, 2, 0)).astype(np.uint8))
        return res


def postproprecess(image):
    # clip output_tile
    if args.tile_pad != 0:
        image = image[:,
                      args.tile_pad * args.upscale_ratio: -args.tile_pad * args.upscale_ratio,
                      args.tile_pad * args.upscale_ratio: -args.tile_pad * args.upscale_ratio]

    image = image * 255
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype(np.uint8)
    return image


def multi_calc_niqe(idx, input_size, output_tile_list):
    idx += 1
    h, w, ofs = input_size

    output_height = h * args.upscale_ratio
    output_width = w * args.upscale_ratio

    tiles_x = math.ceil(w / args.tile_x)
    tiles_y = math.ceil(h / args.tile_y)

    # CHW
    output_shape = (3, tiles_y * args.tile_y * args.upscale_ratio,
                    tiles_x * args.tile_x * args.upscale_ratio)
    output = np.zeros(output_shape, dtype=np.uint8)

    for y in range(tiles_y):
        for x in range(tiles_x):
            ofs_x = x * args.tile_x
            ofs_y = y * args.tile_y

            output_start_x = ofs_x * args.upscale_ratio
            output_end_x = (ofs_x + args.tile_x) * args.upscale_ratio
            output_start_y = ofs_y * args.upscale_ratio
            output_end_y = (ofs_y + args.tile_y) * args.upscale_ratio

            # CHW
            output_tile = output_tile_list[y * tiles_x + x]
            # output tile area on total image
            output[:, output_start_y: output_end_y,
                   output_start_x: output_end_x] = output_tile

    output = output[:, :output_height, :output_width]
    # CHW -> HWC
    output = np.transpose(output, (1, 2, 0))

    output_path = os.path.join(
        args.output, str(idx).zfill(4) + tile_extension_dict[idx])
    try:
        cv2.imwrite(output_path, output)
    except Exception as e:
        print("output path: ", output_path)
        print("imwrite error: ", e)
    runtime = format(sum(tile_runtime[ofs: ofs + tiles_x * tiles_y]), '.4f')

    # 计算niqe
    # output = cv2.imread(output_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        niqe_output = calculate_niqe(
            output, 0, input_order='HWC', convert_to='y')
    niqe_output = format(niqe_output, '.4f')

    return {"img_name": str(idx).zfill(4),
            "runtime": runtime,
            "niqe": niqe_output}


def thread_calc_niqe(idx_start, input_size_list):
    global output_tile_list, tile_extension_dict, tile_runtime

    result = []
    for idx, input_size in enumerate(input_size_list):
        idx += (idx_start + 1)
        h, w, ofs = input_size

        output_height = h * args.upscale_ratio
        output_width = w * args.upscale_ratio

        tiles_x = math.ceil(w / args.tile_x)
        tiles_y = math.ceil(h / args.tile_y)

        # CHW
        output_shape = (3, tiles_y * args.tile_y * args.upscale_ratio,
                        tiles_x * args.tile_x * args.upscale_ratio)
        output = np.zeros(output_shape, dtype=np.uint8)

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * args.tile_x
                ofs_y = y * args.tile_y

                output_start_x = ofs_x * args.upscale_ratio
                output_end_x = (ofs_x + args.tile_x) * args.upscale_ratio
                output_start_y = ofs_y * args.upscale_ratio
                output_end_y = (ofs_y + args.tile_y) * args.upscale_ratio

                # CHW
                output_tile = output_tile_list[ofs + y * tiles_x + x]
                # output tile area on total image
                output[:, output_start_y: output_end_y,
                       output_start_x: output_end_x] = output_tile
        output = output[:, :output_height, :output_width]
        # CHW -> HWC
        output = np.transpose(output, (1, 2, 0))

        output_path = os.path.join(
            args.output, str(idx).zfill(4) + tile_extension_dict[idx])
        try:
            cv2.imwrite(output_path, output)
        except Exception as e:
            print("output path: ", output_path)
            print("imwrite error: ", e)
        runtime = format(
            sum(tile_runtime[ofs: ofs + tiles_x * tiles_y]), '.4f')

        # 计算niqe
        # output = cv2.imread(output_path)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            niqe_output = calculate_niqe(
                output, 0, input_order='HWC', convert_to='y')
        niqe_output = format(niqe_output, '.4f')

        result.append({"img_name": str(idx).zfill(4),
                       "runtime": runtime,
                       "niqe": niqe_output})
    return result


def multipro_calc_niqe(idx_start, start_ofs, input_size_list, output_tile_name):
    global tile_extension_dict, tile_runtime

    result = []
    _output_tile_list = np.load(output_tile_name)
    for idx, input_size in enumerate(input_size_list):
        h, w, ofs = input_size
        idx += (idx_start + 1)

        output_height = h * args.upscale_ratio
        output_width = w * args.upscale_ratio

        tiles_x = math.ceil(w / args.tile_x)
        tiles_y = math.ceil(h / args.tile_y)

        # CHW
        output_shape = (3, tiles_y * args.tile_y * args.upscale_ratio,
                        tiles_x * args.tile_x * args.upscale_ratio)
        output = np.zeros(output_shape, dtype=np.uint8)

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * args.tile_x
                ofs_y = y * args.tile_y

                output_start_x = ofs_x * args.upscale_ratio
                output_end_x = (ofs_x + args.tile_x) * args.upscale_ratio
                output_start_y = ofs_y * args.upscale_ratio
                output_end_y = (ofs_y + args.tile_y) * args.upscale_ratio

                # CHW
                output_tile = _output_tile_list[ofs -
                                                start_ofs + y * tiles_x + x]
                # output tile area on total image
                output[:, output_start_y: output_end_y,
                       output_start_x: output_end_x] = output_tile

        output = output[:, :output_height, :output_width]
        # CHW -> HWC
        output = np.transpose(output, (1, 2, 0))

        output_path = os.path.join(
            args.output, str(idx).zfill(4) + tile_extension_dict[idx])
        try:
            cv2.imwrite(output_path, output)
        except Exception as e:
            print("output path: ", output_path)
            print("imwrite error: ", e)

        runtime = format(
            sum(tile_runtime[ofs: ofs + tiles_x * tiles_y]), '.4f')
        # 计算niqe
        # output = cv2.imread(output_path)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            niqe_output = calculate_niqe(
                output, 0, input_order='HWC', convert_to='y')
        niqe_output = format(niqe_output, '.4f')

        result.append({"img_name": str(idx).zfill(4),
                       "runtime": runtime,
                       "niqe": niqe_output})

    # delete output_tile_i.npy file
    os.remove(output_tile_name)

    return result


def main():
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, "*")))

    # set models
    model = args.model_path
    upmodel = UpscaleModel(model=model, model_size=(
        200, 200), upscale_rate=4, tile_size=(args.tile_x, args.tile_y), padding=20)

    global tile_extension_dict, tile_runtime

    start_all = time.time()
    result, runtime, niqe = [], [], []
    tile_runtime = []

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # paths = paths[:20]
    if args.predict_method == "dataset":

        input_paths = []
        dataset_size = min(len(paths), 50)
        for ids in range(int(len(paths)/dataset_size)):
            if ((ids + 1) * dataset_size) > len(paths):
                input_paths = paths[ids * dataset_size:]
            else:
                input_paths = paths[ids *
                                    dataset_size: (ids + 1) * dataset_size]

            print("input paths len: ", len(input_paths))

            dataset = CustomDatasetv1(filenames=input_paths,
                                      batch_size=args.batch_size,
                                      tile_size=(args.tile_x, args.tile_y),
                                      upscale=4,
                                      tile_pad=args.tile_pad)
            print("dataset size: ", dataset.data_size)
            dataloader = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=2)

            output_tile_list = []

            for batch in dataloader:
                tile_img = batch
                tile_img = tile_img.detach().numpy()
                # print("tile img shape: ", tile_img.shape)
                start = time.time()
                # shape:  (8, 3, 64, 64)
                # inference (n, c, h, w)
                output_tiles = upmodel.model([tile_img])[0]
                end = time.time() - start

                # CHW
                output_tile_list += [postproprecess(output_tiles[i])
                                     for i in range(args.batch_size)]
                # end = time.time() - start
                tile_runtime += [round(end / args.batch_size, 5)
                                 ] * args.batch_size

            input_size_list = copy.copy(dataset.input_size_list)
            tile_extension_dict = copy.copy(dataset.extension_dict)

            # gc
            del dataset, dataloader
            gc.collect()

            # save output_tile_list
            file_size = 4
            size_per_file = int(len(input_size_list)/file_size)
            for i in range(file_size):
                start_ofs = input_size_list[i * size_per_file][-1]

                if i != (file_size - 1):
                    end_ofs = input_size_list[(i + 1) * size_per_file][-1]
                    print(
                        f"index: {i}, start ofs: {start_ofs}, end ofs: {end_ofs}")
                    np.save(os.path.join(args.output, f"output_tile_{i}.npy"),
                            output_tile_list[start_ofs: end_ofs])
                else:
                    print(
                        f"index: {i}, start ofs: {start_ofs}, end ofs: {end_ofs}")
                    np.save(os.path.join(args.output,
                            f"output_tile_{i}.npy"), output_tile_list[start_ofs:])

            del output_tile_list
            gc.collect()

            pool = multiprocessing.Pool(processes=file_size)
            _result = []
            for i in range(file_size):
                print("multipro idx: ", i)
                start_ofs = input_size_list[i * size_per_file][-1]
                if i != (file_size - 1):
                    _input_size_list = input_size_list[i *
                                                       size_per_file: (i + 1) * size_per_file]
                else:
                    _input_size_list = input_size_list[i * size_per_file:]

                res = pool.apply_async(multipro_calc_niqe, (ids * dataset_size + i * size_per_file,
                                                            start_ofs, _input_size_list, os.path.join(args.output, f"output_tile_{i}.npy")))
                _result.append(res)

            pool.close()
            pool.join()

            for res in _result:
                result += res.get()

    else:
        for idx, path in enumerate(paths[:10]):
            img_name, extension = os.path.splitext(os.path.basename(path))
            output_path = os.path.join(args.output, img_name + extension)

            if args.predict_method == "new":
                img = cv2.imread(path)
                start = time.time()
                res = upmodel.predict(img, upscale_ratio=4)
                end = format((time.time() - start), '.4f')
                runtime.append(end)

                cv2.imwrite(output_path, res)

            elif args.predict_method == "batch":

                img = cv2.imread(path)
                start = time.time()
                res = upmodel.batch_predict(
                    img, upscale_ratio=4, batch_size=args.batch_size)
                end = format((time.time() - start), '.4f')
                runtime.append(end)

                cv2.imwrite(output_path, res)

            else:
                img = Image.open(path)
                print("Testing", idx, img_name)

                start = time.time()
                res = upmodel.extract_and_enhance_tiles(img, upscale_ratio=4.0)
                end = format((time.time() - start), '.4f')
                runtime.append(end)

                res.save(output_path)

            # 计算niqe
            output = cv2.imread(output_path)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                niqe_output = calculate_niqe(
                    output, 0, input_order='HWC', convert_to='y')
            niqe_output = format(niqe_output, '.4f')
            niqe.append(niqe_output)

            result.append(
                {"img_name": img_name, "runtime": end, "niqe": niqe_output})

    model_size = os.path.getsize(model)

    runtime, niqe = [], []
    for _res in result:
        runtime.append(float(_res["runtime"]))
        niqe.append(float(_res["niqe"]))

    runtime_avg = np.mean(np.array(runtime, dtype=float))
    niqe_avg = np.mean(np.array(niqe, dtype=float))

    end_all = time.time()
    time_all = end_all - start_all
    print('time_all:', time_all)
    params = {"B": [{"model_size": model_size, "time_all": time_all, "runtime_avg": format(runtime_avg, '.4f'),
                     "niqe_avg": format(niqe_avg, '.4f'), "images": result}]}
    print("params: ", params)

    output_fn = f'{args.report}'
    with open(output_fn, 'w') as f:
        json.dump(params, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, default="./models/out.bmodel",
                        help="Model names")
    parser.add_argument("-i", "--input", type=str, default="./dataset/val",
                        help="Input image or folder")
    parser.add_argument("-o", "--output", type=str, default="./results/test_result",
                        help="Output image folder")
    parser.add_argument("-r", "--report", type=str, default="./results/result.json",
                        help="report model runtime to json file")
    parser.add_argument("-n", "--predict_method", type=str, default="dataset",
                        help="predict method (old/new/batch/dataset)")
    parser.add_argument("-b", "--batch_size", type=int, default=400,
                        help="inference batch size")
    parser.add_argument("-x", "--tile_x", type=int, default=10,
                        help="x tile size (width)")
    parser.add_argument("-y", "--tile_y", type=int, default=10,
                        help="y tile size (height)")
    parser.add_argument("-t", "--tile_pad", type=int, default=0,
                        help="tile padding")
    parser.add_argument("-u", "--upscale_ratio", type=int, default=4,
                        help="upscale ratio")
    args = parser.parse_args()
    main()
