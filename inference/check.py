import os
import json
import glob
import argparse
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="./dataset/val",
                        help="Input image or folder")
    parser.add_argument("-o", "--output", type=str, default="./results/test_result",
                        help="Output image folder")
    parser.add_argument("-r", "--report", type=str, default="./results/result.json",
                        help="report model runtime to json file")
    args = parser.parse_args()

    input_images = sorted(glob.glob(os.path.join(args.input, "*")))
    output_images = sorted(glob.glob(os.path.join(args.output, "*")))

    with open(args.report, "r") as file:
        result = json.load(file)["B"][0]
    results = sorted(result["images"], key=lambda x: x['img_name'])

    try:
        for idx, out_img_path in enumerate(output_images):
            out_img_name, out_img_extension = os.path.splitext(
                os.path.basename(out_img_path))
            in_img_path = input_images[idx]
            in_img_name, in_img_extension = os.path.splitext(
                os.path.basename(in_img_path))

            # check 输入输出文件名是否一致
            assert out_img_extension == in_img_extension, "image extension not same"
            assert out_img_name == in_img_name, "image name not same"

            out_h, out_w, _ = cv2.imread(out_img_path).shape
            in_h, in_w, _ = cv2.imread(in_img_path).shape

            # check 输出图像尺寸是输入图像尺寸的 4 倍
            assert out_h == 4 * in_h, "height not 4x"
            assert out_w == 4 * in_w, "width not 4x"

            # check runtime 和 niqe 是否等于 0 或者小于 1 ms (0.001)
            runtime, niqe = float(results[idx]["runtime"]), float(
                results[idx]["niqe"])
            assert runtime > 0.001, "runtime <= 0.001 (1ms)"
            assert niqe > 0, "niqe < 0"

            print(f"image {idx + 1} is valid.")
    except AssertionError as e:
        print("Assertion Error! ", e)
    else:
        print("Congratulations, the result is legal !!")
