"""
-*- coding: utf-8 -*-
__author__:Steve Zhang
2023/3/14 16:24
"""
import os
import time
import base64
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image

from net import Net
from option import Options
import utils
from utils import StyleLoader


def style_select(index):
    style_path = ''
    if index == 'candy':
        style_path = 'images\\21styles\\candy.jpg'
    elif index == 'composition':
        style_path = 'images\\21styles\\composition_vii.jpg'
    elif index == 'sphere':
        style_path = 'images\\21styles\\escher_sphere.jpg'
    elif index == 'feathers':
        style_path = 'images\\21styles\\feathers.jpg'
    elif index == 'frida':
        style_path = 'images\\21styles\\frida_kahlo.jpg'
    elif index == 'muse':
        style_path = 'images\\21styles\\la_muse.jpg'
    elif index == 'mosaic':
        style_path = 'images\\21styles\\mosaic.jpg'
    elif index == 'massimo':
        style_path = 'images\\21styles\\mosaic_ducks_massimo.jpg'
    elif index == 'pencil':
        style_path = 'images\\21styles\\pencil.jpg'
    elif index == 'picasso':
        style_path = 'images\\21styles\\picasso_selfport1907.jpg'
    elif index == 'rain':
        style_path = 'images\\21styles\\rain_princess.jpg'
    elif index == 'robert':
        style_path = 'images\\21styles\\Robert_Delaunay,_1906,_Portrait.jpg'
    elif index == 'nude':
        style_path = 'images\\21styles\\seated-nude.jpg'
    elif index == 'shipwreck':
        style_path = 'images\\21styles\\shipwreck.jpg'
    elif index == 'starry':
        style_path = 'images\\21styles\\starry_night.jpg'
    elif index == 'stars2':
        style_path = 'images\\21styles\\stars2.jpg'
    elif index == 'strip':
        style_path = 'images\\21styles\\strip.jpg'
    elif index == 'scream':
        style_path = 'images\\21styles\\the_scream.jpg'
    elif index == 'udnie':
        style_path = 'images\\21styles\\udnie.jpg'
    elif index == 'wave':
        style_path = 'images\\21styles\\wave.jpg'
    elif index == 'woman':
        style_path = 'images\\21styles\\woman-with-hat-matisse.jpg'

    return style_path


def camera_demo(style_idx, mirror=False):
    style_path = style_select(style_idx)
    checkpoint_path = 'models/21styles.model'
    style_model = Net(ngf=128)
    # model_dict = torch.load(args.model)
    model_dict = torch.load(checkpoint_path)
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)
    style_model.eval()
    style_loader = StyleLoader('images/21styles/', 512)
    style_model.cuda()

    # Define the codec and create VideoWriter object
    height = 480
    width = int(4.0 / 3 * 480)
    swidth = int(width / 4)
    sheight = int(height / 4)

    # if args.record:
    #     fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    #     out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (2 * width, height))

    cam = cv2.VideoCapture(0)
    cam.set(3, width)
    cam.set(4, height)
    key = 0
    idx = 0
    while True:
        # read frame
        idx += 1
        ret_val, img = cam.read()
        # img = cv2.imread('./images/input_realtime/origin_frame.jpeg')
        # print(img.shape)
        if mirror:
            img = cv2.flip(img, 1)
        cimg = img.copy()
        img = np.array(img).transpose(2, 0, 1)

        time1 = time.time()
        # selecting a style
        style_v = style_loader.getStyle(style_path)
        style_v = Variable(style_v.data)
        style_model.setTarget(style_v)

        # # changing style
        # if idx % 20 == 1:
        #     style_v = style_loader.get(int(idx / 20))
        #     style_v = Variable(style_v.data)
        #     style_model.setTarget(style_v)

        img = torch.from_numpy(img).unsqueeze(0).float()
        img = img.cuda()

        img = Variable(img)
        # print(img.shape)
        img = style_model(img)

        simg = style_v.cpu().data[0].numpy()
        img = img.cpu().clamp(0, 255).data[0].numpy()

        simg = np.squeeze(simg)
        img = img.transpose(1, 2, 0).astype('uint8')
        cv2.imwrite('stylized.jpg', img)

        time2 = time.time()
        print("load model: ", int(round(time2 * 1000)) - int(round(time1 * 1000)))
        simg = simg.transpose(1, 2, 0).astype('uint8')

        # display
        simg = cv2.resize(simg, (swidth, sheight), interpolation=cv2.INTER_CUBIC)
        cimg[0:sheight, 0:swidth, :] = simg
        img = np.concatenate((cimg, img), axis=1)
        cv2.imshow('MSG Demo', img)
        # cv2.imwrite('stylized/%i.jpg'%idx,img)
        key = cv2.waitKey(1)
        # if args.record:
        #     out.write(img)
        if key == 27:
            break
    cam.release()
    # if args.record:
    #     out.release()
    cv2.destroyAllWindows()


def evaluate_single_frame(content, style_idx):
    t0 = time.time()

    checkpoint_path = 'models/21styles.model'
    style = style_select(style_idx)
    content_image = utils.tensor_load_rgbimage(content, size=512, keep_asp=True)
    # print(content_image.shape)
    content_image = content_image.unsqueeze(0)
    # print(content_image.shape)
    style = utils.tensor_load_rgbimage(style, size=512)
    style = style.unsqueeze(0)
    style = utils.preprocess_batch(style)

    t1 = time.time()  # load content and style
    print("load content and style: ", int(round(t1 * 1000)) - int(round(t0 * 1000)))

    style_model = Net(ngf=128)
    model_dict = torch.load(checkpoint_path)
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)
    style_model.cuda()

    t2 = time.time()
    print("load model: ", int(round(t2 * 1000)) - int(round(t1 * 1000)))

    with torch.no_grad():
        content_image = content_image.cuda()
        style = style.cuda()
        style_v = Variable(style)
        content_image = Variable(utils.preprocess_batch(content_image))
        style_model.setTarget(style_v)
        output = style_model(content_image)

    t3 = time.time()
    print("inference: ", int(round(t3 * 1000)) - int(round(t2 * 1000)))

    # output = utils.color_match(output, style_v)
    utils.tensor_save_bgrimage(output.data[0], "./images/output_frame/output_frame.jpg", 1)

    t4 = time.time()
    print("save: ", int(round(t4 * 1000)) - int(round(t3 * 1000)))


def evaluate_video(style_idx, style_loader, style_model):
    style_path = style_select(style_idx)
    style_v = style_loader.getStyle(style_path)
    style_v = Variable(style_v.data)
    style_model.setTarget(style_v)

    input_video_path = "./images/input_video/origin_video1.mp4"
    output_video_path = "./images/output_video/processed_video.mp4"
    capture = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (240, 320), True)
    if capture.isOpened():
        while True:
            ret, input_frame = capture.read()
            if not ret:
                break
            input_frame = np.array(input_frame).transpose(2, 0, 1)
            input_frame = torch.from_numpy(input_frame).unsqueeze(0).float()
            input_frame = input_frame.cuda()
            input_frame = Variable(input_frame)
            # print(input_frame.shape)
            with torch.no_grad():
                input_frame = style_model(input_frame)
            input_frame = input_frame.cpu().clamp(0, 255).data[0].numpy()
            input_frame = input_frame.transpose(1, 2, 0).astype('uint8')
            writer.write(input_frame)
    else:
        print("error occurs from video opening")
    writer.release()


def decoder(code):
    with open('./images/input_frame/image.jpg', 'wb') as file:
        img = base64.b64decode(code)
        file.write(img)
        file.close()


def encoder(img_path):
    with open(img_path, 'rb') as img:
        img_data = img.read()
        base64_data = base64.b64encode(img_data)
        return base64_data.decode()


def video_encoder(video_path):
    with open(video_path, 'rb') as video_file:
        data = video_file.read()
        base64_code = base64.b64encode(data)
        return base64_code.decode()


def img_preprocess(path):
    image = Image.open(path)
    image = image.transpose(Image.ROTATE_270)
    image = image.resize((120, 160), Image.ANTIALIAS)
    image.save('./images/input_realtime/origin_frame.jpeg', quality=10)


def video_preprocess(path):
    video_size = round(os.path.getsize(path)/1024/1024, 1)
    print(video_size)
    if video_size > 1.0:
        changed_video_path = "./images/input_video/origin_video1.mp4"
        capture = cv2.VideoCapture(path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(changed_video_path, fourcc, 20.0, (240, 320), True)
        if capture.isOpened():
            while True:
                ret, input_frame = capture.read()
                try:
                    input_frame = cv2.resize(input_frame, (240, 320))
                except Exception as e:
                    break

                # if not ret:
                #     break
                writer.write(input_frame)
        else:
            print("error occurs from video opening")
        writer.release()


def frame_fast_evaluate(content, style_idx, style_model, style_loader):
    t1 = time.time()
    style_path = style_select(style_idx)

    with torch.no_grad():
        style_v = style_loader.getStyle(style_path)
        style_v = Variable(style_v.data)
        style_model.setTarget(style_v)

    img = cv2.imread(content, 1)
    with torch.no_grad():
        img = np.array(img).transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).float()
        img = img.cuda()
        img = Variable(img)
        img = style_model(img)
        img = img.cpu().clamp(0, 255).data[0].numpy()
        img = img.transpose(1, 2, 0).astype('uint8')
    cv2.imwrite('./images/output_realtime/output_frame.jpeg', img)
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    t2 = time.time()
    print("time: ", int(round(t2 * 1000)) - int(round(t1 * 1000)))


if __name__ == '__main__':
    # # real time model inference
    # camera_demo(style_idx='shipwreck', mirror=True)

    # # video inference
    # video_path = "./images/input_video/origin_video.mp4"
    # video_preprocess(video_path)
    # evaluate_video("candy")

    # image inference
    content_path = "./images/test/test.jpg"
    style_name = "pencil"
    evaluate_single_frame(content_path, style_name)





