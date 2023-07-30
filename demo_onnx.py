import argparse
import math
from threading import Thread

import onnxruntime as ort
from torch.backends import cudnn
from torchvision import transforms
from tqdm import tqdm

from utils.utils import *


def detect(opt):
    if not opt.source.isnumeric():
        opt.save_dir = Path(increment_path(Path(opt.save_dir) / opt.name, exist_ok=False))  # increment run
        opt.save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    inf_time = AverageMeter()
    nms_time = AverageMeter()

    device = select_device(device=opt.device)
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False

    # Load model
    ort.set_default_logger_severity(4)
    ort_session = ort.InferenceSession(opt.weights, None, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f"Load {opt.weights} done!")

    outputs_info = ort_session.get_outputs()
    inputs_info = ort_session.get_inputs()

    for ii in inputs_info:
        print("Input: ", ii)
    for oo in outputs_info:
        print("Output: ", oo)

    print("num outputs: ", len(outputs_info))

    # Set Dataloader
    vid_path, vid_writer = None, None
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size, batch=opt.batch_size)

    t0 = time.time()
    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        img = img.to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        img = img.cpu().numpy()

        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out, ll_seg_out = ort_session.run(
            ['detect_output', 'driving_area_segmentation', 'lane_line_segmentation'], input_feed={"images": img}
        )
        t2 = time_synchronized()

        det_out = torch.from_numpy(det_out).float()
        da_seg_out = torch.from_numpy(da_seg_out).float()
        ll_seg_out = torch.from_numpy(ll_seg_out).float()

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(det_out, opt.conf_thres, opt.iou_thres)
        t4 = time_synchronized()

        height, width = img.shape[2], img.shape[3]

        inf_time.update((t2 - t1) / img.shape[0], img.shape[0])
        nms_time.update((t4 - t3) / img.shape[0], img.shape[0])

        for id in range(img.shape[0]):
            pad_w, pad_h = shapes[id][1][1]
            pad_w, pad_h = int(pad_w), int(pad_h)
            da_seg_mask = driving_area_mask(da_seg_out[id].unsqueeze(0), width, height, pad_w, pad_h, width / img_det[id].shape[1])
            ll_seg_mask = lane_line_mask(ll_seg_out[id].unsqueeze(0), width, height, pad_w, pad_h, width / img_det[id].shape[1])
            img_det_out = show_seg_result(img_det[id], (da_seg_mask, ll_seg_mask), batch=0, is_demo=True)

            det = det_pred[id]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_det_out.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label_det_pred = 'Car ' + f'{conf * 100:.2f}' + '%' if opt.show_detect_label else None
                    img_det_out = plot_one_box(xyxy, img_det_out, label=label_det_pred, color=[0, 255, 255], line_thickness=1)

            if dataset.mode != 'stream':
                save_path = str(Path(opt.save_dir) / Path(path[id]).name)
            else:
                save_path = str(Path(opt.save_dir) / "web.mp4")

            if opt.original_shape:
                ori_height = int(opt.img_size / shapes[id][1][0][0])
                ori_width = int(opt.img_size / shapes[id][1][0][1])
                img_det_out = cv2.resize(img_det_out, (ori_width, ori_height), interpolation=cv2.INTER_LINEAR)

            if dataset.mode == 'images':
                cv2.imwrite(save_path, img_det_out)
            elif dataset.mode == 'video':
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    h, w = img_det_out.shape[0], img_det_out.shape[1]
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(img_det_out)

            else:
                cv2.imshow('image', img_det_out)

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, batch=10):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist' % p)

        img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
        vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']
        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'images'
        self.frame = 0
        self.nframes = 0
        self.batch = batch
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        batch_path, batch_img, batch_img0, batch_shapes = [], None, None, []
        for _ in range(min(self.batch, self.nframes - self.count if self.video_flag[self.count] else self.nf - self.count)):
            path = self.files[self.count]

            if self.video_flag[self.count]:
                # Read video
                self.mode = 'video'
                ret_val, img0 = self.cap.read()
                if not ret_val:
                    self.count += 1
                    self.cap.release()
                    if self.count == self.nf:  # last video
                        raise StopIteration
                    else:
                        path = self.files[self.count]
                        self.new_video(path)
                        ret_val, img0 = self.cap.read()
                h0, w0 = img0.shape[:2]

                self.frame += 1
            else:
                # Read image
                self.count += 1
                img0 = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
                assert img0 is not None, 'Image Not Found ' + path
                h0, w0 = img0.shape[:2]

            # Padded resize
            img0 = cv2.resize(img0, (1280, 720), interpolation=cv2.INTER_LINEAR)
            img, ratio, pad = letterbox_for_img(img0, new_shape=self.img_size, auto=False)
            h, w = img.shape[:2]
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            # Convert
            img = np.ascontiguousarray(img[:, :, ::-1])
            img = self.transform(img)

            batch_path.append(path)
            batch_img = img.unsqueeze(0) if batch_img is None else torch.cat([batch_img, img.unsqueeze(0)], dim=0)
            batch_img0 = img0[np.newaxis, :] if batch_img0 is None else np.append(batch_img0, img0[np.newaxis, :], axis=0)
            batch_shapes.append(shapes)

        if self.count < len(self.video_flag):
            if self.video_flag[self.count]:
                print('\n video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nf, self.frame, self.nframes, path), end='')
            else:
                print('image %g/%g %s: \n' % (self.count, self.nf, path), end='')

        return batch_path, batch_img, batch_img0, self.cap, batch_shapes

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return math.ceil(self.nf / self.batch)  # number of files


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, auto=True):
        self.mode = 'stream'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print('')  # newline

        # check for common shapes

        s = np.stack([letterbox_for_img(x, self.img_size, auto=self.auto)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def update(self, i, cap):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        batch_path, batch_img, batch_img0, batch_shapes = [], None, None, []
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        h0, w0 = img0[0].shape[:2]

        img0[0] = cv2.resize(img0[0], (1280, 720), interpolation=cv2.INTER_LINEAR)
        img, _, pad = letterbox_for_img(img0[0], self.img_size, auto=False)

        # Stack
        h, w = img.shape[:2]
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        # Convert
        img = np.ascontiguousarray(img[..., ::-1])
        img = self.transform(img)

        batch_path.append(self.sources)
        batch_img = img.unsqueeze(0) if batch_img is None else torch.cat([batch_img, img.unsqueeze(0)], dim=0)
        batch_img0 = img0[0][np.newaxis, :] if batch_img0 is None else np.append(batch_img0, img0[0][np.newaxis, :],
                                                                                 axis=0)
        batch_shapes.append(shapes)
        return batch_path, batch_img, batch_img0, None, batch_shapes

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/mtpnet.onnx', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='file/folder  ex:inference/images')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images processed at once')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_dir', type=str, default='runs/detect', help='directory to save results')
    parser.add_argument('--name', default='results', help='save results to directory/name')
    parser.add_argument('--original_shape', default=True, help='maintain original shape')
    parser.add_argument('--show_detect_label', default=False, help='show detect labels or not')
    opt = parser.parse_args()

    with torch.no_grad():
        detect(opt)
