import os

import numpy as np
import torch

from image_utils import save_image


class SwinIrMagnifier:
    model="test"
    device = None
    args = None

    def load(self, config):
        import sys
        assert os.path.isdir("./libs"), os.path.abspath("./libs")
        sys.path.append("../..")
        try:
            from SwinIR.models.network_swinir import Mlp
            from SwinIR.main_test_swinir import define_model
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Could not find swinIR lib. From the lib directory run `git clone "
                                      "https://github.com/JingyunLiang/SwinIR.git`")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class Args(object):
            pass

        args = Args()
        args.task = "real_sr"
        args.large_model = False
        args.window_size = 8
        args.scale = 4
        args.tile = None
        args.tile_overlap = 32
        args.model_path = os.path.join('./models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth')  # set model path
        # 003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth
        model = define_model(args=args)
        self.args = args
        model.eval()
        self.model = model.to(self.device)
        torch.cuda.empty_cache()

    def transform(self, config, input_path, output_path):
        from SwinIR.main_test_swinir import get_image_pair, test
        # read image
        imgname, img_lq, img_gt = get_image_pair(self.args, input_path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                              (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

        # inference
        window_size = self.args.window_size or 8
        scale = self.args.scale or 4

        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, self.model, args=self.args, window_size=window_size)
            output = output[..., :h_old * scale, :w_old * scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        save_image(image=output, save_path=output_path)


    def destroy(self, config):
        self.model = None
