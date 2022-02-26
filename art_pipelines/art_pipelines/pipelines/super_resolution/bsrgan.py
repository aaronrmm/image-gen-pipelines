import os

import torch

from image_utils import save_image


class BsrganMagnifier:
    model="test"
    netscale = 4
    device = None

    def load(self, config):
        import sys
        assert os.path.isdir("./libs"), os.path.abspath("./libs")
        sys.path.append("../..")
        try:
            from BSRGAN.models.network_rrdbnet import RRDBNet
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Could not find bsrgan lib. From the lib directory run `git clone "
                                      "https://github.com/cszn/BSRGAN.git`")
        print(self.model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"BSRGAN on {self.device}")
        sf = 4
        model_path = os.path.join('./models/BSRGAN.pth')  # set model path
        torch.cuda.empty_cache()

        # --------------------------------
        # define network and load model
        # --------------------------------
        self.model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network

        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        for k, v in self.model.named_parameters():
            v.requires_grad = False
        self.model = self.model.to(self.device)
        torch.cuda.empty_cache()

    def transform(self, config, input_path, output_path):
        from BSRGAN.utils.utils_image import imread_uint, uint2tensor4, imsave
        img_L = imread_uint(input_path, n_channels=3)
        img_L = uint2tensor4(img_L)
        img_L = img_L.to(self.device)
        img_E = self.model(img_L)
        save_image(image=img_E, save_path=output_path)
        print(f"saved to {os.path.abspath(output_path)}")
        #imsave(img_E, output_path)


    def destroy(self, config):
        self.model = None
