import torch
from torch import nn


from DECA.decalib.deca import DECA
from DECA.decalib.utils.config import cfg as deca_cfg

OUTPUT_NAMES = ["shape", "tex", "exp", "pose", "cam", "light"]


class DECAWrapper(DECA):
    def forward(self, images):
        if self.use_detail:
            # use_detail is for training detail model, need to set coarse model as eval mode
            with torch.no_grad():
                parameters = self.E_flame(images)
        else:
            parameters = self.E_flame(images)
        codedict = self.decompose_code(parameters, self.param_dict)
        if self.use_detail:
            detailcode = self.E_detail(images)
            codedict["detail"] = detailcode
            return tuple([codedict[k] for k in OUTPUT_NAMES + ["detail"]])
        else:
            return tuple([codedict[k] for k in OUTPUT_NAMES])


def get_deca(device, deca_cfg):
    return DECAWrapper(config=deca_cfg, device=device).to(device).eval()


@torch.no_grad()
def trace_deca(deca_cfg, device, use_detail=True):
    images = torch.rand(1, 3, 224, 224).to(device)
    deca = get_deca(device, deca_cfg)
    deca.use_detail = use_detail
    jit_model = torch.jit.trace(deca, images, strict=False)
    return jit_model


@torch.no_grad()
def deca_to_onnx(outpath, deca_cfg, device, use_detail=True):
    images = torch.rand(1, 3, 224, 224).to(device)
    deca = get_deca(device, deca_cfg)
    deca.use_detail = use_detail
    input_names = ["image"]
    output_names = OUTPUT_NAMES
    output_names += ["detail"] if use_detail else []
    torch.onnx.export(
        deca,
        images,
        outpath,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = "pytorch3d"
    deca_cfg.model.extract_tex = False

    deca_trace_no_detail = trace_deca(deca_cfg, device, use_detail=False)
    deca_trace_no_detail.save("deca_no_detail.pt")
    print("Saved deca_no_detail.pt")

    deca_trace_detail = trace_deca(deca_cfg, device, use_detail=True)
    deca_trace_detail.save("deca_detail.pt")
    print("Saved deca_detail.pt")

    deca_to_onnx("deca_no_detail.onnx", deca_cfg, device, use_detail=False)
    print("Saved deca_no_detail.onnx")
    deca_to_onnx("deca_detail.onnx", deca_cfg, device, use_detail=True)
    print("Saved deca_detail.onnx")
