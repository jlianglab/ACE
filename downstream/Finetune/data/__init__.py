from .build import build_loader as _build_loader
from .build_NIHchest import build_loader_NIHchest
from .build_RSNA import build_loader_RSNA
from .build_shenzhenCXR import build_loader_shenzhenCXR
from .build_CheXpert import build_loader_CheXpert
from .data_simmim_ft import build_loader_finetune
from .data_simmim_pt import build_loader_simmim
from .build_JSRT import build_loader_JSRT
from .build_ChestXdet import build_loader_ChestXdet
from .build_SIIM import build_loader_SIIM
from .build_Montgomery import build_loader_Montgomery
from .build_VindrRib import build_loader_VindrRib
from .build_EyePACS import build_loader_eyepacs
from .build_CovidQuEx import build_loader_CovidQuEx
from .build_SIIM_cls import build_loader_SIIM_cls
from .build_NIHchest_keypoint import build_loader_NIHchest_keypoint


def build_loader(config, dataset, simmim=False, is_pretrain=False, ddp=False):
    if dataset=="NIHchest":
        return build_loader_NIHchest(config, ddp)
    elif dataset=='shenzhenCXR':
        return build_loader_shenzhenCXR(config, ddp)
    elif dataset=='RSNA':
        return build_loader_RSNA(config, ddp)
    elif dataset=='CheXpert':
        return build_loader_CheXpert(config, ddp)
    elif dataset=='JSRT':
        return build_loader_JSRT(config, ddp)
    elif dataset=='ChestXdet':
        return build_loader_ChestXdet(config, ddp)
    elif dataset=='SIIM':
        return build_loader_SIIM(config, ddp)
    elif dataset=='Montgomery':
        return build_loader_Montgomery(config, ddp)
    elif dataset=='VindrRib':
        return build_loader_VindrRib(config, ddp)
    elif dataset=='eyepacs':
        return build_loader_eyepacs(config, ddp)
    elif dataset=='CovidQuEx':
        return build_loader_CovidQuEx(config, ddp)
    elif dataset=='SIIM_cls':
        return build_loader_SIIM_cls(config, ddp)
    elif dataset=='NIHchest_keypoint':
        return build_loader_NIHchest_keypoint(config)
    # if not simmim:
    #     return _build_loader(config)
    # if is_pretrain:
    #     return build_loader_simmim(config)
    # else:
    #     return build_loader_finetune(config)
