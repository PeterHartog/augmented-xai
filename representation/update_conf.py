import os
import pathlib

import pyrootutils
from aidd_codebase.utils import utils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

if __name__ == "__main__":
    from aidd_codebase.registries import AIDD

    from representation.src.datamodules.ames import TDCommonsDatamodule
    from representation.src.datamodules.chembl import TDCommonsDatamodule
    from representation.src.models.encoder_decoder import EncoderDecoder
    from representation.src.models.encoder_only import EncoderOnly
    from representation.src.models.mlp import LightningMLP
    from representation.src.models.transformer_cnn import TransformerCNN
    from representation.src.models.transformer_nn import TransformerNN
    from representation.src.tokenizer.character import CharTokenizer

    print([registry for registry in AIDD.get_registries().keys()])
    config_check = utils.ConfigChecker(os.path.join(pathlib.Path(__file__).parent.absolute(), "conf"))
