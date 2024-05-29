from typing import Optional
import tempfile
import logging

from src.data import get_dataset
from src.algorithms.compilation import compile
from src.algorithms.quantization import quantize
from src.algorithms.pruning import prune
from src.algorithms.factorization import factorize
from src.loggings.formatters import CustomFormatter
from pruna_engine.PrunaModel import PrunaModel

# Set up logging
pruna_logger = logging.getLogger("pruna_logger")
pruna_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(CustomFormatter("%(asctime)s - %(levelname)s - %(message)s - (%(filename)s:%(lineno)d)"))
pruna_logger.propagate = False
pruna_logger.addHandler(console_handler)


def smash(
    model,
    api_key: str,
    smash_config,  # Contains any compression parameters including pruner, quantizer....
    dataloader=None,
) -> PrunaModel:
    """
    Prunes, factorizes, quantizes, and compiles a PyTorch model for inference using the Pruna library.

    Args:
        model (torch.nn.Module): The PyTorch model to be pruned, factorized, quantized, and compiled.
        dataloader (LightningDataModule): The PyTorch Lightning data module used to load the training data.
        api_key (str): The API key used to log the request.
        smash_config: smash configurations.
        device: device to use for smashing

    Returns:
        PrunaModel: The pruned, factorized, quantized, and compiled model wrapped in a PrunaModel object.
    """
    pruna_logger.info("Verify API key")
    PrunaModel.verify_api_key(api_key)

    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name  # get the path to the temporary directory

    if "tokenizer_name" not in smash_config:
        tokenizer_name = "placeholder"
    elif hasattr(smash_config["tokenizer_name"], "tokenizer"):
        tokenizer_name = smash_config["tokenizer_name"].tokenizer.name_or_path
    else:
        tokenizer_name = smash_config["tokenizer_name"].name_or_path

    # TODO: Fix Tokenizer
    if isinstance(dataloader, str):
        data_module = get_dataset(
            dataset_name=dataloader,
            directory_dataset=temp_dir,
            batch_size=smash_config["batch_size"],
            tokenizer_name=tokenizer_name,
            seed=123,
        )
        dataloader = data_module.val_dataloader()
    temp_dir_obj.cleanup()

    # Check if the smash_config is valid
    smash_config.validate()

    # Prune
    pruners = smash_config["pruners"]
    if pruners:
        pruna_logger.info("Prune...")
        model, save_load_fn = prune(model, dataloader=dataloader, smash_config=smash_config)

    # Factorize
    factorizers = smash_config["factorizers"]
    if factorizers:
        pruna_logger.info("Factorize...")
        model, save_load_fn = factorize(model, dataloader=dataloader, smash_config=smash_config)

    # Quantize
    quantizers = smash_config["quantizers"]
    if quantizers:
        pruna_logger.info("Quantize...")
        model, save_load_fn = quantize(model, dataloader=dataloader, smash_config=smash_config)

    # Compile
    compilers = smash_config["compilers"]
    if compilers:
        pruna_logger.info("Compile...")
        model, save_load_fn = compile(model, dataloader=dataloader, smash_config=smash_config)

    smash_config["save_load_fn"] = save_load_fn
    model = PrunaModel(model, api_key=api_key, smash_config=smash_config)

    # Log smashing
    PrunaModel.log_request_static(api_key, 0)

    return model
