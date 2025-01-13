import logging

import click
import torch
import torch.nn.functional as F

from src.models.dataset import TimeSeriesDataset, load_data
from src.models.model import load_model


def predict(
    model,
    src,
    output_sequence_length: int,
    start_token: float = 0.0,
    device="cpu",
):
    """
    1ステップごとに予測して結果を返す

    params
      model: torch module
      x: torch.tensor
        size -> B, inSec, inF
      output_sequence_length: int
      start_token: int

    returns
      torch.tensor
        size -> B, output_sequence_length(outSec), outF
    """

    logger = logging.getLogger(__name__)
    logger.debug(f"{src.size()=}")

    src_proj = model.src_input_projection(src)  # B, inSeq, E
    src_proj = model.positional_encoding(src_proj)  # B, inSeq, E
    memory = model.encoder(src_proj)  # B, inSeq, E

    tgt = src[:, -1, 1:5].unsqueeze(1)
    logger.debug(f"{tgt.size()=}")

    output_list = []
    for _ in range(output_sequence_length):
        tgt_proj = model.tgt_input_projection(tgt)  # B, t, E
        tgt_proj = model.positional_encoding(tgt_proj)  # B, t, E
        out_proj = model.decoder(tgt_proj, memory)  # B, t, E
        out = model.tgt_output_projection(out_proj)  # B, t, outF
        out = F.relu(out)

        output_list.append(out[:, -1, :].unsqueeze(1))  # B, 1, outF
        tgt = torch.cat(
            (tgt, out[:, -1, :].unsqueeze(1)), dim=1
        )  # B, t+1, outF

    output = torch.cat(output_list, dim=1)  # B, outSec, outF
    return output


def load_model_and_dataset(input_filepath, model_filepath, target_columns):

    # load model
    model, model_config = load_model(model_filepath)

    model.eval()
    # load data
    input_df = load_data(input_filepath)

    # make dataset
    dataset = TimeSeriesDataset(
        data=input_df,
        input_sequence_length=model_config.input_sequence_length,
        output_sequence_length=model_config.output_sequence_length,
        target_columns=target_columns,
    )

    return model, dataset


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("model_filepath", type=click.Path(exists=True))
@click.option("--data_index", type=int, default=0)
def main(**kwargs):
    logger = logging.getLogger(__name__)

    target_columns = ["watt_black", "watt_red", "watt_living", "watt_kitchen"]
    model, dataset = load_model_and_dataset(
        kwargs["input_filepath"], kwargs["model_filepath"], target_columns
    )

    x, actual = dataset[kwargs["data_index"]]  # (inSec, inF), (outSec, outF)
    output_sequence_length = actual.size()[0]
    logger.info(f"{output_sequence_length=}")
    predicted = predict(model, x.unsqueeze(0), output_sequence_length)

    logger.info(f"{x=}")
    logger.info(f"{x[-output_sequence_length:,1:5]=}")  # noqa: E231
    logger.info(f"{actual=}")
    logger.info(f"{predicted=}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # log_level = logging.DEBUG
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=log_fmt)
    main()
