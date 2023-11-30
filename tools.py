import random
import time
from typing import Union

import numpy as np
import pynvml
import pytorch_lightning as pl
import torch
from torchinfo import summary


def convert_str_params_list(str_params: str, params_type: str, split_marker: str = '#') -> list:
    params_list = str_params.split(split_marker)
    params_list = [eval(params_type)(param) for param in params_list]
    return params_list


def set_random_seed(random_seed: Union[float, int]):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    pl.seed_everything(random_seed)


def predict_model_memory_usage(
        model: torch.nn.Module,
        input_shape: list[...],
) -> float:
    """
    calculate model memory usage
    :param model: model
    :param input_shape: input shape
    :return: predicted memory usage (GB)
    """
    summary_info = summary(model, input_size=input_shape, device=torch.device('cpu'), mode='train')
    memory_usage = summary_info.total_input + summary_info.total_output_bytes + summary_info.total_param_bytes
    # detach input tensor and model, then release memory
    del model
    del summary_info
    return memory_usage / 1024 / 1024 / 1024


def auto_find_memory_free_card(
        card_list: list[int, ...],
        model_memory_usage: float,
        idle: bool = False,
        idle_max_seconds: int = 60 * 60 * 24,
) -> int:
    """
    auto find memory free card
    :param card_list: card list to choose
    :param model_memory_usage: model memory usage (GB)
    :param idle: if True, waiting until there is a card with free memory
    :param idle_max_seconds: max waiting seconds
    :return: card id
    """
    device_count = torch.cuda.device_count()
    print(f'[Info] device count: {device_count}')
    pynvml.nvmlInit()
    if idle:
        start_time = time.time()
        print('[Info] waiting for idle card, waiting begin time:',
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
        while True:
            if time.time() - start_time > idle_max_seconds:
                # shutdown pynvml
                pynvml.nvmlShutdown()
                raise TimeoutError(
                    f'[Error] no card has enough memory to load model, model memory usage is {model_memory_usage} MB')
            # get all card free memory and use the card with max free memory
            print('=' * 50)
            print('[Info] waiting for idle card, waiting time: {:.0f} seconds'.format(time.time() - start_time))
            free_memory_list = []
            for card_id in card_list:
                # get card free memory by pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(card_id)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_memory = info.free / 1024 / 1024 / 1024
                print('[Info] cuda:{}, free memory: {:.1f} GB, model needs: {:.1f} GB'.format(
                    card_id,
                    free_memory,
                    model_memory_usage,
                ))
                if free_memory > model_memory_usage:
                    free_memory_list.append({
                        'card_id': card_id,
                        'free_memory': free_memory,
                    })
            print('=' * 50)
            if len(free_memory_list) != 0:
                # shutdown pynvml
                pynvml.nvmlShutdown()
                # sort free memory list by free memory descending
                free_memory_list.sort(key=lambda x: x['free_memory'], reverse=True)
                max_free_memory_card_id = free_memory_list[0]['card_id']
                print('[Info] find free card cuda:{}, training begin time: {}'.format(
                    max_free_memory_card_id,
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                )
                return max_free_memory_card_id
            else:
                print('[Info] no card has enough memory to load model, waiting for 60 seconds')
                time.sleep(60)

    else:
        # get all card free memory and use the card with max free memory
        free_memory_list = []
        for card_id in card_list:
            # get card free memory by pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(card_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory = info.free / 1024 / 1024 / 1024
            if free_memory > model_memory_usage:
                free_memory_list.append({
                    'card_id': card_id,
                    'free_memory': free_memory,
                })
        if len(free_memory_list) == 0:
            # shutdown pynvml
            pynvml.nvmlShutdown()
            raise RuntimeError('no card has enough memory to load model, model memory usage is {:.1f} GB'.format(
                model_memory_usage))
        # shutdown pynvml
        pynvml.nvmlShutdown()
        # sort free memory list by free memory descending
        free_memory_list.sort(key=lambda x: x['free_memory'], reverse=True)
        max_free_memory_card_id = free_memory_list[0]['card_id']
        return max_free_memory_card_id
