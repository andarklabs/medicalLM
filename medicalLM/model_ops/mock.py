# from dotenv.main import dotenv_values

import json
import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


def get_hf_info(*args) -> List:  # ideally this just works
    # TODO: should take a model name and look in blobs (but stupid blobs ACK i hate this dont do it
    # this way)
    info = [json.loads(os.getenv("hf_info"))[key] for key in args]
    return info


print(*get_hf_info("model", "test"))
