from dataclasses import dataclass
from loguru import logger
import json


class ReadConfigFile:

    def __init__(self) -> None:
        pass

    def read_config(self, path):
        try:
            with open(path, "r") as jsonfile:
                data = json.load(jsonfile)
        except Exception as error:
            logger.error(f"[CFG] Failed to load config file [path={path} error={error}]")

        @dataclass
        class config:
            camera_url: str

        config.camera_url = data["camera_url"]

        return config
