from dataclasses import dataclass
import json


class ReadConfigFile:

    def __init__(self) -> None:
        pass

    def read_config(self, path):
        with open(path, "r") as jsonfile:
            data = json.load(jsonfile)

        @dataclass
        class config:
            camera_url: str

        config.camera_url = data["camera_url"]

        return config
