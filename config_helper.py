import json
from dataclasses import dataclass

from loguru import logger


class ReadConfigFile:
    def __init__(self) -> None:
        pass

    def read_config(self, path):
        try:
            with open(path, "r") as jsonfile:
                data = json.load(jsonfile)
        except Exception as error:
            logger.error(
                f"[CFG] Failed to load config file [path={path} error={error}]"
            )

        # fmt: off
        @dataclass
        class config:
            camera_url:                 str
            lpd_weights:                str
            lpd_network:                str
            lpd_classes:                str
            lpd_confidence:             float
            lpd_nms_threshold:          float
            lpd_width:                  int
            lpd_height:                 int
            counter_bbox:               bool
            counter_video_save_path:    str

        config.camera_url               = data["camera_url"]
        config.lpd_weights              = data["lpd_weights"]
        config.lpd_network              = data["lpd_network"]
        config.lpd_classes              = data["lpd_classes"]
        config.lpd_confidence           = data["lpd_confidence"]
        config.lpd_nms_threshold        = data["lpd_nms_threshold"]
        config.lpd_width                = data["lpd_width"]
        config.lpd_height               = data["lpd_height"]
        config.counter_bbox             = data["counter_bbox"]
        config.counter_video_save_path  = data["counter_video_save_path"]

        return config
        # fmt: off
