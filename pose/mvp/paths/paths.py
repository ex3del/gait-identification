import json
from os import PathLike
from pathlib import Path


class _DataPath:
    def __init__(
        self,
        prefix: str | PathLike,
        bag: str | PathLike,
    ):
        project_root = Path(__file__).parent.parent.parent.parent

        if isinstance(prefix, str):
            prefix = project_root / prefix
        if isinstance(bag, str):
            bag = project_root / bag

        assert isinstance(prefix, PathLike), TypeError(
            "Given prefix path is typed nor str neither type!"
        )
        assert prefix.exists(), RuntimeError(f"Path '{prefix}' does not exist!")
        assert isinstance(bag, PathLike), TypeError(
            "Given bag path is typed nor str neither type!"
        )
        assert bag.exists(), RuntimeError(f"Path '{bag}' does not exist!")

        self._prefix = prefix
        self._bag = bag

    @property
    def BAG(self):
        return self._bag

    @property
    def MP4(self):
        return self._prefix / "mp4"

    @property
    def KP_PIXEL(self):
        return self._prefix / "kp_pixel"

    @property
    def KP_3D(self):
        return self._prefix / "kp_3d"

    @property
    def FEATURES(self):
        return self._prefix / "features"

    def check_tree(self):
        """Checks or creates subdirectory tree"""
        for p in [
            self.MP4,
            self.KP_PIXEL,
            self.KP_3D,
            self.FEATURES,
        ]:
            if not p.exists():
                p.mkdir(exist_ok=True, parents=True)
            else:
                if not p.is_dir():
                    raise RuntimeError(f"Path '{p}' exists but is not a directory!!!")


project_root = Path(__file__).parent.parent.parent.parent

TRAIN = _DataPath(prefix="data/rsvid/train", bag="data/rsvid/bag")
EVAL = _DataPath(prefix="data/rsvid/eval", bag="data/rsvid/bag")
MODELS = project_root / "data/models"
PLOTS = project_root / "data/plots"

current_dir = Path(__file__).parent
with open(current_dir / "names_best_result.json", "r") as names_json:
    NAMES = json.load(names_json)

with open(current_dir / "good_features.json", "r") as gf_json:
    GOOD_FEATURES = json.load(gf_json)
