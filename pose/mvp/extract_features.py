import numpy as np
from features.extract_lengths import (
    KeypointScheme,
    extract_lengths,
    extract_near_cosines,
)

from .paths.paths import NAMES, OUT_PATH


def main(eval_: bool = False):
    for person in NAMES:
        print(person)
        name = person["out"]
        if eval_:
            name = person["out"]
        kp3d_path = (
            OUT_PATH / "rsvid" / "npys_with_features" / f"full_features_{name}.npy"
        )
        data = np.load(kp3d_path)
        print(data.shape)
        ls = extract_lengths(data, KeypointScheme._17)
        # ls = np.sqrt(np.sum(ls*ls, axis=(2,)))
        np.save(OUT_PATH / "rsvid" / f"lengths_{name}.npy", ls)
        print("Lengths saved:", ls.shape)
        cs = extract_near_cosines(data, KeypointScheme._17)
        np.save(OUT_PATH / "rsvid" / f"angles_{name}.npy", cs)
        print("Angles saved:", cs.shape)


if __name__ == "__main__":
    main()
