"""
Generate traning and evaluating bash scripts.
"""

import os
import itertools

MODEL_INFO = {
    #"SRCNN": {"lr": 1e-3, "wd": 1e-5, "batch_size": 32, "epochs": 25},
    #"ESPCN": {"lr": 1e-4, "wd": 1e-5, "batch_size": 32, "epochs": 25},
    "FNO": {"lr": 1e-3, "wd": 1e-5, "batch_size": 32, "epochs": 25, "modes": [12, 16, 20, 24], "width": [32, 48, 64], "layers": [3, 4, 5]}
}

UPSCALE_FACTORS = [8]

TRAIN_SCRIPT = "train_all.sh"
EVAL_SCRIPT = "eval_all.sh"

if __name__ == "__main__":
    
    # Clear output files
    open(TRAIN_SCRIPT, "w").close()
    open(EVAL_SCRIPT, "w").close()

    with open(TRAIN_SCRIPT, "a") as f_train, open(EVAL_SCRIPT, "a") as f_eval:
        for upscale in UPSCALE_FACTORS:
            for model_name, params in MODEL_INFO.items():
                base_cmd = (
                    f"python train.py"
                    f" --model {model_name}"
                    f" --upscale_factor {upscale}"
                    f" --lr {params['lr']}"
                    f" --wd {params['wd']}"
                    f" --batch_size {params['batch_size']}"
                    f" --epochs {params['epochs']}"
                )

                if model_name == "FNO":
                    for modes, width, layers in itertools.product(
                        params["modes"], params["width"], params["layers"]
                    ):
                        fno_cmd = (
                            base_cmd +
                            f" --modes {modes} --width {width} --layers {layers}"
                        )
                        f_train.write(fno_cmd + "\n")
                        f_eval.write(fno_cmd + " --evaluate=True\n")
                else:
                    f_train.write(base_cmd + "\n")
                    f_eval.write(base_cmd + " --evaluate=True\n")

    os.chmod(TRAIN_SCRIPT, 0o755)
    os.chmod(EVAL_SCRIPT, 0o755)
