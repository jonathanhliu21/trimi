# TriMi

TriMi brings together hand and Piper control, demonstration collection,
generative policy training, and learned policy deployment in one repository.

| Folder | Purpose | Documentation |
| --- | --- | --- |
| `diffusion-policies/` | Diffusion, flow-matching, and shortcut policy training; shared vision encoders and inference wrappers | [Training README](diffusion-policies/README.md), [shortcut policy guide](diffusion-policies/shortcut_policy/README.md) |
| `trimorph-piper-control/` | TRIMORPH/MIDAS bring-up, retargeting, force feedback, Piper/OAK/Quest/Vicon collection, replay, and policy execution | [Control README](trimorph-piper-control/README.md), [collection pipeline](trimorph-piper-control/data_collection/PIPELINE.md), [inference guide](trimorph-piper-control/inference/README.md) |
| `trimorph_hand_api/` | Dynamixel hand control, homing/calibration, and Paxini tactile support | [Hand API README](trimorph_hand_api/README.md) |

The component READMEs retain their detailed instructions, examples, credits,
and citations. This page connects them into a single workflow. Run component
commands from the indicated folder; “repository root” in a component guide
means that component's folder.

## Setup

Keep hardware control and GPU training in separate Python environments. The
hand API folder is named `trimorph_hand_api`, but its Python import remains
`midas_hand_api` and its distribution remains `midas-hand`.

### Hardware control and hand API

From the `trimi/` root:

```bash
cd trimorph-piper-control
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ../trimorph_hand_api
python -c 'import midas_hand_api; print(midas_hand_api.__file__)'
```

The control requirements include the vendor Piper SDK. See the
[control setup](trimorph-piper-control/README.md#setup) for SocketCAN, camera,
and UI requirements, and the [hand API setup](trimorph_hand_api/README.md#setup)
for serial permissions, USB latency, calibration, and optional tactile tools.
The hand API can also be installed independently from its own folder.

Review hand configuration, motor mapping, limits, gains, current limits, and
serial ports before enabling torque. Use calibrated local configurations and
follow the [hardware safety instructions](trimorph-piper-control/README.md#safety-and-configuration).

### GPU training

From the `trimi/` root, in a separate shell/environment:

```bash
cd diffusion-policies
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-train.txt
python -m pip install -e .
wandb login
```

See the [training README](diffusion-policies/README.md) for the alternative
Conda setup and the [shortcut guide](diffusion-policies/shortcut_policy/README.md)
for flow matching, shortcut policies, and DINOv3 dependencies and weights.

## Collection to deployment

1. Bring up and calibrate the hand using the
   [hand API guide](trimorph_hand_api/README.md) and
   [control diagnostics](trimorph-piper-control/README.md).
2. Record demonstrations and check stream quality using the
   [collection pipeline](trimorph-piper-control/data_collection/PIPELINE.md).
   Postprocess with `data_collection/postprocess_dataset.py` from
   `trimorph-piper-control/`; Quest and mocap pose sources need the appropriate
   calibration. The [dataset specification](trimorph-piper-control/data_collection/DATASET_SPEC.md)
   describes the training inputs.
3. Train from `diffusion-policies/`, pointing at the processed dataset:

   ```bash
   python diffusion_policy/unet/train_unet_image_policy.py \
     --config-name=train_unet_image_trimi_hand_cond_dinov2_policy \
     tasks.dataset.dataset_path=/absolute/path/to/processed_dataset
   ```

4. On the GPU machine, activate the training environment, then run from
   `trimorph-piper-control/`:

   ```bash
   python inference/predictor/predictor.py \
     --policy-repo ../diffusion-policies \
     --checkpoint /absolute/path/to/latest.ckpt \
     --device cuda:0
   ```

5. On the robot machine, activate the control environment and test the link
   from `trimorph-piper-control/`:

   ```bash
   python inference/executor/executor.py \
     --predictor GPU_HOST --fake --max-cycles 20
   ```

   Fake mode uses synthetic observations without connecting to hardware; the
   predictor still runs the actual model. Follow the
   [inference guide](trimorph-piper-control/inference/README.md) for real execution,
   hardware configuration, synchronous/async/RTC modes, and timing limitations.

Collection, training, and execution must agree on the calibrated 13-joint hand
order and units, pose frames, camera settings, and effective sample rate after
downsampling. See the inference guide's coordinate contract before deployment.

## Repository provenance

The following local branch tips were imported with full reachable Git history
using `git subtree add` without squashing. All three folders are ordinary tracked
directories in this repository; a normal clone includes all three.

| Folder | Source | Branch | Imported commit |
| --- | --- | --- | --- |
| `diffusion-policies/` | [alvister88/diffusion-policies](https://github.com/alvister88/diffusion-policies) | `trimi` | `e6c4b003f38914f1ef49c7a71cf5ddb4e0508ab5` |
| `trimorph-piper-control/` | [jonathanhliu21/trimorph-piper-control](https://github.com/jonathanhliu21/trimorph-piper-control) | `rss` | `4e727c1cfccc0ffee18b0279d3a2f17ce06541c4` |
| `trimorph_hand_api/` | [jonathanhliu21/trimorph-hand-api](https://github.com/jonathanhliu21/trimorph-hand-api) | `more-lenient-tactile` | `9c1f53ce85c3add8acca4d377e0d647332376767` |

Imports contain committed source files. Existing virtual environments, datasets,
checkpoints, and ignored local configuration files were not copied. Recreate
environments and supply local data/calibration paths on each machine. Original
license files and attribution remain in their component folders.
