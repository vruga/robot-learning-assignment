# Fixed environment and method interface

## What is fixed

- Upstream BiGym2 commit: `d71501359b215b9d78e27dcb70b220412d474dfa`.
- MuJoCo: 3.8.1; all other resolved dependencies are in `uv.lock`.
- Task: `move_plate`; episode length: 17,000 upstream env steps / 1,700 policy control steps / 34 simulated seconds, after the 200-control-step warmup.
- 50 Hz control, frozen G1/Dex1-1 model, Balance/Walk ONNX controller.
- Three RGB 84×84 cameras; 50-dimensional proprioception; 20-dimensional actions.
- Source slot randomization and ±5 cm x/y rack variation, inherited unchanged.
- Success: any destination slot within 5 cm, orientation within 20° toward the robot's right, destination-rack contact, no table contact, no gripper holding; continuously true for one second. Floor contact fails; a fall at any time invalidates success.

The source selection is sampled exactly as upstream. The plan deterministically finds unique seeds giving the same number of starts from every slot, and checks those starts in the real simulator. It never relocates the plate, shortens warmup or changes physics to balance the trials.

## Policy observations

The only evaluated inputs are `rgb` and `state`. There is no simulator object or seed in the policy process. Training tools can access the public environment locally, but evaluated inference must use only those observations.

State layout (zero-based indices):

- 0–6: left arm joint positions.
- 7–8: left finger joint positions.
- 9–15: right arm joint positions.
- 16–17: right finger joint positions.
- 18–21: pelvis x/y/z/yaw.
- 22–43: corresponding velocities.
- 44–45: left/right gripper states.
- 46–49: base-controller pelvis x/y/z/yaw.

`action_layout.json` lists arm names and raw bounds. Normalize with `2*(raw-low)/(high-low)-1`. A normalized zero is **the middle of a joint's range**, not a neutral robot pose. Preserve this distinction when constructing model adapters.

## Action chunks

Return one action or up to 50 actions (one simulated second). The evaluator executes each action with full physics and success/fall checks and requests fresh observations after the chunk. It stops immediately on episode termination. Choose your chunk length and explain the feedback/latency tradeoff. Inference time does not advance simulated time; it is measured separately.

## What you may change

Your method, models, policy-side resizing/normalization, data processing, training architecture/objectives, controller/planner above the common action API, and recovery logic are yours. Use a separate training environment if desired and disclose differences. Evaluation always uses the frozen common environment.

Do not replace the supplied lower-body controller or add actuators, weld the plate to a gripper, teleport bodies, alter friction/contact parameters or loosen success criteria. Such changes defeat the common task.

## Evaluation result meaning

30, 36, 42, 48, 54, 60... are scored counts. At least 90% of **all requested trials** must succeed. Policy errors are failures; infrastructure errors make the run non-qualifying. Missing trials never disappear from the denominator. Incomplete, modified or smoke runs cannot pass.

Reports and action traces support reproduction and diagnosis. They are not cryptographically attested results. A reviewer must run the candidate policy using a clean trusted copy of this repository.
