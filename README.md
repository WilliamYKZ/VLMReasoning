# VLMReasoning

## Config.yaml Change

### What I change

1. Switch the `train_files`, `val_files`, `format_prompt`
   1. `rollout n = 1` For the tesing I change to 1 example. 
   2. `limit_images = 1` Since we are just using the tool once. 
2. Add
   1. `image_key` for image path.
   2. `format_prompt` Using the Refocus prompt style. I add a new file write in the prompt, where is `EasyR1/examples/format_prompt/chartQA.jinja` (Might need change format)
   3. `execute_once` Because we are execute once right now, and I already change in the fsdp_worker.py. (Please check the training pipeline you want and then change this logic)
   4. `reward_function` Change to the Refocus style reward fucntion. Using rule base reward, add the file to `EasyR1/examples/reward_function/chartqa_reward.py`. There might be error please do experiments.  





## Rollout Worker Change

Main Target: `EasyR1/verl/workers/fsdp_worker.py`

Secondary Target: `EasyR1/verl/rollout/vllm_rollout_spmd.py` Because if we need lower-level control over how vLLMRollout processes multimodal inputs. 

### Change in the `fsdp_worker.py`

 







## Data Loader Change









## Re-encode the Edited Image via Vision Encoder

