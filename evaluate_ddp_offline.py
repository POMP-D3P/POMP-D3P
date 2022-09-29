import torch
import os
import gym
import env
from agent_maac2 import Agent
import logging
import sys
import re
import argparse
import glob
import json
import math

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


def parse_range(range_str):
    pattern = re.compile("start(\d+)end(\d+)")
    results = re.search(pattern, range_str)
    if results is None:
        return (0, 150000)
    else:
        start = results.group(1)
        end = results.group(2)
        return (int(start), int(end))


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("cktdir", type=str)
parser.add_argument("--range", type=parse_range, default="start5000end100000")
parser.add_argument("--iter-range", type=parse_range, default="start0end2")
parser.add_argument("--step", type=int, default=None)

parser.add_argument("--penalize-var", action="store_true", default=False)
parser.add_argument("--penalty-coeff", type=float, default=1.0)
parser.add_argument("--penalty-model-var", action="store_true", default=False)
parser.add_argument("--overwrite", action="store_true", default=False)
parser.add_argument("--ddp_clipk", type=float, default=None)
parser.add_argument("--reverse-logpi", action="store_true", default=False)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--logpi-alpha", type=float, default=None)
parser.add_argument("--logpi-each-step", action="store_true", default=False)
parser.add_argument("--random-init", action="store_true", default=False)
args = parser.parse_args()

cktdir = args.cktdir
start, end = args.range
iter_start, iter_end = args.iter_range
evaluate_step = args.step

penalize_var = args.penalize_var
penalty_coeff = args.penalty_coeff
penalty_model_var = args.penalty_model_var
random_init = args.random_init

overwrite = args.overwrite
ddp_clipk = args.ddp_clipk
reverse_logpi = args.reverse_logpi
alpha = args.alpha
logpi_alpha = args.logpi_alpha
logpi_each_step = args.logpi_each_step

assert os.path.exists(cktdir)
cktfns = glob.glob(os.path.join(cktdir, "checkpoint_*.pt"))


def ckt2step(cktfn):
    pattern = "checkpoint_(\d+).pt"
    result = re.search(pattern, cktfn)
    assert result is not None
    step = int(result.group(1))
    return step


cktfn2step = {cktfn: ckt2step(cktfn) for cktfn in cktfns}

filtered_cktfn2step = dict()
for k, v in cktfn2step.items():
    if evaluate_step is not None:
        start = evaluate_step - 1
        end = evaluate_step
    if v > start and v <= end:
        filtered_cktfn2step[k] = v

if len(filtered_cktfn2step) == 0:
    print("no ckt!")
    exit()

print(f"evaluate {len(filtered_cktfn2step)} checkpoints")
# print(sorted(list(filtered_cktfn2step.values())))

sorted_fns = sorted(list(filtered_cktfn2step.keys()), key=lambda x: filtered_cktfn2step[x])
print(sorted_fns)

print(f"iter {iter_start+1}~{iter_end}")


def evaluate(env_e, agent, ddpiter=0, random_init=False):
    avg_reward = 0.0
    avg_steps = 0.0
    episodes = 10
    for ep in range(episodes):
        episode_reward_e = 0
        episode_steps_e = 0
        done_e = False
        state_e = env_e.reset()
        while not done_e:
            action_e = agent.select_action(
                state_e,
                evaluate=True,
                ddp=ddpiter > 0,
                ddp_iters=ddpiter,
                init_action=env_e.action_space.sample() if random_init else None,
            )
            episode_steps_e += 1
            next_state_e, reward_e, done_e, _ = env_e.step(action_e)  # fix bug
            episode_reward_e += reward_e
            state_e = next_state_e
        avg_reward += episode_reward_e
        avg_steps += episode_steps_e
    avg_reward /= episodes
    avg_steps /= episodes
    return (avg_reward, avg_steps)


for fn in sorted_fns:
    if penalize_var:
        suffix = "pc{}_{}".format(penalty_coeff, "model" if penalty_model_var else "learned")
        result_fn = f"{fn}_s{iter_start}_e{iter_end}_{suffix}.result"
    else:
        result_fn = f"{fn}_s{iter_start}_e{iter_end}.result"

    if reverse_logpi:
        result_fn = result_fn[: -len(".result")] + f"_re{alpha}" + result_fn[-len(".result") :]

    if logpi_alpha is not None:
        result_fn = (
            result_fn[: -len(".result")]
            + f"_logpialpha{logpi_alpha}"
            + (f"_eachstep" if logpi_each_step else "")
            + result_fn[-len(".result") :]
        )
    if random_init:
        result_fn = result_fn[: -len(".result")] + "_raninit" + result_fn[-len(".result") :]

    if os.path.exists(result_fn) and not overwrite:
        continue

    print(f"saving to {result_fn}")

    checkpoint = torch.load(fn)
    args = checkpoint["args"]
    args.penalize_var = penalize_var
    args.penalty_coeff = penalty_coeff
    args.penalty_model_var = penalty_model_var
    args.ddp_clipk = ddp_clipk if ddp_clipk is not None else args.ddp_clipk
    args.reverse_logpi = reverse_logpi
    args.logpi_alpha = logpi_alpha
    args.logpi_each_step = logpi_each_step

    env_e = gym.make(args.env_name)
    env_e.seed(1234)
    env_e.action_space.seed(1234)

    agent = Agent(env_e.observation_space.shape[0], env_e.action_space, args)
    agent.load_checkpoint(checkpoint)
    if reverse_logpi:
        agent.log_alpha = torch.zeros_like(agent.log_alpha) + math.log(alpha)

    results = dict()
    print("start eval policy...")
    results["policy"] = evaluate(env_e, agent, ddpiter=0)
    print("policy {}".format(results["policy"]))
    print()
    for iternum in range(iter_start + 1, iter_end + 1):
        print(f"start eval ddp {iternum}...")
        results[f"ddp_{iternum}"] = evaluate(
            env_e,
            agent,
            ddpiter=iternum,
            random_init=random_init,
        )
        print("ddp {} {}".format(iternum, results[f"ddp_{iternum}"]))
        print()

    with open(result_fn, "w") as fn:
        json.dump(results, fn)
