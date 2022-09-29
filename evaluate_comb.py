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
parser.add_argument(
    "--cktdir",
    type=str,
    default="",
)
parser.add_argument("--iter-range", type=parse_range, default="start1end10")
parser.add_argument("--policy-step", type=int, default=10000)
parser.add_argument("--model-step", type=int, default=50000)
parser.add_argument("--overwrite", action="store_true", default=False)
parser.add_argument("--logpi-alpha", type=float, default=None)
parser.add_argument("--logpi-each-step", action="store_true", default=False)

args = parser.parse_args()

cktdir = args.cktdir
iter_start, iter_end = args.iter_range
policy_step = args.policy_step
model_step = args.model_step
overwrite = args.overwrite
logpi_alpha = args.logpi_alpha
logpi_each_step = args.logpi_each_step

assert os.path.exists(cktdir)

model_ckt = os.path.join(cktdir, f"checkpoint_{model_step}.pt")
assert os.path.exists(model_ckt)
policy_ckt = os.path.join(cktdir, f"checkpoint_{policy_step}.pt")
assert os.path.exists(policy_ckt)

logger.info(f"model ckt {model_ckt}")
logger.info(f"policy ckt {policy_ckt}")


def evaluate(
    env_e, agent, ddpiter=0,
):
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
                state_e, evaluate=True, ddp=ddpiter > 0, ddp_iters=ddpiter
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


result_fn = f"comb_p{policy_step}_m{model_step}_s{iter_start}_e{iter_end}_logpi_{logpi_alpha}_es{logpi_each_step}.result"
result_fn = os.path.join(cktdir, result_fn)
if os.path.exists(result_fn) and not overwrite:
    exit(0)

logger.info(f"save to {result_fn}")


checkpoint = torch.load(model_ckt)
args = checkpoint["args"]
args.logpi_alpha = logpi_alpha
args.logpi_each_step = logpi_each_step

env_e = gym.make(args.env_name)
env_e.seed(1234)
env_e.action_space.seed(1234)

agent = Agent(env_e.observation_space.shape[0], env_e.action_space, args)
agent.load_checkpoint(checkpoint)

# load policy
checkpoint_polciy = torch.load(policy_ckt)
agent.log_alpha.data.copy_(checkpoint_polciy["log_alpha"])
agent.alpha = agent.log_alpha.exp()
agent.policy.load_state_dict(checkpoint_polciy["policy"])
agent.critic.load_state_dict(checkpoint_polciy["critic"])
agent.critic_target.load_state_dict(checkpoint_polciy["critic_target"])


results = dict()
logger.info("start eval policy...")

results["policy"] = evaluate(env_e, agent, ddpiter=0)
logger.info("policy {}".format(results["policy"]))
logger.info("")

for iternum in range(iter_start, iter_end + 1):
    logger.info(f"start eval ddp {iternum}...")
    results[f"ddp_{iternum}"] = evaluate(env_e, agent, ddpiter=iternum)
    logger.info("ddp {} {}".format(iternum, results[f"ddp_{iternum}"]))
    logger.info("")

with open(result_fn, "w") as fn:
    json.dump(results, fn)
