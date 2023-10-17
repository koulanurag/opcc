import argparse
from pathlib import Path
import os
import torch
import hashlib
import random
import numpy as np
import logging
import pickle
import wandb
import gym
from opcc.model import ActorNetwork
import copy
import torch.nn.functional as F
import torch.nn as nn
import d4rl
from PIL import Image


class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return {
            "state": torch.FloatTensor(self.state[ind]).to(self.device),
            "action": torch.FloatTensor(self.action[ind]).to(self.device),
            "next_state": torch.FloatTensor(self.next_state[ind]).to(self.device),
            "reward": torch.FloatTensor(self.reward[ind]).to(self.device),
            "not_done": torch.FloatTensor(self.not_done[ind]).to(self.device),
        }


class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim=64,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        device="cpu",
    ):
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim, max_action)
        self.actor = self.actor.to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = QNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_target_2 = copy.deepcopy(self.critic_2)
        self.critic_optimizer = torch.optim.Adam(
            [
                {"params": self.critic_1.parameters()},
                {"params": self.critic_2.parameters()},
            ],
            lr=3e-4,
        )

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.device = device

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()

    def eval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    def update(self, batch):
        self.total_it += 1

        state = batch["state"]
        action = batch["action"]
        next_state = batch["next_state"]
        reward = batch["reward"]
        not_done = batch["not_done"]

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_q1 = self.critic_target_1(next_state, next_action)
            target_q2 = self.critic_target_2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q)
        critic_loss += F.mse_loss(current_q2, target_q)
        loss_info = {"critic": critic_loss.item()}

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            loss_info["actor"] = actor_loss.item()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic_1.parameters(), self.critic_target_1.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.critic_2.parameters(), self.critic_target_2.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        return loss_info

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic_1.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])

        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_target_2 = copy.deepcopy(self.critic_2)
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])


class Trainer(object):
    def __init__(self, model, expr_dir, use_wandb):
        self.model = model

        # paths
        self.expr_dir = expr_dir
        self.model_path = os.path.join(expr_dir, "model.p")
        self.checkpoint_dir = os.path.join(expr_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # job specific loggers
        self.use_wandb = use_wandb
        train_log_path = init_logger(expr_dir, "train")
        train_eval_log_path = init_logger(expr_dir, "train-eval")
        if self.use_wandb:
            wandb.save(glob_str=train_log_path, policy="live")
            wandb.save(glob_str=train_eval_log_path, policy="live")

    def train(
        self,
        env_fn,
        replay_buffer,
        batch_size,
        start_time_steps,
        max_time_steps,
        expl_noise,
        seed,
        num_test_episodes=1,
        checkpoint_interval=1,
        eval_interval=1,
        device="cpu",
    ):
        # Evaluate untrained policy
        eval_info = self.eval(env_fn, seed, eval_episodes=num_test_episodes)
        evaluations = {
            "original-task-score": [
                np.mean([np.sum(_) for _ in eval_info["original-task-reward"]])
            ]
        }

        # initialize episode and trackers
        env = env_fn()
        state, done = env.reset(), False
        episode_reward = {"original-task": 0}
        episode_time_steps = 0
        episode_num = 0

        # interact and train
        for t in range(int(max_time_steps)):
            self.model.train()

            episode_time_steps += 1

            # Select action randomly or according to policy
            if t < start_time_steps:
                action = env.action_space.sample()
            else:
                action = self.model.select_action(np.array(state))
                action_noise = np.random.normal(
                    0, self.model.max_action * expl_noise, size=self.model.action_dim
                )
                action = (action + action_noise).clip(
                    -self.model.max_action, self.model.max_action
                )

            # Perform action
            next_state, reward, done, info = env.step(action)
            if episode_time_steps < env._max_episode_steps:
                done_bool = float(done)
            else:
                done_bool = 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            # transition to next-state
            state = next_state

            # store episode info
            episode_reward["original-task"] += reward

            if done:
                # +1 to account for 0 indexing.
                # +0 on ep_time_steps since it will increment +1
                # even if done=True

                # log to file/console
                episode_info = {
                    "episode-num": episode_num + 1,
                    "episode-time-steps": episode_time_steps,
                    "episode-org-score": episode_reward["original-task"],
                }
                log(
                    {
                        "env-steps": t + 1,
                        **{f"train-data/{k}": v for k, v in episode_info.items()},
                    },
                    logging.getLogger("train-eval"),
                    use_wandb=self.use_wandb,
                )

                # Reset environment
                state, done = env.reset(), False
                episode_reward = {"original-task": 0.0, "task": 0.0}
                episode_time_steps = 0
                episode_num += 1

            # Train agent after collecting sufficient data
            if t >= start_time_steps:
                batch = replay_buffer.sample(batch_size)
                self.model.update({k: v.to(device) for k, v in batch.items()})

            # save model
            if (t + 1) % checkpoint_interval == 0:
                self.save_checkpoint(time_step=t, info={"env-steps": t})

            # Evaluate policy
            if (t + 1) % eval_interval == 0:
                eval_info = self.eval(env_fn, seed, eval_episodes=num_test_episodes)

                evaluations["original-task-score"].append(
                    np.mean([np.sum(_) for _ in eval_info["original-task-reward"]])
                )

                np.save(os.path.join(self.expr_dir, "evaluations"), evaluations)

                # log to file/console
                log(
                    {
                        "env-steps": t + 1,
                        **{f"train-eval/{k}": v[-1] for k, v in evaluations.items()},
                    },
                    logging.getLogger("train-eval"),
                    use_wandb=self.use_wandb,
                )

    def save_checkpoint(self, time_step: int, info: dict = None):
        if info is None:
            info = {}

        # create checkpoint
        checkpoint = {"model": self.model.state_dict(), "info": info}

        # save locally
        torch.save(checkpoint, self.model_path)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{time_step}.p")
        torch.save(checkpoint, checkpoint_path)

        # save on wandb
        if self.use_wandb:
            wandb.save(glob_str=self.model_path, policy="now")
            wandb.save(glob_str=checkpoint_path, policy="now")

    def load_checkpoint(self):
        checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict=checkpoint["model"])

        return checkpoint["info"]

    def eval(self, env_fn, seed=0, seed_offset=100, eval_episodes=10, render=False):
        eval_env = env_fn()
        eval_env.seed(seed + seed_offset)

        eval_info = {"original-task-reward": [], "images": []}
        for _ in range(eval_episodes):
            # init logging structure
            for k in eval_info:
                eval_info[k].append([])

            state, done = eval_env.reset(), False
            while not done:
                if render:
                    img = eval_env.render(mode="rgb_array")
                    eval_info["images"][-1].append(img)

                action = self.model.select_action(np.array(state))
                next_state, reward, done, info = eval_env.step(action)

                # estimate custom reward
                eval_info["original-task-reward"][-1].append(reward)

                state = next_state

        return eval_info


def get_args():
    parser = argparse.ArgumentParser("TD3")
    parser.add_argument(
        "--job",
        default="train",
        type=str,
        choices=["train", "eval"],
        help="job to be performed",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="if enabled, disables usage of cuda",
    )

    # wandb setup
    wandb_args = parser.add_argument_group("wandb setup")
    wandb_args.add_argument(
        "--wandb-project-name",
        default="opcc-td3",
        help="name of the wandb project",
    )
    wandb_args.add_argument(
        "--use-wandb",
        action="store_true",
        help="if enabled, use Weight and bias visualization lib",
    )

    # paths
    path_args = parser.add_argument_group("paths setup")
    path_args.add_argument(
        "--result-dir",
        type=Path,
        default=Path("./results"),
        help="directory to store results",
    )

    # env-args
    env_args = parser.add_argument_group("env args")
    env_args.add_argument(
        "--env",
        default="hopper-medium-v2",
        help="name of the environment",
    )

    # training args
    train_args = parser.add_argument_group("train args")
    train_args.add_argument("--policy", default="TD3")
    train_args.add_argument("--seed", default=0, type=int)
    train_args.add_argument("--num-test-episodes", default=10, type=int)
    train_args.add_argument("--checkpoint-interval", default=500, type=int)
    train_args.add_argument("--log-interval", default=500, type=int)
    train_args.add_argument("--eval-interval", default=int(5e3), type=int)

    # TD3
    train_args.add_argument("--hidden-dim", default=256, type=int)
    train_args.add_argument("--start-time-steps", default=25e3, type=int)
    train_args.add_argument("--max-time-steps", default=1e6, type=int)
    train_args.add_argument("--expl-noise", default=0.1)
    train_args.add_argument("--batch-size", default=256, type=int)
    train_args.add_argument("--discount", default=0.99)
    train_args.add_argument("--tau", default=0.005)
    train_args.add_argument("--policy-noise", default=0.2)
    train_args.add_argument("--noise-clip", default=0.5)
    train_args.add_argument("--policy-frequency", default=2, type=int)
    train_args.add_argument("--lr", default=1e-4, type=float)

    # process arguments
    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    # setup device
    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
    elif (
        not args.no_cuda
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")

    # setup experiment path ( creates unique directory name based on hash of relevant
    # arguments)
    key_args = env_args._group_actions + train_args._group_actions
    sorted_args = sorted(key_args, key=lambda x: x.dest)
    hp_str = [str(vars(args)[hp.dest]) for hp in sorted_args]  # hyper-parameter-string
    hp_byte = bytes("".join(hp_str), "ascii")  # hyper-parameter-byte
    hp_hash = hashlib.sha224(hp_byte).hexdigest()  # hyper-parameter-hash
    args.expr_dir = os.path.join(
        args.result_dir,
        args.env,
        hp_hash,
    )  # experiment-directory`
    os.makedirs(args.expr_dir, exist_ok=True)
    print("# Experiment directory:", args.expr_dir)

    return args, wandb_args, path_args, env_args, train_args


def init_logger(base_path: str, name: str, file_mode="w", console=True, file=True):
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s]"
        "[%(filename)s>%(funcName)s] => %(message)s"
    )
    file_path = os.path.join(base_path, name + ".log")
    logger = logging.getLogger(name)
    logging.getLogger().handlers.clear()

    if console:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if file:
        handler = logging.FileHandler(file_path, mode=file_mode)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.DEBUG)

    return file_path


def log(info, logger, use_wandb=False):
    _msg = ""
    for _i, (k, v) in enumerate(info.items()):
        _msg += f" {k}:{v:<8.5f} "

        # add comma seperator for ease of log processing
        if _i < len(info.keys()) - 1:
            _msg += ","
    logger.info(_msg)

    if use_wandb:
        wandb.log(info)


def write_gif(episode_numpy_images, gif_path, save_mp4=True):
    episode_images = [Image.fromarray(_img) for _img in episode_numpy_images]
    # save as gif
    episode_images[0].save(
        gif_path,
        save_all=True,
        append_images=episode_images[1:],
        optimize=False,
        loop=1,
    )

    # save video
    if save_mp4:
        import moviepy.editor as mp

        clip = mp.VideoFileClip(gif_path)
        clip.write_videofile(gif_path.replace(".gif", ".mp4"))


def main():
    # get arguments and seeding
    args, _, _, env_args, train_args = get_args()

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # ##################################################################################
    # Create Loggers
    # ##################################################################################
    init_logger(args.expr_dir, "args")  # for logging args data
    logging.getLogger("args").info(
        f'\n\n{"=" * 100}\n'
        + f'{"Argument Name":<50}\tValue'
        + f'\n{"------------":<50}\t-----\n'
        + "\n".join(
            f"{arg_name:<50}\t{getattr(args, arg_name)}" for arg_name in vars(args)
        )
        + f'\n\n{"=" * 100}\n'
    )
    # save args
    pickle.dump(args, open(os.path.join(args.expr_dir, "args.p"), "wb"))

    # wandb (online logger)
    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, save_code=True, job_type=args.job)
        wandb.config.update(
            {x.dest: vars(args)[x.dest] for x in env_args._group_actions}
        )
        wandb.config.update(
            {x.dest: vars(args)[x.dest] for x in train_args._group_actions}
        )
        wandb.run.log_code(
            root=".",
            include_fn=lambda path: True,
            exclude_fn=lambda path: "results" in path
            or "__pycache__" in path
            or "datasets" in path
            or "wandb" in path,
        )

    # ##################################################################################
    # Setup env/ dataLoader
    # ##################################################################################
    def env_fn():
        if "antmaze" in args.env:
            _env = gym.make(args.env, reward_type="dense")
        else:
            _env = gym.make(args.env)

        # custom render function for adroit envs
        if "door" in args.env or "pen" in args.env:

            def _render_fn(mode):
                _frame_size = (640, 480)
                if mode == "rgb_array":
                    # list available camera names:
                    # >>> env.sim.model.camera_names
                    # ('fixed', 'vil_camera')
                    return _env.sim.render(
                        width=_frame_size[0],
                        height=_frame_size[1],
                        mode="offscreen",
                        camera_name=None,
                        device_id=0,
                    )
                else:
                    raise NotImplementedError()

        # custom render function for franka kitchen
        elif "kitchen" in args.env:
            from dm_control.mujoco import engine

            def _render_fn(mode="rgb_array"):
                if mode == "rgb_array":
                    camera = engine.MovableCamera(_env.sim, 1920, 2560)
                    camera.set_pose(
                        distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35
                    )
                    img = camera.render()
                    return img
                else:
                    raise NotImplementedError()

        _env.render = _render_fn

        return _env

    env = env_fn()
    env.action_space.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # ##################################################################################
    # Setup model
    # ##################################################################################
    kwargs = {
        "device": args.device,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # TD3
        "hidden_dim": args.hidden_dim,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_frequency,
    }

    model = TD3(**kwargs)

    # ##################################################################################
    # Job: Train Model
    # ##################################################################################
    trainer = Trainer(model=model, expr_dir=args.expr_dir, use_wandb=args.use_wandb)

    if args.job == "train":
        replay_buffer = ReplayBuffer(state_dim, action_dim)
        trainer.train(
            env_fn=env_fn,
            replay_buffer=replay_buffer,
            batch_size=args.batch_size,
            start_time_steps=args.start_time_steps,
            max_time_steps=args.max_time_steps,
            expl_noise=args.expl_noise,
            seed=args.seed,
            num_test_episodes=args.num_test_episodes,
            checkpoint_interval=args.checkpoint_interval,
            eval_interval=args.eval_interval,
            device=args.device,
        )

    # ##################################################################################
    # Job: Evaluate Model
    # ##################################################################################
    elif args.job == "eval":
        # job specific logger
        init_logger(args.expr_dir, "eval")

        # load model
        trainer.load_checkpoint()

        # eval
        eval_info = trainer.eval(
            env_fn=env_fn,
            seed=args.seed,
            seed_offset=100,
            eval_episodes=args.num_test_episodes,
            render=True,
        )

        # save video of 1st episode
        write_gif(
            eval_info["images"][0],
            gif_path=os.path.join(args.expr_dir, "eval_episode.gif"),
            save_mp4=True,
        )
        eval_info.pop("images")

        # log to file/console
        log(
            {
                f"eval/{k}": np.mean([np.sum(_) for _ in v])
                for k, v in eval_info.items()
            },
            logging.getLogger("eval"),
        )

    else:
        raise ValueError(f"{args.job} is not supported")

    # safe-finish
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
