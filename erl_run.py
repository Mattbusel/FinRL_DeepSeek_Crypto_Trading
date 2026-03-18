"""ElegantRL multi-process training runner for the LARSA trading system.

This module provides the :func:`train_agent` and :func:`valid_agent` entry
points as well as the :class:`Learner`, :class:`Worker`, and
:class:`EvaluatorProc` multiprocessing classes that together implement
parallelised off-policy / on-policy DRL training.

Typical usage::

    python erl_run.py          # CPU
    python erl_run.py 0        # GPU 0
"""

from __future__ import annotations

import os
import time
from multiprocessing import Process
from typing import Any

import numpy as np
import torch
import torch.nn.modules.container as container

import erl_net
from erl_config import Config, build_env
from erl_evaluator import Evaluator
from erl_replay_buffer import ReplayBuffer
from logger import get_logger
from trade_simulator import EvalTradeSimulator, TradeSimulator

log = get_logger(__name__)

# Register safe globals required for torch.load with weights_only=True.
torch.serialization.add_safe_globals(
    [
        erl_net.QNetBase,
        erl_net.QNetTwin,
        erl_net.QNetTwinDuel,
        container.Sequential,
        torch.nn.modules.linear.Linear,
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.activation.Softmax,
    ]
)

if os.name == "nt":
    # Fix Anaconda OMP duplicate-DLL bug on Windows NT.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Learner(Process):
    """Multiprocessing learner that updates the policy network.

    The Learner receives trajectory batches from :class:`Worker` processes,
    updates the agent, and relays actors and training statistics to
    :class:`EvaluatorProc`.

    Args:
        learner_pipe: Bidirectional pipe endpoint for the Learner side.
        worker_pipes: List of pipe endpoints for each Worker process.
        evaluator_pipe: Pipe endpoint shared with :class:`EvaluatorProc`.
        args: :class:`erl_config.Config` configuration object.
    """

    def __init__(
        self,
        learner_pipe: tuple[Any, Any],
        worker_pipes: list[tuple[Any, Any]],
        evaluator_pipe: tuple[Any, Any],
        args: Config,
    ) -> None:
        super().__init__()
        self.recv_pipe = learner_pipe[0]
        self.send_pipes = [wp[1] for wp in worker_pipes]
        self.eval_pipe = evaluator_pipe[1]
        self.args = args

    def run(self) -> None:
        """Execute the learner training loop inside the child process."""
        args = self.args
        torch.set_grad_enabled(False)

        agent = args.agent_class(
            args.net_dims,
            args.state_dim,
            args.action_dim,
            gpu_id=args.gpu_id,
            args=args,
        )
        agent.save_or_load_agent(args.cwd, if_save=False)

        if args.if_off_policy:
            buffer: ReplayBuffer | list = ReplayBuffer(
                gpu_id=args.gpu_id,
                num_seqs=args.num_envs * args.num_workers,
                max_size=args.buffer_size,
                state_dim=args.state_dim,
                action_dim=1 if args.if_discrete else args.action_dim,
            )
        else:
            buffer = []

        if_off_policy = args.if_off_policy
        if_save_buffer = args.if_save_buffer
        num_workers = args.num_workers
        num_envs = args.num_envs
        state_dim = args.state_dim
        action_dim = args.action_dim
        horizon_len = args.horizon_len
        num_seqs = args.num_envs * args.num_workers
        num_steps = args.horizon_len * args.num_workers
        cwd = args.cwd
        del args

        agent.last_state = torch.empty(
            (num_seqs, state_dim), dtype=torch.float32, device=agent.device
        )

        states = torch.empty(
            (horizon_len, num_seqs, state_dim), dtype=torch.float32, device=agent.device
        )
        actions = torch.empty(
            (horizon_len, num_seqs, action_dim),
            dtype=torch.float32,
            device=agent.device,
        )
        rewards = torch.empty((horizon_len, num_seqs), dtype=torch.float32, device=agent.device)
        undones = torch.empty((horizon_len, num_seqs), dtype=torch.bool, device=agent.device)
        if if_off_policy:
            buffer_items_tensor: tuple = (states, actions, rewards, undones)
        else:
            logprobs = torch.empty(
                (horizon_len, num_seqs), dtype=torch.float32, device=agent.device
            )
            buffer_items_tensor = (states, actions, logprobs, rewards, undones)

        if_train = True
        while if_train:
            for send_pipe in self.send_pipes:
                send_pipe.send(agent.act)

            for _ in range(num_workers):
                worker_id, buffer_items, last_state = self.recv_pipe.recv()
                buf_i = worker_id * num_envs
                buf_j = buf_i + num_envs
                for buffer_item, buffer_tensor in zip(buffer_items, buffer_items_tensor):
                    buffer_tensor[:, buf_i:buf_j] = buffer_item
                agent.last_state[buf_i:buf_j] = last_state

            if if_off_policy:
                buffer.update(buffer_items_tensor)
            else:
                buffer[:] = buffer_items_tensor

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            if self.eval_pipe.poll():
                if_train = self.eval_pipe.recv()
                actor: Any | None = agent.act
            else:
                actor = None

            exp_r = float(buffer_items_tensor[2].mean().item())
            self.eval_pipe.send((actor, num_steps, exp_r, logging_tuple))

        for send_pipe in self.send_pipes:
            send_pipe.send(None)

        agent.save_or_load_agent(cwd, if_save=True)
        if if_save_buffer and hasattr(buffer, "save_or_load_history"):
            log.info("buffer.saving", cwd=cwd)
            buffer.save_or_load_history(cwd, if_save=True)
            log.info("buffer.saved", cwd=cwd)


class Worker(Process):
    """Multiprocessing rollout worker that collects environment trajectories.

    Args:
        worker_pipe: Pipe endpoint for receiving actors from the Learner.
        learner_pipe: Pipe endpoint for sending trajectory batches to the
            Learner.
        worker_id: Integer identifier of this worker (used for buffer
            slicing on the Learner side).
        args: :class:`erl_config.Config` configuration object.
    """

    def __init__(
        self,
        worker_pipe: tuple[Any, Any],
        learner_pipe: tuple[Any, Any],
        worker_id: int,
        args: Config,
    ) -> None:
        super().__init__()
        self.recv_pipe = worker_pipe[0]
        self.send_pipe = learner_pipe[1]
        self.worker_id = worker_id
        self.args = args

    def run(self) -> None:
        """Execute the rollout collection loop inside the child process."""
        args = self.args
        worker_id = self.worker_id
        torch.set_grad_enabled(False)

        env = build_env(args.env_class, args.env_args, args.gpu_id)

        agent = args.agent_class(
            args.net_dims,
            args.state_dim,
            args.action_dim,
            gpu_id=args.gpu_id,
            args=args,
        )
        agent.save_or_load_agent(args.cwd, if_save=False)

        state = env.reset()
        if args.num_envs == 1:
            assert state.shape == (args.state_dim,)
            assert isinstance(state, np.ndarray)
            state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        else:
            assert state.shape == (args.num_envs, args.state_dim)
            assert isinstance(state, torch.Tensor)
            state = state.to(agent.device)
        assert state.shape == (args.num_envs, args.state_dim)
        assert isinstance(state, torch.Tensor)
        agent.last_state = state.detach()

        horizon_len = args.horizon_len
        if args.if_off_policy:
            buffer_items = agent.explore_env(env, args.horizon_len, if_random=True)
            self.send_pipe.send((worker_id, buffer_items, agent.last_state))

        del args

        while True:
            actor = self.recv_pipe.recv()
            if actor is None:
                break
            agent.act = actor
            buffer_items = agent.explore_env(env, horizon_len)
            self.send_pipe.send((worker_id, buffer_items, agent.last_state))

        if hasattr(env, "close"):
            env.close()


class EvaluatorProc(Process):
    """Multiprocessing evaluator that periodically tests the current policy.

    Args:
        evaluator_pipe: Pipe endpoint for communicating with the Learner.
        args: :class:`erl_config.Config` configuration object.
    """

    def __init__(
        self,
        evaluator_pipe: tuple[Any, Any],
        args: Config,
    ) -> None:
        super().__init__()
        self.pipe = evaluator_pipe[0]
        self.args = args

    def run(self) -> None:
        """Execute the evaluation loop inside the child process."""
        args = self.args
        torch.set_grad_enabled(False)

        eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
        eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
        eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
        evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args)

        cwd = args.cwd
        break_step = args.break_step
        device = torch.device(
            f"cuda:{args.gpu_id}" if (torch.cuda.is_available() and (args.gpu_id >= 0)) else "cpu"
        )
        del args

        if_train = True
        while if_train:
            actor, steps, exp_r, logging_tuple = self.pipe.recv()

            if actor is None:
                evaluator.total_step += steps
            else:
                actor = actor.to(device)
                evaluator.evaluate_and_save(actor, steps, exp_r, logging_tuple)

            if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))
            self.pipe.send(if_train)

        evaluator.save_training_curve_jpg()
        elapsed = time.time() - evaluator.start_time
        log.info("training.complete", elapsed_s=round(elapsed), cwd=cwd)

        if hasattr(eval_env, "close"):
            eval_env.close()


def train_agent(args: Config) -> None:
    """Run single-process RL training.

    Initialises the environment, agent, replay buffer, and evaluator, then
    iterates the training loop until ``break_step`` is reached or a ``stop``
    sentinel file is created under ``args.cwd``.

    Args:
        args: Fully populated :class:`erl_config.Config` instance.
    """
    args.init_before_training()
    torch.set_grad_enabled(False)

    env = build_env(args.env_class, args.env_args, args.gpu_id)

    agent = args.agent_class(
        args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args
    )
    agent.save_or_load_agent(args.cwd, if_save=False)

    state = env.reset()
    if args.num_envs == 1:
        assert state.shape == (args.state_dim,)
        assert isinstance(state, np.ndarray)
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    else:
        if state.shape != (args.num_envs, args.state_dim):
            raise ValueError(
                f"state.shape == (num_envs, state_dim): {state.shape, args.num_envs, args.state_dim}"
            )
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"isinstance(state, torch.Tensor): {repr(state)}")
        state = state.to(agent.device)
    assert state.shape == (args.num_envs, args.state_dim)
    assert isinstance(state, torch.Tensor)
    agent.last_state = state.detach()

    if args.if_off_policy:
        buffer: ReplayBuffer | list = ReplayBuffer(
            gpu_id=args.gpu_id,
            num_seqs=args.num_envs,
            max_size=args.buffer_size,
            state_dim=args.state_dim,
            action_dim=1 if args.if_discrete else args.action_dim,
        )
        buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
        buffer.update(buffer_items)
    else:
        buffer = []

    eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
    eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
    eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
    evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args)

    cwd = args.cwd
    break_step = args.break_step
    horizon_len = args.horizon_len
    if_off_policy = args.if_off_policy
    if_save_buffer = args.if_save_buffer
    del args

    import torch as th

    if_train = True
    while if_train:
        buffer_items = agent.explore_env(env, horizon_len)

        action = buffer_items[1].flatten()
        action_count = th.bincount(action).data.cpu().numpy() / action.shape[0]
        action_count = np.ceil(action_count * 998).astype(int)

        position = buffer_items[0][:, :, 0].long().flatten().float()
        position_count = torch.histc(position, bins=env.max_position * 2 + 1, min=-2, max=2)
        position_count = position_count.data.cpu().numpy() / position.shape[0]
        position_count = np.ceil(position_count * 998).astype(int)

        log.debug(
            "train.step",
            action_dist=action_count.tolist(),
            position_dist=position_count.tolist(),
        )

        exp_r = buffer_items[2].mean().item()
        if if_off_policy:
            buffer.update(buffer_items)
        else:
            buffer[:] = buffer_items

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        evaluator.evaluate_and_save(
            actor=agent.act, steps=horizon_len, exp_r=exp_r, logging_tuple=logging_tuple
        )
        if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))

    elapsed = time.time() - evaluator.start_time
    log.info("training.complete", elapsed_s=round(elapsed), cwd=cwd)

    if hasattr(env, "close"):
        env.close()
    evaluator.save_training_curve_jpg()
    agent.save_or_load_agent(cwd, if_save=True)
    if if_save_buffer and hasattr(buffer, "save_or_load_history"):
        buffer.save_or_load_history(cwd, if_save=True)


def valid_agent(args: Config) -> None:
    """Run a single validation episode using the best saved actor checkpoint.

    Loads the newest actor file from the checkpoint directory, steps through
    a full :class:`EvalTradeSimulator` episode, and saves position arrays to
    ``erl_run_valid_position.npy``.

    Args:
        args: Configuration object; ``eval_env_class`` and ``eval_env_args``
            must be set.
    """

    cwd = f"{args.env_name}_D3QN_{args.gpu_id}"
    thresh = 0.001

    eval_env_class = args.eval_env_class
    eval_env_args = args.eval_env_args

    agent_class = args.agent_class
    net_dims = args.net_dims

    sim: TradeSimulator = build_env(eval_env_class, eval_env_args, gpu_id=args.gpu_id)

    state_dim = eval_env_args["state_dim"]
    action_dim = eval_env_args["action_dim"]
    agent = agent_class(net_dims, state_dim, action_dim, gpu_id=args.gpu_id)

    agent.save_or_load_agent(cwd=cwd, if_save=False)
    agent_path = sorted(
        [f for f in os.listdir(cwd) if len(f) == len("actor_00154050_000.664.pth")]
    )[-1]
    agent.act.load_state_dict(
        torch.load(
            f"{cwd}/{agent_path}", map_location=agent.device, weights_only=False
        ).state_dict()  # nosec B614
    )

    actor = agent.act
    device = agent.device
    del agent

    state = sim.reset()

    position_ary: list = []
    trade_ary: list = []
    q_values_ary: list = []

    for i in range(sim.max_step):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device)
        tensor_q_values = actor(tensor_state)
        tensor_action = tensor_q_values.argmax(dim=1)

        mask_zero_position = sim.position.eq(0)
        mask_q_values = (tensor_q_values.max(dim=1)[0] - tensor_q_values.mean(dim=1)).lt(
            torch.where(tensor_action.eq(2), thresh, thresh)
        )
        mask = torch.logical_and(mask_zero_position, mask_q_values)
        tensor_action[mask] = 1

        action = tensor_action.detach().cpu().unsqueeze(1)
        state, reward, done, info_dict = sim.step(action=action)

        trade_ary.append(sim.action_int.data.cpu().numpy())
        position_ary.append(sim.position.data.cpu().numpy())
        q_values_ary.append(tensor_q_values.data.cpu().numpy())

    save_path = "erl_run_valid_position.npy"
    position_ary_np = np.stack(position_ary, axis=0)
    np.save(save_path, position_ary_np)
    log.info("valid.position_saved", path=save_path)


def run() -> None:
    """Configure and launch single-process training followed by validation.

    GPU ID is read from the first CLI argument (default: -1 for CPU).
    """
    import sys

    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1

    from erl_agent import AgentD3QN

    num_sims = 512
    num_ignore_step = 60
    max_position = 1
    step_gap = 2
    slippage = 7e-7

    max_step = (4800 - num_ignore_step) // step_gap

    env_args: dict = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": 2 + 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
    }
    args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.net_dims = (128, 128, 128)

    args.gamma = 0.995
    args.explore_rate = 0.005
    args.state_value_tau = 0.01
    args.soft_update_tau = 2e-6
    args.learning_rate = 2e-6
    args.batch_size = 512
    args.break_step = int(32e4)
    args.buffer_size = int(max_step * 32)
    args.repeat_times = 2
    args.horizon_len = int(max_step * 4)
    args.eval_per_step = int(max_step)
    args.num_workers = 1
    args.save_gap = 8

    args.eval_env_class = EvalTradeSimulator
    args.eval_env_args = env_args.copy()

    train_agent(args=args)
    valid_agent(args=args)


if __name__ == "__main__":
    run()
