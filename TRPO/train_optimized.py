"""
TRPO 优化版训练脚本

主要优化:
1. 多环境采样（可选 SubprocVecEnv 真并行）
2. 奖励归一化 - 稳定训练过程
3. 更大的熵系数 (0.05) - 鼓励探索
4. 放宽 KL 约束 (0.02) - 允许更大步长更新
5. 增加 rollout_steps (4096) - 每次更新收集更多数据
6. 熵系数衰减 - 从探索逐渐过渡到利用
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import argparse

from trpo import TRPOAgent
from trpo.rollout_buffer import RolloutBuffer
from trpo.vec_env import DummyVecEnv, SubprocVecEnv, FrameStackVecEnv
from envs.games.breakout import BreakoutEnv


def make_env():
    """创建单个环境实例"""
    return BreakoutEnv(frame_skip=4)


def evaluate(agent, env_fn, n_episodes=5):
    """评估智能体性能"""
    env = env_fn()

    # 帧堆叠
    frames = deque(maxlen=4)

    total_rewards = []

    for _ in range(n_episodes):
        frame = env.reset()
        for _ in range(4):
            frames.append(frame)
        state = np.array(frames, dtype=np.uint8)

        episode_reward = 0
        done = False

        while not done:
            action, _, _ = agent.select_action(state, eval_mode=True)
            frame, reward, done, _ = env.step(action)
            frames.append(frame)
            state = np.array(frames, dtype=np.uint8)
            episode_reward += reward

        total_rewards.append(episode_reward)

    if hasattr(env, 'close'):
        env.close()

    return np.mean(total_rewards), np.std(total_rewards)


def train(args):
    """主训练循环"""

    # ========== 创建多环境 ==========
    print(f"创建 {args.n_envs} 个并行环境...")
    env_fns = [make_env for _ in range(args.n_envs)]
    if args.use_subproc:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    env = FrameStackVecEnv(vec_env, n_frames=4)
    print(f"VecEnv: {vec_env.__class__.__name__}")

    n_actions = vec_env.n_actions
    print(f"动作空间: {n_actions}")

    # ========== 创建智能体 ==========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    agent = TRPOAgent(
        n_actions=n_actions,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_kl=args.max_kl,
        damping=args.damping,
        cg_iters=args.cg_iters,
        backtrack_iters=args.backtrack_iters,
        backtrack_coeff=args.backtrack_coeff,
        critic_lr=args.critic_lr,
        critic_train_iters=args.critic_train_iters,
        entropy_coef=args.entropy_coef,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        device=device,
    )

    # 创建支持多环境的 Rollout Buffer
    rollout_buffer = RolloutBuffer(
        buffer_size=args.rollout_steps,
        state_shape=(4, 84, 84),
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=device,
        n_envs=args.n_envs,
        normalize_rewards=True
    )

    print(f"\nActor 参数量: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"Critic 参数量: {sum(p.numel() for p in agent.critic.parameters()):,}")

    # ========== 训练记录 ==========
    episode_rewards = []
    eval_rewards = []
    policy_losses = []
    kl_divergences = []
    entropies = []
    step_fractions = []
    line_search_successes = []
    policy_update_rates = []
    max_kl_history = []
    damping_history = []
    consecutive_line_search_failures = 0

    # 每个环境的 episode 奖励跟踪
    env_episode_rewards = np.zeros(args.n_envs)

    # 熵系数衰减
    current_entropy_coef = args.entropy_coef

    # ========== 开始训练 ==========
    os.makedirs('checkpoints', exist_ok=True)

    states = env.reset()  # (n_envs, 4, 84, 84)
    total_steps = 0
    n_updates = 0
    best_eval_reward = -float('inf')
    start_time = time.time()

    print(f"\n开始训练 TRPO (优化版)...")
    print(f"总步数: {args.total_steps:,}")
    print(f"每次更新步数: {args.rollout_steps * args.n_envs}")
    print(f"预计更新次数: {args.total_steps // (args.rollout_steps * args.n_envs)}")
    print("-" * 60)

    pbar = tqdm(total=args.total_steps, desc="Training")

    while total_steps < args.total_steps:
        # 收集 rollout 数据
        rollout_buffer.reset()

        for step in range(args.rollout_steps):
            # 批量选择动作
            actions, log_probs, values = agent.select_action_batch(states)

            # 环境步进
            next_states, rewards, dones, infos = env.step(actions)

            # 存储经验
            rollout_buffer.add_batch(
                states=states,
                actions=actions,
                rewards=rewards,
                values=values,
                log_probs=log_probs,
                dones=dones
            )

            # 更新 episode 奖励跟踪
            env_episode_rewards += rewards
            for i, done in enumerate(dones):
                if done:
                    episode_rewards.append(env_episode_rewards[i])
                    env_episode_rewards[i] = 0

            states = next_states
            total_steps += args.n_envs
            pbar.update(args.n_envs)

        # 同步到 agent，确保 checkpoint 中总步数正确
        agent.total_steps = total_steps

        # 计算最后状态的价值 (用于 GAE bootstrap)
        with torch.no_grad():
            state_tensor = torch.from_numpy(states).to(device)
            last_values = agent.critic(state_tensor).squeeze(-1).cpu().numpy()

        # 计算 GAE
        rollout_buffer.compute_gae_batch(last_values)

        # 临时替换 agent 的 buffer
        original_buffer = agent.rollout_buffer
        agent.rollout_buffer = rollout_buffer

        # 更新熵系数
        agent.entropy_coef = current_entropy_coef

        # TRPO 更新
        update_info = agent.update()

        # 恢复原 buffer
        agent.rollout_buffer = original_buffer

        # 记录训练信息
        policy_losses.append(update_info['policy_loss'])
        kl_divergences.append(update_info['kl_divergence'])
        entropies.append(update_info['entropy'])
        step_fractions.append(update_info.get('step_fraction', 0.0))
        ls_success = bool(update_info.get('line_search_success', False))
        policy_updated = bool(update_info.get('policy_updated', ls_success))
        line_search_successes.append(1.0 if ls_success else 0.0)
        policy_update_rates.append(1.0 if policy_updated else 0.0)
        n_updates += 1

        # 自适应稳定器: 连续线性搜索失败时，收紧 trust region 并增加阻尼
        if ls_success:
            consecutive_line_search_failures = 0
        else:
            consecutive_line_search_failures += 1

        if consecutive_line_search_failures >= args.ls_fail_threshold:
            old_max_kl = agent.max_kl
            old_damping = agent.damping
            agent.max_kl = max(args.min_max_kl, agent.max_kl * args.max_kl_decay)
            agent.damping = min(args.max_damping, agent.damping * args.damping_growth)
            consecutive_line_search_failures = 0

            tqdm.write(
                f"  [AutoTune] 连续线性搜索失败达到阈值，"
                f"max_kl: {old_max_kl:.4f} -> {agent.max_kl:.4f}, "
                f"damping: {old_damping:.4f} -> {agent.damping:.4f}"
            )

        max_kl_history.append(agent.max_kl)
        damping_history.append(agent.damping)

        # 熵系数衰减
        current_entropy_coef = max(
            args.min_entropy_coef,
            current_entropy_coef * args.entropy_decay
        )
        # 同步到 agent，确保保存时使用最新值
        agent.entropy_coef = current_entropy_coef

        # 定期输出
        if n_updates % args.log_interval == 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            recent_ls_success = np.mean(line_search_successes[-args.log_interval:]) * 100
            recent_policy_updated = np.mean(policy_update_rates[-args.log_interval:]) * 100
            recent_step_fraction = np.mean(step_fractions[-args.log_interval:])

            tqdm.write(
                f"Updates: {n_updates} | Steps: {total_steps:,} | "
                f"Avg Reward: {avg_reward:.1f} | "
                f"KL: {update_info['kl_divergence']:.4f} | "
                f"StepFrac: {recent_step_fraction:.3f} | "
                f"LS_OK: {recent_ls_success:.0f}% | "
                f"PolicyUpd: {recent_policy_updated:.0f}% | "
                f"LS_Mode: {update_info.get('line_search_mode', 'n/a')} | "
                f"max_kl: {agent.max_kl:.4f} | "
                f"damping: {agent.damping:.4f} | "
                f"Entropy: {update_info['entropy']:.3f} | "
                f"FPS: {fps:.0f}"
            )

        # 评估
        if n_updates % args.eval_interval == 0:
            eval_mean, eval_std = evaluate(agent, make_env, n_episodes=5)
            eval_rewards.append((total_steps, eval_mean, eval_std))

            tqdm.write(f"  [Eval] Reward: {eval_mean:.1f} ± {eval_std:.1f}")

            if eval_mean > best_eval_reward:
                best_eval_reward = eval_mean
                agent.save('checkpoints/trpo_optimized_best.pt')
                tqdm.write(f"  *** New best: {eval_mean:.1f} ***")

        # 保存检查点
        if n_updates % args.save_interval == 0:
            agent.save(f'checkpoints/trpo_optimized_{total_steps}.pt')

    pbar.close()
    env.close()

    # ========== 绘制训练曲线 ==========
    print("\n绘制训练曲线...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Episode Rewards
    ax = axes[0, 0]
    ax.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    if len(episode_rewards) >= 100:
        smoothed = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        ax.plot(range(99, len(episode_rewards)), smoothed, label='Moving Avg (100)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Evaluation Rewards
    ax = axes[0, 1]
    if eval_rewards:
        steps, means, stds = zip(*eval_rewards)
        ax.errorbar(steps, means, yerr=stds, capsize=3, marker='o')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')
    ax.set_title(f'Evaluation Rewards (Best: {best_eval_reward:.1f})')
    ax.grid(True, alpha=0.3)

    # KL Divergence
    ax = axes[1, 0]
    ax.plot(kl_divergences, label='Observed KL')
    if max_kl_history:
        ax.plot(max_kl_history, color='r', linestyle='--', label='Max KL (Adaptive)')
    else:
        ax.axhline(y=args.max_kl, color='r', linestyle='--', label=f'Max KL = {args.max_kl}')
    ax.set_xlabel('Update')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence (Trust Region)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 1]
    ax.plot(entropies)
    ax.set_xlabel('Update')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves_optimized.png', dpi=150)
    plt.show()

    print(f"\n训练完成!")
    print(f"总步数: {total_steps:,}")
    print(f"总更新次数: {n_updates}")
    print(f"最佳评估奖励: {best_eval_reward:.1f}")
    if line_search_successes:
        print(f"线性搜索成功率: {np.mean(line_search_successes) * 100:.1f}%")
    if policy_update_rates:
        print(f"策略更新率(accept+fallback): {np.mean(policy_update_rates) * 100:.1f}%")
    print(f"最终 max_kl: {agent.max_kl:.4f}")
    print(f"最终 damping: {agent.damping:.4f}")
    print(f"总耗时: {(time.time() - start_time) / 3600:.2f} 小时")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRPO Optimized Training')

    # 训练参数
    parser.add_argument('--total_steps', type=int, default=10_000_000, help='总训练步数')
    parser.add_argument('--n_envs', type=int, default=8, help='并行环境数量')
    parser.add_argument('--use_subproc', action='store_true', help='使用 SubprocVecEnv 真并行采样')
    parser.add_argument('--rollout_steps', type=int, default=512, help='每个环境每次更新的步数')

    # TRPO 超参数 (优化后)
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.98, help='GAE lambda')
    parser.add_argument('--max_kl', type=float, default=0.02, help='最大 KL 散度约束')
    parser.add_argument('--damping', type=float, default=0.05, help='Fisher 矩阵阻尼')
    parser.add_argument('--cg_iters', type=int, default=15, help='共轭梯度迭代次数')
    parser.add_argument('--backtrack_iters', type=int, default=15, help='线性搜索回溯次数')
    parser.add_argument('--backtrack_coeff', type=float, default=0.6, help='回溯系数')

    # Critic 参数
    parser.add_argument('--critic_lr', type=float, default=5e-4, help='Critic 学习率')
    parser.add_argument('--critic_train_iters', type=int, default=10, help='Critic 训练迭代次数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')

    # 探索参数
    parser.add_argument('--entropy_coef', type=float, default=0.05, help='初始熵系数')
    parser.add_argument('--entropy_decay', type=float, default=0.9995, help='熵系数衰减率')
    parser.add_argument('--min_entropy_coef', type=float, default=0.01, help='最小熵系数')

    # 稳定性参数
    parser.add_argument('--ls_fail_threshold', type=int, default=20, help='连续线性搜索失败阈值')
    parser.add_argument('--max_kl_decay', type=float, default=0.8, help='触发自适应时 max_kl 衰减系数')
    parser.add_argument('--min_max_kl', type=float, default=0.005, help='max_kl 最小值')
    parser.add_argument('--damping_growth', type=float, default=1.2, help='触发自适应时 damping 增长系数')
    parser.add_argument('--max_damping', type=float, default=0.2, help='damping 最大值')

    # 日志参数
    parser.add_argument('--log_interval', type=int, default=10, help='日志输出间隔 (更新次数)')
    parser.add_argument('--eval_interval', type=int, default=50, help='评估间隔 (更新次数)')
    parser.add_argument('--save_interval', type=int, default=200, help='保存间隔 (更新次数)')

    args = parser.parse_args()

    # 打印配置
    print("=" * 60)
    print("TRPO 优化版训练配置")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 60)

    train(args)
