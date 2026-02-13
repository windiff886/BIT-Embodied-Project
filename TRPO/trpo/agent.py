"""
TRPO 智能体 (Trust Region Policy Optimization)

实现 2015 年 Schulman 等人的 TRPO 论文核心算法:
- 自然策略梯度 (Natural Policy Gradient)
- 共轭梯度法 (Conjugate Gradient) 求解 Fisher 信息矩阵逆
- 线性搜索 (Line Search) 确保 KL 散度约束

论文: "Trust Region Policy Optimization"
https://arxiv.org/abs/1502.05477
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from .network import ActorNetwork, CriticNetwork
from .rollout_buffer import RolloutBuffer


def get_device() -> str:
    """自动检测最佳可用设备"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


@dataclass
class TRPOAgent:
    """
    TRPO (Trust Region Policy Optimization) 智能体

    核心思想:
    - 在策略更新时限制 KL 散度，确保新旧策略足够接近
    - 使用共轭梯度法高效求解自然梯度方向
    - 通过线性搜索找到满足约束的最大步长
    """

    # 环境参数
    n_actions: int

    # 超参数 (优化后)
    gamma: float = 0.99                    # 折扣因子
    gae_lambda: float = 0.98               # GAE lambda (增大以获得更长期的优势估计)
    max_kl: float = 0.02                   # 放宽 KL 约束，允许更大步长更新
    damping: float = 0.05                  # 降低阻尼系数，允许更激进的更新
    cg_iters: int = 15                     # 增加共轭梯度迭代次数，更精确求解
    backtrack_iters: int = 15              # 增加线性搜索回溯次数
    backtrack_coeff: float = 0.6           # 降低回溯系数，更快找到有效步长
    ls_accept_tolerance: float = 1e-4      # 线性搜索改进容忍度（允许极小负改进）
    ls_fallback_tolerance: float = 5e-4    # 线性搜索回退容忍度（选取最优可行步）

    # Critic 参数
    critic_lr: float = 5e-4                # Critic 学习率
    critic_train_iters: int = 10           # 增加 Critic 训练迭代次数
    entropy_coef: float = 0.05             # 大幅增加熵系数，鼓励更多探索
    entropy_decay: float = 0.9999          # 熵系数衰减率
    min_entropy_coef: float = 0.01         # 最小熵系数

    # 训练参数
    rollout_steps: int = 4096              # 增加每次更新收集的步数
    batch_size: int = 128                  # 增大批次大小

    # 设备
    device: str = 'cuda'

    # 内部状态
    actor: ActorNetwork = field(init=False)
    critic: CriticNetwork = field(init=False)
    critic_optimizer: optim.Optimizer = field(init=False)
    rollout_buffer: RolloutBuffer = field(init=False)
    total_steps: int = field(init=False, default=0)

    def __post_init__(self):
        """初始化网络和优化器"""
        # Actor 网络 (策略网络)
        self.actor = ActorNetwork(self.n_actions).to(self.device)

        # Critic 网络 (价值网络)
        self.critic = CriticNetwork().to(self.device)

        # 只有 Critic 使用梯度下降优化
        # Actor 使用 TRPO 的自然梯度更新
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr
        )

        # Rollout Buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.rollout_steps,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.device
        )

        self.total_steps = 0

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> Tuple[int, float, float]:
        """
        根据当前策略选择动作

        Args:
            state: 当前状态
            eval_mode: 是否为评估模式 (使用确定性策略)

        Returns:
            action: 选择的动作
            log_prob: 动作的 log 概率
            value: 状态价值估计
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

            # 获取动作概率
            action, log_prob = self.actor.get_action(state_tensor, deterministic=eval_mode)

            # 获取价值估计
            value = self.critic(state_tensor)

        return action.item(), log_prob.item(), value.item()

    def select_action_batch(
        self,
        states: np.ndarray,
        eval_mode: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量选择动作 (用于多环境并行)

        Args:
            states: 状态批次 (n_envs, *state_shape)
            eval_mode: 是否为评估模式

        Returns:
            actions: (n_envs,)
            log_probs: (n_envs,)
            values: (n_envs,)
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(states).to(self.device)

            # 获取动作和 log 概率
            actions, log_probs = self.actor.get_action(state_tensor, deterministic=eval_mode)

            # 获取价值估计
            values = self.critic(state_tensor).squeeze(-1)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """存储一步经验到 rollout buffer"""
        self.rollout_buffer.add(state, action, reward, value, log_prob, done)
        self.total_steps += 1

    def finish_path(self, last_value: float = 0.0):
        """完成当前轨迹"""
        self.rollout_buffer.finish_path(last_value)

    def _get_flat_params(self, model: nn.Module) -> torch.Tensor:
        """获取模型参数的扁平化向量"""
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    def _set_flat_params(self, model: nn.Module, flat_params: torch.Tensor):
        """设置模型参数从扁平化向量"""
        pointer = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[pointer:pointer + numel].view_as(p))
            pointer += numel

    def _get_flat_grads(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        create_graph: bool = False
    ) -> torch.Tensor:
        """获取损失关于模型参数的扁平化梯度"""
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=create_graph)
        return torch.cat([g.view(-1) for g in grads])

    def _fisher_vector_product(
        self,
        vector: torch.Tensor,
        states: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Fisher 信息矩阵与向量的乘积: F @ v

        使用 Hessian-vector product 技巧，避免显式计算 Fisher 矩阵

        Fisher 矩阵定义为 KL 散度的 Hessian:
        F = E[∇log π(a|s) ∇log π(a|s)^T]
        """
        # 确保 vector 不需要梯度追踪
        vector = vector.detach()

        # 计算当前策略的 log 概率
        log_probs = self.actor(states)
        probs = torch.exp(log_probs)

        # KL 散度 (新策略 vs 旧策略，这里旧策略 = 新策略)
        # KL(π_old || π_new) = Σ π_old * (log π_old - log π_new)
        # 在 π_old = π_new 时，KL = 0，但我们需要它的 Hessian
        kl = (probs * (log_probs - log_probs.detach())).sum(dim=-1).mean()

        # 计算 KL 关于参数的梯度
        kl_grad = self._get_flat_grads(kl, self.actor, create_graph=True)

        # 计算 Hessian-vector product: H @ v = ∇(∇KL · v)
        kl_grad_v = (kl_grad * vector).sum()
        hessian_v = torch.autograd.grad(kl_grad_v, self.actor.parameters(), retain_graph=True)
        flat_hessian_v = torch.cat([g.contiguous().view(-1) for g in hessian_v])

        # 添加阻尼以确保数值稳定性，并 detach 防止计算图累积
        return flat_hessian_v.detach() + self.damping * vector

    def _conjugate_gradient(
        self,
        fvp_fn,
        b: torch.Tensor,
        nsteps: int = 10,
        residual_tol: float = 1e-10
    ) -> torch.Tensor:
        """
        共轭梯度法求解 Fx = b

        用于求解自然梯度方向: x = F^{-1} g
        其中 F 是 Fisher 信息矩阵，g 是策略梯度

        Args:
            fvp_fn: Fisher-vector product 函数
            b: 右侧向量 (策略梯度)
            nsteps: 最大迭代次数
            residual_tol: 残差容忍度

        Returns:
            近似解 x ≈ F^{-1} b
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)

        for _ in range(nsteps):
            fvp = fvp_fn(p)
            denom = torch.dot(p, fvp)
            denom_value = denom.item()
            if (not np.isfinite(denom_value)) or denom_value <= 1e-12:
                break

            alpha = rdotr / denom
            if not torch.isfinite(alpha):
                break

            x += alpha * p
            r -= alpha * fvp
            new_rdotr = torch.dot(r, r)
            if not torch.isfinite(new_rdotr):
                break

            if new_rdotr < residual_tol:
                break

            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def _compute_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算策略损失 (surrogate objective) 和熵

        L(θ) = E[π(a|s) / π_old(a|s) * A(s, a)] + entropy_coef * H(π)
        """
        new_log_probs, entropy = self.actor.evaluate_actions(states, actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        policy_loss = (ratio * advantages).mean()
        mean_entropy = entropy.mean()
        return policy_loss, mean_entropy

    def _compute_kl(
        self,
        states: torch.Tensor,
        old_log_probs_all: torch.Tensor
    ) -> torch.Tensor:
        """
        计算新旧策略之间的 KL 散度

        KL(π_old || π_new) = Σ π_old * (log π_old - log π_new)
        """
        new_log_probs = self.actor(states)
        old_probs = torch.exp(old_log_probs_all)
        kl = (old_probs * (old_log_probs_all - new_log_probs)).sum(dim=-1).mean()
        return kl

    def update(self) -> dict:
        """
        执行 TRPO 策略更新

        Returns:
            包含训练统计信息的字典
        """
        # 获取 rollout 数据
        states, actions, old_log_probs, advantages, returns = self.rollout_buffer.get()

        # ========== Actor 更新 (TRPO) ==========

        # 保存旧参数
        old_params = self._get_flat_params(self.actor)

        # 获取旧策略的完整 log 概率分布 (用于 KL 计算)
        with torch.no_grad():
            old_log_probs_all = self.actor(states)

        # 计算策略梯度 (加入熵正则化鼓励探索)
        policy_loss, entropy = self._compute_policy_loss(states, actions, old_log_probs, advantages)
        # TRPO 方向与线性搜索必须优化同一个目标，避免“方向与验收条件不一致”
        policy_objective = policy_loss + self.entropy_coef * entropy
        policy_loss_value = policy_loss.item()  # 仅用于日志
        entropy_value = entropy.item()
        policy_objective_value = policy_objective.item()
        policy_grad = self._get_flat_grads(policy_objective, self.actor, create_graph=False)

        # 关键修复：detach 策略梯度，防止计算图在 CG 迭代中累积
        policy_grad = policy_grad.detach()

        # 释放 policy_loss 的计算图
        del policy_loss, entropy, policy_objective
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 使用共轭梯度法计算自然梯度方向
        # natural_grad = F^{-1} @ policy_grad
        def fvp_fn(v):
            return self._fisher_vector_product(v, states)

        natural_grad = self._conjugate_gradient(fvp_fn, policy_grad, nsteps=self.cg_iters)

        # 线性搜索，找到满足 KL 约束的最大步长
        step_frac = 1.0
        line_search_success = False   # 严格意义上的线性搜索成功（accept）
        policy_updated = False        # 策略参数是否实际更新（accept/fallback）
        accepted_step_fraction = 0.0
        line_search_iters = 0
        kl_value = float('nan')
        line_search_mode = 'revert'
        best_feasible_params = None
        best_feasible_kl = float('nan')
        best_feasible_improve = -float('inf')
        best_feasible_step_frac = 0.0

        if not torch.isfinite(natural_grad).all():
            line_search_mode = 'invalid_natural_grad'
            self._set_flat_params(self.actor, old_params)
        else:
            # 计算步长
            # max_step = sqrt(2 * max_kl / (g^T F^{-1} g))
            grad_fvp = self._fisher_vector_product(natural_grad, states)
            shs = torch.dot(natural_grad, grad_fvp)
            shs_value = shs.item()

            if (not np.isfinite(shs_value)) or shs_value <= 0.0:
                line_search_mode = 'invalid_curvature'
                self._set_flat_params(self.actor, old_params)
            else:
                max_step_value = np.sqrt((2.0 * self.max_kl) / (shs_value + 1e-8))
                if (not np.isfinite(max_step_value)) or max_step_value <= 0.0:
                    line_search_mode = 'invalid_step'
                    self._set_flat_params(self.actor, old_params)
                else:
                    step_dir = natural_grad * max_step_value

                    for i in range(self.backtrack_iters):
                        line_search_iters = i + 1
                        new_params = old_params + step_frac * step_dir
                        self._set_flat_params(self.actor, new_params)

                        with torch.no_grad():
                            new_policy_loss, new_entropy = self._compute_policy_loss(
                                states, actions, old_log_probs, advantages
                            )
                            new_policy_objective = new_policy_loss + self.entropy_coef * new_entropy
                            kl = self._compute_kl(states, old_log_probs_all)

                        actual_improve = new_policy_objective.item() - policy_objective_value

                        # 先记录满足 KL 约束的“最优可行步”，供失败时回退使用
                        if kl.item() < self.max_kl and actual_improve > best_feasible_improve:
                            best_feasible_params = new_params.clone()
                            best_feasible_kl = kl.item()
                            best_feasible_improve = actual_improve
                            best_feasible_step_frac = step_frac

                        # 检查是否满足约束（允许极小负改进，提升鲁棒性）
                        if kl.item() < self.max_kl and actual_improve > -self.ls_accept_tolerance:
                            line_search_success = True
                            policy_updated = True
                            accepted_step_fraction = step_frac
                            kl_value = kl.item()
                            line_search_mode = 'accept'
                            break

                        step_frac *= self.backtrack_coeff

        if not line_search_success:
            # 若存在可行步且退化很小，则使用最优可行步，避免长期“零更新”
            if (
                best_feasible_params is not None
                and best_feasible_improve > -self.ls_fallback_tolerance
            ):
                self._set_flat_params(self.actor, best_feasible_params)
                policy_updated = True
                accepted_step_fraction = best_feasible_step_frac
                kl_value = best_feasible_kl
                line_search_mode = 'fallback'
            else:
                # 线性搜索失败，恢复旧参数
                self._set_flat_params(self.actor, old_params)

        # 清理 Actor 更新的中间变量
        del states, old_log_probs_all, policy_grad, natural_grad

        # ========== Critic 更新 ==========
        # 重新获取数据用于 Critic 训练
        critic_losses = []
        for _ in range(self.critic_train_iters):
            for batch in self.rollout_buffer.sample_batch(self.batch_size):
                batch_states, _, _, _, batch_returns = batch

                values = self.critic(batch_states).squeeze(-1)
                critic_loss = nn.functional.mse_loss(values, batch_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                critic_losses.append(critic_loss.item())

        # 重置 buffer
        self.rollout_buffer.reset()

        # 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'policy_loss': policy_loss_value,
            'critic_loss': np.mean(critic_losses),
            'kl_divergence': kl_value,
            'entropy': entropy_value,
            'step_fraction': accepted_step_fraction,
            'line_search_success': line_search_success,
            'policy_updated': policy_updated,
            'line_search_iters': line_search_iters,
            'line_search_mode': line_search_mode,
        }

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'max_kl': self.max_kl,
            'damping': self.damping,
            'entropy_coef': self.entropy_coef,
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.max_kl = checkpoint.get('max_kl', self.max_kl)
        self.damping = checkpoint.get('damping', self.damping)
        self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
