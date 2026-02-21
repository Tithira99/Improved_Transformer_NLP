# rigl_scheduler.py
import numpy as np
import torch
import torch.distributed as dist

from utils_rigl import get_W


class IndexMaskHook:
    """
    各 weight テンソル w.grad に mask をかける backward hook ＋
    RigL 用の dense_grad （dense gradient のローリング平均）を溜める。
    """
    def __init__(self, layer, scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None

    def __name__(self):
        return "IndexMaskHook"

    @torch.no_grad()
    def __call__(self, grad):
        mask = self.scheduler.backward_masks[self.layer]

        # 必要なタイミングでのみ dense_grad を蓄積
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad += grad / self.scheduler.grad_accumulation_n
        else:
            self.dense_grad = None

        # mask された部分の勾配を 0 にする
        return grad * mask


def _create_step_wrapper(scheduler, optimizer):
    """
    optimizer.step を wrap して、
    step のたびに mask を weights と momentum に適用する。
    """
    _unwrapped_step = optimizer.step

    def _wrapped_step():
        _unwrapped_step()
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()

    optimizer.step = _wrapped_step


class RigLScheduler:
    """
    RigL 本体。
    - モデルの重みをランダムに初期 sparsify
    - backward hook で dense_grad を集める
    - delta_T ごとに drop/grow でトポロジを更新
    """

    def __init__(
        self,
        model,
        optimizer,
        dense_allocation=1.0,
        T_end=None,
        sparsity_distribution="uniform",
        ignore_linear_layers=False,
        delta=100,
        alpha=0.3,
        static_topo=False,
        grad_accumulation_n=1,
        state_dict=None,
    ):
        if dense_allocation <= 0 or dense_allocation > 1:
            raise Exception(
                "Dense allocation must be on the interval (0, 1]. Got: %f"
                % dense_allocation
            )

        self.model = model
        self.optimizer = optimizer

        # W: list of weight tensors
        self.W, self._linear_layers_mask = get_W(model, return_linear_layers_mask=True)
        _create_step_wrapper(self, optimizer)

        self.dense_allocation = dense_allocation
        self.N = [torch.numel(w) for w in self.W]

        if state_dict is not None:
            self.load_state_dict(state_dict)
            self.apply_mask_to_weights()
        else:
            self.sparsity_distribution = sparsity_distribution
            self.static_topo = static_topo
            self.grad_accumulation_n = grad_accumulation_n
            self.ignore_linear_layers = ignore_linear_layers
            self.backward_masks = None

            # 層ごとのターゲット sparsity S[l] を定義
            self.S = []
            for i, (W, is_linear) in enumerate(zip(self.W, self._linear_layers_mask)):
                is_first_layer = i == 0
                if (
                    is_first_layer
                    and self.sparsity_distribution == "uniform"
                    and len(self.W) > 1
                ):
                    # uniform の場合、最初の層は dense に (論文実装準拠)
                    self.S.append(0.0)
                elif is_linear and self.ignore_linear_layers:
                    # Linear を無視する場合は dense のまま
                    self.S.append(0.0)
                else:
                    self.S.append(1.0 - dense_allocation)

            # 初期ランダム sparsify
            self.random_sparsify()

            self.step = 0
            self.rigl_steps = 0

            self.delta_T = delta
            self.alpha = alpha
            self.T_end = T_end

        # backward hook 登録
        self.backward_hook_objects = []
        for i, w in enumerate(self.W):
            if self.S[i] <= 0:
                self.backward_hook_objects.append(None)
                continue

            if getattr(w, "_has_rigl_backward_hook", False):
                raise Exception(
                    "This model already has been registered to a RigLScheduler."
                )

            hook_obj = IndexMaskHook(i, self)
            self.backward_hook_objects.append(hook_obj)
            w.register_hook(hook_obj)
            setattr(w, "_has_rigl_backward_hook", True)

        assert self.grad_accumulation_n > 0 and self.grad_accumulation_n < delta
        assert self.sparsity_distribution in ("uniform",)

    # ---------------- Stats / state_dict -----------------

    def state_dict(self):
        return {
            "dense_allocation": self.dense_allocation,
            "S": self.S,
            "N": self.N,
            "hyperparams": {
                "delta_T": self.delta_T,
                "alpha": self.alpha,
                "T_end": self.T_end,
                "ignore_linear_layers": self.ignore_linear_layers,
                "static_topo": self.static_topo,
                "sparsity_distribution": self.sparsity_distribution,
                "grad_accumulation_n": self.grad_accumulation_n,
            },
            "step": self.step,
            "rigl_steps": self.rigl_steps,
            "backward_masks": self.backward_masks,
            "_linear_layers_mask": self._linear_layers_mask,
        }

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if isinstance(v, dict):
                self.load_state_dict(v)
            else:
                setattr(self, k, v)

    # ---------------- Initial random sparsify -----------------

    @torch.no_grad()
    def random_sparsify(self):
        is_dist = dist.is_initialized()
        self.backward_masks = []
        for l, w in enumerate(self.W):
            if self.S[l] <= 0:
                self.backward_masks.append(None)
                continue

            n = self.N[l]
            s = int(self.S[l] * n)
            perm = torch.randperm(n, device=w.device)[:s]
            flat_mask = torch.ones(n, device=w.device)
            flat_mask[perm] = 0
            mask = flat_mask.view_as(w)

            if is_dist:
                dist.broadcast(mask, 0)

            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)

    def __str__(self):
        s = "RigLScheduler(\n"
        s += "layers=%i,\n" % len(self.N)

        N_str = "["
        S_str = "["
        total_params = 0
        total_nonzero = 0
        total_conv_params = 0
        total_conv_nonzero = 0

        for N, S, mask, W, is_linear in zip(
            self.N, self.S, self.backward_masks, self.W, self._linear_layers_mask
        ):
            if S <= 0:
                actual_S = 0
            else:
                actual_S = torch.sum(W[mask == 0] == 0).item()
            N_str += "%i/%i, " % (N - actual_S, N)
            sp_p = float(N - actual_S) / float(N) * 100.0
            S_str += "%.2f%%, " % sp_p
            total_params += N
            total_nonzero += N - actual_S
            if not is_linear:
                total_conv_nonzero += N - actual_S
                total_conv_params += N

        N_str = N_str[:-2] + "]"
        S_str = S_str[:-2] + "]"

        s += "nonzero_params=" + N_str + ",\n"
        s += "nonzero_percentages=" + S_str + ",\n"
        s += "total_nonzero_params=" + (
            "%i/%i (%.2f%%)"
            % (
                total_nonzero,
                total_params,
                float(total_nonzero) / float(total_params) * 100.0,
            )
        ) + ",\n"
        s += "total_CONV_nonzero_params=" + (
            "%i/%i (%.2f%%)"
            % (
                total_conv_nonzero,
                total_conv_params if total_conv_params > 0 else 1,
                float(total_conv_nonzero)
                / float(total_conv_params if total_conv_params > 0 else 1)
                * 100.0,
            )
        ) + ",\n"
        s += "step=" + str(self.step) + ",\n"
        s += "num_rigl_steps=" + str(self.rigl_steps) + ",\n"
        s += "ignoring_linear_layers=" + str(self.ignore_linear_layers) + ",\n"
        s += "sparsity_distribution=" + str(self.sparsity_distribution) + ",\n"
        return s + ")"

    # ---------------- Mask / momentum handling -----------------

    @torch.no_grad()
    def reset_momentum(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            if s <= 0:
                continue
            param_state = self.optimizer.state.get(w, {})
            if "momentum_buffer" in param_state:
                buf = param_state["momentum_buffer"]
                buf *= mask

    @torch.no_grad()
    def apply_mask_to_weights(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            if s <= 0:
                continue
            w *= mask

    @torch.no_grad()
    def apply_mask_to_gradients(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            if s <= 0:
                continue
            if w.grad is not None:
                w.grad *= mask

    # ---------------- Scheduling logic -----------------

    def check_if_backward_hook_should_accumulate_grad(self):
        """
        RigL step の直前 self.grad_accumulation_n ステップだけ
        dense_grad を貯める。
        """
        if self.step >= self.T_end:
            return False
        steps_til_next = self.delta_T - (self.step % self.delta_T)
        return steps_til_next <= self.grad_accumulation_n

    def cosine_annealing(self):
        return self.alpha / 2.0 * (1.0 + np.cos((self.step * np.pi) / self.T_end))

    def __call__(self):
        """
        1 ミニバッチごとに呼ぶ。
        delta_T ごとに _rigl_step() を発火させる。
        """
        self.step += 1
        if self.static_topo:
            return True
        if (self.step % self.delta_T) == 0 and self.step < self.T_end:
            self._rigl_step()
            self.rigl_steps += 1
            return False
        return True

    # ---------------- Core RigL step -----------------

    @torch.no_grad()
    def _rigl_step(self):
        drop_fraction = self.cosine_annealing()

        is_dist = dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else None

        for l, w in enumerate(self.W):
            if self.S[l] <= 0:
                continue

            current_mask = self.backward_masks[l]
            score_drop = torch.abs(w)
            score_grow = torch.abs(self.backward_hook_objects[l].dense_grad)

            if is_dist:
                dist.all_reduce(score_drop)
                score_drop /= world_size
                dist.all_reduce(score_grow)
                score_grow /= world_size

            n_total = self.N[l]
            n_ones = torch.sum(current_mask).item()
            n_prune = int(n_ones * drop_fraction)
            n_keep = n_ones - n_prune

            # drop mask
            _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
            new_values = torch.where(
                torch.arange(n_total, device=w.device) < n_keep,
                torch.ones_like(sorted_indices),
                torch.zeros_like(sorted_indices),
            )
            mask1 = new_values.scatter(0, sorted_indices, new_values)

            # grow mask
            score_grow = score_grow.view(-1)
            score_grow_lifted = torch.where(
                mask1 == 1,
                torch.ones_like(mask1) * (torch.min(score_grow) - 1),
                score_grow,
            )
            _, sorted_indices = torch.topk(score_grow_lifted, k=n_total)
            new_values = torch.where(
                torch.arange(n_total, device=w.device) < n_prune,
                torch.ones_like(sorted_indices),
                torch.zeros_like(sorted_indices),
            )
            mask2 = new_values.scatter(0, sorted_indices, new_values)

            mask2_reshaped = mask2.view_as(current_mask)
            grow_tensor = torch.zeros_like(w)

            new_connections = (mask2_reshaped == 1) & (current_mask == 0)
            new_weights = torch.where(new_connections.to(w.device), grow_tensor, w)
            w.data = new_weights

            mask_combined = (mask1 + mask2).view_as(current_mask).bool()
            current_mask.data = mask_combined

        # マスク更新後に momentum, weight, grad をマスク
        self.reset_momentum()
        self.apply_mask_to_weights()
        self.apply_mask_to_gradients()

