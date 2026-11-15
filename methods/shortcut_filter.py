import torch
import numpy as np
import logging

class UnifiedShortcutFilter:
    """
    Unified Multi-View Shortcut Purging & Null-Space Deflation Filter.
    Merges:
    1. Multi-View Augmentation Shortcut Purging (from Job 65113):
       Purges color and blur sensitivity shortcuts from activation covariance.
       This clean, un-deflated covariance is fed to DualGPM so DualGPM updates surgically
       (adding only 2-10 genuine visual dimensions per task), preserving 75%+ old-task retention.
    2. Null-Space Orthogonal Deflation (from Job 65261):
       Projects purged activation covariance onto (I - M M^T) for LoRA Matrix A initialization.
       Strictly guarantees A @ M_past == 0 and keeps Matrix A permanently frozen.
    3. Commonality Weight Estimation (User's Idea):
       Estimates persistent overlap across tasks to scale the subspace barrier on LoRA Matrix B
       up to (1.0 + beta)x for shared shortcut dimensions, preventing cross-task interference.
    """
    def __init__(self, num_layers=12, embed_dim=768, delta_color=0.35, delta_blur=0.35,
                 gamma_persist=0.85, max_shortcut_ratio=0.20, commonality_threshold=0.30,
                 use_cross_persist=False):
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.delta_color = delta_color
        self.delta_blur = delta_blur
        self.gamma_persist = gamma_persist
        self.max_shortcut_ratio = max_shortcut_ratio
        self.commonality_threshold = commonality_threshold
        self.use_cross_persist = use_cross_persist
        self.history_bases = [[] for _ in range(num_layers)]

    def purge_augmentation_shortcuts(self, clean_mat, color_mat=None, blur_mat=None, layer_idx=0, cur_task=0):
        """
        Step 1: Removes color and blur shortcuts from clean_mat.
        Returns cleaned covariance matrix C_purged to be used for DualGPM update.
        """
        device = clean_mat.device if isinstance(clean_mat, torch.Tensor) else torch.device('cpu')
        clean_np = clean_mat.detach().cpu().numpy().astype(np.float32) if isinstance(clean_mat, torch.Tensor) else np.array(clean_mat, dtype=np.float32)

        if color_mat is None or blur_mat is None:
            diag = {
                'k_evaluated': 0,
                'n_shortcuts': 0,
                'shortcut_indices': [],
                'shortcut_reasons': []
            }
            return torch.from_numpy(clean_np).float().to(device), diag

        color_np = color_mat.detach().cpu().numpy().astype(np.float32) if isinstance(color_mat, torch.Tensor) else np.array(color_mat, dtype=np.float32)
        blur_np  = blur_mat.detach().cpu().numpy().astype(np.float32) if isinstance(blur_mat, torch.Tensor) else np.array(blur_mat, dtype=np.float32)

        U, S, Vh = np.linalg.svd(clean_np, full_matrices=False)
        total_energy = float(np.sum(S**2)) + 1e-12
        cumsum_ratio = np.cumsum(S**2) / total_energy
        k_eval = int(np.sum(cumsum_ratio < 0.99)) + 1
        k_eval = min(k_eval, int(self.embed_dim * self.max_shortcut_ratio))

        shortcut_indices = []
        shortcut_reasons = []

        for i in range(k_eval):
            u_i = U[:, i:i+1]
            clean_energy = float(u_i.T @ clean_np @ u_i) + 1e-12
            color_energy = float(u_i.T @ color_np @ u_i)
            blur_energy  = float(u_i.T @ blur_np @ u_i)

            s_color = color_energy / clean_energy
            s_blur  = blur_energy / clean_energy

            is_color_shortcut = abs(s_color - 1.0) > self.delta_color
            is_blur_shortcut  = abs(s_blur - 1.0) > self.delta_blur

            is_persistent_shortcut = False
            max_sim = 0.0
            if getattr(self, 'use_cross_persist', False) and cur_task > 0 and len(self.history_bases[layer_idx]) > 0:
                for past_clean_base in self.history_bases[layer_idx]:
                    projections = np.abs(u_i.T @ past_clean_base)
                    local_max = float(np.max(projections)) if projections.size > 0 else 0.0
                    if local_max > max_sim:
                        max_sim = local_max
                if max_sim > self.gamma_persist:
                    is_persistent_shortcut = True

            flagged = False
            reasons = []
            if is_color_shortcut:
                flagged = True
                reasons.append(f"color_sens({s_color:.2f})")
            if is_blur_shortcut:
                flagged = True
                reasons.append(f"blur_sens({s_blur:.2f})")
            if is_persistent_shortcut:
                flagged = True
                reasons.append(f"cross_persist({max_sim:.2f})")

            if flagged:
                shortcut_indices.append(i)
                shortcut_reasons.append(f"dir_{i}:" + "+".join(reasons))

        if len(shortcut_indices) > 0:
            U_sc = U[:, shortcut_indices]
            P_sc = U_sc @ U_sc.T
            deflated_np = clean_np - P_sc @ clean_np - clean_np @ P_sc + P_sc @ clean_np @ P_sc
        else:
            deflated_np = clean_np

        clean_indices = [i for i in range(k_eval) if i not in shortcut_indices]
        if len(clean_indices) > 0:
            U_clean = U[:, clean_indices]
            self.history_bases[layer_idx].append(U_clean)

        deflated_np = 0.5 * (deflated_np + deflated_np.T)
        purged_mat = torch.from_numpy(deflated_np).float().to(device)
        diag = {
            'k_evaluated': k_eval,
            'n_shortcuts': len(shortcut_indices),
            'shortcut_indices': shortcut_indices,
            'shortcut_reasons': shortcut_reasons[:5]
        }
        return purged_mat, diag

    def deflate_novel_space_for_lora(self, purged_mat, past_base):
        """
        Step 2: Deflates purged_mat against past task memory basis (I - M_past M_past^T).
        Used exclusively for initializing frozen LoRA Matrix A so that A @ M_past == 0!
        """
        if past_base is None or (isinstance(past_base, np.ndarray) and past_base.size == 0) or (isinstance(past_base, torch.Tensor) and past_base.numel() == 0):
            return purged_mat

        device = purged_mat.device if isinstance(purged_mat, torch.Tensor) else torch.device('cpu')
        mat_np = purged_mat.detach().cpu().numpy().astype(np.float32) if isinstance(purged_mat, torch.Tensor) else np.array(purged_mat, dtype=np.float32)
        M_past = past_base.detach().cpu().numpy().astype(np.float32) if isinstance(past_base, torch.Tensor) else np.array(past_base, dtype=np.float32)

        D = mat_np.shape[0]
        P_perp = np.eye(D, dtype=np.float32) - (M_past @ M_past.T)
        novel_np = P_perp @ mat_np @ P_perp
        novel_np = 0.5 * (novel_np + novel_np.T)
        return torch.from_numpy(novel_np).float().to(device)

    def compute_commonality_weights(self, clean_mat, past_base, past_task_covariances=None, beta=4.0):
        """
        Step 3: Estimates cross-task commonality weights for the subspace barrier on LoRA B.
        w_k = 1.0 + beta * (score_norm)^2, clamped to [1.0, 1.0 + beta].
        """
        if past_base is None or (isinstance(past_base, np.ndarray) and past_base.size == 0) or (isinstance(past_base, torch.Tensor) and past_base.numel() == 0):
            return None

        device = past_base.device if isinstance(past_base, torch.Tensor) else (clean_mat.device if isinstance(clean_mat, torch.Tensor) else torch.device('cpu'))
        M = past_base.to(device) if isinstance(past_base, torch.Tensor) else torch.from_numpy(past_base).float().to(device)
        cur_cov = clean_mat.to(device) if isinstance(clean_mat, torch.Tensor) else torch.from_numpy(clean_mat).float().to(device)

        cur_trace = torch.trace(cur_cov) + 1e-12
        cur_cov_norm = cur_cov / cur_trace
        overlap_cur = torch.diag(torch.mm(torch.mm(M.t(), cur_cov_norm), M))

        if past_task_covariances is not None and len(past_task_covariances) > 0:
            overlap_past = torch.zeros_like(overlap_cur)
            for past_cov in past_task_covariances:
                p_cov = past_cov.to(device) if isinstance(past_cov, torch.Tensor) else torch.from_numpy(past_cov).float().to(device)
                p_cov_norm = p_cov / (torch.trace(p_cov) + 1e-12)
                overlap_past = overlap_past + torch.diag(torch.mm(torch.mm(M.t(), p_cov_norm), M))
            overlap_past = overlap_past / float(len(past_task_covariances))
        else:
            overlap_past = torch.zeros_like(overlap_cur)

        raw_score = overlap_cur + overlap_past
        max_score = torch.max(raw_score)
        if max_score > 1e-8:
            score_norm = raw_score / max_score
        else:
            score_norm = torch.zeros_like(raw_score)

        weights = 1.0 + beta * (score_norm ** 2)
        weights = torch.clamp(weights, min=1.0, max=1.0 + beta)
        return weights.to(device)


# Backward-compatibility aliases
TaskCommonalityFilter = UnifiedShortcutFilter
ShortcutSubspaceFilter = UnifiedShortcutFilter
AugmentationShortcutFilter = UnifiedShortcutFilter
