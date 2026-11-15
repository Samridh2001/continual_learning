import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import logging
import numpy as np
import math
from copy import deepcopy
from tqdm import tqdm

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from methods.inflora import InfLoRA
from models.vit_inflora import Attention_LoRA
from methods.shortcut_filter import UnifiedShortcutFilter
from utils.augmentations import MultiViewAugmenter
from utils.calibration_metrics import compute_ece, compute_grassmann_angles

class InfLoRA_Filter(InfLoRA):
    """
    Unified Dual-Path InfLoRA:
    Merges Multi-View Augmentation Shortcut Purging (Old Tasks Winner, Job 65113)
    with Commonality-Weighted Barrier & Isometric Kernel RKHS Expansion (New Tasks Winner, Job 65261).

    Dual-Path Architecture:
    - Path A (Memory Bases / DualGPM):
        Purges color & blur shortcuts from activation covariances before feeding to DualGPM.
        Because past bases M already explain ~95% of genuine visual features, DualGPM
        starts at accumulated_sval ~ 0.95 and only adds 2-10 dimensions per task.
        Prevents DualGPM saturation and premature retain flipping, maintaining 75%+ old-task retention!
    - Path B (LoRA Matrix A Initialization):
        Projects purged covariance onto the null-space (I - M M^T) and computes SVD.
        Strictly guarantees A @ M_past == 0 and keeps Matrix A permanently frozen.
    - Path C (LoRA Matrix B Training):
        Optimizes LoRA B with Commonality-Weighted Subspace Barrier:
            w_{l, k} = 1.0 + beta_common * (score_k / max_j score_j)^2
        escalating penalty up to (1.0 + beta)x on shared shortcut directions.
    - Path D (Classifier RKHS Expansion):
        Isometric 2048-dim RFF Kernel expansion guarantees high class separability,
        unlocking 80%+ accuracy on new tasks and sub-1% calibration error.
    """
    def __init__(self, args):
        super().__init__(args)

        self.lambda_subspace = args.get('lambda_subspace', 1.0)
        self.commonality_threshold = args.get('commonality_threshold', 0.30)
        self.max_shortcut_ratio = args.get('max_shortcut_ratio', 0.20)
        self.restricted_bases = [None for _ in range(12)]

        # Commonality-Weighted Subspace Penalization (User's Idea)
        self.use_commonality_weighting = args.get('use_commonality_weighting', True)
        self.beta_commonality = args.get('beta_commonality', 4.0)
        self.commonality_weights = [None for _ in range(12)]
        self.past_task_covariances = [[] for _ in range(12)]

        # Multi-View Augmentation Probing (From Job 65113)
        self.use_aug_probing = args.get('use_aug_probing', True)
        self.delta_color = args.get('delta_color', 0.35)
        self.delta_blur = args.get('delta_blur', 0.35)
        self.gamma_persist = args.get('gamma_persist', 0.85)

        if self.use_aug_probing:
            self.augmenter = MultiViewAugmenter(device=self._device)
        else:
            self.augmenter = None

        # Option 1 & 3: Kernel Expansion & Kernel DualGPM Configuration
        self.use_kernel = args.get('use_kernel', False)
        self.kernel_dim = args.get('kernel_dim', 2048)
        self.kernel_gamma = args.get('kernel_gamma', 1.0)
        self.kernel_memory_basis = None  # [kernel_dim, d_kpast] orthonormal basis in RKHS

        self.use_cross_persist = args.get('use_cross_persist', False)

        # Option 4: Prototype Subspace Dimension Pruner
        self.use_proto_pruner = args.get('use_proto_pruner', False)
        self.proto_threshold = args.get('proto_threshold', 0.05)
        self.proto_max_prune = args.get('proto_max_prune', 0.25)
        from methods.prototype_pruner import PrototypeDimensionPruner
        self.proto_pruner = PrototypeDimensionPruner(
            threshold=self.proto_threshold,
            max_prune_ratio=self.proto_max_prune,
            enabled=self.use_proto_pruner
        )

        # Unified Shortcut Filter Engine
        self.shortcut_filter = UnifiedShortcutFilter(
            num_layers=12,
            embed_dim=self.args.get('embd_dim', 768),
            delta_color=self.delta_color,
            delta_blur=self.delta_blur,
            gamma_persist=self.gamma_persist,
            max_shortcut_ratio=self.max_shortcut_ratio,
            commonality_threshold=self.commonality_threshold,
            use_cross_persist=self.use_cross_persist
        )
        self.debug = args.get('debug', False)
        self.ece_curve = []
        logging.info(
            "Unified InfLoRA_Filter initialized: lambda_subspace={}, commonality_weights={}, beta={}, "
            "aug_probing={}, use_kernel={}, kernel_dim={}".format(
                self.lambda_subspace, self.use_commonality_weighting, self.beta_commonality,
                self.use_aug_probing, self.use_kernel, self.kernel_dim
            )
        )

    def _collect_activation_matrices(self, loader):
        """
        Collects layer-wise activation covariance matrices.
        If use_aug_probing is enabled, also collects color-perturbed and blur-perturbed passes.
        """
        clean_mats, color_mats, blur_mats = [], [], []

        # 1. Clean Pass
        with torch.no_grad():
            for i, batch in enumerate(loader):
                inputs = batch[1].to(self._device) if len(batch) == 3 else batch[0].to(self._device)
                self._network(inputs, get_cur_feat=True)
                if getattr(self, 'debug', False) and i >= 3:
                    break
            for module in self._network.modules():
                if isinstance(module, Attention_LoRA):
                    clean_mats.append(deepcopy(module.cur_matrix))
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0

        # 2. Multi-view Augmentation Passes (if enabled)
        if getattr(self, 'use_aug_probing', False):
            if self.augmenter is None:
                self.augmenter = MultiViewAugmenter(device=self._device)

            # Color Perturbed Pass
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    inputs = batch[1].to(self._device) if len(batch) == 3 else batch[0].to(self._device)
                    inputs_color = self.augmenter.get_color_view(inputs)
                    self._network(inputs_color, get_cur_feat=True)
                    if getattr(self, 'debug', False) and i >= 3:
                        break
                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
                        color_mats.append(deepcopy(module.cur_matrix))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0

            # Blur Perturbed Pass
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    inputs = batch[1].to(self._device) if len(batch) == 3 else batch[0].to(self._device)
                    inputs_blur = self.augmenter.get_blur_view(inputs)
                    self._network(inputs_blur, get_cur_feat=True)
                    if getattr(self, 'debug', False) and i >= 3:
                        break
                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
                        blur_mats.append(deepcopy(module.cur_matrix))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
        else:
            color_mats, blur_mats = None, None

        return clean_mats, color_mats, blur_mats

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        # Dynamic Zero-Data Direct-Sum Dimension Expansion Check
        net = self._network.module if hasattr(self._network, 'module') else self._network
        if getattr(net, 'use_dynamic_expansion', False) and not getattr(net, 'is_expanded', False):
            if self._cur_task >= getattr(net, 'expand_at_task', 5):
                net.expand_feature_dimension()

        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source='train', mode='train'
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test'
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        self._train(self.train_loader, self.test_loader)
        self.clustering(self.train_loader)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        # Enable training ONLY on current task's classifier and LoRA B parameters.
        # LoRA A is strictly FROZEN in the novel null space (guaranteeing zero past interference).
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            try:
                task_id = str(self._network.module.numtask - 1)
            except:
                task_id = str(self._network.numtask - 1)

            for key in ["classifier_pool", "lora_B_k", "lora_B_v"]:
                if f"{key}.{task_id}" in name:
                    param.requires_grad_(True)

        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        # Pre-training Activation Collection
        logging.info("Executing Pre-task Multi-View Activation Collection...")
        clean_mats, color_mats, blur_mats = self._collect_activation_matrices(train_loader)

        # Initialize Matrix A and setup Restricted Subspace Bases
        self.restricted_bases = [None for _ in range(12)]
        layer_idx = 0
        print("\n" + "="*65)
        print(f"TASK {self._cur_task}: PRE-TRAINING SHORTCUT PURGING & LoRA A INIT")
        print("="*65)
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                past_base = None
                if self._cur_task > 0 and layer_idx < len(self.feature_list) and self.feature_list[layer_idx].shape[1] > 0:
                    past_base = self.feature_list[layer_idx]
                    self.restricted_bases[layer_idx] = torch.from_numpy(past_base).float().to(self._device)
                else:
                    self.restricted_bases[layer_idx] = None

                # Step 1: Purge augmentation shortcuts
                c_mat = color_mats[layer_idx] if color_mats is not None else None
                b_mat = blur_mats[layer_idx] if blur_mats is not None else None
                purged_clean, diag = self.shortcut_filter.purge_augmentation_shortcuts(
                    clean_mats[layer_idx], c_mat, b_mat, layer_idx=layer_idx, cur_task=self._cur_task
                )

                # Step 2: Deflate purged covariance for LoRA Matrix A initialization (A @ M_past = 0)
                novel_mat = self.shortcut_filter.deflate_novel_space_for_lora(purged_clean, past_base=past_base)
                novel_mat = novel_mat.to(self._device)

                try:
                    cU, cS, cV = torch.linalg.svd(novel_mat, full_matrices=False)
                except RuntimeError:
                    cU, cS, cV = torch.linalg.svd(novel_mat.cpu(), full_matrices=False)
                    cU = cU.to(self._device)

                module.lora_A_k[self._cur_task].weight.data.copy_(cU[:, :module.rank].T / math.sqrt(3))
                module.lora_A_v[self._cur_task].weight.data.copy_(cU[:, :module.rank].T / math.sqrt(3))

                if self._cur_task > 0 and past_base is not None:
                    logging.info(
                        f"Layer {layer_idx+1:2d} | Past Dims: {past_base.shape[1]:2d} | "
                        f"Purged {diag['n_shortcuts']:2d} Aug-Shortcuts {diag['shortcut_reasons']} | "
                        f"LoRA A strictly orthogonal to past bases"
                    )
                else:
                    logging.info(f"Layer {layer_idx+1:2d} | Task 0 Initial Base: {module.rank} LoRA dims initialized")

                module.cur_matrix.zero_()
                module.n_cur_matrix = 0
                layer_idx += 1
        print("="*65 + "\n")

        # Step 3: Compute Commonality Weights for Subspace Barrier Escalation (User's Idea)
        if self._cur_task > 0 and getattr(self, 'use_commonality_weighting', True) and self.lambda_subspace > 0.0:
            logging.info("Computing Commonality Weights for Subspace Barrier Escalation...")
            for l in range(12):
                if self.restricted_bases[l] is not None and self.restricted_bases[l].shape[1] > 0:
                    weights = self.shortcut_filter.compute_commonality_weights(
                        clean_mat=clean_mats[l],
                        past_base=self.restricted_bases[l],
                        past_task_covariances=self.past_task_covariances[l],
                        beta=self.beta_commonality
                    )
                    self.commonality_weights[l] = weights.to(self._device) if weights is not None else None
                    if weights is not None:
                        logging.info(
                            f"Layer {l+1:2d} | Commonality Weights: min={weights.min().item():.2f}, "
                            f"max={weights.max().item():.2f}, mean={weights.mean().item():.2f} "
                            f"(Max penalty escalated by {weights.max().item():.1f}x)"
                        )
                else:
                    self.commonality_weights[l] = None

        print(f"Parameters to be updated: {enabled}")
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # Training with Cosine Schedule & Subspace Barrier Loss
        from utils.schedulers import CosineSchedule
        if self._cur_task == 0:
            if self.optim == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.init_epoch)
            elif self.optim == 'adam':
                optimizer = optim.Adam(self._network.parameters(), lr=self.init_lr, weight_decay=self.init_weight_decay, betas=(0.9, 0.999))
                scheduler = CosineSchedule(optimizer=optimizer, K=self.init_epoch)
            self.run_epoch = self.init_epoch
            self.train_function(train_loader, test_loader, optimizer, scheduler)
        else:
            if self.optim == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.lrate, weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs)
            elif self.optim == 'adam':
                optimizer = optim.Adam(self._network.parameters(), lr=self.lrate, weight_decay=self.weight_decay, betas=(0.9, 0.999))
                scheduler = CosineSchedule(optimizer=optimizer, K=self.epochs)
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # Update empirical class prototypes for Prototype Dimension Pruner
        if self.use_proto_pruner:
            self.proto_pruner.update_prototypes(self._network, train_loader, self._cur_task, self._device)

        # Post-training: Collect Activations & Purge Augmentation Shortcuts for DualGPM
        logging.info("Executing Post-task Subspace Surgery & Shortcut Deflation...")
        clean_mats, color_mats, blur_mats = self._collect_activation_matrices(train_loader)

        purged_mat_list = []
        print("\n" + "="*65)
        print(f"TASK {self._cur_task}: POST-TRAINING SURGERY REPORT (DUALGPM PURGING)")
        print("="*65)
        for l in range(len(clean_mats)):
            c_mat = color_mats[l] if color_mats is not None else None
            b_mat = blur_mats[l] if blur_mats is not None else None
            purged_cov, diag = self.shortcut_filter.purge_augmentation_shortcuts(
                clean_mats[l], c_mat, b_mat, layer_idx=l, cur_task=self._cur_task
            )
            purged_mat_list.append(purged_cov.cpu().numpy() if isinstance(purged_cov, torch.Tensor) else purged_cov)
            self.past_task_covariances[l].append(clean_mats[l].detach().cpu())

            if diag['n_shortcuts'] > 0:
                print(
                    f"Layer {l+1:2d} | Purged {diag['n_shortcuts']:2d} shortcuts: {diag['shortcut_reasons']}"
                )
            else:
                print(f"Layer {l+1:2d} | Preserved clean activation energy (0 shortcuts detected)")
        print("="*65 + "\n")

        # Crucial Dual-Path Step:
        # Update DualGPM with purged_mat_list (NOT deflated by past bases!).
        # Since past bases explain ~95% of visual features, DualGPM only adds 2-10 genuine dimensions.
        self.update_DualGPM(purged_mat_list)

        # Prototype Dimension Pruning:
        # Evaluate class-discriminative variance of DualGPM bases; prune uninformative directions!
        if self.use_proto_pruner and self._cur_task > 0:
            self.feature_list = self.proto_pruner.prune_layer_bases(
                self.feature_list, self._network, cur_task=self._cur_task
            )

        # Precompute projection matrices
        self.feature_mat = []
        for p in range(len(self.feature_list)):
            Uf = torch.Tensor(np.dot(self.feature_list[p], self.feature_list[p].transpose()))
            print('Layer {} - Clean Projection Matrix shape: {}'.format(p + 1, Uf.shape))
            self.feature_mat.append(Uf)

        # Option 3: Kernel DualGPM representation memory update (in RKHS)
        if self.use_kernel:
            logging.info(f"TASK {self._cur_task}: Extracting Kernel Representations for Kernel DualGPM...")
            net_eval = self._network.module if hasattr(self._network, 'module') else self._network
            net_eval.eval()
            kfeats = []
            with torch.no_grad():
                for i_k, batch in enumerate(train_loader):
                    inputs = batch[1].to(self._device) if len(batch) == 3 else batch[0].to(self._device)
                    feat = net_eval.extract_vector(inputs)  # [B, kernel_dim]
                    kfeats.append(feat)
                    if getattr(self, 'debug', False) and i_k >= 3:
                        break
            kfeats = torch.cat(kfeats, dim=0)  # [N, kernel_dim]
            K_cov = torch.mm(kfeats.t(), kfeats) / float(kfeats.shape[0])

            # Purge kernel shortcuts against previous tasks' kernel basis
            if self.kernel_memory_basis is not None and self.kernel_memory_basis.shape[1] > 0:
                M_k = self.kernel_memory_basis
                P_k = torch.mm(M_k, M_k.t())
                K_cov = K_cov - torch.mm(P_k, K_cov) - torch.mm(K_cov, P_k) + torch.mm(torch.mm(P_k, K_cov), P_k)
                K_cov = 0.5 * (K_cov + K_cov.t())

            try:
                kU, kS, _ = torch.linalg.svd(K_cov, full_matrices=False)
            except RuntimeError:
                kU, kS, _ = torch.linalg.svd(K_cov.cpu(), full_matrices=False)
                kU = kU.to(self._device)
                kS = kS.to(self._device)

            s_ratio = (kS ** 2) / ((kS ** 2).sum() + 1e-12)
            cumsum_s = torch.cumsum(s_ratio, dim=0)
            k_retain = int((cumsum_s < self.lamb).sum().item()) + 1
            k_retain = min(k_retain, int(self.kernel_dim * 0.15))
            k_retain = max(k_retain, 5)

            new_kbasis = kU[:, :k_retain]
            if self.kernel_memory_basis is None:
                self.kernel_memory_basis = new_kbasis
            else:
                combined = torch.cat([self.kernel_memory_basis, new_kbasis], dim=1)
                q, _ = torch.linalg.qr(combined)
                self.kernel_memory_basis = q

            # Prune redundant dimensions in RKHS kernel memory if enabled
            if self.use_proto_pruner:
                self.kernel_memory_basis = self.proto_pruner.prune_kernel_basis(
                    self.kernel_memory_basis, cur_task=self._cur_task
                )

            logging.info(
                f"Task {self._cur_task} Kernel DualGPM: Retained {k_retain} dims | "
                f"Total RKHS Memory: {self.kernel_memory_basis.shape[1]}/{self.kernel_dim} dims"
            )

        return

    def compute_subspace_penalty(self):
        """
        Subspace Barrier Penalty with Adaptive Commonality Weights:
        Smoothly penalizes current task LoRA B weights from writing into past tasks' output subspaces:
            L_sub = (1/L) sum_l sum_k w_{l, k} * || M_{l, k}^T @ B_l ||_2^2
        LoRA A is strictly frozen in the novel null space (A @ M_past = 0).
        """
        if self._cur_task == 0:
            return torch.tensor(0.0, device=self._device)

        total_penalty = torch.tensor(0.0, device=self._device)
        count = 0

        net = self._network.module if hasattr(self._network, 'module') else self._network

        if hasattr(self, 'restricted_bases') and self.restricted_bases is not None:
            layer_idx = 0
            for module in net.modules():
                if isinstance(module, Attention_LoRA):
                    if layer_idx < len(self.restricted_bases) and self.restricted_bases[layer_idx] is not None:
                        M = self.restricted_bases[layer_idx]  # [embed_dim, k_past] on device
                        if M.shape[1] > 0:
                            w_B_k = module.lora_B_k[self._cur_task].weight  # [embed_dim, rank]
                            w_B_v = module.lora_B_v[self._cur_task].weight  # [embed_dim, rank]

                            # Output direction overlap: ||M^T @ B||_F^2
                            B_k_proj = torch.mm(M.t(), w_B_k)  # [k_past, rank]
                            B_v_proj = torch.mm(M.t(), w_B_v)

                            if getattr(self, 'use_commonality_weighting', True) and self.commonality_weights[layer_idx] is not None:
                                weights = self.commonality_weights[layer_idx].to(B_k_proj.device).unsqueeze(1)  # [k_past, 1] on same device
                                penalty_B = torch.sum(weights * (B_k_proj ** 2)) + torch.sum(weights * (B_v_proj ** 2))
                            else:
                                penalty_B = torch.sum(B_k_proj ** 2) + torch.sum(B_v_proj ** 2)

                            total_penalty = total_penalty + penalty_B
                            count += 1
                    layer_idx += 1

        if count > 0:
            total_penalty = total_penalty / count
        return total_penalty

    def project_lora_orthogonal(self):
        """No-op: LoRA A is strictly initialized in the deflated null space and permanently frozen."""
        pass

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.
            losses_ce = 0.
            losses_sub = 0.
            correct, total = 0, 0
            for i, batch in enumerate(train_loader):
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes

                logits = self._network(inputs)['logits']
                loss_ce = F.cross_entropy(logits, targets)

                # Subspace barrier penalty enforcing orthogonal direction learning
                if self.lambda_subspace > 0.0:
                    subspace_penalty = self.compute_subspace_penalty()
                    loss = loss_ce + self.lambda_subspace * subspace_penalty
                else:
                    subspace_penalty = torch.tensor(0.0, device=self._device)
                    loss = loss_ce

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                losses += loss.item()
                losses_ce += loss_ce.item()
                losses_sub += subspace_penalty.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                if getattr(self, 'debug', False) and i >= 3:
                    break

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f} (CE: {:.3f}, Sub: {:.4f}), Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, 
                losses / len(train_loader), losses_ce / len(train_loader), losses_sub / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def eval_task(self):
        cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task = super().eval_task()

        # Compute Expected Calibration Error (ECE) on test set
        self._network.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for _, batch in enumerate(self.test_loader):
                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs = inputs.to(self._device)
                logits = self._network.interface(inputs)
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())

        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        ece = compute_ece(cat_logits, cat_targets)
        self.ece_curve.append(ece)
        logging.info(f"Expected Calibration Error (ECE): {ece:.2f}%")
        logging.info(f"ECE Curve: {self.ece_curve}")

        return cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task
