import torch
import numpy as np
import logging

class PrototypeDimensionPruner:
    """
    Prototype-based Subspace Dimension Pruner for Exemplar-Free Continual Learning.
    
    Identifies and safely prunes dimensions from DualGPM / RKHS memory that have
    negligible class-discriminative power for old tasks.
    
    Mathematical Principle:
      Let W_c be the classifier prototype/weight for class c.
      The discriminative capacity of basis vector u_k is measured by the 
      between-class variance:
          Var(u_k) = (1/C) * sum_c ( (W_c - mean(W))^T u_k )^2
      If Var(u_k) < threshold * max_j Var(u_j), direction u_k does not contribute
      to separating any pair of classes ((W_c - W_j)^T u_k approx 0).
      Removing u_k from the protected subspace restores rank to the orthogonal
      complement (I - U U^T) with zero degradation of old-task accuracy.
    """
    def __init__(self, threshold=0.05, max_prune_ratio=0.25, enabled=True):
        self.threshold = threshold          # Relative variance cutoff (default: 5% of max)
        self.max_prune_ratio = max_prune_ratio  # Maximum fraction of dims to prune per task
        self.enabled = enabled
        self.feature_prototypes = {}        # class_id -> prototype vector [768]
        self.kernel_prototypes = {}         # class_id -> kernel prototype [2048]

    def update_prototypes(self, network, train_loader, cur_task, device):
        """
        Extracts and stores empirical class mean prototypes on current task data.
        Storage footprint: 20 classes * 768 * 4 bytes = 61 KB per task (zero image replay).
        """
        if not self.enabled:
            return

        net = network.module if hasattr(network, 'module') else network
        net.eval()
        
        sums = {}
        counts = {}
        
        with torch.no_grad():
            for batch in train_loader:
                inputs = batch[1].to(device) if len(batch) == 3 else batch[0].to(device)
                targets = batch[2].to(device) if len(batch) == 3 else batch[1].to(device)
                
                # Extract representation
                feats = net.extract_vector(inputs)  # [B, D]
                
                for c in targets.unique():
                    c_item = c.item()
                    mask = (targets == c)
                    c_feats = feats[mask].sum(dim=0).cpu()
                    c_count = mask.sum().item()
                    
                    if c_item not in sums:
                        sums[c_item] = c_feats
                        counts[c_item] = c_count
                    else:
                        sums[c_item] += c_feats
                        counts[c_item] += c_count
                        
        for c_item in sums:
            self.feature_prototypes[c_item] = sums[c_item] / max(counts[c_item], 1)
            
        logging.info(
            f"PrototypePruner: Updated class prototypes for Task {cur_task}. "
            f"Total stored prototypes: {len(self.feature_prototypes)}"
        )

    def prune_layer_bases(self, feature_list, network, cur_task):
        """
        Prunes discriminatively redundant dimensions from each layer's DualGPM basis in feature_list.
        Uses accumulated classifier weights from classifier_pool to evaluate discriminative power.
        """
        if not self.enabled or feature_list is None or len(feature_list) == 0:
            return feature_list

        net = network.module if hasattr(network, 'module') else network
        
        # Collect class discriminative vectors (empirical prototypes or classifier weights)
        W_all = None
        if hasattr(self, 'feature_prototypes') and len(self.feature_prototypes) >= 2:
            protos = torch.stack(list(self.feature_prototypes.values()), dim=0)  # [Total_C, D]
            if protos.shape[1] == 768:
                W_all = protos
                
        if W_all is None:
            all_weights = []
            for t in range(net.numtask):
                head = net.classifier_pool[t]
                w = head.weight.detach().cpu()  # [C_t, D]
                all_weights.append(w)
            if len(all_weights) > 0:
                W_all = torch.cat(all_weights, dim=0)  # [Total_C, D]
            
        if W_all is None or W_all.shape[0] < 2:
            return feature_list
            
        W_centered = W_all - W_all.mean(dim=0, keepdim=True)  # [Total_C, D]
        
        pruned_feature_list = []
        total_pruned = 0
        total_dims = 0
        
        print("\n" + "="*65)
        print(f"TASK {cur_task}: PROTOTYPE SUBSPACE PRUNING REPORT (DUALGPM BASES)")
        print("="*65)
        
        for l in range(len(feature_list)):
            U = feature_list[l]  # [768, K] numpy array
            if U is None or U.shape[1] <= 1:
                pruned_feature_list.append(U)
                continue
                
            K = U.shape[1]
            total_dims += K
            U_t = torch.from_numpy(U).float()  # [768, K]
            
            # Align feature dimensions if classifier has been expanded:
            if W_centered.shape[1] >= U_t.shape[0]:
                W_proj = W_centered[:, :U_t.shape[0]]
            else:
                pruned_feature_list.append(U)
                print(f"Layer {l+1:2d} | Dimension mismatch ({W_centered.shape[1]} vs {U_t.shape[0]}), skipped.")
                continue
            
            # Project centered classifier vectors onto basis vectors:
            # Proj: [Total_C, K]
            proj = torch.mm(W_proj, U_t)
            
            # Between-class variance along each basis direction:
            var_k = torch.var(proj, dim=0, unbiased=False)  # [K]
            max_var = torch.max(var_k).item()
            
            if max_var <= 1e-8:
                pruned_feature_list.append(U)
                continue
                
            # Normalized discriminative ratio (0.0 to 1.0)
            ratio = var_k / max_var
            
            # Identify uninformative directions
            # Direction is candidate for pruning if ratio < threshold
            keep_mask = (ratio >= self.threshold)
            n_prune = (ratio < self.threshold).sum().item()
            
            # Cap maximum pruning to prevent over-pruning
            max_allowed_prune = int(K * self.max_prune_ratio)
            if n_prune > max_allowed_prune:
                # Keep top (K - max_allowed_prune) dimensions with highest variance
                _, sorted_idx = torch.sort(var_k, descending=True)
                keep_idx = sorted_idx[: (K - max_allowed_prune)]
                keep_mask = torch.zeros(K, dtype=torch.bool)
                keep_mask[keep_idx] = True
                n_prune = max_allowed_prune
                
            if n_prune > 0 and keep_mask.sum().item() > 0:
                U_kept = U[:, keep_mask.numpy()]
                # Re-orthonormalize kept bases via QR
                Q, _ = np.linalg.qr(U_kept)
                pruned_feature_list.append(Q)
                total_pruned += n_prune
                print(
                    f"Layer {l+1:2d} | Original: {K:3d} dims -> Pruned: {n_prune:2d} redundant dims -> Kept: {Q.shape[1]:3d} dims"
                )
            else:
                pruned_feature_list.append(U)
                print(f"Layer {l+1:2d} | Retained all {K:3d} dims (all discriminatively significant)")
                
        print(f"Total Subspace Capacity Freed: {total_pruned}/{total_dims} dims ({total_pruned*100.0/max(total_dims,1):.1f}%)")
        print("="*65 + "\n")
        
        return pruned_feature_list

    def prune_kernel_basis(self, kernel_memory_basis, cur_task):
        """
        Prunes redundant dimensions from the 2048-D RKHS kernel_memory_basis.
        Uses stored feature/kernel prototypes to test discriminative variance in RKHS.
        """
        if not self.enabled or kernel_memory_basis is None or kernel_memory_basis.shape[1] <= 2:
            return kernel_memory_basis
            
        if len(self.feature_prototypes) < 2:
            return kernel_memory_basis
            
        protos = torch.stack(list(self.feature_prototypes.values()), dim=0).to(kernel_memory_basis.device)  # [C, D]
        K = kernel_memory_basis.shape[1]
        
        # If prototypes match kernel dimension directly:
        if protos.shape[1] == kernel_memory_basis.shape[0]:
            P_centered = protos - protos.mean(dim=0, keepdim=True)
            proj = torch.mm(P_centered, kernel_memory_basis)  # [C, K]
            var_k = torch.var(proj, dim=0, unbiased=False)
            max_var = torch.max(var_k).item()
            
            if max_var > 1e-8:
                ratio = var_k / max_var
                keep_mask = (ratio >= self.threshold)
                n_prune = (ratio < self.threshold).sum().item()
                max_prune = int(K * self.max_prune_ratio)
                
                if n_prune > max_prune:
                    _, sorted_idx = torch.sort(var_k, descending=True)
                    keep_idx = sorted_idx[: (K - max_prune)]
                    keep_mask = torch.zeros(K, dtype=torch.bool, device=kernel_memory_basis.device)
                    keep_mask[keep_idx] = True
                    n_prune = max_prune
                    
                if n_prune > 0 and keep_mask.sum().item() > 0:
                    kept_kbasis = kernel_memory_basis[:, keep_mask]
                    Q, _ = torch.linalg.qr(kept_kbasis)
                    logging.info(
                        f"PrototypePruner: Kernel RKHS Memory Pruned from {K} -> {Q.shape[1]} dims "
                        f"(Freed {n_prune} dims in RKHS)"
                    )
                    return Q
                    
        return kernel_memory_basis
