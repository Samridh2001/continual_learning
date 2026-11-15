import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import logging

from models.vit_inflora import VisionTransformer, PatchEmbed, Block, resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.zoo import CodaPrompt

class RandomFourierKernel(nn.Module):
    """
    Random Fourier Feature (RFF) Kernel Expansion with Isometric Norm Preservation.
    Maps representation z in R^{in_dim} -> Phi(z) in R^{kernel_dim}
    approximating an infinite-dimensional Gaussian/RBF Reproducing Kernel Hilbert Space (RKHS).
    Preserves input feature magnitude so that downstream classifier gradients remain healthy and well-conditioned:
        Phi(z) = ||z|| * sqrt(2 / D_kernel) * cos(Omega @ (z / ||z||) + bias)
    """
    def __init__(self, in_dim=768, kernel_dim=2048, gamma=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_dim = kernel_dim
        self.gamma = gamma

        # Sample random projection matrix Omega and bias on unit hypersphere
        omega = torch.randn(kernel_dim, in_dim) * math.sqrt(2.0 * gamma)
        bias = torch.rand(kernel_dim) * (2.0 * math.pi)

        self.register_buffer("omega", omega)
        self.register_buffer("bias", bias)
        self.scale = math.sqrt(2.0 / kernel_dim)

    def forward(self, x):
        # x: [..., in_dim] -> [..., kernel_dim]
        in_norm = x.norm(dim=-1, keepdim=True) + 1e-8
        x_unit = x / in_norm
        proj = F.linear(x_unit, self.omega, self.bias)
        phi = torch.cos(proj) * self.scale
        return phi * in_norm


class ViT_lora_co(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, n_tasks=n_tasks, rank=rank)


    def forward(self, x, task_id, register_blk=-1, get_feat=False, get_cur_feat=False):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)
        for i, blk in enumerate(self.blocks):
            x = blk(x, task_id, register_blk==i, get_feat=get_feat, get_cur_feat=get_cur_feat)

        x = self.norm(x)
        
        return x, prompt_loss



def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT_lora_co, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model



class SiNet(nn.Module):

    def __init__(self, args):
        super(SiNet, self).__init__()

        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, n_tasks=args["total_sessions"], rank=args["rank"])
        self.image_encoder = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=True, **model_kwargs)

        self.class_num = args["init_cls"]

        # Option 1: Random Fourier Features Kernel Expansion
        self.use_kernel = args.get("use_kernel", False)
        self.kernel_dim = args.get("kernel_dim", 2048)
        self.kernel_gamma = args.get("kernel_gamma", 0.01)

        # Dynamic Direct-Sum Dimension Expansion
        self.use_dynamic_expansion = args.get("use_dynamic_expansion", False)
        self.expand_at_task = args.get("expand_at_task", 5)
        self.expand_dim = args.get("expand_dim", 1280)
        self.expand_gamma = args.get("expand_gamma", 1.0)
        self.is_expanded = False
        self.expansion_module = None
        self.current_feature_dim = args["embd_dim"]

        if self.use_kernel:
            self.kernel_expansion = RandomFourierKernel(
                in_dim=args["embd_dim"],
                kernel_dim=self.kernel_dim,
                gamma=self.kernel_gamma
            )
            classifier_in_dim = self.kernel_dim
        else:
            self.kernel_expansion = nn.Identity()
            classifier_in_dim = args["embd_dim"]

        self.classifier_pool = nn.ModuleList([
            nn.Linear(classifier_in_dim, self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        self.classifier_pool_backup = nn.ModuleList([
            nn.Linear(classifier_in_dim, self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        self.numtask = 0

    def expand_feature_dimension(self, expand_dim=None, gamma=None):
        """
        Zero-Data Direct-Sum Dimension Expansion:
        Expands the representation space by concatenating an orthogonal Random Fourier Feature bank.
        Past classifier weights are zero-padded: W_new = [W_old, 0].
        This strictly guarantees that old-task logits remain 100% bit-exact without retraining or replay data!
        """
        if self.is_expanded:
            return
        if expand_dim is None:
            expand_dim = self.expand_dim
        if gamma is None:
            gamma = self.expand_gamma

        dev = next(self.parameters()).device
        old_dim = self.current_feature_dim
        new_dim = old_dim + expand_dim

        logging.info("=" * 65)
        logging.info(f"*** ZERO-DATA DIRECT-SUM DIMENSION EXPANSION TRIGGERED ***")
        logging.info(f"Expanding Feature Dimension: {old_dim} -> {new_dim} (+{expand_dim} dimensions)")
        logging.info(f"Zero-padding past {self.numtask} classifier heads to guarantee 100% bit-exact logit preservation.")
        logging.info("=" * 65)

        self.expansion_module = RandomFourierKernel(
            in_dim=self.image_encoder.out_dim,
            kernel_dim=expand_dim,
            gamma=gamma
        ).to(dev)

        # Zero-pad past classifier heads: W_new = [W_old, 0]
        for t in range(self.numtask):
            old_head = self.classifier_pool[t]
            h_dev = old_head.weight.device
            new_head = nn.Linear(new_dim, self.class_num, bias=True).to(h_dev)
            with torch.no_grad():
                new_head.weight[:, :old_dim] = old_head.weight
                new_head.weight[:, old_dim:] = 0.0
                if old_head.bias is not None:
                    new_head.bias.copy_(old_head.bias)
            self.classifier_pool[t] = new_head

            if t < len(self.classifier_pool_backup):
                old_b = self.classifier_pool_backup[t]
                new_b = nn.Linear(new_dim, self.class_num, bias=True).to(h_dev)
                with torch.no_grad():
                    new_b.weight[:, :old_dim] = old_b.weight
                    new_b.weight[:, old_dim:] = 0.0
                    if old_b.bias is not None:
                        new_b.bias.copy_(old_b.bias)
                self.classifier_pool_backup[t] = new_b

        # Resize upcoming uninstantiated heads in pool
        for t in range(self.numtask, len(self.classifier_pool)):
            h_dev = self.classifier_pool[t].weight.device
            self.classifier_pool[t] = nn.Linear(new_dim, self.class_num, bias=True).to(h_dev)
            if t < len(self.classifier_pool_backup):
                self.classifier_pool_backup[t] = nn.Linear(new_dim, self.class_num, bias=True).to(h_dev)

        self.is_expanded = True
        self.current_feature_dim = new_dim
        logging.info(f"Zero-data expansion complete: Feature Dimension is now {self.current_feature_dim}.")

    @property
    def feature_dim(self):
        if self.use_kernel:
            return self.kernel_dim
        elif self.is_expanded:
            return self.current_feature_dim
        return self.image_encoder.out_dim

    def extract_vector(self, image, task=None):
        if task is None:
            image_features, _ = self.image_encoder(image, self.numtask-1)
        else:
            image_features, _ = self.image_encoder(image, task)
        image_features = image_features[:, 0, :]
        if self.use_kernel:
            image_features = self.kernel_expansion(image_features)
        elif self.is_expanded and self.expansion_module is not None:
            phi_ext = self.expansion_module(image_features)
            image_features = torch.cat([image_features, phi_ext], dim=-1)
        return image_features

    def forward(self, image, get_feat=False, get_cur_feat=False, fc_only=False):
        if fc_only:
            fc_outs = []
            for ti in range(self.numtask):
                fc_out = self.classifier_pool[ti](image)
                fc_outs.append(fc_out)
            return torch.cat(fc_outs, dim=1)

        logits = []
        image_features, prompt_loss = self.image_encoder(image, task_id=self.numtask-1, get_feat=get_feat, get_cur_feat=get_cur_feat)
        image_features = image_features[:, 0, :]
        image_features = image_features.view(image_features.size(0), -1)
        if self.use_kernel:
            image_features = self.kernel_expansion(image_features)
        elif self.is_expanded and self.expansion_module is not None:
            phi_ext = self.expansion_module(image_features)
            image_features = torch.cat([image_features, phi_ext], dim=-1)

        for prompts in [self.classifier_pool[self.numtask-1]]:
            logits.append(prompts(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
            'prompt_loss': prompt_loss
        }

    def interface(self, image, task_id = None):
        image_features, _ = self.image_encoder(image, task_id=self.numtask-1 if task_id is None else task_id)

        image_features = image_features[:, 0, :]
        image_features = image_features.view(image_features.size(0), -1)
        if self.use_kernel:
            image_features = self.kernel_expansion(image_features)
        elif self.is_expanded and self.expansion_module is not None:
            phi_ext = self.expansion_module(image_features)
            image_features = torch.cat([image_features, phi_ext], dim=-1)

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits, 1)
        return logits
    
    def interface1(self, image, task_ids):
        logits = []
        for index in range(len(task_ids)):
            image_features, _ = self.image_encoder(image[index:index+1], task_id=task_ids[index].item())
            image_features = image_features[:, 0, :]
            image_features = image_features.view(image_features.size(0), -1)
            if self.use_kernel:
                image_features = self.kernel_expansion(image_features)
            elif self.is_expanded and self.expansion_module is not None:
                phi_ext = self.expansion_module(image_features)
                image_features = torch.cat([image_features, phi_ext], dim=-1)

            logits.append(self.classifier_pool_backup[task_ids[index].item()](image_features))

        logits = torch.cat(logits, 0)
        return logits

    def interface2(self, image_features):
        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits, 1)
        return logits

    def update_fc(self, nb_classes):
        self.numtask += 1

    def classifier_backup(self, task_id):
        self.classifier_pool_backup[task_id].load_state_dict(self.classifier_pool[task_id].state_dict())

    def classifier_recall(self):
        self.classifier_pool.load_state_dict(self.old_state_dict)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
