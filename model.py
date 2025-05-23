import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast
import logging
from typing import Optional, Dict
from transformers import logging as hf_logging
hf_logging.set_verbosity_warning()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model")

class Biaffine(nn.Module):
    def __init__(self, input_dim, output_dim, bias_x=True, bias_y=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias_x = bias_x
        self.bias_y = bias_y
        
        self.weight = nn.Parameter(torch.empty(output_dim, 
                                           input_dim + bias_x, 
                                           input_dim + bias_y))
        nn.init.kaiming_normal_(self.weight, nonlinearity='linear')
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y):
        x = self.dropout(x)
        y = self.dropout(y)
        
        if self.bias_x:
            x = torch.cat([x, torch.ones_like(x[..., :1])], -1)
        if self.bias_y:
            y = torch.cat([y, torch.ones_like(y[..., :1])], -1)

        scores = torch.einsum("bxi,oij,byj->bxyo", x, self.weight, y)
        return scores

class DependencyParser(nn.Module):
    def __init__(self, model_name="roberta-base", num_labels=50, hidden_dim=768):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name, add_pooling_layer=False)
        self.hidden_dim = hidden_dim

        self.head_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),
        )
        
        self.dep_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),
        )

        self.edge_predictor = Biaffine(hidden_dim, 1)
        self.label_predictor = Biaffine(hidden_dim, num_labels)
        self._init_weights()

    def _init_weights(self):
        for module in [self.head_mlp, self.dep_mlp]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.normal_(layer.bias, std=1e-6)

    def forward(self, input_ids, attention_mask, word_to_subword_map):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        batch_size, seq_len = word_to_subword_map.shape
        device = input_ids.device
        
        word_embeddings = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)
        valid_mask = word_to_subword_map != -1
        valid_indices = word_to_subword_map[valid_mask]
        word_embeddings[valid_mask] = hidden_states[torch.where(valid_mask)[0], valid_indices]

        head_repr = self.head_mlp(word_embeddings)
        dep_repr = self.dep_mlp(word_embeddings)

        edge_scores = self.edge_predictor(head_repr, dep_repr).squeeze(-1)
        label_scores = self.label_predictor(head_repr, dep_repr)

        return edge_scores, label_scores

class ArcMarginLoss(nn.Module):
    """Margin-based loss for edge prediction with integrated label loss"""
    def __init__(self, margin: float = 0.3, ignore_index: int = -100, 
                 alpha: float = 0.5, label_smoothing: float = 0.1):
        super().__init__()
        self.margin = margin
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.label_loss = LabelSmoothCrossEntropy(
            smoothing=label_smoothing, 
            ignore_index=ignore_index
        )

    def forward(self, edge_scores: torch.Tensor, label_scores: torch.Tensor,
                gold_heads: torch.Tensor, gold_labels: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = gold_heads.shape
        device = edge_scores.device
        
        # Create masks for valid tokens and positive edges
        valid_mask = gold_heads != self.ignore_index
        pos_mask = torch.zeros_like(edge_scores, dtype=torch.bool)
        
        # Vectorized positive mask creation
        batch_indices = torch.arange(batch_size, device=device)[:, None, None]
        seq_indices = torch.arange(seq_len, device=device)[None, :, None]
        head_indices = gold_heads[..., None].clamp(0, seq_len-1)
        pos_mask.scatter_(2, head_indices, valid_mask.unsqueeze(-1))
        
        # Calculate margin loss
        pos_scores = edge_scores[pos_mask]
        neg_scores = edge_scores[~pos_mask & valid_mask.unsqueeze(-1)]
        
        # Sample hard negatives
        num_neg = min(5 * pos_scores.size(0), neg_scores.size(0))
        neg_scores = neg_scores[torch.topk(neg_scores, num_neg, largest=True).indices]
        
        margin_loss = F.relu(neg_scores - pos_scores.unsqueeze(-1) + self.margin).mean()
        
        # Label prediction loss
        batch_idx = torch.arange(batch_size, device=device)[:, None].expand(-1, seq_len)
        token_idx = torch.arange(seq_len, device=device)[None, :].expand(batch_size, -1)
        gathered_labels = label_scores[batch_idx, token_idx, gold_heads.clamp(0, seq_len-1)]
        
        label_loss = self.label_loss(
            gathered_labels[valid_mask],
            gold_labels[valid_mask]
        )
        
        return self.alpha * margin_loss + (1 - self.alpha) * label_loss




class MultiTaskLoss(nn.Module):
    def __init__(self, label2id: Dict[str, int], loss_type: str = 'arc_margin', 
                 alpha: float = 0.5, margin: float = 0.3, 
                 gamma: float = 2.0, smoothing: float = 0.1):
        super().__init__()
        self.label2id = label2id
        self.edge_pad_id = -1
        self.label_pad_id = label2id.get("[PAD]", -100)
        self.loss_type = loss_type
        
        if loss_type == 'arc_margin':
            self.loss_fn = ArcMarginLoss(
                margin=margin,
                ignore_index=self.edge_pad_id,
                alpha=alpha,
                label_smoothing=smoothing
            )
        else:
            self.ce_loss_fn = CrossEntropyLoss(ignore_index=self.edge_pad_id)
            self.label_loss_fn = self._get_label_loss_fn(loss_type, gamma, smoothing)
            self.alpha = alpha

    def _get_label_loss_fn(self, loss_type: str, gamma: float, smoothing: float):
        if loss_type == 'focal':
            return FocalLoss(ignore_index=self.label_pad_id, gamma=gamma)
        elif loss_type == 'label_smooth':
            return LabelSmoothCrossEntropy(smoothing=smoothing, ignore_index=self.label_pad_id)
        return CrossEntropyLoss(ignore_index=self.label_pad_id)

    def forward(self, edge_scores, label_scores, gold_heads, gold_labels):
        # Input validation
        assert edge_scores.dim() == 3, f"Edge scores must be 3D, got {edge_scores.shape}"
        assert label_scores.dim() == 4, f"Label scores must be 4D, got {label_scores.shape}"
        
        if self.loss_type == 'arc_margin':
            loss = self.loss_fn(edge_scores, label_scores, gold_heads, gold_labels)
            logger.debug(f"ArcMargin Loss: {loss.item():.4f}")
            return loss
        
        # Standard CE implementation
        edge_loss = self.ce_loss_fn(
            edge_scores.view(-1, edge_scores.size(-1)),
            gold_heads.view(-1)
        )
        
        batch_size, seq_len = gold_heads.shape
        valid_mask = (gold_heads != self.edge_pad_id)
        
        if valid_mask.sum() == 0:
            logger.warning("No valid tokens in batch")
            return edge_loss

        batch_idx = torch.arange(batch_size, device=gold_heads.device)[:, None].expand(-1, seq_len)
        token_idx = torch.arange(seq_len, device=gold_heads.device)[None, :].expand(batch_size, -1)
        gathered_scores = label_scores[batch_idx, token_idx, gold_heads]
        
        label_loss = self.label_loss_fn(
            gathered_scores[valid_mask],
            gold_labels[valid_mask]
        )
        
        total_loss = self.alpha * edge_loss + (1 - self.alpha) * label_loss
        logger.debug(f"Edge Loss: {edge_loss.item():.4f}, Label Loss: {label_loss.item():.4f}")
        return total_loss


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance"""
    def __init__(self, gamma: float = 2.0, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            reduction='none', 
            ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss for regularization"""
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(x, dim=-1)
        n_classes = x.size(-1)
        
        # Create smoothed target distribution
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (n_classes - 1))
        
        # Handle valid targets
        valid_mask = target != self.ignore_index
        valid_targets = target[valid_mask]
        
        if valid_targets.numel() > 0:
            true_dist.scatter_(
                dim=-1,
                index=valid_targets.unsqueeze(1),
                value=1 - self.smoothing
            )
        
        # Zero out ignored indices
        true_dist[target == self.ignore_index] = 0
        
        # Calculate loss
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss