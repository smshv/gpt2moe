import torch
from torch import nn
from copy import deepcopy
from transformers.models.switch_transformers.modeling_switch_transformers import router_z_loss_func, load_balancing_loss_func, SwitchTransformersTop1Router
from transformers.pytorch_utils import Conv1D
from transformers.activations import ACT2FN
from typing import Optional, Tuple, Union

class Expert(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)
    
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GPT2SparseMLP(nn.Module):
    """
        router config params:
        num_experts:int, expert_capacity:int,
        hidden_size:int, router_jitter_noise:bool,
        router_ignore_padding_tokens: bool, router_dtype:torch.dtype
    """
    def __init__(self, intermediate_size, config, init_expert=None,router_loss=None):
        super().__init__()
        self.router = SwitchTransformersTop1Router(config)
        self.router.classifier = Conv1D(config.num_experts, config.hidden_size)
        self.experts = nn.ModuleList([deepcopy(init_expert) for _ in range(config.num_experts)]) if init_expert else nn.ModuleList([Expert(intermediate_size, config) for _ in range(config.num_experts)])
        #print("Number of params in expert: ", sum(p.numel() for p in init_expert.parameters() if p.requires_grad))
        self.router_loss = router_loss

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]):
        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_indices = torch.argmax(router_mask, dim=-1)
        next_states = hidden_states.clone()
        for idx, expert in enumerate(self.experts):
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = expert(hidden_states[token_indices]).to(next_states.dtype)

        hidden_states = router_probs * next_states

        if self.router_loss and self.router_loss.compute:
            router_loss = self.router_loss
            router_loss.z_value = router_z_loss_func(router_logits)*router_loss.z_coef
            router_loss.aux_value = load_balancing_loss_func(nn.Softmax(dim=-1)(router_logits), expert_indices)*router_loss.aux_coef
        
        return hidden_states


