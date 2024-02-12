from transformers import GPT2Config

class GPT2MoEConfig(GPT2Config):
    def __init__(self, num_experts = 2, expert_capacity = 64, router_jitter_noise = 0.01,
                 router_ignore_padding_tokens = False, router_bias = False, router_dtype="float32",
                 router_z_loss_coef=0.001,
                 router_aux_loss_coef=0.001
                 ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.router_jitter_noise = router_jitter_noise
        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        self.router_bias = router_bias # filler attributed. No use.
        self.router_dtype = router_dtype
        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef