from gpt2_moe_layer import GPT2SparseMLP
from gpt2_moe_config import GPT2MoEConfig

class RouterLoss:
    def __init__(self, config) -> None:
        self.z_value = 0
        self.z_coef = config.router_z_loss_coef
        self.aux_value = 0
        self.aux_coef = config.router_aux_loss_coef
        self.compute = True

def createMoELayer(config, init_expert = None, router_loss = False):
    intermediate_size = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    router_loss = RouterLoss(config) if router_loss else None
    return GPT2SparseMLP(intermediate_size, config, init_expert, router_loss)

def createGPT2MoE(baseGPT2Model, num_experts, copy_mlp=True, router_loss=False):
    for param in baseGPT2Model.parameters():
        param.requires_grad = False

    config = GPT2MoEConfig(num_experts=num_experts)
    router_losses = []
    idx = 0
    for _, module in baseGPT2Model.named_modules():
        if "GPT2MLP" in module.__class__.__name__:
            if idx % 2 == 1:
                moe_mlp = createMoELayer(config, module, router_loss) if copy_mlp else createMoELayer(config, router_loss=router_loss)
                for param in moe_mlp.parameters():
                    param.requires_grad = True
                setattr(baseGPT2Model.transformer.h[idx], "mlp", moe_mlp)
                router_losses.append(moe_mlp.router_loss)
            idx += 1
    return baseGPT2Model, router_losses, config