import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from convert_to_moe import createGPT2MoE

if __name__ =="__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    print("Total params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model, router_losses, config = createGPT2MoE(model, num_experts=2, router_loss=True)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model.generate(**encoded_input, max_length = 100)
    print(tokenizer.decode(output[0], skip_special_tokens=True))