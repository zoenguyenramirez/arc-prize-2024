# activation_hooks.py

import torch
from src.model import TransformerBlock, Transformer

class ActivationStore:
    def __init__(self):
        self.activations = {}

    def store_activation(self, name, activation):
        if name not in self.activations:
            self.activations[name] = []
        self.activations[name].append(activation)

    def clear(self):
        self.activations.clear()

    def __getitem__(self, key):
        return self.activations[key]
    
    def save(self, filepath):
        """Save the activation store to a file"""
        torch.save(self.activations, filepath)    
    
def register_transformer_hooks(model, activation_store):
    def attention_hook(module, input, output):
        if not module.training:
            activation_store.store_activation('attention_out', output[0].detach())
            activation_store.store_activation('attention_weights', output[1].detach())

    def embedding_hook(module, input, output):
        if not module.training:
            activation_store.store_activation('embedding_out', output.detach())

    def norm1_hook(module, input, output):
        if not module.training:
            activation_store.store_activation('norm1_out', output.detach())

    def feed_forward_hook(module, input, output):
        if not module.training:
            activation_store.store_activation('feed_forward_out', output.detach())

    def norm2_hook(module, input, output):
        if not module.training:
            activation_store.store_activation('norm2_out', output.detach())

    def initial_dropout_hook(module, input, output):
        if not module.training:
            activation_store.store_activation('initial_dropout_out', output.detach())

    def transformer_input_hook(module, input, output):
        if not module.training:
            activation_store.store_activation('transformer_input_x', input[0].detach())
            activation_store.store_activation('transformer_input_mask', input[1].detach())
            
    for name, module in model.named_modules():
        if isinstance(module, TransformerBlock):
            module.attention.register_forward_hook(attention_hook)
            module.norm1.register_forward_hook(norm1_hook)
            module.feed_forward.register_forward_hook(feed_forward_hook)
            module.norm2.register_forward_hook(norm2_hook)

        if isinstance(module, Transformer):
            module.register_forward_hook(transformer_input_hook)
            module.dropout.register_forward_hook(initial_dropout_hook)            
            module.embedding.register_forward_hook(embedding_hook)