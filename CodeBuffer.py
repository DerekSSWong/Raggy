import torch.nn.functional as F
from chromadb import Documents, Embeddings
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


class Qwen3Embedder:
    def last_token_pool(self, last_hidden_states: Tensor,
        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
        self.model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')

    def __call__(self, input: Documents) -> Embeddings:
        self.batch_dict = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        self.batch_dict.to(self.model.device)
        self.outputs = self.model(**self.batch_dict)
        self.embeddings = self.last_token_pool(self.outputs.last_hidden_state, self.batch_dict['attention_mask'])

        # normalize embeddings
        self.embeddings = F.normalize(self.embeddings, p=2, dim=1)
        return self.embeddings.tolist()
    def name(self):
        return 'Qwen3Embedder'