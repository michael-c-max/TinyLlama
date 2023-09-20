from transformers import AutoModel,LlamaForCausalLM

import torch
from pathlib import Path
from typing import Optional
class ByteTokenizer:
    def __init__(self) -> None:
        self.backend = "byte"
        self.bos_id = 256
        self.eos_id = 257

    @property
    def vocab_size(self) -> int:
        return 258  # Byte values can range from 0 to 255, plus BOS (256) and EOS (257) tokens

    def token_to_id(self, token: str) -> int:
        if token == '<BOS>':
            return self.bos_id
        elif token == '<EOS>':
            return self.eos_id
        elif len(token.encode('utf-8')) != 1:
            raise ValueError(f"token {token!r} not valid for byte tokenizer")
        return ord(token)

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: bool = False,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        tokens = list(string.encode('utf-8'))
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]

        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        new_tokens = []
        for tok in tokens:
            if tok in {self.bos_id, self.eos_id}:
                continue
            new_tokens.append(tok)
        return bytes(new_tokens).decode('utf-8')

tokenizer = ByteTokenizer()
prompt = "China's imports of seafood from Japan slumped last month as Tokyo started to release treated waste water from the damaged Fukushima nuclear power plant."

model = LlamaForCausalLM.from_pretrained("TinyLlama/bytellama-320M-step-5000").to("cuda")

prompt_id = tokenizer.encode(prompt).unsqueeze(0).to("cuda")
out = model.generate(prompt_id, max_length=1000, do_sample=True, num_return_sequences=1)
import pdb; pdb.set_trace() 