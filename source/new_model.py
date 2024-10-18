# Model structure:
# 1. Define the model structure
# 2. Define the forward pass
# 3. Define the loss function
# 4. Define the optimizer
# 5. Define the training loop
# 6. Define the evaluation loop

# 1. Define the model structure
# encoder - RoBERTa with a pooler
# clustering module for encoder output that
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from torch.nn import AvgPool3d


def minmax_norm(arr, a=0.0, b=1.0):
    """
    Min-Max Normalization

    Args:
        arr: torch.Tensor
        a: float
        b: float

    Returns:
        torch.Tensor
    """
    min_val, max_val = torch.min(arr), torch.max(arr)
    return (arr - min_val) * ((b - a) / (max_val - min_val)) + a


class ClusteringModule(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        self.roberta = RobertaModel.from_pretrained("roberta-base")

    def encode_and_pool(self, input_ids, attention_mask):
        """
        Encode and pool the input using cls pooling

        Args:
            input_ids: torch.Tensor
            attention_mask: torch.Tensor

        Returns:
            torch.Tensor
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        return pooled_output

    def tokenize_and_encode(self, text):
        """
        Tokenize and encode the text

        Args:
            text: str

        Returns:
            torch.Tensor
        """
        tokens = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        pooled = self.encoder(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
        )
        return pooled[0].tolist()
