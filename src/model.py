"""Hybrid RoBERTa + BiLSTM model for fake news detection."""

import json
import os
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from typing import Optional, Dict, Any, Union


class RoBERTaBiLSTM(nn.Module):
    """Hybrid RoBERTa + BiLSTM model for binary classification."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 2,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers 
        self.dropout = dropout
        self.use_attention = use_attention

        # Config and RoBERTa encoder
        self.roberta_config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name)

        # Fine-tune RoBERTa end-to-end (NOT frozen). An earlier version of
        # this comment said "Freeze RoBERTa base" while the code below set
        # requires_grad_(True) — that was a stale/incorrect comment, not a
        # frozen encoder. Full fine-tuning is the actual, intended behavior.
        # Set requires_grad_(False) here instead if you want a frozen,
        # feature-extraction-only encoder.
        self.roberta.requires_grad_(True)

        # BiLSTM on top of RoBERTa last hidden state
        self.lstm = nn.LSTM(
            input_size=self.roberta_config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Attention layer components
        if self.use_attention:
            self.attention_weights = nn.Linear(lstm_hidden_size * 2, 1)

        # Dropout and linear classifier head
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # Pass through RoBERTa base
        roberta_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = roberta_outputs.last_hidden_state

        # Pass through BiLSTM
        lstm_output, _ = self.lstm(sequence_output)

        if self.use_attention:
            att_scores = self.attention_weights(lstm_output)  # (B, T, 1)

            # Mask out padding positions before the softmax. Without this,
            # padded timesteps (attention_mask == 0) can still receive
            # nonzero attention weight and dilute the context vector,
            # especially for short inputs with heavy padding.
            mask = attention_mask.unsqueeze(-1).to(dtype=att_scores.dtype)  # (B, T, 1)
            att_scores = att_scores.masked_fill(mask == 0, float("-inf"))

            att_weights = torch.softmax(att_scores, dim=1)
            context_vector = torch.sum(lstm_output * att_weights, dim=1)
        else:
            # Mean over real tokens only, excluding padding.
            mask = attention_mask.unsqueeze(-1).to(dtype=lstm_output.dtype)
            summed = torch.sum(lstm_output * mask, dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            context_vector = summed / counts

        # Apply dropout and classification head
        context_vector = self.dropout_layer(context_vector)
        logits = self.classifier(context_vector)

        # Fixed: Return a dictionary matching your train.py loop requirements
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}

        return logits

    def save_pretrained(self, save_directory: str):
        """Save model weights, the RoBERTa sub-config, and this class's own
        constructor arguments (lstm_hidden_size, lstm_layers, dropout,
        use_attention, num_labels) so from_pretrained can rebuild an
        identical architecture before loading weights into it."""
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        self.roberta_config.save_pretrained(save_directory)

        hybrid_config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "lstm_hidden_size": self.lstm_hidden_size,
            "lstm_layers": self.lstm_layers,
            "dropout": self.dropout,
            "use_attention": self.use_attention,
        }
        with open(os.path.join(save_directory, "hybrid_config.json"), "w") as f:
            json.dump(hybrid_config, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "RoBERTaBiLSTM":
        """Rebuild the exact architecture used at save time (from
        hybrid_config.json) and load the saved weights into it.

        This was previously missing entirely — evaluate.py and predict.py
        both called RoBERTaBiLSTM.from_pretrained(...), which doesn't exist
        on a plain nn.Module, so both scripts crashed immediately on
        import/use. This closes that gap.
        """
        config_path = os.path.join(load_directory, "hybrid_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                hybrid_config = json.load(f)
        else:
            # Backward-compatible fallback for checkpoints saved before
            # hybrid_config.json existed: assume the defaults train.py used.
            hybrid_config = {
                "model_name": "roberta-base",
                "num_labels": 2,
                "lstm_hidden_size": 256,
                "lstm_layers": 1,
                "dropout": 0.3,
                "use_attention": True,
            }

        model = cls(**hybrid_config)
        state_dict_path = os.path.join(load_directory, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model
