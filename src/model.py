"""Hybrid RoBERTa + BiLSTM model for fake news detection."""

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

        # Freeze RoBERTa base
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
            att_scores = self.attention_weights(lstm_output)  
            att_weights = torch.softmax(att_scores, dim=1)     
            context_vector = torch.sum(lstm_output * att_weights, dim=1)  
        else:
            context_vector = torch.mean(lstm_output, dim=1)

        # Apply dropout and classification head
        context_vector = self.dropout_layer(context_vector)
        logits = self.classifier(context_vector)

        # Fixed: Return a dictionary matching your train.py loop requirements
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}

        return logits

    def save_pretrained(self, save_directory: str):
        """Save both model weights and configurations."""
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        self.roberta_config.save_pretrained(save_directory)
