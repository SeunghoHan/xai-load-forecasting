import torch
import torch.nn as nn

from .base import BaseModel

class GRUModel(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(self.dropout(out[:, -1, :]))
        return out
        
    def predict(self, x):
        """
        Perform prediction with the GRU model.
        """
        self.eval()  # Evaluation mode
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)
            outputs = self.forward(x)
            return outputs.cpu().numpy()

