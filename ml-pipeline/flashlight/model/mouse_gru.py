class MouseGRUModelV2(nn.Module):
    def __init__(
        self,
        seq_size: int = 7,
        static_size: int = 10,
        hidden: int = 32,
        layers: int = 1,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=seq_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        self.static_mlp = nn.Sequential(
            nn.Linear(static_size, 32),
            nn.LayerNorm(32),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        self.fc_final = nn.Sequential(
            nn.Linear(hidden + 32, 64),
            nn.LayerNorm(64),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, lengths, x_static):
        packed = pack_padded_sequence(
            x_seq,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hn = self.gru(packed)
        gru_out = hn[-1]

        static_out = self.static_mlp(x_static)
        combined = torch.cat([gru_out, static_out], dim=1)
        logits = self.fc_final(combined).view(-1)

        return logits