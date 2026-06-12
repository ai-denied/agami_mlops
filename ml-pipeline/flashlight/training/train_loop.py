def make_loader(
    samples: List[Dict],
    normalizer: MouseFeatureNormalizer,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seq_noise_std: float,
    static_noise_std: float,
    training: bool,
):
    return data.DataLoader(
        MouseDataset(
            samples,
            normalizer,
            seq_noise_std=seq_noise_std,
            static_noise_std=static_noise_std,
            training=training,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_pos_weight(samples: List[Dict], device):
    labels = np.array([int(s.get("label", 0)) for s in samples])
    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)

    if pos == 0:
        return torch.tensor([1.0], device=device)

    return torch.tensor([neg / pos], dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip: float):
    model.train()

    total_loss = 0.0
    total_count = 0

    for x_seq, lengths, x_static, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        x_static = x_static.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x_seq, lengths, x_static)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_count = 0

    for x_seq, lengths, x_static, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        x_static = x_static.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x_seq, lengths, x_static)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_risk_scores(model, loader, device):
    """
    기존 bot_probability 표현 대신 bot_risk_score로 사용한다.
    출력값은 확률 보정된 진짜 확률이라기보다 모델 기반 위험도 점수로 해석한다.
    """
    model.eval()

    all_scores = []
    all_labels = []

    for x_seq, lengths, x_static, y in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        x_static = x_static.to(device, non_blocking=True)

        logits = model(x_seq, lengths, x_static)
        scores = torch.sigmoid(logits)

        all_scores.extend(scores.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

    return np.array(all_scores), np.array(all_labels)