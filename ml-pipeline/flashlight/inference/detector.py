class MouseBotRiskDetector:
    """
    FastAPI에 붙일 때 사용할 수 있는 추론용 클래스.
    bot_probability가 아니라 bot_risk_score로 반환한다.
    """

    def __init__(
        self,
        model_path: str,
        normalizer_path: str,
        metadata_path: str,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device("auto")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.normalizer = joblib.load(normalizer_path)

        self.model = MouseGRUModelV2(
            seq_size=len(SEQ_FEATURES),
            static_size=len(STATIC_FEATURES),
            hidden=self.metadata["hidden"],
            layers=self.metadata["layers"],
            dropout=self.metadata["dropout"],
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.low_risk_threshold = float(self.metadata["low_risk_threshold"])
        self.high_risk_threshold = float(self.metadata["high_risk_threshold"])

    @torch.no_grad()
    def predict_one(self, sample: Dict) -> Dict:
        seq = self.normalizer.transform_seq(sample)
        static = self.normalizer.transform_static(sample)

        x_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(seq)], dtype=torch.long).to(self.device)
        x_static = torch.tensor(static, dtype=torch.float32).unsqueeze(0).to(self.device)

        logits = self.model(x_seq, lengths, x_static)
        bot_risk_score = torch.sigmoid(logits)[0].detach().cpu().item()

        risk_band = classify_single_attempt_risk(
            bot_risk_score,
            low_risk_threshold=self.low_risk_threshold,
            high_risk_threshold=self.high_risk_threshold,
        )

        return {
            "bot_risk_score": float(bot_risk_score),
            "risk_band": risk_band,
            "low_risk_threshold": self.low_risk_threshold,
            "high_risk_threshold": self.high_risk_threshold,
        }

    def decide_three_attempts(self, scores: List[float]) -> Dict[str, Any]:
        policy = self.metadata["three_attempt_policy"]
        return apply_three_attempt_policy(
            scores,
            low_risk_threshold=self.low_risk_threshold,
            high_risk_threshold=self.high_risk_threshold,
            block_suspicious_count=int(policy["block_suspicious_count"]),
            block_high_risk_count=int(policy["block_high_risk_count"]),
            block_total_score=float(policy["block_total_score"]),
        )