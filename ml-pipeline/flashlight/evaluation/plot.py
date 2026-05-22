def plot_loss(train_losses, val_losses, out_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.title("Training / Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Loss 그래프 저장: {path}")


def plot_score_distribution(y_true, scores, out_dir):
    human_scores = scores[y_true == 0]
    bot_scores = scores[y_true == 1]

    plt.figure(figsize=(10, 5))
    plt.hist(human_scores, bins=30, alpha=0.6, label="Human")
    plt.hist(bot_scores, bins=30, alpha=0.6, label="Bot")
    plt.title("Predicted Bot Risk Score Distribution")
    plt.xlabel("Bot Risk Score")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "risk_score_distribution_test.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"위험도 점수 분포 그래프 저장: {path}")


def plot_threshold_metrics(threshold_df, out_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(threshold_df["threshold"], threshold_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(threshold_df["threshold"], threshold_df["f1_bot"], marker="o", label="Bot F1")
    plt.plot(threshold_df["threshold"], threshold_df["human_block_rate"], marker="o", label="Human Block Rate")
    plt.plot(threshold_df["threshold"], threshold_df["bot_miss_rate"], marker="o", label="Bot Miss Rate")
    plt.title("Threshold Metrics")
    plt.xlabel("Threshold")
    plt.ylabel("Score / Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "threshold_metrics_test.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Threshold 그래프 저장: {path}")


def plot_roc_curve(y_true, scores, out_dir):
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "roc_curve_test.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"ROC 그래프 저장: {path}")


def plot_pr_curve(y_true, scores, out_dir):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    auc = average_precision_score(y_true, scores)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"PR-AUC={auc:.4f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "pr_curve_test.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"PR 그래프 저장: {path}")