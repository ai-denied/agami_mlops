def check_data_path(data_path: str):
    if not os.path.exists(data_path):
        print("\n[에러] 데이터 파일을 찾을 수 없습니다.")
        print(f"입력한 경로: {data_path}")
        raise FileNotFoundError(data_path)


def diagnose_dataset(samples: List[Dict], out_dir: str):
    rows = []
    for d in samples:
        rows.append({
            "label": d.get("label"),
            "user_id": d.get("user_id"),
            "participant_id": d.get("participant_id"),
            "real_user_id": d.get("real_user_id"),
            "bot_type": d.get("bot_type"),
            "image_id": d.get("image_id"),
            "original_file": d.get("original_file"),
            "source_file": d.get("source_file"),
        })

    df = pd.DataFrame(rows)

    print("\n[데이터 진단]")
    print("전체 데이터 수:", len(df))
    print("label 분포:", df["label"].value_counts(dropna=False).to_dict())
    print("user_id 고유 개수:", df["user_id"].nunique(dropna=False))
    print("participant_id 고유 개수:", df["participant_id"].nunique(dropna=False))
    print("real_user_id 고유 개수:", df["real_user_id"].nunique(dropna=False))
    print("bot_type 고유 개수:", df["bot_type"].nunique(dropna=False))
    print("image_id 고유 개수:", df["image_id"].nunique(dropna=False))
    print("original_file 고유 개수:", df["original_file"].nunique(dropna=False))
    print("source_file 고유 개수:", df["source_file"].nunique(dropna=False))

    path = os.path.join(out_dir, "dataset_diagnostics.csv")
    df.to_csv(path, index=False)
    print(f"데이터 진단 CSV 저장: {path}")