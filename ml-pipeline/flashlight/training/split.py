def resolve_group(sample: Dict, idx: int, group_key: str) -> str:
    label = int(sample.get("label", 0))

    if group_key == "auto":
        if label == 0:
            return str(
                sample.get("participant_id")
                or sample.get("real_user_id")
                or sample.get("user_id")
                or sample.get("original_file")
                or sample.get("source_file")
                or f"sample_{idx}"
            )
        return str(
            sample.get("bot_type")
            or sample.get("original_file")
            or sample.get("source_file")
            or f"sample_{idx}"
        )

    return str(
        sample.get(group_key)
        or sample.get("original_file")
        or sample.get("source_file")
        or f"sample_{idx}"
    )


def has_both_classes(samples: List[Dict]) -> bool:
    labels = [int(s.get("label", 0)) for s in samples]
    return len(set(labels)) == 2


def group_split_once(
    samples: List[Dict],
    test_size: float,
    group_key: str,
    seed: int,
) -> Tuple[List[Dict], List[Dict], str]:
    labels = np.array([int(s.get("label", 0)) for s in samples])
    groups = np.array([resolve_group(s, i, group_key) for i, s in enumerate(samples)])

    if len(np.unique(groups)) < 4:
        train_part, test_part = train_test_split(
            samples,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
        return train_part, test_part, "stratified_random_fallback"

    splitter = GroupShuffleSplit(
        n_splits=20,
        test_size=test_size,
        random_state=seed,
    )

    for train_idx, test_idx in splitter.split(samples, labels, groups):
        train_part = [samples[i] for i in train_idx]
        test_part = [samples[i] for i in test_idx]

        if has_both_classes(train_part) and has_both_classes(test_part):
            return train_part, test_part, f"group_split_{group_key}"

    train_part, test_part = train_test_split(
        samples,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return train_part, test_part, "stratified_random_fallback"


def make_train_val_test_split(samples: List[Dict], group_key: str, seed: int):
    train_val, test, mode1 = group_split_once(
        samples=samples,
        test_size=0.2,
        group_key=group_key,
        seed=seed,
    )

    train, val, mode2 = group_split_once(
        samples=train_val,
        test_size=0.2,
        group_key=group_key,
        seed=seed + 1,
    )

    print("\n[Split 정보]")
    print(f"1차 split 방식: {mode1}")
    print(f"2차 split 방식: {mode2}")
    print(f"Train: {len(train)}")
    print(f"Val  : {len(val)}")
    print(f"Test : {len(test)}")

    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        labels = [int(s.get("label", 0)) for s in part]
        print(f"{name} label count:", pd.Series(labels).value_counts().to_dict())

    return train, val, test