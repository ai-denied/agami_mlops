#!/usr/bin/env python3
"""current/를 archive/의 이전 버전으로 되돌린다.

promote_model.py가 매 승격 전에 archive/{timestamp}_{old_version}/으로
백업을 남기기 때문에, rollback은 단순히 "가장 최근 archive를 current로
다시 복사"하는 것과 같다 - flashlight에는 이 스크립트가 없다 (archive는
쌓이지만 되돌리는 자동화가 없었음). 운영 중인 모델이 사고를 내면 사람이
기다릴 시간이 없으므로 여기서는 처음부터 만들어 둔다.

사용법:
    python -m context_emotion.deployment.rollback_model --list
    python -m context_emotion.deployment.rollback_model --to-latest-archive
    python -m context_emotion.deployment.rollback_model --to-archive 20260701_120000_v1
"""
import argparse
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from context_emotion.deployment import model_store  # noqa: E402


def list_archives(archive_dir: str):
    if not os.path.isdir(archive_dir):
        return []
    return sorted(os.listdir(archive_dir), reverse=True)


def rollback(
    archive_name: str = None,
    dry: bool = False,
    store_root_override: str = None,
) -> dict:
    """archive/{archive_name}/ -> current/. archive_name이 None이면 가장
    최근 archive를 쓴다. 현재 current는 promote_model.py와 같은 패턴으로
    한 번 더 archive에 백업한 다음 교체한다 - 즉 rollback도 흔적을 남긴다."""
    store_root, _, current_dir, archive_dir = model_store.resolve_store_paths(store_root_override)

    archives = list_archives(archive_dir)
    if not archives:
        raise FileNotFoundError(f"archive가 비어 있습니다: {archive_dir}")

    target = archive_name or archives[0]
    target_dir = os.path.join(archive_dir, target)
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"archive를 찾을 수 없습니다: {target_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # microseconds - see promote_model.py for why
    print("=" * 60)
    print("  context_emotion model-store ROLLBACK")
    print("=" * 60)
    print(f"  대상 archive: {target}")
    print(f"  current:      {current_dir}")
    if dry:
        print("  모드:          DRY-RUN (파일 변경 없음)")
    print()

    print("[1/3] rollback 전 current를 한 번 더 archive에 백업")
    pre_rollback_backup = os.path.join(archive_dir, f"{timestamp}_pre_rollback")
    if not dry and os.path.isdir(current_dir):
        shutil.copytree(current_dir, pre_rollback_backup)
        print(f"  [OK] archive/{os.path.basename(pre_rollback_backup)}/")
    elif dry:
        print(f"  [DRY-RUN] archive/{os.path.basename(pre_rollback_backup)}/ 생성 예정")

    print("\n[2/3] staging 준비 (archive -> staging)")
    staging_dir = os.path.join(store_root, f"current_staging_{timestamp}")
    if not dry:
        shutil.copytree(target_dir, staging_dir)
        print(f"  [OK] {os.path.basename(staging_dir)}/")
    else:
        print(f"  [DRY-RUN] {os.path.basename(staging_dir)}/ 생성 예정")

    print("\n[3/3] current 교체")
    if dry:
        print("  [DRY-RUN] 실제 파일 변경 없음")
    else:
        old_tmp = os.path.join(store_root, "_current_old_rollback")
        if os.path.isdir(current_dir):
            os.rename(current_dir, old_tmp)
        try:
            os.rename(staging_dir, current_dir)
        except Exception:
            if os.path.isdir(old_tmp):
                os.rename(old_tmp, current_dir)
            raise
        if os.path.isdir(old_tmp):
            shutil.rmtree(old_tmp, ignore_errors=True)
        print(f"  [OK] current <- archive/{target}/")

    print()
    print("=" * 60)
    print("  [DRY-RUN 완료]" if dry else "  ROLLBACK 완료")
    print("=" * 60)
    return {"rolled_back": not dry, "from_archive": target}


def _parse_args():
    ap = argparse.ArgumentParser(description="current/를 archive/의 이전 버전으로 롤백")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="archive 목록만 출력")
    group.add_argument("--to-latest-archive", action="store_true")
    group.add_argument("--to-archive", metavar="ARCHIVE_NAME")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = _parse_args()
    _, _, _, archive_dir = model_store.resolve_store_paths()

    if args.list:
        archives = list_archives(archive_dir)
        if not archives:
            print(f"archive가 비어 있습니다: {archive_dir}")
        else:
            print(f"archive ({len(archives)}개, 최신순):")
            print(f"{'NAME':<40} {'metadata.json version':<20} {'promoted_at'}")
            for a in archives:
                # 디렉터리 이름의 {old_version} 부분은 promote_model.py가 그때
                # current였던 모델의 버전을 적은 것 - 'pre_rollback'이면 진짜
                # 모델 버전이 아니라 "rollback 직전 current 스냅샷"이라는 뜻이니
                # metadata.json을 직접 읽어서 실제 버전을 보여준다 (디렉터리
                # 이름만 보고 고르면 헷갈릴 수 있음).
                meta_path = os.path.join(archive_dir, a, "metadata.json")
                version = promoted_at = "?"
                if os.path.isfile(meta_path):
                    try:
                        meta = model_store.load_json(meta_path)
                        version = meta.get("version", "?")
                        promoted_at = meta.get("promoted_at", "?")
                    except Exception:
                        pass
                print(f"{a:<40} {version:<20} {promoted_at}")
        return

    # mutually_exclusive_group(required=True)이 --list/--to-latest-archive/
    # --to-archive 중 정확히 하나를 보장하므로, 여기 도달했다면
    # --to-latest-archive 또는 --to-archive 둘 중 하나다. archive_name=None은
    # rollback()에게 "가장 최근 archive를 써라"라는 뜻 - --to-latest-archive일
    # 때도 args.to_archive가 그냥 None이라 같은 분기를 타지만, 의도를
    # 명시적으로 남겨서 나중에 그룹 옵션이 늘어나도 헷갈리지 않게 한다.
    assert args.to_latest_archive or args.to_archive, "argparse mutually_exclusive_group(required=True) 위반"

    try:
        rollback(
            archive_name=args.to_archive if args.to_archive else None,
            dry=args.dry_run,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n[FAILED] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
