import FlashlightCaptcha from './FlashlightCaptcha';
import FaceMissionCaptcha from './FaceMissionCaptcha';
import ImageGridCaptcha from './ImageGridCaptcha';

// =============================================================================
// 캡챠 종류별 라우터.
// 백엔드 ChallengeKind 와 1:1 대응.
//   flashlight        → FlashlightCaptcha
//   face_mission      → FaceMissionCaptcha
//   context_inference → ImageGridCaptcha
// =============================================================================

export default function CaptchaRouter({
  kind,
  spec,
  status,
  error,
  onSubmit,
  onRefresh,
}) {
  if (kind === 'flashlight') {
    return (
      <FlashlightCaptcha
        spec={spec}
        status={status}
        error={error}
        onSubmit={onSubmit}
        onRefresh={onRefresh}
      />
    );
  }

  if (kind === 'face_mission') {
    return (
      <FaceMissionCaptcha
        spec={spec}
        status={status}
        error={error}
        onSubmit={onSubmit}
        onRefresh={onRefresh}
      />
    );
  }

  if (kind === 'context_inference') {
    return (
      <ImageGridCaptcha
        spec={spec}
        status={status}
        error={error}
        onSubmit={onSubmit}
        onRefresh={onRefresh}
      />
    );
  }

  return (
    <div className="mx-auto w-full max-w-[480px] rounded-xl bg-white p-10 text-center shadow-[0_20px_60px_rgba(70,130,255,0.15)]">
      <div className="mb-2 text-3xl">🚧</div>
      <div className="text-lg font-bold text-[#2563eb]">
        "{kind}" 캡챠는 준비 중입니다.
      </div>
      <p className="mt-2 text-sm text-[#8a96ad]">
        현재는 손전등(flashlight) / 안면 미션(face_mission) / 감정 추론(context_inference) 이 지원됩니다.
      </p>
    </div>
  );
}
