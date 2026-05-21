# 1. Python 3.11 슬림 이미지 사용 (가볍고 안정적임)
FROM python:3.11-slim

# 2. 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# 3. 환경 변수 설정 (Python 출력 버퍼링 방지 및 pyc 파일 생성 억제)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 4. 의존성 패키지 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 프로젝트 소스 코드 전체 복사
COPY . .

# 6. FastAPI 앱 실행 (Uvicorn 서버 사용)
# (주의: 엔트리포인트 파일명이 main.py이고 FastAPI 객체가 app인 경우)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]