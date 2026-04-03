# 교통 이상징후 탐지 시스템 v2.0
## 기술 스택 및 AI 모델 검토 보고서
*Technology Stack & AI Model Review Report*

| 항목 | 내용 |
|------|------|
| 프로젝트명 | CCTV AI 교통 이상징후 탐지 시스템 v2.0 |
| 작성자 | 박대원 · 이수빈 |
| 작성일 | 2026년 3월 |
| 버전 | v1.0 (최초 작성) |
| 문서 목적 | AI 모델, 백엔드, 프론트엔드, DB 기술 선택 근거 및 비교 검토 |

---

## 목차

1. 문서 개요
2. AI 객체 탐지 모델 검토
3. 백엔드 프레임워크 검토 — Flask
4. 프론트엔드 프레임워크 검토 — React
5. 데이터베이스 검토 — MySQL
6. 전체 기술 스택 종합 검토
7. 핵심 알고리즘 검토
8. 결론 및 향후 개선 방향

---

## 1. 문서 개요

본 문서는 고속도로 CCTV 영상 기반 교통 이상징후 탐지 시스템 v2.0 개발에 적용된 핵심 기술 스택의 선정 배경과 대안 비교 검토 결과를 정리한 보고서이다. AI 객체 탐지 모델(YOLO), 백엔드 프레임워크(Flask), 프론트엔드(React), 데이터베이스(MySQL) 각 영역에 대해 주요 대안과의 정량적·정성적 비교를 수행하고 최종 선택 근거를 명시한다.

### 1.1 시스템 요구사항 요약

기술 스택 선정 기준이 된 핵심 요구사항은 아래와 같다.

| 구분 | 요구사항 | 비고 |
|------|----------|------|
| **실시간성** | FPS ≥ 15 (실시간 처리 기준) | CCTV 영상 30fps 기준 |
| **경량화** | 엣지/CPU 환경에서도 동작 가능 | 고속도로 관제 서버 사양 고려 |
| **정확도** | 역주행·정체 탐지 오탐률 최소화 | 히스테리시스·투표 기법 보완 |
| **확장성** | 다중 CCTV, REST API, 실시간 스트리밍 | Phase 3 풀스택 통합 목표 |
| **개발 효율성** | Python 중심 기술 스택 통일 | AI 엔진과 백엔드 언어 일치 |

---

## 2. AI 객체 탐지 모델 검토

### 2.1 검토 배경

본 시스템은 CCTV 영상에서 차량을 실시간으로 탐지·추적하여 역주행, 교통 정체, 급정거, 사고 의심 등 4종 이상징후를 감지해야 한다. 객체 탐지 모델의 속도와 정확도는 시스템 전체 성능에 직결되므로, 다음 5가지 주요 모델을 비교 검토하였다.

### 2.2 후보 모델 비교 분석

| 모델 | mAP50 (%) | 속도 (FPS) | 파라미터 | 모델크기 | CPU 동작 | 특징 요약 |
|------|-----------|-----------|---------|---------|---------|-----------|
| **★ YOLO11n (선택)** | 56.1 | ~60+ | 2.6M | 5.4MB | ✅ 가능 | 최신 아키텍처, ByteTrack 내장 지원 |
| YOLOv8n | 52.9 | ~55 | 3.2M | 6.3MB | ✅ 가능 | 안정적, YOLO11n 전세대 |
| YOLOv8s | 61.8 | ~40 | 11.2M | 22MB | ⚠️ 제한적 | 정확도 향상, 속도 감소 |
| RT-DETR | 53.1 | ~25 | 32M | 67MB | ❌ 어려움 | Transformer 기반, GPU 필수 |
| EfficientDet-D0 | 33.8 | ~30 | 3.9M | 15MB | ⚠️ 제한적 | 균형형, mAP 낮음 |

> ※ FPS는 GPU(RTX 3080) 기준이며, CPU 환경에서는 약 1/3~1/4 수준. mAP50은 COCO 데이터셋 기준.

### 2.3 YOLO11n 선택 근거

#### (1) 속도와 정확도의 최적 균형

YOLO11n은 전 세대 YOLOv8n 대비 파라미터 수를 22% 줄이면서(3.2M → 2.6M) mAP50은 오히려 3.2%p 향상(52.9 → 56.1)되었다. FPS ≥ 15 요구사항을 CPU 환경에서도 충족하며, GPU 사용 시 60FPS 이상으로 실시간 처리가 가능하다.

#### (2) ByteTrack 추적기 내장 연동

Ultralytics YOLO11n은 `model.track(tracker="bytetrack.yaml", persist=True)` 단일 호출로 탐지와 ID 추적을 동시에 수행한다. ByteTrack은 저확신 검출 박스도 이중 연관(high-score + low-score)으로 처리하여 가림(occlusion) 상황에서의 ID 전환(ID switch)을 최소화한다. 이는 역주행 판정 히스테리시스(8프레임 연속) 알고리즘의 신뢰성에 직접 기여한다.

> ⚠️ config.py 실제값: `wrong_count_threshold=8`

#### (3) 경량 모델로 엣지/관제 서버 배포 적합

모델 크기 5.4MB는 고속도로 관제 서버의 제한된 사양(CPU-only 또는 저사양 GPU) 환경에 적합하다. RT-DETR(67MB), YOLOv8s(22MB) 대비 압도적으로 경량이며, Docker 컨테이너 이미지 크기도 최소화할 수 있다.

#### (4) Python/Ultralytics 생태계 완전 지원

AI 엔진(Python)과 동일한 언어·패키지 생태계에서 학습, 파인튜닝, 추론, 후처리가 일관되게 구현된다. 커스텀 best.pt 모델을 별도 변환 없이 바로 적용할 수 있으며, 향후 교통 특화 데이터셋으로 파인튜닝도 용이하다.

> **결론**: 실시간성(FPS≥15), 경량화(CPU 동작), ByteTrack 내장, Python 생태계 통합 — 4가지 핵심 요구사항을 동시에 충족하는 모델은 YOLO11n이 유일하다.

### 2.4 ByteTrack 추적 알고리즘 검토

| 추적기 | MOTA (%) | 속도 | ID Switch | 특징 |
|--------|---------|------|-----------|------|
| **★ ByteTrack (선택)** | 80.3 | 빠름 | 최소 | 저확신 박스 이중연관, YOLO 내장 |
| DeepSORT | 75.4 | 보통 | 많음 | Re-ID 특징 추출 필요, 추가 모델 필요 |
| SORT | 59.8 | 매우 빠름 | 많음 | 단순 IoU 매칭, ID switch 빈발 |
| StrongSORT | 79.6 | 느림 | 매우 적음 | 정확하지만 연산 부하 큼 |

---

## 3. 백엔드 프레임워크 검토 — Flask

### 3.1 검토 배경

백엔드는 AI 엔진(Python)과의 연동, REST API 제공, WebSocket 실시간 이벤트 푸시, MJPEG CCTV 스트리밍, MySQL 연동의 5가지 역할을 수행해야 한다. Python 기반 주요 웹 프레임워크를 비교 검토하였다.

### 3.2 후보 프레임워크 비교

| 프레임워크 | 성능 | 난이도 | Python AI 연동 | 주요 특징 |
|-----------|------|--------|---------------|-----------|
| **★ Flask (선택)** | 충분 | 낮음 | ✅ 최고 | 마이크로 프레임워크, Python 네이티브, 빠른 프로토타이핑 |
| FastAPI | 높음 | 중간 | ✅ 우수 | 비동기(async), 자동 OpenAPI 문서, 타입 검증 |
| Django | 중간 | 높음 | ✅ 가능 | 풀 프레임워크, ORM 내장, 과사양 |
| Node.js (Express) | 높음 | 중간 | ❌ 어려움 | JavaScript 기반, AI 엔진과 언어 불일치, IPC 필요 |

### 3.3 Flask 선택 근거

#### (1) AI 엔진과의 직접 통합 — 언어 통일

Flask와 AI 엔진(YOLO, OpenCV, NumPy) 모두 Python으로 구현되어 있어 별도 IPC(프로세스간 통신)나 REST 중계 없이 직접 함수 호출·객체 공유가 가능하다. Node.js(Express) 선택 시 Python AI 엔진과의 통신을 위해 별도 마이크로서비스 구성이 필요하다.

#### (2) 프로젝트 규모에 최적화

8주 개발 일정 내 단일 팀(2인)이 구현하는 프로토타입에 Django의 관리자 패널, ORM 마이그레이션, 앱 구조 등 풀 프레임워크 기능은 과사양이다. Flask는 필요한 기능(REST API + WebSocket + 스트리밍)만 최소 구성으로 구현할 수 있다.

#### (3) Flask-SocketIO로 WebSocket 실시간 이벤트

`/ws/traffic`(1초 간격 상태 갱신)와 `/ws/alerts`(이상징후 즉시 푸시) 두 채널을 Flask-SocketIO로 구현한다. FastAPI도 WebSocket을 지원하지만, Flask-SocketIO는 재연결 로직·룸(room) 관리·이벤트 네임스페이스를 더 직관적으로 제공한다.

#### (4) SQLAlchemy ORM + MySQL 연동 용이

Flask + SQLAlchemy 조합은 Python 생태계에서 가장 성숙한 ORM 패턴이다. Flask-SQLAlchemy 확장으로 MySQL 연결, 트랜잭션 관리, 모델 정의를 간결하게 처리한다.

> **결론**: Python AI 엔진과의 직접 통합, 프로젝트 규모 적합성, Flask-SocketIO 실시간 이벤트 — 세 가지 핵심 이유로 Flask를 선택한다. 성능 요구사항이 높아지는 Phase 4 이후에는 FastAPI 전환을 검토할 수 있다.

---

## 4. 프론트엔드 프레임워크 검토 — React

### 4.1 검토 배경

대시보드는 CCTV 그리드 뷰(멀티 스트림), 실시간 지표 차트(Chart.js/Recharts), 팝업 알림, 이벤트 로그 뷰어 4가지 주요 UI 컴포넌트로 구성된다. 동적 데이터 갱신(WebSocket 수신 → UI 업데이트)이 핵심 요구사항이다.

### 4.2 후보 프레임워크 비교

| 프레임워크 | 학습 곡선 | 실시간 UI | 생태계 | 특징 |
|-----------|---------|---------|--------|------|
| **★ React (선택)** | 중간 | ✅ 최고 | ✅ 최대 | 컴포넌트 재사용, 상태 관리, 방대한 차트 라이브러리 |
| Vue.js | 낮음 | ✅ 우수 | 중간 | 학습 쉬움, React 대비 생태계 작음 |
| Angular | 높음 | ✅ 우수 | 중간 | TypeScript 강제, 엔터프라이즈 과사양 |
| Vanilla JS | 낮음 | ⚠️ 제한적 | 없음 | 복잡 UI에서 유지보수 불가능 |

### 4.3 React 선택 근거

- 컴포넌트 기반 구조로 CCTV 그리드, 알림 패널, 차트, 로그 뷰어를 독립 컴포넌트로 개발·재사용 가능하다.
- `useState` / `useEffect` Hook으로 WebSocket 메시지 수신 시 즉각적인 UI 업데이트(상태 변경 → 리렌더링)가 직관적으로 구현된다.
- Recharts, Chart.js 모두 React 전용 래퍼를 제공하여 실시간 지표 차트 구현이 용이하다.
- 방대한 npm 생태계로 WebSocket 클라이언트(socket.io-client), MJPEG 스트림 뷰어 등 필요 라이브러리를 즉시 활용할 수 있다.

---

## 5. 데이터베이스 검토 — MySQL

### 5.1 검토 배경

시스템이 저장해야 하는 데이터는 이벤트 로그(역주행/정체 발생 시각·위치·레벨), 프레임별 분석 결과(frame_log.csv 대응), 객체 추적 이력(track_log.csv 대응), CCTV 메타 정보 4가지이다. 데이터 구조가 명확하고 정형화되어 있어 관계형 DB가 적합하다.

### 5.2 후보 DB 비교

| DB | 성능 | 설치 용이성 | Flask 연동 | 특징 |
|----|------|-----------|-----------|------|
| **★ MySQL 8.0 (선택)** | 높음 | ✅ 용이 | ✅ SQLAlchemy | 검증된 RDBMS, 대용량 로그 처리, Docker 공식 이미지 |
| PostgreSQL | 높음 | ✅ 용이 | ✅ SQLAlchemy | 고급 기능 풍부, MySQL 대비 설정 복잡 |
| SQLite | 낮음 | ✅ 내장 | ✅ 내장 | 파일 기반, 다중 쓰기 불가, 프로덕션 부적합 |
| MongoDB | 높음 | 보통 | ⚠️ PyMongo | NoSQL, 스키마 유연하지만 정형 데이터에 과사양 |

### 5.3 MySQL 선택 근거

- frame_log, track_log, events_log 3종 로그는 모두 정형 컬럼 구조를 가지며 CSV 스키마와 1:1 매핑된다. 관계형 모델이 최적이다.
- SQLAlchemy ORM과의 조합으로 Flask 코드에서 SQL 없이 Python 객체로 DB를 조작할 수 있다.
- Docker Compose에서 `mysql:8.0` 공식 이미지를 사용하여 개발/배포 환경을 일관되게 유지한다.
- PostgreSQL도 동급 성능이지만, 팀의 기존 MySQL 경험 및 국내 교통 관제 인프라의 MySQL 표준화를 고려한다.

---

## 6. 전체 기술 스택 종합 검토

### 6.1 최종 선정 스택 요약

| 영역 | 선택 기술 | 버전 | 선택 이유 요약 |
|------|-----------|------|---------------|
| **AI 탐지 모델** | YOLO11n | Ultralytics 8.x | 최고 경량·고속, ByteTrack 내장, CPU 동작 |
| **추적 알고리즘** | ByteTrack | bytetrack.yaml | 저확신 박스 이중연관, ID switch 최소화 |
| **백엔드** | Python Flask | 3.x | AI 엔진 직접 통합, 마이크로 프레임워크, 프로토타이핑 |
| **실시간 통신** | WebSocket (Flask-SocketIO) | 5.x | /ws/traffic (1초 간격) + /ws/alerts (즉시 푸시) |
| **API 문서** | Swagger (Flask-RESTX) | 1.x | 자동 REST API 문서화, 테스트 UI 제공 |
| **프론트엔드** | React | 18.x | 컴포넌트 재사용, WebSocket 상태 관리, 차트 생태계 |
| **차트/시각화** | Chart.js + Recharts | 4.x / 2.x | 실시간 갱신 차트, React 전용 래퍼 제공 |
| **데이터베이스** | MySQL | 8.0 | 정형 로그 데이터, SQLAlchemy ORM, Docker 공식 이미지 |
| **ORM** | SQLAlchemy (Flask-SQLAlchemy) | 3.x | Python 표준 ORM, MySQL 완전 지원 |
| **영상 처리** | OpenCV | 4.x | CCTV 프레임 처리, MJPEG 인코딩 |
| **수치 연산** | NumPy | 1.x | Grid Flow Map EMA 연산, 벡터 계산 |
| **배포** | Docker + Compose | 24.x | AI엔진·백엔드·MySQL·프론트 일괄 컨테이너화 |

### 6.2 기술 스택 아키텍처 흐름

```
CCTV 영상 입력  →  YOLO11n + ByteTrack (Python)  →  Flask 백엔드
→  WebSocket (/ws/traffic, /ws/alerts)  →  React Dashboard
Flask  →  SQLAlchemy ORM  →  MySQL (이벤트·프레임·추적 로그)
```

---

## 7. 핵심 알고리즘 검토

### 7.1 Grid Flow Map + Cosine Similarity 역주행 판정

역주행 탐지의 핵심 알고리즘은 20×20 Grid Flow Map과 코사인 유사도 투표 방식이다. 이 알고리즘은 사전에 ROI(관심 영역)를 설정하지 않아도 정상 흐름을 자동으로 학습하는 무감독(Unsupervised) 방식으로, 다양한 카메라 설치 환경에 즉시 적용 가능하다.

> ⚠️ config.py 실제값: `grid_size=20` (문서 원본 15×15에서 수정)

| 단계 | 내용 | 수식/기준 |
|------|------|-----------|
| 학습 단계 (LEARNING) | 정상 차량 이동벡터를 20×20 셀에 EMA(α=0.1)로 누적 | flow[r,c] = (1-α)·flow[r,c] + α·(ndx, ndy) |
| 탐지 단계 (DETECTING) | 차량 이동벡터와 Flow Map의 코사인 유사도를 8포인트 샘플링하여 투표 | cos θ = (ndx × flow_vx) + (ndy × flow_vy) |
| 판정 단계 (JUDGMENT) | disagree 비율 ≥ **0.7**이면 의심 카운트 증가, **8**회 연속 시 역주행 확정 | cos θ < **-0.75** → disagree / disagree_ratio ≥ **0.7** → 의심 |
| 히스테리시스 (오탐 억제) | 비의심 프레임에서는 카운트를 2씩 감소하여 일시적 오탐 억제 | count -= 2 (비의심 시) / count >= **8** → 확정 |

> ⚠️ config.py 실제값: `cos_threshold=-0.75`, `vote_threshold=0.7`, `wrong_count_threshold=8`

### 7.2 교통 정체 감지 — LOS 분류 알고리즘

신규 개발하는 `congestion_judge.py` 모듈은 교통류 기본 다이어그램(Fundamental Diagram)의 속도-밀도 역관계 원리를 기반으로 3단계 서비스 수준(LOS)을 자동 분류한다.

| 레벨 | 상태 | 판정 기준 (config.py 실제값) | 조치 |
|------|------|---------------------------|------|
| 🟢 **SMOOTH** | 원활 (LOS A~B) | jam_score < **0.30** | 모니터링 유지 |
| 🟡 **SLOW** | 서행 (LOS C~D) | **0.30** ≤ jam_score < **0.60** | CAUTION 경보 발송 |
| 🔴 **CONGESTED** | 정체 (LOS E~F) | jam_score ≥ **0.60** | WARNING/CRITICAL 경보 |

---

## 8. 결론 및 향후 개선 방향

### 8.1 기술 선택 결론

본 시스템의 기술 스택은 실시간성·경량화·Python 생태계 통합·개발 효율성 4가지 핵심 요구사항을 기준으로 각 영역별 최적 기술을 선정하였다.

| 영역 | 선택 기술 | 핵심 선택 근거 |
|------|-----------|---------------|
| **AI 탐지** | YOLO11n + ByteTrack | CPU 환경 FPS≥15, 파라미터 2.6M(최경량), ID switch 최소화 |
| **백엔드** | Flask + SQLAlchemy | Python AI 직접 통합, WebSocket 실시간, 8주 일정 내 구현 적합 |
| **프론트엔드** | React + Recharts | 컴포넌트 재사용, WebSocket 상태 관리, 실시간 차트 생태계 |
| **데이터베이스** | MySQL 8.0 | 정형 로그 구조 최적, SQLAlchemy ORM, Docker 공식 이미지 |

### 8.2 향후 개선 방향

**단기 (Phase 3~4, ~5월)**

- YOLO11n 커스텀 파인튜닝: 한국 고속도로 CCTV 특화 데이터셋으로 재학습하여 mAP 추가 향상
- FastAPI 전환 검토: 다중 CCTV(4개+) 동시 처리 시 Flask의 GIL 한계 → 비동기 FastAPI 전환
- Redis 도입: 실시간 이벤트 큐(Pub/Sub) 및 세션 캐시로 WebSocket 부하 분산

**중장기 (Phase 5~, ~2026 하반기)**

- LSTM/GRU 정체 예측 모델 통합: 시계열 속도 버퍼로 5분 후 정체 발생 예측
- YOLO11n → YOLO11s 업그레이드: GPU 서버 확보 시 정확도 최우선 전환
- TimescaleDB 또는 InfluxDB 도입: 시계열 교통 데이터에 최적화된 DB로 전환 검토

> 본 보고서의 모든 기술 선택은 2026년 3월 기준 최신 버전 및 벤치마크를 기반으로 작성되었다. 기술 환경 변화에 따라 주기적 재검토를 권장한다.

---

## 참고문헌

1. Liu et al. (2025) — Vehicle Flow Detection and Tracking Based on Improved YOLOv8n and ByteTrack. World Electric Vehicle Journal, MDPI.
2. Real Time Object Detection for Traffic Analysis using YOLOv8 (2024). IJIRSET.
3. Chakraborty et al. (2018) — YOLO vs AlexNet for Congestion Detection. ScienceDirect.
4. Ultralytics YOLO11 Documentation. https://docs.ultralytics.com (2026).
5. ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV 2022.
6. Fundamental Diagram of Traffic Flow. Wikipedia. https://en.m.wikipedia.org/wiki/Fundamental_diagram_of_traffic_flow
7. Flask Documentation. https://flask.palletsprojects.com (2026).
8. React Documentation. https://react.dev (2026).
