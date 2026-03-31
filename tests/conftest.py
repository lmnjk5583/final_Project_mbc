# 파일 경로: 최종 프로젝트/tests/conftest.py
# 역할: pytest 설정 — legacy 테스트 파일을 수집 대상에서 제외한다.

collect_ignore = [
    "test_traffic_analyzer_legacy.py",  # Phase 1 교체 전 구버전 테스트 (비활성화)
]
