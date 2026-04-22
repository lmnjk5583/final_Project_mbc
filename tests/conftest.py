# 파일 경로: 최종 프로젝트/tests/conftest.py
# 역할: pytest 설정 — legacy 테스트 파일을 수집 대상에서 제외한다.
# 비활성화 보관소
# pytest tests/ 로 한번에 tests 폴더 내의 테스트 코드들 모두 실행 가능

collect_ignore = [
    "test_traffic_analyzer_legacy.py",  # Phase 1 교체 전 구버전 테스트 (비활성화)
    "test_passage_tracker.py",          # passage_tracker 모듈 삭제됨 (125차 이후)
    "test_nm_measurement.py",           # nm 로직 복제본 테스트 — 실제 소스 미임포트, slow_upper_nm 상수 구버전(0.50) 불일치
]
