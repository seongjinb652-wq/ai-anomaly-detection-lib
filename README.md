# ai-anomaly-detection-lib
anomaly-detection-toolkit | AI 기반 이상 탐지 라이브러리 


최신 침입 탐지 데이터셋들

데이터셋	공개 시점	특징	활용 포인트

NSL-KDD	(2009) **	KDD99의 문제점(중복, 불균형)을 개선한 버전	여전히 벤치마크로 자주 사용

CICIDS2017	(2017)	캐나다 사이버보안 연구소(UNB) 제작, 최신 공격 패턴 포함	DoS, DDoS, Brute Force, Web 공격 등 다양한 트래픽

CSE-CIC-IDS2018	(2018)	CICIDS2017 확장판, 더 많은 공격 시나리오	VPN, Botnet, Web 공격 등 현대적 위협 반영

RT-IoT2022	(2022) ** 	IoT 환경 기반, 다양한 IoT 디바이스와 공격 포함	IoT 보안 연구에 적합 (Brute Force SSH, DDoS, Nmap 등)

HIKARI-2021	(2021)	실제 네트워크 환경 기반 데이터셋	Botnet, Ransomware, Cloud 보안 등 다양한 시나리오

Applications of AI for Anomaly Detection

## Dataset 준비

이 프로젝트는 **KDD Cup 1999 Intrusion Detection Dataset**을 사용합니다.  
데이터 파일(`kddcup.data.corrected`)은 용량이 크기 때문에 GitHub 저장소에는 포함하지 않았습니다.

### 다운로드 방법
1. [KDD Cup 1999 Dataset 공식 페이지](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)에서 `kddcup.data.corrected` 파일을 다운로드합니다.
2. 다운로드한 파일을 프로젝트의 `./data/` 디렉토리에 저장합니다.


### 참고
- 데이터셋은 수백 MB 이상으로 매우 크기 때문에 GitHub에 직접 업로드하지 않습니다.
- 전처리 및 학습 스크립트는 `data_preprocessing_kdd.py`와 `preprocess_data.py`에서 자동으로 이 파일을 불러옵니다.
- 다른 데이터셋(RT-IoT2022 등)을 사용할 경우, 해당 전처리 스크립트를 참고하세요.

