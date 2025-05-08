# PBVS 2025 Multi-modal Aerial View Image Challenge Submission

이 저장소는 PBVS 2025 Multi-modal Aerial View Image Challenge - SAR Classification 제출을 위한 LaTeX 파일을 포함하고 있습니다.

## 파일 구조

- `team_submission.tex`: 제출용 메인 LaTeX 문서
- `TEAM-architecture.tex`: 아키텍처 다이어그램 생성용 LaTeX 파일
- `README.md`: 이 파일

## 제출 전 필요한 작업

1. **저자 정보 완성**: 
   - `team_submission.tex` 파일에 저자 이름과 소속 기관 정보 추가
   - 주석에 제공된 형식을 따라 작성

2. **아키텍처 다이어그램 생성**:
   - `TEAM-architecture.tex` 파일을 컴파일하여 PDF 생성:
     ```
     pdflatex TEAM-architecture.tex
     ```
   - 또는 자체 아키텍처 다이어그램을 `TEAM-architecture.pdf` 이름으로 대체

3. **최종 제출물 컴파일**:
   - 메인 문서 컴파일:
     ```
     pdflatex team_submission.tex
     pdflatex team_submission.tex  # 참조를 위해 두 번 실행
     ```

4. **추가 필요 정보**:
   - Codalab 사용자 이름 및 챌린지 제출자 이메일
   - 소스 코드 링크 (GitHub 또는 Google Colab 선호)
   - 제출물이 한 페이지 제한을 준수하는지 확인

## 제출 방법

다음 파일들을 3월 13일까지 justice.wheelwright.ext@afresearchlab.com으로 이메일 전송:
- LaTeX 문서
- 컴파일된 PDF
- 아키텍처 다이어그램 PDF
- 기타 추가 정보 파일

## 모델 개요

KNUNIST 모델은 SAR 이미지 분류를 위해 EO(Electro-Optical) 이미지에서 SAR(Synthetic Aperture Radar) 이미지로의 지식 증류를 활용합니다. 주요 특징:

1. **지식 증류 아키텍처**:
   - ResNet101 백본을 사용한 교사-학생 프레임워크
   - 사전 학습된 EO 모델(교사)이 SAR 모델(학생)을 가이드
   - 예측 신뢰도를 추정하는 추가 헤드

2. **손실 함수**:
   - 분류 손실: 기본 분류 작업을 위한 교차 엔트로피 손실
   - 특징 매칭 손실: SAR와 EO 도메인 간 특징 표현 정렬
   - 대조 손실: SupCon(Supervised Contrastive Learning)을 사용한 특징 구분 강화
   - 신뢰도 손실: 과신뢰를 방지하기 위한 신뢰도 예측 정규화

3. **학습 전략**:
   - EO 사전 학습: EO 도메인에서 사전 학습된 가중치 활용
   - 적응형 람다: 신뢰도 손실 가중치의 동적 조정
   - 클래스 가중치: 데이터셋의 클래스 불균형 해결
   - 다중 GPU 학습: 효율성을 위한 분산 학습

## 참고사항

현재 템플릿은 제공된 정보를 기반으로 작성되었습니다. 모든 필수 정보를 유지하면서 한 페이지 제한에 맞게 내용을 조정해야 할 수 있습니다. # CTD
