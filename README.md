# ☀ IPS 도드니 딥러닝
**장애 아동 부모를 위한 네트워킹 모바일 서비스**<br/>
<br/><br/>

## 🛠 기술 스택
> **DL Model** : Python, Colab, Visual Studio Code, Flask<br/>

<br/><br/>

## 💻 딥러닝 모델 코드 실행

### KoBERT 모델 'model.pt' 다운로드
```bash
1. https://colab.research.google.com/drive/1AExblxZFg6XVobRefePtkF3zxK2oYmjF?usp=sharing 코랩에
   '한국어_단발성_데이터셋.xslx' 파일 업로드
2. 전체 코드 실행
3. 'model.pt'를 'KoBERT-master' 폴더(전체 파일과 같은 디렉토리)에 저장
```
<br/>

### 분석 예측 결과 및 Flask 코드 'app.py' 실행
```bash
1. https://huggingface.co/skt/kobert-base-v1/tree/main 링크 안에 있는 7개의 파일 다운
2. 'KoBERT-master/kobert-base-v1' 위치에 저장
```

```bash
※app.py 사용 시 사용된 패키지 모듈 (설치 필수)※
python 3.7 버전 사용
pip install boto3==1.15.18 
pip install gluonnlp==0.8.0 
pip install mxnet==1.7.0.post2 
pip install onnxruntime==1.8.0 
pip install sentencepiece==0.1.96 
pip install torch==1.10.1
pip install transformers==4.29.2
pip install flask

* 272번째 라인의 host에는 로컬IP 주소를 적어주세요
```



<br/><br/>

## 👩🏻‍💻 도드니 팀 딥러닝 개발자 
| 송재희 | 
| :-: |
| [@zzaehee](https://github.com/zzaehee) |
|<img src="https://github.com/zzaehee.png" style="width:150px; height:150px;">|
