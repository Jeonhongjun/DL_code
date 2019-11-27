# View Count Prediction for New Contents

## 1. 개요
본 모듈은 contents provider가 제공하는 VOD 컨텐츠의 정보를 활용하여 아래의 기능을 제공하는 모듈이다.

1) 신규 컨텐츠 여부를 확인한 후, 데이터를 전처리하고
2) 신규 컨텐츠의 hot 또는 cold여부를 예측한다.

본 모듈은 REST API 형태로 개발되어 resource relocator등 다른 서비스가 연결된다.

## 2. 가정
본 모듈은 개발에 있어 아래와 같은 가정하에 개발되었다.

- 신규 컨텐츠는 최근 이틀내에 새롭게 출시된 컨텐츠를 의미한다.
- 컨텐츠의 hot과 cold는 인기도를 의미하며, 전체 상위 20% 컨텐츠는 hot, 나머지는 cold 컨텐츠를 의미한다.
- 사용되는 컨텐츠 데이터는 추후 신규 고객사에 따라 변경가능하나 본 모듈의 코드는 CJ사의 데이터에 기반한다.

본 가정은 향후 본 모듈은 활용되는 고객사에 따라 변경가능하다.

## 3. 구성
본 모듈은 아래와 같이 3가지로 구성되어 있다.

|모듈명 | 파일명 | 역할 |
| --| --| --|
|preprocessor   |preprocessor.py   | 저장된 컨텐츠 데이터 베이스로 접속하여 입력된 날짜를 기준으로 신규 컨텐츠를 추출하고 전처리를 실행한다.  |
|build_model   |build_model.py   | 저장된 컨텐츠 데이터 베이스에서 데이터를 추출하여, 사용가능한 모델을 생성한다. 생성된 모델은 pickle로 영속화한다.  |
|run_api   |run_api.py   | 외부에서 접속 가능하도록 REST API를 실행한다.  |

![모델의 구성](./image/model.png)

## 4. 사용방법
본 모듈의 사용하기 위해선 RESTfull API를 사용해야 하며, 인터페이스는 아래와 같이 정의된다.

#### 1) 호출명 - New contents prediction (/contents/new/:DATE)

###### - Type 및 설명
![](https://img.shields.io/badge/TYPE-GET-blue.svg)
특정 날짜에 대한 신규 컨텐츠의 hot/cold 여부를 반환함

###### - 파라메터
|parameter   | 설명  |  예시 |
| -- | -- | -- |
| DATE  | 분석 대상이 되는 날짜를 입력함, 형식은 yyyymmdd  | 20180301 |

###### - 출력결과
아래와 같은 결과가 출력되며, 각 파라메터에 대한 설명은 아래 표와 같음
```json
{"items":[{"contentID":"E000628113","viewCount":0.04081472358689787}]}
```
|parameter   | 설명  |  예시 |
| -- | -- | -- |
| contentID  | 특정 컨텐츠의 ID 정보를 표시함  | E000628113 |
| viewCount  | 해당 컨텐츠의 hot 컨텐츠가 될 확률 | 0.040814 |


## 5. 테스트 방법
본 모듈은 아래 세 가지에 대한 테스트를 실행하며 그 테스트의 실행 방법은 아래와 같음

| 테스트명 | 설명 |
| --- | ---|
| 전처리 모듈 테스트  | 매일 생성되는 컨텐츠 정보로 부터 신규 컨텐츠를 분류 하고 전처리를 수행함  |
| 모델 생성 및 모델 성능 결과  | 생성된 모델의 성능이 F1-score 기준으로 75% 이상을 달성함   |
| REST API 작동 여부  | 생성된 모델을 사용하여 REST API 운영함   |

#### 1) 전처리 모듈 테스트

- 본 모듈은 아래와 같은 Unit test 를 실시하였음.

| test case id | test description |
| --- | ---|
| VCN_TC_001  | 학습 전에 데이터 재구성에 필요한 로그데이터를 데이터베이스로부터 전송받을 수 있는지 확인 한다.  |
| VCN_TC_002  | 데이터베이스로 부터 받은 정보를 사용하여, 머신러닝 모델 개발을 위한 전처리 데이터를 생성한다.   |

#### 2) 모델 생성 및 모델 성능
- 본 모듈은 GCS 사업계획서상 아래와 같이 성능 측정 지표를 기술하였음

> 1개월의 로그를 이용하여 신규 콘텐트 입수 시 뷰 카운트 예측을 하여 이 값이 실제 예측치와 같은지 로그를 통하여 확인. 2차년도 목표는 정확도 75% 이상

- 이를 기준으로 1개월의 로그 데이터를 기준으로 Viewcount 상위 20% 의 Hot contents에 대한 평균 precision과 recall의 조화평균인 **F1-score가 75% 이상** 을 성공 기준으로 설정함

- Unit Test 로 Test 가 수행되며 전처리된 데이터들 중 무작위로 한달가량의 30일을 random sample 로서 성능을 측정한다.
expected result 로 아래와 같은 30일에 대한 평균 예측성능이 전송되어야한다.
```json
F1-score : 0.834032986234
```

- 본 모듈은 아래와 같은 Unit test 를 실시하였음.

| test case id | test description |
| --- | ---|
| VCN_TC_003  | 특정 날짜의 데이터를 사용해서 신규 영상에 대한 선호 예측치를 생성한다.  |
| VCN_TC_004  | 요청된 모든 예측 결과에 대해 json 형식으로 content relocater 에게 제대로 전송이 되는지 확인한다   |
| VCN_TC_005  | 분석이 진행된 30일의 데이터에 대한 평균 성능을 보여주는지 확인한다.   |


#### 3) REST API 작동 여부  

- Viewcount Predictor 는 Contents Relocater 와의 interface 로써 REST API 를 사용한다.

- 본 모듈은 아래와 같은 Unit test 를 실시하였음.

| test case id | test description |
| --- | ---|
| VCN_TC_006  | 서버와의 통신 내용을 외부에서 볼 수 없도록 보호되는지 확인한다.  |
| VCN_TC_007  | 각 날짜 예측 모델의 결과 서버에 저장되어야 한다.   |
| VCN_TC_008  | REST API 요청에대해 평균 20초이내 응답한다.   |

## 6. 테스트 결과
#### 1) 전처리 모듈 테스트 결과 - 합격
> 사진등 관련자료 업데이트 예정

#### 2) 모델 생성 및 모델 성능 결과 - 합격

- VCN_TC_003
> Unit test 결과 **4.사용방법** 에서 기재한 출력결과와 동일한 형식의 결과값이 생성됨이 확인됨.

![REST API](./image/VCN_TC_03.png)

- VCN_TC_004
> Contents Relocater 와의 intercace 를 통해 정상적으로 json 형식으로 결과값이 제공되는 것이 확인됨.

![REST API2](./image/VCN_TC_004.png)

- VCN_TC_005
> Unit test 결과 날짜에 대한 Random sampling 및 모델 구성, 평균 성능 message 제출 모두 통과됨이 확인됨.

![모델 성능](./image/VCN_TC_005.png)

> 전처리된 모든 날짜 데이터에 대한 최종 성능은 아래와 같이 성능 기준값을 상회함

Metric | Performance | 성능 기준값 |
-------- | -------- | ---- |
Recall | 0.773 | N/A |
Precision | 0.956 | N/A |
F1-score | 0.868 | 0.75이상 |

#### 3) REST API 작동 여부 결과 - 합격

- VCN_TC_006
> 메세지를 암호화할 수 있게 기존 환경인 HTTP 가 아닌 HTTPS 환경으로 json 결과값이 전송되는 것을 확인함. 

![REST API3](./image/VCN_TC_006.png)

- VCN_TC_007
> 각 날짜 예측 모델의 결과 서버에 저장됨이 확인됨.

![REST API4](./image/VCN_TC_007.png)

- VCN_TC_008
> REST API 요청에 대해 1일치의 Test 결과 예측에 15.716s가 소요되어 20초이내 응답함이 확인됨.

![REST API5](./image/VCN_TC_008.png)





## APPENDIX I - 시스템 설치
### 1) AWS 계정 접속
- [로그인](https://082351608649.signin.aws.amazon.com/console)
- Username: gcs_g
- Password: awsg1020*

> 인스턴스 Launch 를 위한 AWS 계정 정보 입니다.

### 2) AMI Launch
- AMI Name : gcs_g_AMI
- AMI ID : ami-09e535a404bcfd31d
> 1. EC2 console 을 엽니다.
> 2. Dash board 에서 [AMI]를 선택합니다.
> 3. 새 인스턴스를 시작할 때 사용할 AMI를 찾습니다. 위의 기재한 정보의 AMI 를 선택합니다.
> 4. AMI를 선택하고 [Launch]를 선택합니다. (pem키는 key 폴더에 저장되어있습니다.)

- 인스턴스 스펙
  - name: t2.large
  - core_n: 2
  - mem_size: 8G
  - EBS_size: 16G
  - Elastic IP: 54.208.190.38:5000
> 이미지 생성(AMI) 시킨 서버 인스턴스 spec 입니다.

- AWS configure
```
sudo pip install awscli
aws configure
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE  
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

> Local 환경에서 boto3 를 통해 해당 AWS 서버로 access 하기 위해서는 AWS CLI 를 구성해줘야합니다.
위의 Access Key 와 Secret Access Key 를 통해 boto3 를 통한 서버 자동화 Process 가 구현 가능합니다.

### 3) RDS Information
- Instance: gcs_database
- Hostname: gcsdbinstance.cu0nuaw6yxna.us-east-1.rds.amazonaws.com
- Port: 3306
- Username: gcs_g
- Password: awsg1020*
> Test 에 쓰일 CJ log data 가 있는 데이터베이스 입니다. test 에서 pymysql library 를 통해 connect 됩니다.


### 4) Install package
ViewCount Prediction 모델 전처리 및 분석, api 구축에 필요한 library, package 를 설치합니다.

```
pip install virtualenv
virtualenv 가상환경이름
source 가상환경이름/bin/activate
```
> 먼저 가상환경을 구성해줍니다.

```
sudo apt-get update

pip install flask
pip install numpy
pip install pandas
pip install xgboost
pip install sklearn
pip install pymysql
pip install boto3
```
> 그리고 필요한 library 들을 install 해줍니다.

### 5) API
- [API url](http://54.208.190.38:5000/contents/new/20160402)
> contents relocater 에게 전달될 program episode 들의 viewcount 값에 대한 API url 입니다.


## APPENDIX II - 모듈 파일 구성

| 파일명 | 설명 |
| --- | --- |
|preprocessor.py  | 서버로부터 전송받은 로그데이터를 전처리하는 파이썬 스크립트 |
|build_model.py  | ViewCount 를 계산하는 모델의 스크립트 |
|viewcount_predictor.py| api resource 가 될 model class 가 있는 파이썬 스크립트|
|viewcount_unittest.py| 모델 검증을 위한 Unit test code|
|evaluate_viewcount.py| 모델의 Predict Performance 를 계산하는 파이썬 스크립트|
|run_api.py   |플라스크 서버를 올리는 파이썬 스크립트   |
|data   | 모델 구성에 필요한 임시저장파일 및 episode 와 매핑할 meta data 파일  |
|key/*   | pem키 전용 보관 폴더  |
