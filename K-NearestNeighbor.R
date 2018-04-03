
#=============================================================================================================================================#
#                                                  DataSet1 : 신용카드 사기 고객 분류(분류 문제)                                              #
#                                                                                                                                             #
#                  input : 거래 데이터로 총 30개의 설명변수가 존재 (pca된 28개 변수와 각 거래간 간격, 사용 금액)                              #
#                  ouput : 사기 거래 분류(사기 = 1, 정상 = 0)                                                                                 #
#=============================================================================================================================================#


#############################################################################################################################
########################################################준비단계#############################################################
#############################################################################################################################

#========= 패키지 설치하기 =========#
#install.packages(c("ggplot2","caret","doBy","data.table","class"))


#========= 패키지 불러오기 =========#
library(data.table)
library(doBy)
library(class)
library(ggplot2)
library(caret)

#========= 초기 옵션 지정하기 =========#
options(header=T)
options(stringsAsFactors=F)
# 성능지표계산
perf <- function(pred, real){
  result_table <- table(pred, real)
  if(length(unique(pred))==1){
    if(unique(pred)==0){
      result_table <- rbind(result_table, c(0,0))
    }else{
      result_table <- rbind(c(0,0), result_table)
    }
  }
  TN <- result_table[1,1]
  FN <- result_table[1,2]
  FP <- result_table[2,1]
  TP <- result_table[2,2]
  
  acc <- (TP+TN)/(TN+FN+FP+TP)
  recall <- TP/(TP+FN)
  precision <- TP/(TP+FP+10^-10)
  f1 <- 2*recall*precision/(recall+precision+10^-10)
  
  return(c(acc, recall, precision, f1))
}
setwd("C:\\Users\\jeonyunsu\\Google 드라이브\\[내 공부]\\knn실습\\dataset")

#========= 데이터 불러오기 =========#
credit_fraud<-fread("creditcard.csv")
credit_fraud<-credit_fraud[order(credit_fraud$Class),]
credit_fraud$Class<-as.factor(credit_fraud$Class)

#========= 데이터 나누기 =========#
# 데이터 일부만 사용하기(너무 많아서 학습시간이 오래걸린다.)
set.seed(50)
credit_fraud2 <- sampleBy(~Class, frac=0.5, data=credit_fraud) # 20만건 거래 데이터 중 10만건만 사용

#========= 데이터 특성 확인하기 =========#
dim(credit_fraud2) # nrow, ncol
str(credit_fraud2) # 변수 타입 확인
summary(credit_fraud2) # 기초 통계량 확인 

## knn은 input변수들이 수치형일 때를 생각하고 알고리즘이 개발되었다. 

#========= target 비율 확인하기 =========#
table(credit_fraud2$Class)/nrow(credit_fraud2)

#========= 데이터 분포 확인하기 =========#
circle_size=(as.integer(credit_fraud2$Class)-1)*10 # 사기건이 너무 적어서 안보임. 포인트 크기를 늘려서 분포 확인
p6 <- ggplot(credit_fraud2, aes(x = Amount, y = Time, col=Class, size = circle_size)) +  geom_point()

# 학습 데이터
train_idx <- sampleBy(~Class, frac=c(0.5,0.5), data=cbind(credit_fraud2$Class,(1:nrow(credit_fraud2))))[,2] #학습 - 0.7 / 평가 - 0.3
train.x<-credit_fraud2[as.numeric(train_idx),-"Class"]
train.y<-credit_fraud2[as.numeric(train_idx),"Class"]

# 검증 데이터(적절한 k개를 찾기 위한)
non_train<-credit_fraud2[-as.numeric(train_idx),]
valid_idx <- sampleBy(~Class, frac=c(0.6,0.4), data=cbind(non_train$Class,(1:nrow(non_train))))[,2] #학습 - 0.7 / 평가 - 0.3
valid.x<-non_train[as.numeric(valid_idx),-"Class"]
valid.y<-non_train[as.numeric(valid_idx),"Class"]

# 평가 데이터
test.x<-non_train[-as.numeric(valid_idx),-"Class"]
test.y<-non_train[-as.numeric(valid_idx),"Class"]

#========= 나눠진 데이터의 비율 확인하기 =========#
table(train.y$Class)/nrow(train.y) # 학습
table(valid.y$Class)/nrow(valid.y) # 검증
table(test.y$Class)/nrow(test.y)   # 평가


#############################################################################################################################
######################################################모델링-KNN#############################################################
#############################################################################################################################

#=========1. knn으로 예측하기 =========#
perfor_set<-as.data.frame(matrix(nrow=0, ncol = 5))

# 약 10분 소요 
system.time(
for (k_neighbors in 1:5){
  #========= modeling KNN =========#
  knn.pred<-knn(train.x, valid.x, train.y$Class, k=k_neighbors)
  
  #========= knn으로 예측하기 =========#
  perfor_set<-rbind(perfor_set, append(k_neighbors, perf(pred=knn.pred,real=valid.y$Class))) # ACC, Recall, Precision, F1
}
)
colnames(perfor_set)<-c("k_num", "ACC", "Recall", "Precision", "F1")
best_k<-perfor_set[which.max(perfor_set$Recall),"k_num"]


#========= 평가데이터에 best_k를 적용하여 KNN =========#
best_k = 1
system.time(
  knn.pred<-knn(train.x, test.x, train.y$Class, k=best_k)
)
table(pred=knn.pred, real=test.y$Class)
origin_perf<-perf(pred=knn.pred,real=test.y$Class)


#=========2. 불균형 데이터 보완 - sampling =========#

# Under(Down) Sampling : Majority(다수) Class 군집에 속한 개체(Record) 중 일부만 무작위 추출하여 모델 구축에 사용

raw_data_down <- downSample(train.x, train.y$Class)
train.x_down <- raw_data_down[,-match( "Class",colnames(raw_data_down))]
train.y_down <- raw_data_down[,match( "Class",colnames(raw_data_down))]

#========= best_k 찾기 =========#
perfor_set<-as.data.frame(matrix(nrow=0, ncol = 5))

system.time(
  for (k_neighbors in 1:5){
    #========= modeling KNN =========#
    knn.pred<-knn(train.x_down, valid.x, train.y_down, k=k_neighbors)
    
    #========= knn으로 예측하기 =========#
    table(pred=knn.pred, real=valid.y$Class) # 분류테이블
    perfor_set<-rbind(perfor_set, append(k_neighbors, perf(pred=knn.pred,real=valid.y$Class))) # ACC, Recall, Precision, F1
  }
)
colnames(perfor_set)<-c("k_num", "ACC", "Recall", "Precision", "F1")
best_k<-perfor_set[which.max(perfor_set$Recall),"k_num"]

#========= bset_k로 평가 데이터 예측 =========#
system.time(
  knn.pred<-knn(train.x_down, test.x, train.y_down, k=best_k)
)
table(pred=knn.pred, real=test.y$Class)
downsample_perf<-perf(pred=knn.pred,real=test.y$Class)



#############################################################################################################################
######################################################예측 성능 #############################################################
#############################################################################################################################

#========= preformance 비교 =========#
origin_perf; downsample_perf











#=============================================================================================================================================#
#                                                      DataSet2 : 우유 생산량 예측(예측 문제)                                                 #
#                                                                                                                                             #
#                                   input : 1~t-1 시점의 우유 생산량(마리당)                                                                  #
#                                   ouput : t시점의 우유 생산량(마리당)                                                                       #
#=============================================================================================================================================#

#############################################################################################################################
########################################################준비단계#############################################################
#############################################################################################################################

#========= 초기 옵션 지정하기 =========#
# 이동평균 시켜주는 함수
moveAvg2 <- function(ts, freq=12){
  # freq는 계절 주기 (N), ts는 시계열 데이터
  ma<-c() #이동평균 벡터
  for(t in (freq+1):length(ts)){ ma<-c(ma, mean(ts[(t-freq):(t-1)])) } # t<-13; freq<-12
  return(c(rep(NA,freq),ma))
}

# 계절지수 산출하는 함수
seasonIndex2 <- function(ts, freq=12){ #ts: 원시계열, freq: 계절주기
  ts_ma <- ts[1:length(ts)]/moveAvg2(ts)
  ts_ma <- ts_ma[(freq+1):length(ts_ma)]
  ssi <- c() #계절지수 벡터
  for(i in 1:freq){ # i<-1
    if(i<=(length(ts_ma)%%freq)){
      s<-c(ts_ma[seq(i,(length(ts_ma)-(length(ts_ma)%%freq)), freq)])
      s<-c(s, ts_ma[length(ts_ma)-freq+1+i])
      s<-sum(s)/length(s)
    }else{
      s<-c(ts_ma[seq(i,(length(ts_ma)-(length(ts_ma)%%freq)),freq)])
      s<-sum(s)/length(s)
    }
    ssi<-c(ssi, s)
  }
  return(ssi)
}

# 에러 계산함수
MAPE <- function(real, pred){ return(mean(abs(real-pred)/real)*100) }


#========= 데이터 불러오기 =========#
milk<-fread("milk.csv")
milk$Month<-as.Date(milk$Month)
milk<-milk[order(milk$Month),] # 날짜순으로 정렬

#========= 데이터 특성 확인하기 =========#
dim(milk) # nrow, ncol
str(milk) # 변수 타입 확인
summary(milk) # 기초 통계량 확인 

#========= 분포 확인하기 =========#
plot(milk$Month, milk$Pounds,type="l") # 전체 시계열 
plot(milk$Month[1:12], milk$Pounds[1:12],type="l") # 계절 변화 그래프 

#========= 이동평균 시계열 산출 ========#
mvdata <- moveAvg2(milk$Pounds)     # default = 12개월
mvdata[13] ; mean(milk$Pounds[1:12]) 

mvdata <- mvdata[!is.na(mvdata)] # 62년은 학습데이터에서 제외 

plot(milk$Month[13:nrow(milk)], mvdata, type="l", ylim = c(min(milk$Pounds),max(milk$Pounds))) # 예측하고자 하는 데이터(검은선)
lines(milk$Month[13:nrow(milk)],milk$Pounds[13:nrow(milk)], col="red")

#========= 계절지수 구하기 =========#
season.value<-seasonIndex2(milk$Pounds)      # default = 12개월
plot(c(1:12), season.value, type="l", xlab = "month", ylab = "value", main="계절지수")

#========= 데이터 분할하기 =========#
tst_idx <- (length(mvdata)-11):length(mvdata) # 평가용 우유 생산량 -  최근 1년 [75년 1월 ~ 75년 12월 (12months)]
val_idx <- (tst_idx[1]-12):(tst_idx[length(tst_idx)]-12) # 검증용 우유 생산량 -  최근 2년전 ~ 최근 1년전 [74년 1월 ~ 74년 12월 (12months)]
trn_idx <- 1:(val_idx[1]-1) # 학습용 우유 생산량 -  이전 12년 (144months) [63년 1월 ~ 73년 12월 (12months)] 


#========= 벡터변화 예시 =========#
k=5
r=3

#### 학습 데이터 생성 ####
range3_train.x<-as.data.frame(matrix(nrow=0,ncol=r))
range3_train.y<-as.data.frame(matrix(nrow=0,ncol=1))

for (inx in trn_idx){
  if ((inx-r-1) %in% trn_idx){
    
    range3_train.y<-rbind(range3_train.y,(mvdata[inx]-mvdata[(inx-1)])/mvdata[(inx-1)])
    
    i=1
    range_X<-c()
    while ((r-i)>=0){
      before_inx=(inx-i)
      
      xset<-(mvdata[before_inx]-mvdata[before_inx-1])/mvdata[before_inx-1]
      range_X<-append(range_X, xset)
      
      i=i+1
    }
    
    range3_train.x<-rbind(range3_train.x,range_X)
  }else{
    next
  }
}
colnames(range3_train.x)<-paste("R", r:1, "'" ,sep="")
colnames(range3_train.y)<-"R0'"

#### 평가 데이터 생성 ####

range3_valid.x<-as.data.frame(matrix(nrow=0,ncol=r))
range3_valid.y<-as.data.frame(matrix(nrow=0,ncol=1))

for (inx in tst_idx){
  if ((inx-r-1) %in% 1:(max(tst_idx)-r-1)){
    
    range3_valid.y<-rbind(range3_valid.y,(mvdata[inx]-mvdata[(inx-1)])/mvdata[(inx-1)])
    
    i=1
    range_X<-c()
    while ((r-i)>=0){
      before_inx=(inx-i)
      
      xset<-(mvdata[before_inx]-mvdata[before_inx-1])/mvdata[before_inx-1]
      range_X<-append(range_X, xset)
      
      i=i+1
    }
    
    range3_valid.x<-rbind(range3_valid.x, range_X)
  }else{
    next
  }
}
colnames(range3_valid.x)<-paste("R", r:1, sep="")
colnames(range3_valid.y)<-"R0"

fit<-knnreg(range3_train.x, as.numeric(range3_train.y$`R0'`), k=k)
pred<-predict(fit, range3_valid.x)
pred<-mvdata[(tst_idx-1)]*pred+mvdata[(tst_idx-1)]
pred<-pred*season.value
real<-milk$Pounds[(nrow(milk)):(nrow(milk)-11)]

perf_rk<-MAPE(real,pred)
perf_rk

#############################################################################################################################
######################################################모델링-KNN#############################################################
#############################################################################################################################


#========= 파라미터(k = 근접이웃수, r = 비교할 이전 달 범위) =========#
k_range <- 1:5; r_range <- 3:10
perf <- data.frame(matrix(nrow=0, ncol=3))

#========= 파라미터 검증 - valid set 사용 =========#

for(k in k_range){ #k<-1
  for(r in r_range){ #r<-3
        
    train.x<-as.data.frame(matrix(nrow=0,ncol=r))
    train.y<-as.data.frame(matrix(nrow=0,ncol=1))
    
    for (inx in trn_idx){
      if ((inx-r-1) %in% trn_idx){
        
        train.y<-rbind(train.y,(mvdata[inx]-mvdata[(inx-1)])/mvdata[(inx-1)])
        
        i=1
        range_X<-c()
        while ((r-i)>=0){
          before_inx=(inx-i)
          
          xset<-(mvdata[before_inx]-mvdata[before_inx-1])/mvdata[before_inx-1]
          range_X<-append(range_X, xset)
          
          i=i+1
        }
        
        train.x<-rbind(train.x,range_X)
      }else{
        next
      }
    }
    colnames(train.x)<-paste("R", r:1, "'" ,sep="")
    colnames(train.y)<-"R0'"
    
    #### 평가 데이터 생성 ####
    
    valid.x<-as.data.frame(matrix(nrow=0,ncol=r))
    valid.y<-as.data.frame(matrix(nrow=0,ncol=1))
    
    for (inx in val_idx){
      if ((inx-r-1) %in% 1:(max(val_idx)-r-1)){
        
        valid.y<-rbind(valid.y,(mvdata[inx]-mvdata[(inx-1)])/mvdata[(inx-1)])
        
        i=1
        range_X<-c()
        while ((r-i)>=0){
          before_inx=(inx-i)
          
          xset<-(mvdata[before_inx]-mvdata[before_inx-1])/mvdata[before_inx-1]
          range_X<-append(range_X, xset)
          
          i=i+1
        }
        
        valid.x<-rbind(valid.x, range_X)
      }else{
        next
      }
    }
    colnames(valid.x)<-paste("R", r:1, sep="")
    colnames(valid.y)<-"R0"
    
    fit<-knnreg(train.x, as.numeric(train.y$`R0'`), k=k)
    pred<-predict(fit, valid.x)
    pred<-(mvdata[(val_idx-1)]*pred)+mvdata[(val_idx-1)]
    pred<-pred*season.value
    real<-milk$Pounds[val_idx+12]
    
    perf_rk<-MAPE(real,pred)
    
    perf<-rbind(perf,c(k, r, perf_rk))
  }
}
colnames(perf)<-c("k_num","r_num","mape")

best_k<-perf[which.min(perf$mape),"k_num"]
best_r<-perf[which.min(perf$mape),"r_num"]



#========= best_k와 best_r을 사용하여 예측 - test set 사용 =========#

##best_r범위의 input을 갖는 학습데이터 생성 
train.x<-as.data.frame(matrix(nrow=0,ncol=best_r))
train.y<-as.data.frame(matrix(nrow=0,ncol=1))

for (inx in trn_idx){
  if ((inx-best_r-1) %in% trn_idx){
    
    train.y<-rbind(train.y,(mvdata[inx]-mvdata[(inx-1)])/mvdata[(inx-1)])
    
    i=1
    range_X<-c()
    while ((best_r-i)>=0){
      before_inx=(inx-i)
      
      xset<-(mvdata[before_inx]-mvdata[before_inx-1])/mvdata[before_inx-1]
      range_X<-append(range_X, xset)
      
      i=i+1
    }
    
    train.x<-rbind(train.x,range_X)
  }else{
    next
  }
}
colnames(train.x)<-paste("R", best_r:1, "'" ,sep="")
colnames(train.y)<-"R0'"



##best_r범위의 input을 갖는 평가데이터 생성 
test.x<-as.data.frame(matrix(nrow=0,ncol=best_r))
test.y<-as.data.frame(matrix(nrow=0,ncol=1))

for (inx in tst_idx){
  if ((inx-best_r-1) %in% 1:(max(tst_idx)-best_r-1)){
    
    test.y<-rbind(test.y,(mvdata[inx]-mvdata[(inx-1)])/mvdata[(inx-1)])
    
    i=1
    range_X<-c()
    while ((best_r-i)>=0){
      before_inx=(inx-i)
      
      xset<-(mvdata[before_inx]-mvdata[before_inx-1])/mvdata[before_inx-1]
      range_X<-append(range_X, xset)
      
      i=i+1
    }
    
    test.x<-rbind(test.x, range_X)
  }else{
    next
  }
}
colnames(test.x)<-paste("R", best_r:1, sep="")
colnames(test.y)<-"R0"


##best_k로 knn regression 

fit<-knnreg(train.x, as.numeric(train.y$`R0'`), k=best_k)

pred<-predict(fit, test.x) # knn모델로 test셋 예측
pred<-(mvdata[(tst_idx-1)]*pred)+mvdata[(tst_idx-1)] # 예측된 이동평균 변동 값을 가지고 t시점의 이동평균 값으로 변환
pred<-pred*season.value # 계절변동값 부여

real<-milk$Pounds[tst_idx+12] # 원래 시계열의 실제 값 




#############################################################################################################################
######################################################예측 성능 #############################################################
#############################################################################################################################

##예측 성능 
perf_rk<-MAPE(real,pred)
perf_rk


##예측 성능 - 그래프
plot(milk$Month, milk$Pounds, type = "l")
lines(milk$Month,append(rep(NA, nrow(milk)-12), pred), col="red",lwd=3)


