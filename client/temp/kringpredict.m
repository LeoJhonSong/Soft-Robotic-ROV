function Pre=kringpredict(dmodel,L0,L1,L2,L3)
 
         QT0=L1-L0;
         QT1=L2-L1;
         QT2=L3-L2;
         EE=(QT0==QT1)&(QT2==QT1);
         XCC=[L1',L2',QT1',EE'];
         [Pre dyx mse dmse]= predictor(XCC, dmodel);
         %Pre=medfilt1(Pre,50);%利用中值滤波器进行平滑处理