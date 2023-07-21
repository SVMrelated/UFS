

 
 load lapointe-2004-v2.txt
 
lap_data=lapointe_2004_v2';

trainX=lap_data(:,2:2497);
trainY=lap_data(:,1);

 trainX= normalizemeanstd(trainX);







%--------------------------paramater set--------------------

MAX_acc=zeros(4,1);
Best_N=zeros(4,1);
Best_C=zeros(4,1);
Best_S=zeros(4,1);

mod=1;
Scale=2^0;
N=200;
C=-1;
%numEliminate=1;
nStop=60;
rmvRatio = .1; % ratio of num of removed features before nStop

%-----------------set--------------------

        option1.N=N;
        option1.C=2^C;
        option1.Scale=Scale;
		option1.mode=mod;		
        option1.Scalemode=3;
        option1.bias=0;
        option1.link=0;


        option2.N=N;
        option2.C=2^C;
        option2.Scale=Scale;
		option2.mode=mod;		
        option2.Scalemode=3;
        option2.bias=1;
        option2.link=0;

       
        option3.N=N;
        option3.C=2^C;
        option3.Scale=Scale;
		option3.mode=mod;		
        option3.Scalemode=3;
        option3.bias=0;
        option3.link=1;

 
        option4.N=N;
        option4.C=2^C;
        option4.Scale=Scale;
		option4.mode=mod;		
        option4.Scalemode=3;
        option4.bias=1;
        option4.link=1;



allFeatIndex = 1:size(trainX,2); 
rankedFeat = []; 

tic
while length(rankedFeat) < size(trainX,2)
 feat_tmp = allFeatIndex();
 feat_tmp(rankedFeat) = [];
 feat_tmp_index = allFeatIndex(setdiff(allFeatIndex, rankedFeat )); 
 
 feat_remain=size(feat_tmp,2);
	if feat_remain > nStop
		numEliminate = floor(feat_remain*rmvRatio);;
		else
			numEliminate = 1;
	end 
 
 
 
 
 if size(feat_tmp,2)==1
    rankedFeat = [feat_tmp rankedFeat]; 
     break
 end
 
 rank = zeros(1,size(feat_tmp,2));
 for i= 1: size(feat_tmp,2)
     feat_tmp_tmp = feat_tmp;
     feat_tmp_tmp(i)=[];
     
	 trainX_tmp=trainX(:,feat_tmp_tmp);
	 
 [beta1]=RVFL_train(trainX_tmp,trainY,trainX,trainY,option1);
  % [beta1]=RVFL_train(trainX_tmp,trainY,trainX,trainY,option2);
   % [beta1]=RVFL_train(trainX_tmp,trainY,trainX,trainY,option3);
%[beta1]=RVFL_train(trainX_tmp,trainY,trainX,trainY,option4);     	 

      rank(i)=norm(beta1);

 end

 
 [val ind] = sort(rank);
  featureIndex = feat_tmp_index(ind(1:min(numEliminate, length(ind)))); 
% featureIndex =  feat_tmp (ind(1)); 
 rankedFeat = [featureIndex rankedFeat]; 
end
toc




