function [Ytest ,probout, info]= MultiClassRegularizedLDA(Xtest,Xtrain,Ytrain,lda_method)
% Performs Multi Class Regularized LDA
% input: 
%       Xtest, matrix with test features
%       Xtrain, matrix with train features
%       Ytrain, vector with train labels
%       lda_method, one of 'sim_diag' or 'LDA' 
% output:
%       Ytest, labels assigned to testing data 
%       probout, probability of outputs
%       info, weights, bias


    if nargin<4
        lda_method='sim_diag';
        fprintf('performing multiclass LDA \n')
    end

    if nargin<3
        fprintf('Not enough input arguments \n')
        return

    end
        
    labels = unique(Ytrain);
    Nclass = length(labels);

    % Train Regularized LDA
    
    
        %Compute class wise means and covariances
        for i=1:Nclass
            mu(:,i) = mean(Xtrain(Ytrain==labels(i),:));
            Covclass(:,:,i) = covariances(Xtrain(Ytrain==labels(i),:)','shcovft');%'cov');%
        end
        %Compute global covariance
        S = mean(Covclass,3);

        %construct LDA classifier
        soft_outputs=0;        
                switch lda_method 
                    
                    case'LDA'
                            for i=1:Nclass
                                for j=i+1:Nclass
                                    mupart(i,j,:)=mean([mu(:,i) mu(:,j)],2);
                                    vpart(i,j,:)=mu(:,j)- mu(:,i);
                                    Sigma(i,j,:,:)=mean(cat(3,Covclass(:,:,i),Covclass(:,:,j)),3); 
                                    weights(i,j,:)=squeeze(Sigma(i,j,:,:))^(-1) *squeeze(vpart(i,j,:));
                                    bias(i,j) = squeeze(weights(i,j,:))'*squeeze(mupart(i,j,:));
                                    s(i,j) = sign(squeeze(weights(i,j,:))'*mu(:,j)-bias(i,j));

                                end
                            end
                                if soft_outputs==1
                                     idx=0;
                                    interval=0:.1:2;
                                    for alpha=interval % the extra parameter controlling sigmoid
                                        idx=idx+1;
                                             for i=1:Nclass
                                                for j=i+1:Nclass
                                                    for tr=1:size(Xtrain,1)  
                                                        dcr = s(i,j)*(squeeze(weights(i,j,:))'*Xtrain(tr,:)'-bias(i,j)) ;
                                                        %out(i,j,tr) = lab((drc>0) + 1);
                                                        prob1(i,tr,idx)= 1/(1+exp(alpha*dcr));
                                                        prob1(j,tr,idx)= 1 - (1/(1+exp(alpha*dcr)));
                                                    end
                                                    variance_resp(i,j,idx)= mean(var(prob1([i j],:,idx)));
                                                    %loglik(i,j,idx)=sum(sum(log(prob1([i j],:,idx))));
                                                    %clear prob1;
                                                end
                                             end
                                    end
                                 for i=1:Nclass
                                    A(i,:)=(squeeze(sum(log(prob1(i,Ytrain==unique(Ytrain(i)),:)),2)));
                                 end
%                                  B=(sum(A));
%                                  plot(B);B=diff(B);
%                                  B=find((B>0));
%                                  alpha_opt=interval(B(end));
                                %figure(1);clf
                                 for i=1:Nclass
                                        for j=i+1:Nclass
                                             B=(sum(A([i j],:)));
                                             %plot(B);hold on;
                                             B=diff(B);
                                             B=find((B>0));
                                             if numel(B)==0;
                                                 alpha_opt(i,j)=1;
                                             else
                                                alpha_opt(i,j)=interval(B(end));
                                             end

                                        end
                                 end
                                for i=1:Nclass
                                    for j=i+1:Nclass
                                        weights(i,j,:)=  alpha_opt(i,j)* weights(i,j,:);
                                        bias(i,j) =   alpha_opt(i,j)* bias(i,j);
                                    end
                                 end
                                    
                                    

                                end

                    case'sim_diag'
                        for i=1:Nclass
                            for j=i+1:Nclass
                                mupart(i,j,:)=mean([mu(:,i) mu(:,j)],2);
                                Sb = (mu(:,i) - squeeze(mupart(i,j,:)))*(mu(:,i)-squeeze(mupart(i,j,:)))' +(mu(:,j) - squeeze(mupart(i,j,:)))*(mu(:,j)-squeeze(mupart(i,j,:)))';
                                [W Lambda] = eig(Sb,S);
                                [dumm, Index] = sort(diag(Lambda),'descend');

                                W = W(:,Index(1));
                                weights(i,j,:)=W(:,1);
                                bias(i,j) = squeeze(weights(i,j,:))'*squeeze(mupart(i,j,:));
                                %sig(i,j) = sign(W(:,1)'*mu(:,2)-b);
                                s(i,j) = sign(squeeze(weights(i,j,:))'*mu(:,j)-bias(i,j));
                            end
                        end
                        

               end

        
       
%test
Stest=Xtest';
    for k=1:size(Xtest,1)
        idx=0;
        %prob1=zeros(Nclass,Nclass,size(Xtest,3));
        probs=[];
        %pairs=[1 2; 1 3; 1 4; 2 3; 2 4 ;3 4];
        for i=1:Nclass
            for j=i+1:Nclass
                idx=idx+1;
                lab=[labels(i) labels(j)]; if lab==[-1 1]; lab=[1 2]; end
                %y1 = s(i,j)*(squeeze(weights(i,j,:))'*Xtest(:,k)'-bias(i,j)) ;
                y1 = s(i,j)*(squeeze(weights(i,j,:))'*Stest(:,k)-bias(i,j)) ;
                out(i,j,k) = lab((y1>0) + 1);
                
                prob1(lab(1),idx,k)= 1/(1+exp(y1));
                prob1(lab(2),idx,k)= 1 - (1/(1+exp(y1)));
                probs=[probs [1/(1+exp(y1)) ; 1 - (1/(1+exp(y1)))] ]; 
                
            end
        end
        prob=sum(prob1(:,:,k),2);
        prob=prob./sum(prob);
        probout(k,:)=prob;
        
        
         
         

    end
    info.w=weights;info.bias=bias;info.s=s;info.mupart=mupart;info.mu=mu;
    
    [dumm Ytest]=max(probout');
end
    
    
    
   