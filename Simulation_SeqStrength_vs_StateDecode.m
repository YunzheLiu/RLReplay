clear;
clc;
close all;
rng('shuffle')

%% Construct Transitions
SeqFwd=zeros(6,18,18);
% Seq 1
SeqFwd(1,1,7)=1;
SeqFwd(1,7,13)=1;
% Seq 2
SeqFwd(2,2,8)=1;
SeqFwd(2,8,14)=1;
% Seq 3
SeqFwd(3,3,9)=1;
SeqFwd(3,9,15)=1;
% Seq 4
SeqFwd(4,4,10)=1;
SeqFwd(4,10,16)=1;
% Seq 5
SeqFwd(5,5,11)=1;
SeqFwd(5,11,17)=1;
% Seq 6
SeqFwd(6,6,12)=1;
SeqFwd(6,12,18)=1;
TF=squeeze(nansum(SeqFwd,1));
TR = TF';

%% Spects
nSensors = 272;
nTrainPerStim = 50; % how many training examples for each stimulus
nstates=18;
nNullExamples = nTrainPerStim*nstates; % how many null examples to use
nSamples = 250; % 2.5 seconds of unlabelled data to predict
nSequences = 100; % how many real sequences to put in the data
maxLag = 50; % evaluate time lag up to 600ms
cTime = 0:10:maxLag*10; % the milliseconds of each cross-correlation time lag
nSubj = 29; % number of subjects to simulate
gamA = 10;
gamB = 0.3;% parameters for the gamma distribution of intervals between states in a sequence
noiseS=50;

sf = cell(noiseS,1);  sb = cell(noiseS,1);
decodeACC= cell(noiseS,1);

%% Core function

for inoise=1:noiseS
    
    sf{inoise} = nan(1, nSubj,maxLag+1);
    sb{inoise} = nan(1, nSubj,maxLag+1);
    decodeACC{inoise} = nan(1, nSubj,nstates);
    
    disp(['iNoise =' num2str(inoise)])
    
    for isub=1:nSubj
        
        %% generate dependence of the sensors
        A = randn(nSensors);
        [U,~] = eig((A+A')/2); 
        covMat = U*diag(abs(randn(nSensors,1)))*U';
        
        %% generate the true patterns
        commonPattern = randn(1,nSensors);    
        patterns = repmat(commonPattern, [nstates 1]) + randn(nstates, nSensors);
        
        %% make training data
        trainingData = inoise*randn(nNullExamples+nstates*nTrainPerStim, nSensors) + [zeros(nNullExamples,nSensors); ...
            repmat(patterns, [nTrainPerStim 1])];
        
        trainingLabels = [zeros(nNullExamples,1); repmat((1:nstates)', [nTrainPerStim 1])];    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %% train classifiers on training data
        betas = nan(nSensors, nstates); intercepts = nan(1,nstates);
        
        for iC=1:nstates
            [betas(:,iC), fitInfo] = lassoglm(trainingData, trainingLabels==iC, 'binomial', 'Alpha', 1, 'Lambda', 0.005, 'Standardize', false);
            intercepts(iC) = fitInfo.Intercept;
        end

        %% decoding accuracy
        testData = inoise*randn(nstates*nTrainPerStim, nSensors) + [repmat(patterns, [nTrainPerStim 1])];        
        testLabels = [repmat((1:nstates)', [nTrainPerStim 1])];    
        
        DecodeP=1./(1+exp(-(testData*betas + repmat(intercepts, [length(testData) 1]))));
        [~,stateind]=max(DecodeP,[],2);
        
        for iC=1:nstates
            decodeACC{inoise}(1,isub,iC)=mean(stateind(find(testLabels==iC))==iC);
        end

        %% make long unlabelled data 
        X = nan(nSamples, nSensors);
        X(1,:) = randn([1 nSensors]);
        
        for iT=2:nSamples
            X(iT,:) = 0.95*(X(iT-1,:) + mvnrnd(zeros(1,nSensors), covMat));% add dependence of the sensors 
        end
        
        %% Injecting Sequence
        for iRS = 1:nSequences
            seqTime = randi([40 nSamples-40]); % pick a random point, not right at the edges       
            state = false(nstates,1); state(randi(nstates)) = true;  % start the sequence in a random state

            for iMv=1:2
                if sum(state)==0
                    X(seqTime,:) = X(seqTime,:);
                else
                    X(seqTime,:) = X(seqTime,:) + patterns(state,:); 
                    state = (state'*TR)'; state2 = false(nstates,1); state2(find(rand < cumsum(state), 1, 'first')) = true; state = state2; % advance states
                    seqTime = seqTime + round(gamrnd(gamA,gamB))+13;
                end
            end
        end
    
        %% make predictions with trained models
        preds = 1./(1+exp(-(X*betas + repmat(intercepts, [nSamples 1]))));
        
        %% calculate sequenceness 
        T1 = TF; T2 = T1'; % backwards is transpose of forwards
        X=preds;
        
        nbins=maxLag+1;
        warning off
        dm=[toeplitz(X(:,1),[zeros(nbins,1)])];
        dm=dm(:,2:end);
        
        for kk=2:nstates
            temp=toeplitz(X(:,kk),[zeros(nbins,1)]);
            temp=temp(:,2:end);
            dm=[dm temp]; 
        end
        warning on
        Y=X;       
        betas = nan(nstates*maxLag, nstates);
        %% GLM: state regression, with other lages       
        
        bins=maxLag; % 
        for ilag=1:bins
            temp_zinds = (1:bins:nstates*maxLag) + ilag - 1; 
            temp = pinv([dm(:,temp_zinds) ones(length(dm(:,temp_zinds)),1)])*Y;
            betas(temp_zinds,:)=temp(1:end-1,:);           
        end
        betasnbins64=reshape(betas,[maxLag nstates^2]);
        bbb=pinv([T1(:) T2(:) squash(eye(nstates)) squash(ones(nstates))])*(betasnbins64'); %squash(ones(nstates))
        
        sf{inoise}(1,isub,2:end) = bbb(1,:); 
        sb{inoise}(1,isub,2:end) = bbb(2,:); 
    end
end


%% Plot
sb2=normalize(sb(:,:,2:end),3,'zscore');
sbmean=squeeze(nanmean(sb2,2));
ACC2=squeeze(nanmean(nanmean(decodeACC,2),3));

tTimes2=10:10:500;

figure
x0=40;
y0=40;
width=1200;
height=600;
set(gcf,'units','points','position',[x0,y0,width,height])

subplot(1,2,1)
set(groot,'defaultAxesColorOrder',jet(size(sbmean,1)));  
plot(tTimes2, sbmean', 'LineWidth', 1)
xlabel('lag (ms)')     
ylabel('sequence strength (zscored)')    
legend(cellfun(@(x) ['Noise Level: ' strtrim(x) ''], cellstr(num2str((1:1:50)')), 'UniformOutput', false))

subplot(1,2,2)
scatter(ACC2,sbmean(:,16),40,'MarkerEdgeColor',[0.7 .7 .7],...
              'MarkerFaceColor',[0.7 .7 .7],...
              'LineWidth',1.5)
xlabel('decoding accuracy');
ylabel('sequence strength (zscored)')
hold on
xline(0.47,'--r')
corr(ACC2,sbmean(:,16))
suptitle('Bkw Replay (ground truth: 160 ms)')
