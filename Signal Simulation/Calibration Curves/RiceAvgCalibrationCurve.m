% MATLAB Script to generate calibration curves for averaging Rice
% distributions at different SNR values


% Save folder
folder = "C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\General VERDICT Code\General-VERDICT-Code\Signal Simulation\Calibration Curves";

% Number of samples
Nsample = 100000;

% NSA values
NSAs = [1, 3, 6, 9,... % NSA=1 1, 1x3, 1x3x2, 1x3x3
    4, 12, 24, 36,... % NSA=4 4, 4x3, 4x3x2, 4x3x3
   18, 54]; % NSA=6 6, 6x3,6x3x2, 6x3x3

NSAs = [1,2,3,4,5,6,7,8,9,10,12,15];

% Define SNR
SNRs = linspace(0.1,10,100);

% save([char(folder) '/SNRs.mat'], 'SNRs')


SNRs = SNRs(50:end);
SNRs = [2];

for SNR = SNRs

    disp(SNR)
    v = 1;
    sigma = 1/SNR;
    
    
    %% Define Rice distibution
    
    % Make Rice distribution
    ricedist = makedist('Rician','s',v,'sigma',sigma);
    
    maxsignal = 10;
    signals = linspace(0,maxsignal,501);
    
    pdf = ricedist.pdf(signals);
    
    
    FitVs = zeros(size(NSAs));
    FitSigmas = zeros(size(NSAs));
    
    
    %% Analysis for each NSA value
    
    for NSAIndx = 1:length(NSAs)
    
        NSA = NSAs(NSAIndx);


        %% Take samples
        
        % Sample distributions
        samples = zeros(Nsample, 1);
        for indx = 1:Nsample
        
            % Average some signal measurements
            sample = 0;
            for AVindx = 1:NSA
                sample = sample + sampleDistribution(pdf, signals);
            end
            sample = sample/NSA;
        
        
            samples(indx, 1) = sample;
        
        end
        
        
        
        %% Display original PDF and histogram of samples
        % 
        % % Define bin edges
        % binedges = signals;
        % binwidth = binedges(2)-binedges(1);
        % bincenters = binedges(1:end-1)+binwidth/2;
        % 
        % figure;
        % plot(signals, Nsample*binwidth*pdf);
        % hold on
        % H = histogram(samples, binedges);
        % counts = H.Values;

        
        
        %% Fit Rician distribution to samples 
        x = (samples) + eps;
        pd = fitdist(x,'Rician');
        
        fitv = pd.s;
        FitVs(NSAIndx) = fitv;
    
        fitsigma = pd.sigma;
        FitSigmas(NSAIndx) = fitsigma;


        % % Display fitted distribution
        % fitricedist = makedist('Rician','s',fitv,'sigma',fitsigma);
        % fitpdf = fitricedist.pdf(signals);
        % 
        % plot(signals, Nsample*binwidth*fitpdf, Color = 'red')

    
     end
    
    % Make dictionarys
    
    try
        mkdir([char(folder) '/SNR ' num2str(SNR)])
    catch
        disp('')
    end

    % MeanDict = dictionary(NSAs, FitVs);
    % save([char(folder) '/SNR ' num2str(SNR) '/MeanCalibration.mat'], 'MeanDict' )
    % 
    % SigmaDict = dictionary(NSAs, FitSigmas/sigma);
    % save([char(folder) '/SNR ' num2str(SNR) '/SigmaCalibration.mat'], 'SigmaDict' )


end




figure;
plot(NSAs, FitVs, 'o-', DisplayName = '\alpha');
hold on
plot(NSAs, FitSigmas/sigma, '*-', DisplayName = '\beta')  
grid on
legend
xlabel('Number of signal averages (NSA)')
ax = gca();
ax.FontSize = 12;

% figure
% plot(NSAs, FitSigmas/sigma, '*-', color = 'b')  





