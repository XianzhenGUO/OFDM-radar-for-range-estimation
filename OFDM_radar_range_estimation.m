clc, clear all, close all;

%% ISAC transmitter

% system parameter
c0 = physconst('LightSpeed');
fc = 30e9; % carrier frequency
lambda = c0 / fc;
B = 400e6; % bandwidth
N = 512; % subcarrier number
delta_f = B / N; % subcarrier spacing 
M = 1; % number of symbol
T = 1 / delta_f; % data symbol duration
Tcp = T / 4; % cp duration
Ts = T + Tcp; % total symbol duration
dunamb = c0 / 2 / delta_f;
%transmit data
bitPerSymbol = 2; % bit per symbol
qam = 2^bitPerSymbol; % qam 4 modulation
data = randi([0, qam - 1], N, M);
TxData = qammod(data, qam, 'UnitAveragePower', true);

%target information 
target_pos = [30, 50, 70];
sigma_rcs = [1, 1, 1];
delay = range2time(target_pos,c0);
K = length(target_pos); %the number of targets
target_speed = [0, 0, 0];
dop = speed2dop(target_speed, lambda);

%received data
SNRdB = 10;
SNR = 10^log10(SNRdB/10);
RxData = zeros(N,M);
b = sqrt(c0 * sigma_rcs ./ ((4*pi)^3 * target_pos.^4 * fc^2));  % Attenuations

for n = 1:N
    for m = 1:M
        for k = 1:K
            RxData(n,m) = RxData(n,m) + b(k) * TxData(n,m) * exp(-1j * 2 * pi * delta_f * n * delay(k)) * exp(1j * 2 * pi * m * dop(k) * Ts);
        end
    end
end

receivedPower = mean(mean(abs(RxData).^2));
sigma2 = receivedPower / SNR;
RxData = RxData + sqrt(sigma2/2) * (randn(N,M) + 1j*randn(N,M));

% remove the effect of transmit data
divideArray = RxData ./ TxData; 

NPer = N;
% FFT method
spectrumPower = abs(ifft(divideArray, NPer, 1)); 
meanspectrumPower = mean(spectrumPower, 2);
meanspectrumPowerMax = max(meanspectrumPower);
normedSpectrumPower = meanspectrumPower / meanspectrumPowerMax;
normedSpectrumPowerdB = 10 * log10(normedSpectrumPower);
[val,rangeIndexFFT] = maxk(normedSpectrumPowerdB, K);
rangeEstFFT = rangeIndexFFT * c0 / (NPer * delta_f * 2);
rangeXlabelFFT = [0:NPer-1] * c0 / (NPer * delta_f * 2);
plot(rangeXlabelFFT, normedSpectrumPowerdB);

% MUSIC method
covMatrix = divideArray * divideArray' / M; % 计算covariance maxtrix
[eigenVec, eigenVal] = eig(covMatrix); % EVD
eigenValDiag = diag(eigenVal); 
[sortedEigenVal, eigenRank] = sort(eigenValDiag, 'descend'); % sorting eigenvalues in a descending order
eigenVec = eigenVec(:,eigenRank); % rearrange the eigen vectors

%取后 N - K 个特征向量作为噪声子空间
G = eigenVec(:,K+1:end);

%遍历频率的间隔
delta_phase = pi/500;
frequencySet = 0 : delta_phase : 2*pi;
lenFrequencySet = length(frequencySet);
carrierSet = 0:1:N-1;
%calculate pseudo spectrum 
for index = 1:lenFrequencySet
    steerVec = exp(-1j * frequencySet(index) * carrierSet).';
    pseudoSpectrum(index) = 1 / (steerVec' * (G * G') * steerVec);
end
pseudoSpectrumPower = abs(pseudoSpectrum);
pseudoSpectrumPowerMax = max(pseudoSpectrumPower);
pseudoSpectrumPower = pseudoSpectrumPower / pseudoSpectrumPowerMax;
pseudoSpectrumPowerdB = 10*log10(pseudoSpectrumPower);
[valMUSIC, rangeIndexMUSIC] = maxk(pseudoSpectrumPowerdB, K);
rangeXlabelMUSIC = frequencySet * c0 / (2 * 2 * pi * delta_f);
rangeEstMUSIC = rangeIndexMUSIC * c0 / (2 * 2 * pi * delta_f) * delta_phase;

figure;
plot(rangeXlabelMUSIC,pseudoSpectrumPowerdB,'k-');
grid on;
% 
% 
% figure
% plot(rangeXlabelFFT, normedSpectrumPowerdB,'b-');
% hold on
% plot(rangeXlabelMUSIC,pseudoSpectrumPowerdB,'r-');
% xline(target_pos,'m--');
% xlabel('Range [m]');
% ylabel('Normalized Range Profile [dB]');
% legend('Periodogram (FFT)','MUSIC','True Distance');
% grid on

%MUSIC + Spatial Smoothing
L = N/4; % number of antennas in each subarray
numSub = N - L + 1; % total number of subarrays

% calculate covariance matrix 
covMatrixSub = zeros(L);
for i = 1:1:numSub
    covMatrixSub = covMatrixSub + divideArray(i : i + L - 1,1) * divideArray(i : i + L - 1,1)';
end
covMatrixSub = covMatrixSub / numSub;


% SVD for noise subspace
[U, S, V] = svd(covMatrixSub);
Sdiag = diag(S);
noiseSubspace = U(:,K + 1 : end);

%遍历频率的间隔
delta_phase = pi/500;
frequencySet = 0 : delta_phase : 2*pi;
lenFrequencySet = length(frequencySet);
carrierSetSS = 0:1:L-1;
%calculate pseudo spectrum 
for index = 1:lenFrequencySet
    steerVecSS = exp(-1j * frequencySet(index) * carrierSetSS).';
    pseudoSpectrumSS(index) = 1 / (steerVecSS' * (noiseSubspace * noiseSubspace') * steerVecSS);
end
pseudoSpectrumPowerSS = abs(pseudoSpectrumSS);
pseudoSpectrumPowerMaxSS = max(pseudoSpectrumPowerSS);
pseudoSpectrumPowerSS = pseudoSpectrumPowerSS / pseudoSpectrumPowerMaxSS;
pseudoSpectrumPowerdBSS = 10*log10(pseudoSpectrumPowerSS);
[valMUSICSS, rangeIndexMUSICSS] = maxk(pseudoSpectrumPowerdBSS, K);
rangeXlabelMUSICSS = frequencySet * c0 / (2 * 2 * pi * delta_f);
rangeEstMUSICSS = rangeIndexMUSICSS * c0 / (2 * 2 * pi * delta_f) * delta_phase;


figure
plot(rangeXlabelFFT, normedSpectrumPowerdB,'b-');
hold on
plot(rangeXlabelMUSIC,pseudoSpectrumPowerdB,'k-');
plot(rangeXlabelMUSICSS,pseudoSpectrumPowerdBSS,'r-');
xline(target_pos,'m--');
xlabel('Range [m]');
ylabel('Normalized Range Profile [dB]');
legend('Periodogram (FFT)','MUSIC','MUSIC + Spatial Smoothing','True Distance');
grid on