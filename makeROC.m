function [ p1d0 , p0d1 , th, thflag ] = makeROC(D0,D1)
% makeROC Empirical ROC plot from experimental/simulation data
%   [p1d0,p0d1] = makeROC(D0,D1) outputs vectors for drawing a ROC plot, 
%   empirically derived from 1-D observation data in D0, D1. 
%   The plot can then be drawn with the command: [...] = plot(p1d0, p0d1, ...) 
%   
%   D0 must contain data gathered under the null hypothesis H0, 
%   D1 must contain data gathered under the complementary hypothesis H1
%   p1d0 collects values of the Type 1 (aka false positive or false alarm) error rates 
%     as the threshold sweeps the whole D0 and D1 data range 
%   p1d0 collects values of the Type 2 (aka false negative or missed detection) error rates 
%     as the threshold sweeps the whole D0 and D1 data range
%   The first and last values of p1d0 (and p0d1) correspond to trivial threshold values 
%   -Inf and +Inf, respectively, and are therefore either 0 ior 1
%
%   [p1d0,p0d1,th] = makeROC(D0,D1) also outputs the vector th of corresponding threshold values  
%
%   [p1d0,p0d1,th,thflag] = makeROC(D0,D1) also specifies the better inequality sign for testing, 
%      whether 'H0 > th, H1 < th' or 'H0 < th, H1 > th'  

%   Nicola Laurenti, University of Padova  nil@dei.unipd.it 	
%   Created 20 October 2017 
%   Last modified 28 November 2019   
 
D0 = D0(:); D1 = D1(:);
N0 = length(D0); N1 = length(D1); N = N0+N1; 
[Ds,i] = sort([D0;D1]);
th = (Ds(1:N-1)+Ds(2:N))/2;
th = [-Inf;th;Inf];
j0 = (i <= N0);
j1 = 1-j0;
i0 = [0;cumsum(j0)];
i1 = N1-[0;cumsum(j1)];
p1d0 = 1-i0/N0;
p0d1 = 1-i1/N1;
if (mean(p1d0+p0d1) > 1), thflag = 'H0 > th, H1 < th';
    p1d0 = i0/N0; 
    p0d1 = i1/N1;
else thflag = 'H0 < th, H1 > th';
end
end

