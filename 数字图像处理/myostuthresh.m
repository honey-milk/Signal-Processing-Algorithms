function [T,SM] = myostuthresh(f)
%MYOSTUTHRESH - Ostu's threshold
%
%   [T,SM] = myostuthresh(f)

%% ���һ���Ҷ�ֱ��ͼ
h = imhist(f);
h = h./sum(h);
h = h(:);

%% ���ۼƸ��ʷֲ�
P = cumsum(h);

%% ���ۼƾ�ֵ
i = (1:length(h))';
m = cumsum(i.*h);
mG = m(end);

%% ������䷽��
sigSquared = ((mG*P-m).^2)./(P.*(1-P)+eps);

%% �������ֵ
maxSigsq = max(sigSquared);
T = mean(find(sigSquared == maxSigsq));
T = (T-1)/(length(h)-1);

%% ����ɷ��Բ��
SM = maxSigsq/(sum(((i-mG).^2).*h)+eps);



