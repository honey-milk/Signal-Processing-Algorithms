function varargout = mypitchtrack(varargin)
%MYPITCHTRACK - Pitch track by cepstrum analysis
%
%   This MATLAB function finds the pitch of the signal frames by 
%   cepstrum analysis.
%
%   [period,T] = mypitchtrack(frame,fs)
%   [period,T] = mypitchtrack(x,fs,nwin)

%% ��������
% ��������Ŀ
narginchk(2,3);
nargoutchk(0,2);

% ��ȡ�������ֵ
% ��ȡ��һ���������ֵ
arg1 = varargin{1};
if isvector(arg1)   % arg1������x
    [x,fs,nwin] = varargin{:};
    frame = framing(x,nwin,0,'truncation');
else                % arg1�Ǿ���frame
    [frame,fs] = varargin{:};
end
[nframe,nwin] = size(frame);

%% ����������ֵ
magnitude = sum(abs(frame),2);
threshmedian = median(magnitude);
threshmean = mean(magnitude);
if threshmean>1.5*threshmedian %% �����ֵ����ֵ�ǳ��ӽ�������Ϊ�󲿷־�Ϊ�����źţ���ֵ��Ϊ0
    threshe = threshmedian;
else
    threshe = 0;
end

%% �����������
%�����˵Ļ������ڷ�Χ2~20ms(50~500Hz)
tstart = round(0.002*fs+1);
tend = min(round(0.02*fs+1),round(nwin/2));
period = zeros(nframe,1);
for i=1:nframe
    if magnitude(i)>=threshe
        c = myrceps(frame(i,:));
        [maximum,maxpos] = max(c(tstart:tend));
        threshold = 4*mean(abs(c(tstart:tend)));
        if maximum>=threshold       %����
            period(i) = (maxpos+tstart-2)/fs;
        else                        %��������
            period(i) = 0;
        end
    else                            %��������
        period(i) = 0;
    end 
end

%% ����ÿһ֡���м��ʱ��(T)
T = zeros(nframe,1);
for i=1:nframe
    start_time = (i-1)*nwin;
    T(i) = start_time+nwin/2;
end
T = T/fs;

%% ����������
if nargout==1
    varargout = {period};
else
    varargout = {period,T};
end
