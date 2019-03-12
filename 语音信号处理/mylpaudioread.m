function varargout = mylpaudioread(varargin)
%MYLPAUDIOREAD - Read audio file compressed by linear predition analysis
%
%   This MATLAB function reads data from the file named filename, and returns
%   sampled data, x, and a sample rate for that data, fs.
%
%   lpcparam = mylpaudioread(filename)
%   [x,fs] = mylpaudioread(filename) 

%% ��������
% ��������Ŀ
narginchk(1,1);
nargoutchk(0,2);

% ��ȡ�������ֵ
filename = varargin{:};


%% ���ļ�
fid = fopen(filename,'r');
if fid>0
    lpcparam.fs = fread(fid,1,'uint16');
    lpcparam.nwin = fread(fid,1,'uint16');
    lpcparam.nframe = fread(fid,1,'uint16');
    lpcparam.p = fread(fid,1,'uint8');
    lpcparam.a = fread(fid,[lpcparam.p+1,lpcparam.nframe],'float');
    lpcparam.period = fread(fid,[1,lpcparam.nframe],'float');
    lpcparam.amp = fread(fid,[1,lpcparam.nframe],'float');
end
fclose(fid);

%% ����ÿһ֡���м��ʱ��(T)
nframe = lpcparam.nframe;
nwin = lpcparam.nwin;
fs = lpcparam.fs;
T = zeros(nframe,1);
for i=1:nframe
    start_time = (i-1)*nwin;
    T(i) = start_time+nwin/2;
end
T = T/fs;
lpcparam.T = T;

%% LP�ϳ�
if nargout==1
    varargout = {lpcparam};
else
    x = lpsynthesis(lpcparam);
    varargout = {x,lpcparam.fs};
end
