function varargout = myspectrogram(x,varargin)
%MYSPECTROGRAM - Spectrogram using a Short-Time Fourier Transform (STFT)
%
%   This MATLAB function returns S, the short time Fourier transform of the input
%   signal vector x.
%
%   S = myspectrogram(x)
%   S = myspectrogram(x,window)
%   S = myspectrogram(x,window,noverlap)
%   S = myspectrogram(x,window,noverlap,nfft)
%   S = myspectrogram(x,window,noverlap,nfft,fs)
%   S = myspectrogram(x,window,noverlap,nfft,fs,[fmin,fmax])
%   S = myspectrogram(x,window,noverlap,nfft,fs,[fmin,fmax,pmin,pmax])
%   [S,F,T,P] = myspectrogram(...)
%   myspectrogram(...)
%   myspectrogram(...,Property)

%   Property��{'xaxis'},'yaxis',{'color'},'gray',{'truncation'},'padding',{'spectall'},'spectpeaks'          

%������Ŀ���
narginchk(1,10);
nargoutchk(0,4);

%��xתΪ������
if isvector(x)==1
    x = x(:);
    nx = length(x);
else
    error('�������''x''����Ϊ1ά����');
end

%������ʾȫ����ֻѡ���ֵѡ��
spectmode = 'spectall';
if (nargin > 1 && ischar(varargin{end})) && any(strcmpi(varargin{end},{'spectall','spectpeaks'})),
    spectmode = varargin{end};
    varargin(end)=[];
end

%�ضϡ�����ѡ��
endprocess = 'truncation';
if (nargin > 1 && ischar(varargin{end})) && any(strcmpi(varargin{end},{'truncation','padding'})),
    endprocess = varargin{end};
    varargin(end)=[];
end

%����ͼ��ɫģʽ
colormode = 'jet';
colorset = {'jet','hsv','hot','gray',...
            'cool','spring','summer','autumn',...
            'winter','gray','bone','copper',...
            'pink','lines'};
if (nargin > 1 && ischar(varargin{end})) && any(strcmpi(varargin{end},colorset)),
    colormode = varargin{end};
    varargin(end)=[];
end

%Ƶ�������ã�Ĭ����x��
freqloc = 'xaxis';
if (nargin > 1 && ischar(varargin{end})) && any(strcmpi(varargin{end},{'yaxis','xaxis'})),
    freqloc = varargin{end};
    varargin(end)=[];
end

%��ȡʣ�����������Ŀ
narg = numel(varargin);
%��ʼ������
window = [];
noverlap = [];
nfft = [];
fs = [];
lim = [];
%��ȡ�������ֵ
switch narg
    case 0
    case 1
        window = varargin{:};      
    case 2
        [window,noverlap] = varargin{:};
    case 3
        [window,noverlap,nfft] = varargin{:};
    case 4
        [window,noverlap,nfft,fs] = varargin{:};
    case 5
        [window,noverlap,nfft,fs,lim] = varargin{:};
    otherwise
        error('�����������');
end

%��ʼ��������win
%û�и�������ʱ��Ĭ�Ͻ��źŷ�Ϊ8֡
if isempty(window)
    if strcmpi(endprocess,'truncation')
        win = hamming(fix(nx*2/9));
    else
        win = hamming(ceil(nx*2/9));
    end
%����һ������    
elseif isscalar(window)==1
    win = hamming(window);
%����һ����ʸ��    
else
    win = window(:);
end
%֡��
nwin = length(win);

%��ʼ��noverlap
if isempty(noverlap)
    noverlap = round(nwin/2);
elseif noverlap >= nwin
    error('''noverlap''��ֵ����С��''window''�ĳ���');
end
 
%��ʼ��nfft
if isempty(nfft)        
   nfft = max(256,power(2,ceil(log2(nwin))));
elseif nfft < nwin
   nfft =  power(2,ceil(log2(nwin)));
end

%��ʼ��fs
%�Ƿ�ʹ�ù�һ��Ƶ��
isFsnormalized = false;
if isempty(fs)    
    fs = 2*pi;
    isFsnormalized = true;
end

%��ʼ��limit
if length(lim)==4
    flim = [lim(1),lim(2)];
    plim = [lim(3),lim(4)];
elseif length(lim)==2
    flim = [lim(1),lim(2)];
    plim = [];
else
    flim = [];
    plim = [];
end

%֡��
nstride=nwin-noverlap; 
%�ź�x���ֳܷ�����֡�����ýضϴ�ʩ 
if strcmpi(endprocess,'truncation')
    %֡��
    nframe=fix((nx-noverlap)/nstride);   
%�ź�x���ֳܷ�����֡�����ò����ʩ   
else
    %֡��
    nframe=ceil((nx-noverlap)/nstride); 
    npadding=nframe*nstride+noverlap-nx;
    %ĩβ����
    x=[x;zeros(npadding,1)];  
end

%��֡
frame=zeros(nwin,nframe);
for i=1:nframe
    start_index=(i-1)*nstride+1;
    end_index=start_index+nwin-1;
    %�Ӵ�
    frame(:,i)=x(start_index:end_index).*win;
end

%��ʱ����Ҷ�任��Ƶ��(F)
F = 0:fs/nfft:fs/2;
F = F';

%�����ʱ����Ҷ�任(S)
nfreq = length(F);
S=zeros(nfreq,nframe);
for i=1:nframe
    X=fft(frame(:,i),nfft);
    S(:,i)=X(1:nfreq);
end

%����ÿһ֡���м��ʱ��(T)
T=zeros(1,nframe);
for i=1:nframe
    start_time=(i-1)*nstride;
    T(i)=start_time+nwin/2;
end
T=T/fs;
    
%�����������ܶ�(P)
P = zeros(nfreq,nframe);
%������
Ewin = sum(win.^2);
%������������Ȩ��ϵ��
k = zeros(nfreq,1);
k(:) = 2/(fs*Ewin);
%��0Ƶ��fs/2��������Ϊ1������Ƶ�ʷ���Ϊ2
k(1) = 1/(fs*Ewin);
if F(end)==fs/2
    k(end) = 1/(fs*Ewin);
end
for i=1:nframe
    P(:,i) = k.*(abs(S(:,i)).^2);
end

%����ֻ��ʾ��ֵ
if strcmpi(spectmode,'spectpeaks')
    %����ÿ֡��������ֵ
    Pframemax = max(P);
    for i=1:nframe
        %�Ƿ�ֵ����
        index = P(:,i)~=Pframemax(i); 
        %�Ƿ�ֵ����������0
        P(index,i) = 0;
    end
end
    

%û������������������ͼ
if nargout==0
    %ʹ�ù�һ��Ƶ��
    if isFsnormalized
        F = F/pi;
        T = T*2*pi;
        flbl = '��һ��Ƶ�� (\times\pi rad/sample)';
        tlbl = 'ʱ�� (sample)';
    else
        flbl = 'Ƶ�� (Hz)';
        tlbl = 'ʱ�� (s)';
    end
    LogP = 10*log10(P+eps);
    if strcmpi(freqloc,'yaxis')
        surf(T,F,LogP,'edgecolor','none');
        xlbl = tlbl;
        ylbl = flbl;
    else
        surf(F,T,LogP','edgecolor','none');
        xlbl = flbl;
        ylbl = tlbl;
    end
    axis xy; 
    axis tight;
    view(0,90);
    colormap(colormode);
    xlabel(xlbl);
    ylabel(ylbl);
    title('����ͼ');
    %����������ʾ��Χ
    if ~isempty(plim)
        zlim(plim); 
    end
    
    %����Ƶ����ʾ��Χ
    if ~isempty(flim)
        if strcmpi(freqloc,'yaxis') 
            ylim(flim); 
        else
            xlim(flim); 
        end        
    end 
else
    switch nargout
        case 1
            varargout = {S};
        case 2
            varargout = {S,F};
        case 3
            varargout = {S,F,T};
        case 4
            varargout = {S,F,T,P};
    end
end
            
    
        

