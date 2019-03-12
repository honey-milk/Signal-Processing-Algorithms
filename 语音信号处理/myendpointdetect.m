function varargout = myendpointdetect(frame,fs,M,Z,thresh,noverlap,disp)
%GETSPEECH - extract speech signal by the time domain feature
%
%   label = myendpointdetect(frame,fs,M,Z,[thresm,thresz])
%   label = myendpointdetect(frame,fs,M,Z,[thresm,thresz],noverlap)
%   label = myendpointdetect(frame,fs,M,Z,[thresm,thresz],noverlap,disp)
%   [label,endpoint] = myendpointdetect(...)

%% ������Ŀ���
narginchk(5,7);
nargoutchk(0,2);

%% ������ʼ��
%֡����֡��
[nframe,nwin] = size(frame);
%֡�ص�����
if nargin<6
    noverlap = round(nwin/2);
end
%�Ƿ���ʾ����
if nargin<7
    disp = false;
end
%֡��
nstride = nwin-noverlap;

%% ɸѡ��������ֵ��֡
threshm = thresh(1);
threshz = thresh(2);
indexm = M >= threshm;
indexz = Z >= threshz;
label = indexm|indexz;

%% Ѱ�Ҷ˵�
current = 0;
endpoint_index = [];
for i=1:nframe
    last = current;
    current = label(i);
    if (last==0) && (current==1) %����֡��ʼ
        start_index = i;
    elseif (last==1) && (current==0) %����֡����
        end_index = i-1;
        index = false(nframe,1);
        index(start_index:end_index) = 1;
        if ~any(index&indexm) %��������֡
            label(start_index:end_index) = 0;
        else
            endpoint_index = cat(1,endpoint_index,[start_index,end_index]);
        end
    elseif (i==nframe) && (current==1) %����֡����
        end_index = i;
        index = false(nframe,1);
        index(start_index:end_index) = 1;
        if ~any(index&indexm) %��������֡
            label(start_index:end_index) = 0;
        else
            endpoint_index = cat(1,endpoint_index,[start_index,end_index]);  
        end
    end
end
%������֡��0
frame(~label,:) = 0;
%�źźϳ�
S = mydeframing(frame,noverlap);

%% �˵�λ��
endpoint = zeros(size(endpoint_index));
endpoint(:,1) = (endpoint_index(:,1)-1)*nstride+1;
endpoint(:,2) = (endpoint_index(:,2)-1)*nstride+nwin;

%% û�����������disp==true������źŲ���
if nargout==0 || disp
    nx = length(S);
    t = (0:(nx-1))/fs;
    plot(t,S);
    hold on;
    xlabel('ʱ��(s)');
    ylabel('����');
    title('��ȡ�������ź�ʱ����');
    for i=1:size(endpoint,1);
        x = (endpoint(i,1)-1)/fs;
        line(x,0,'Marker','.','MarkerSize',20,'Color',[1,0,0]);
        x = (endpoint(i,2)-1)/fs;
        line(x,0,'Marker','.','MarkerSize',20,'Color',[0,1,0]);
    end
    hold off;   
end

%% ���
switch nargout
    case 1
        varargout = {label};
    case 2
        varargout = {label,endpoint};
end


