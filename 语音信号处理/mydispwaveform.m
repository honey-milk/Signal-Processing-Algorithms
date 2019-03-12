function mydispwaveform(varargin)
%MYDISPWAVEFORM - Display waveform
%
%   This MATLAB function plots the waveform of the input vector x.
%
%   mydispwaveform(t,x,fs)
%   mydispwaveform(t,x,fs,Property)
%   mydispwaveform(...,option)
%
%   Property = struct('linespec',[],'xlabel',[],'ylabel',[],'title',[],'xlim',[],'ylim',[])
%   option:{'time'(default),'freq'}

%% ��������
% ��������Ŀ
narginchk(3,10);

% ��ʼ���������
Propertyin = struct();
Property.linespec = '';
Property.xlabel = 'ʱ��(s)';
Property.ylabel = '����';
Property.title = 'ʱ����';
Property.xlim = [];
Property.ylim = [];
domain = 'time';

% ��ȡ�������option
if (nargin > 2 && ischar(varargin{end})) && any(strcmpi(varargin{end},{'time','freq'})),
    domain = varargin{end};
    varargin(end) = [];
end

% ��ȡʣ���������ֵ
narg = numel(varargin); % ��ȡʣ�������������
switch narg
    case 3
        [t,x,fs] = varargin{:};
    case 4
        [t,x,fs,Propertyin] = varargin{:};
end

% ����������ֵ
% ������x
if isvector(x)==1
    x = x(:);   % ��xתΪ������
else
    error('�������x����Ϊ1ά����');
end
% ������domain
f = [];
if strcmpi(domain,'freq')
    Property.xlabel = 'Ƶ��(Hz)';
    Property.title = '������';
    f = t;
end

% ������Property
fieldname = fieldnames(Propertyin);
for i=1:numel(fieldname)
    field = fieldname{i};
    if isfield(Property,field)
        value = getfield(Propertyin,field);
        if ~isempty(value)
            Property = setfield(Property,field,value);
        end
    else
        error(['�ַ�������',field,'Ϊ��Чѡ��']);
    end
end

%% ���Ʋ���
% �ж�ʱ����Ƶ��
nx = length(x);
if isempty(Property.linespec)
    Property.linespec = '';
end
if strcmpi(domain,'time')
    if isempty(t)
        t = (0:(nx-1))/fs;
    end
    plot(t,x,Property.linespec);
else
    if isempty(f)
        f = 0:fs/nx:fs/2;
        x = x(1:length(f));
    end
    plot(f,x,Property.linespec);   
end
xlabel(Property.xlabel);
ylabel(Property.ylabel);
title(Property.title);
if ~isempty(Property.xlim)
    xlim(Property.xlim);
end
if ~isempty(Property.ylim)
    ylim(Property.ylim);
end
