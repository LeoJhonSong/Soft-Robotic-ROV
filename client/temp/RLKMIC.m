
function varargout = RLKMIC(varargin)
% RLKMIC MATLAB code for RLKMIC.fig
%      RLKMIC, by itself, creates x new RLKMIC or raises the existing
%      singleton*.
%
%      H = RLKMIC returns the handle to x new RLKMIC or the handle to
%      the existing singleton*.
%
%      RLKMIC('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RLKMIC.M with the given input arguments.
%
%      RLKMIC('Property','Value',...) creates x new RLKMIC or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RLKMIC_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RLKMIC_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RLKMIC

% Last Modified by GUIDE v2.5 04-Sep-2020 13:04:37

% Begin initialization code - DO NOT EDIT


global dmodel;
load('h17.mat');
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @RLKMIC_OpeningFcn, ...
                   'gui_OutputFcn',  @RLKMIC_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before RLKMIC is made visible.
function RLKMIC_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to RLKMIC (see VARARGIN)

% Choose default command line output for RLKMIC
handles.output = hObject;
%设置按钮背景
P=imread('04.jpg');
set(handles.pushbutton_reset,'Cdata', P);
set(handles.pushbutton_start,'Cdata', P);
% Update handles structure
guidata(hObject, handles);
global COM;  
global rate;
COM = 'COM1';%??GUI??????COM7  
rate = 38400;  
%% tcpip do部分
global dotcp;
global opendo1;
global opendo2;
global opendo3;
global opendo4;
global opendo5;
global opendo6;
global opendo7;
global opendo8;
opendo1 = uint8([ 254, 5, 0, 0, 255, 0, 152, 53] ); %FE 05 00 00 FF 00 98 35 
opendo2 = uint8([ 254, 5, 0, 1, 255, 0, 201, 245]); %FE 05 00 01 FF 00 C9 F5
opendo3 = uint8([ 254, 5, 0, 2, 255, 0, 57, 245] ); %FE 05 00 02 FF 00 39 F5
opendo4 = uint8([ 254, 5, 0, 3, 255, 0, 104, 53] ); %FE 05 00 03 FF 00 68 35
opendo5 = uint8([ 254, 5, 0, 4, 255, 0, 217, 244]); %FE 05 00 04 FF 00 D9 F4
opendo6 = uint8([ 254, 5, 0, 5, 255, 0, 136, 52] ); %FE 05 00 05 FF 00 88 34
opendo7 = uint8([ 254, 5, 0, 6, 255, 0, 120, 52] ); %FE 05 00 06 FF 00 78 34
opendo8 = uint8([ 254, 5, 0, 7, 255, 0, 41, 244] ); %FE 05 00 07 FF 00 29 F4

global closedo1;
global closedo2;
global closedo3;
global closedo4;
global closedo5;
global closedo6;
global closedo7;
global closedo8;
closedo1 = uint8([254, 5, 0, 0, 0, 0, 217, 197]); %FE 05 00 00 00 00 D9 C5
closedo2 = uint8([254, 5, 0, 1, 0, 0, 136, 5]  ); %FE 05 00 01 00 00 88 05
closedo3 = uint8([254, 5, 0, 2, 0, 0, 120, 5]  ); %FE 05 00 02 00 00 78 05
closedo4 = uint8([254, 5, 0, 3, 0, 0, 41, 197] ); %FE 05 00 03 00 00 29 C5
closedo5 = uint8([254, 5, 0, 4, 0, 0, 152, 4]  ); %FE 05 00 04 00 00 98 04
closedo6 = uint8([254, 5, 0, 5, 0, 0, 201, 196]); %FE 05 00 05 00 00 C9 C4
closedo7 = uint8([254, 5, 0, 6, 0, 0, 57, 196] ); %FE 05 00 06 00 00 39 C4
closedo8 = uint8([254, 5, 0, 7, 0, 0, 104, 4]  ); %FE 05 00 07 00 00 68 04
% --- Outputs from this function are returned to the command line.
%% tcpip da部分
global datcp;
global command;
global command_addr;
global command_registername;
global command_registeraddrH;
global command_registeraddrL;
global command_OutputNumH;
global command_OutputNumL;
global command_bytenum;
global command_DAoutput;
global command_crc16Hi;
global command_crc16Lo;
global aucCRCHi;
global aucCRCLo;
command_addr = uint8(254); %0xFE
command_registername = uint8(16); %0x10
command_registeraddrH = uint8(0); %0x00
command_registeraddrL = uint8(0); %0x00
command_OutputNumH = uint8(0); %0x00
command_OutputNumL = uint8(10);%0x0a
command_bytenum = uint8(20); %0x14
aucCRCHi = uint8([ 0;193;129;64;1;192;128;65;1;192;128;65;0;193;129;64;1;192;128;65;0;193;129;64;0;193;129;64;1;192;128;65;1;192;128;65;0;193;129;64;0;193;129;64;1;192;128;65;0;193;129;64;1;192;128;65;1;192;128;65;0;193;129;64;1;192;128;65;0;193;129;64;0;193;129;64;1;192;128;65;0;193;129;64;1;192;128;65;1;192;128;65;0;193;129;64;0;193;129;64;1;192;128;65;1;192;128;65;0;193;129;64;1;192;128;65;0;193;129;64;0;193;129;64;1;192;128;65;1;192;128;65;0;193;129;64;0;193;129;64;1;192;128;65;0;193;129;64;1;192;128;65;1;192;128;65;0;193;129;64;0;193;129;64;1;192;128;65;1;192;128;65;0;193;129;64;1;192;128;65;0;193;129;64;0;193;129;64;1;192;128;65;0;193;129;64;1;192;128;65;1;192;128;65;0;193;129;64;1;192;128;65;0;193;129;64;0;193;129;64;1;192;128;65;1;192;128;65;0;193;129;64;0;193;129;64;1;192;128;65;0;193;129;64;1;192;128;65;1;192;128;65;0;193;129;64 ]);
aucCRCLo = uint8([ 0;192;193;1;195;3;2;194;198;6;7;199;5;197;196;4;204;12;13;205;15;207;206;14;10;202;203;11;201;9;8;200;216;24;25;217;27;219;218;26;30;222;223;31;221;29;28;220;20;212;213;21;215;23;22;214;210;18;19;211;17;209;208;16;240;48;49;241;51;243;242;50;54;246;247;55;245;53;52;244;60;252;253;61;255;63;62;254;250;58;59;251;57;249;248;56;40;232;233;41;235;43;42;234;238;46;47;239;45;237;236;44;228;36;37;229;39;231;230;38;34;226;227;35;225;33;32;224;160;96;97;161;99;163;162;98;102;166;167;103;165;101;100;164;108;172;173;109;175;111;110;174;170;106;107;171;105;169;168;104;120;184;185;121;187;123;122;186;190;126;127;191;125;189;188;124;180;116;117;181;119;183;182;118;114;178;179;115;177;113;112;176;80;144;145;81;147;83;82;146;150;86;87;151;85;149;148;84;156;92;93;157;95;159;158;94;90;154;155;91;153;89;88;152;136;72;73;137;75;139;138;74;78;142;143;79;141;77;76;140;68;132;133;69;135;71;70;134;130;66;67;131;65;129;128;64]);
command = [command_addr,command_registername,command_registeraddrH,command_registeraddrL, command_bytenum,];

function varargout = RLKMIC_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes during object deletion, before destroying properties.
function RLKMIC_DeleteFcn(~, ~, ~)
% hObject    handle to SerialCOM (see GCBO)
% ~  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global datcp;
global dotcp;
global Oadd;
global Opath;
if ~isempty(datcp) && isvalid(datcp)
    if strcmp(datcp.Status, 'open')
        fclose(datcp);
        fclose(dotcp);
    end
    delete(datcp);
    delete(dotcp);
end
Oadd = [];
Opath = [];



% --- Executes on button press in step.
function step_Callback(hObject, eventdata, handles)
% hObject    handle to step (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of step
set(hObject,'value',1);
set(handles.sine,'value',0);
set(handles.square,'value',0);
cla(handles.axes1,'reset');
global x_pr;
global y_pr;
global z_pr z;
global ts;
global h_s;
global dmodel;%预测模型
global L1 L2 L3 L4 L5 L6 L7;%手臂各腔期望长度
global Pre1 Pre2 Pre3 Pre4 Pre5 Pre6 Pre7;
global X_P Y_P Z_P
h_s=str2num(get(handles.h,'string'));%将输入的步长由字符型转为实数
x_pr=str2num(get(handles.X,'string'));%将输入的x方向的位置坐标由字符型转为实数
y_pr=str2num(get(handles.Y,'string'));%将输入的y方向的位置坐标由字符型转为实数
z_pr=str2num(get(handles.Z,'string'));%将输入的z方向的位置坐标由字符型转为实数
ts=str2num(get(handles.T,'string'));%设定手臂运动时间
%手臂运动学求取各腔长度
X_P=x_pr*ones(1,ts/h_s);
Y_P=y_pr*ones(1,ts/h_s);
Z_P=-z_pr*ones(1,ts/h_s);
for j=1:4
    z=100*ones(1,ts/h_s);X=x_pr/2*ones(1,ts/h_s);Y=y_pr/2*ones(1,ts/h_s);Z=z/2;r=24;ZZ=-z_pr*ones(1,ts/h_s);
    theta1=atan(Y./X);
    for i=1:length(z)
       if X(i)>=0&&Y(i)>=0
          theta1(i)=theta1(i);
       end
       if X(i)<0&&Y(i)>0
          theta1(i)=pi+theta1(i);
       end
       if X(i)<0&&Y(i)<=0
          theta1(i)=pi+theta1(i);
       end
       if X(i)>=0&&Y(i)<0
          theta1(i)=2*pi+theta1(i);
       end
    end
    %向心角
    fi1=pi-2.*asin(Z./(X.^2+Y.^2+Z.^2).^(1/2)); 
    %曲率半径
    r1=((X.^2+Y.^2+Z.^2)./(2*(1-cos(fi1)))).^(1/2);
    %第一段充气腔长度变化
    l11=fi1.*(r1-r.*cos(theta1));
    l12=fi1.*(r1-r.*cos(2*pi/3-theta1));
    l13=fi1.*(r1-r.*cos(4*pi/3-theta1)); 
    %针对第二段手臂弯曲关节求取向心角、偏转角及曲率半径
    %偏转角
    theta2=theta1+pi;
    %向心角
    fi2=fi1;
    %曲率半径
    r2=r1;
    %第二段充气腔长度变化
    l21=abs(fi2).*(r2-r.*cos(theta2));
    l22=abs(fi2).*(r2-r.*cos(2*pi/3-theta2));
    l23=abs(fi2).*(r2-r.*cos(4*pi/3-theta2));
    le=abs(ZZ)-z;
    for i=1:length(z)
       while l11(i)>200||l12(i)>200||l13(i)>200||l21(i)>200||l22(i)>200||l23(i)>200||le(i)>170
             z(i)=z(i)+10;
             if le(i)<102
                 break
             end
             [l111 l122 l133 l211 l222 l233 theta11 fi11 r11 theta21 fi21 r21]=inversekinematics(X_P(i),Y_P(i),z(i),r);
             l11(i)=l111;l12(i)=l122;l13(i)=l133;l21(i)=l211;l22(i)=l222;l23(i)=l233;le(i)=abs(ZZ(i))-z(i);
             Z(i)=z(i)/2;
        end
    end
    if j==1
       L10=l11-107;L20=l12-107;L30=l13-107;L40=l21-107;L50=l22-107;L60=l23-107;L70=le-102;
    end
    if j==2
       L11=l11-107;L21=l12-107;L31=l13-107;L41=l21-107;L51=l22-107;L61=l23-107;L71=le-102;
    end
    if j==3
       L12=l11-107;L22=l12-107;L32=l13-107;L42=l21-107;L52=l22-107;L62=l23-107;L72=le-102;
    end
    if j==4
       L13=l11-107;L23=l12-107;L33=l13-107;L43=l21-107;L53=l22-107;L63=l23-107;L73=le-102;
    end 
end
L1=L11;L2=L21;L3=L31;L4=L41;L5=L51;L6=L61;L7=L71;  
Pre1=kringpredict(dmodel,L10,L11,L12,L13);
Pre2=kringpredict(dmodel,L20,L21,L22,L23);
Pre3=kringpredict(dmodel,L30,L31,L32,L33);
Pre4=kringpredict(dmodel,L40,L41,L42,L43);
Pre5=kringpredict(dmodel,L50,L51,L52,L53);
Pre6=kringpredict(dmodel,L60,L61,L62,L63);
Pre7=kringpredict(dmodel,L70,L71,L72,L73);
axes(handles.axes1)
plot3(X_P,Y_P,Z_P,'o')
%
pressure=zeros(1,10);
pressure=uint16(pressure);
PressureSend(pressure);
%}




% --- Executes on button press in sine.
function sine_Callback(hObject, eventdata, handles)
% hObject    handle to sine (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of sine
set(hObject,'value',1);
set(handles.step,'value',0);
set(handles.square,'value',0);
cla(handles.axes1,'reset');
%axes(handles.axes1);
%plot(y);
global x_pr;
global y_pr;
global z_pr z;
global ts;
global h_s;
global dmodel;%预测模型
global L1 L2 L3 L4 L5 L6 L7;%手臂各腔期望长度 
global Pre1 Pre2 Pre3 Pre4 Pre5 Pre6 Pre7;
global X_P Y_P Z_P
h_s=str2num(get(handles.h,'string'));%将输入的步长由字符型转为实数
x_pr=str2num(get(handles.X,'string'));%将输入的x方向的位置坐标由字符型转为实数
y_pr=str2num(get(handles.Y,'string'));%将输入的y方向的位置坐标由字符型转为实数
z_pr=str2num(get(handles.Z,'string'));%将输入的z方向的位置坐标由字符型转为实数
ts=str2num(get(handles.T,'string'));%设定手臂运动时间
%手臂运动学求取各腔长度
t=h_s:h_s:ts;
for i=1:1:ts/(20*h_s)
    %X_P=x_pr*sin(pi*t/0.5);%ones(1,ts/h_s);
    %Y_P=y_pr*cos(pi*t/0.5);%ones(1,ts/h_s);
    %Z_P=-z_pr*ones(1,ts/h_s)-300;
    X_P(20*(i-1)+1:20*i)=x_pr*sin(pi*h_s*(20*(i-1)+1)/0.5)*ones(1,20);
    Y_P(20*(i-1)+1:20*i)=y_pr*cos(pi*h_s*(20*(i-1)+1)/0.5)*ones(1,20);
    Z_P(20*(i-1)+1:20*i)=-z_pr*ones(1,20)-300;
end
%Z_P=z_pr*h_s*t/ts+300;
for j=1:4
    %z=100*ones(1,ts/h_s);X=x_pr/2*sin(pi*t/0.5);Y=y_pr/2*cos(pi*t/0.5);Z=z/2;r=24;ZZ=-z_pr*ones(1,ts/h_s)-300;
    z=100*ones(1,length(Z_P));X=X_P;Y=Y_P;Z=z/2;r=24;ZZ=-Z_P;
    theta1=atan(Y./X);
    for i=1:length(z)
       if X(i)>=0&&Y(i)>=0
          theta1(i)=theta1(i);
       end
       if X(i)<0&&Y(i)>0
          theta1(i)=pi+theta1(i);
       end
       if X(i)<0&&Y(i)<=05
          theta1(i)=pi+theta1(i);
       end
       if X(i)>=0&&Y(i)<0
          theta1(i)=2*pi+theta1(i);
       end
    end
    %向心角
    fi1=pi-2.*asin(Z./(X.^2+Y.^2+Z.^2).^(1/2)); 
    %曲率半径
    r1=((X.^2+Y.^2+Z.^2)./(2*(1-cos(fi1)))).^(1/2);
    %第一段充气腔长度变化
    l11=fi1.*(r1-r.*cos(theta1));
    l12=fi1.*(r1-r.*cos(2*pi/3-theta1));
    l13=fi1.*(r1-r.*cos(4*pi/3-theta1)); 
    %针对第二段手臂弯曲关节求取向心角、偏转角及曲率半径
    %偏转角
    theta2=theta1+pi;
    %向心角
    fi2=fi1;
    %曲率半径
    r2=r1;
    %第二段充气腔长度变化
    l21=abs(fi2).*(r2-r.*cos(theta2));
    l22=abs(fi2).*(r2-r.*cos(2*pi/3-theta2));
    l23=abs(fi2).*(r2-r.*cos(4*pi/3-theta2));
    le=abs(ZZ)-z;
    for i=1:length(z)
       while l11(i)>180||l12(i)>180||l13(i)>180||l21(i)>180||l22(i)>180||l23(i)>180||le(i)>170
             z(i)=z(i)+10;
             if le(i)<102
                 break
             end
             [l111 l122 l133 l211 l222 l233 theta11 fi11 r11 theta21 fi21 r21]=inversekinematics(X_P(i),Y_P(i),z(i),r);
             l11(i)=l111;l12(i)=l122;l13(i)=l133;l21(i)=l211;l22(i)=l222;l23(i)=l233;le(i)=abs(ZZ(i))-z(i);
             Z(i)=z(i)/2;
        end
    end
    if j==1
       L10=l11-107;L20=l12-107;L30=l13-107;L40=l21-107;L50=l22-107;L60=l23-107;L70=le-102;
    end
    if j==2
       L11=l11-107;L21=l12-107;L31=l13-107;L41=l21-107;L51=l22-107;L61=l23-107;L71=le-102;
    end
    if j==3
       L12=l11-107;L22=l12-107;L32=l13-107;L42=l21-107;L52=l22-107;L62=l23-107;L72=le-102;
    end
    if j==4
       L13=l11-107;L23=l12-107;L33=l13-107;L43=l21-107;L53=l22-107;L63=l23-107;L73=le-102;
    end 
end
L1=L11;L2=L21;L3=L31;L4=L41;L5=L51;L6=L61;L7=L71;  
Pre1=kringpredict(dmodel,L10,L11,L12,L13);
Pre2=kringpredict(dmodel,L20,L21,L22,L23);
Pre3=kringpredict(dmodel,L30,L31,L32,L33);
Pre4=kringpredict(dmodel,L40,L41,L42,L43);
Pre5=kringpredict(dmodel,L50,L51,L52,L53);
Pre6=kringpredict(dmodel,L60,L61,L62,L63);
Pre7=kringpredict(dmodel,L70,L71,L72,L73);
Pre1=medfilt1(Pre1,200); Pre2=medfilt1(Pre2,200); Pre3=medfilt1(Pre3,200); Pre4=medfilt1(Pre4,200); Pre5=medfilt1(Pre5,200); Pre6=medfilt1(Pre6,200); Pre7=medfilt1(Pre7,200);
axes(handles.axes1)
plot3(X_P,Y_P,Z_P,'o')






% --- Executes on button press in square.
function square_Callback(hObject, eventdata, handles)
% hObject    handle to square (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of square
set(hObject,'value',1);
set(handles.sine,'value',0);
set(handles.step,'value',0);
cla(handles.axes1,'reset');
global x_pr;
global y_pr;
global z_pr z;
global ts;
global h_s;
global dmodel;%预测模型
global L1 L2 L3 L4 L5 L6 L7;%手臂各腔期望长度
global Pre1 Pre2 Pre3 Pre4 Pre5 Pre6 Pre7;
global X_P Y_P Z_P
h_s=str2num(get(handles.h,'string'));%将输入的步长由字符型转为实数
x_pr=str2num(get(handles.X,'string'));%将输入的x方向的位置坐标由字符型转为实数
y_pr=str2num(get(handles.Y,'string'));%将输入的y方向的位置坐标由字符型转为实数
z_pr=str2num(get(handles.Z,'string'));%将输入的z方向的位置坐标由字符型转为实数
ts=str2num(get(handles.T,'string'));%设定手臂运动时间
%手臂运动学求取各腔长度
t=h_s:h_s:ts;
X_P=x_pr*square(2*pi*t,50);%ones(1,ts/h_s);
Y_P=y_pr*square(2*pi*t,50);%ones(1,ts/h_s);
Z_P=-z_pr*ones(1,ts/h_s);
for j=1:4
    z=100*ones(1,ts/h_s);X=x_pr/2*square(2*pi*t,50);Y=y_pr/2*square(2*pi*t,50);Z=z/2;r=24;ZZ=-z_pr*ones(1,ts/h_s);
    theta1=atan(Y./X);
    for i=1:length(z)
       if X(i)>=0&&Y(i)>=0
          theta1(i)=theta1(i);
       end
       if X(i)<0&&Y(i)>0
          theta1(i)=pi+theta1(i);
       end
       if X(i)<0&&Y(i)<=0
          theta1(i)=pi+theta1(i);
       end
       if X(i)>=0&&Y(i)<0
          theta1(i)=2*pi+theta1(i);
       end
    end
    %向心角
    fi1=pi-2.*asin(Z./(X.^2+Y.^2+Z.^2).^(1/2)); 
    %曲率半径
    r1=((X.^2+Y.^2+Z.^2)./(2*(1-cos(fi1)))).^(1/2);
    %第一段充气腔长度变化
    l11=fi1.*(r1-r.*cos(theta1));
    l12=fi1.*(r1-r.*cos(2*pi/3-theta1));
    l13=fi1.*(r1-r.*cos(4*pi/3-theta1)); 
    %针对第二段手臂弯曲关节求取向心角、偏转角及曲率半径
    %偏转角
    theta2=theta1+pi;
    %向心角
    fi2=fi1;
    %曲率半径
    r2=r1;
    %第二段充气腔长度变化
    l21=abs(fi2).*(r2-r.*cos(theta2));
    l22=abs(fi2).*(r2-r.*cos(2*pi/3-theta2));
    l23=abs(fi2).*(r2-r.*cos(4*pi/3-theta2));
    le=abs(ZZ)-z;
    for i=1:length(z)
       while l11(i)>180||l12(i)>180||l13(i)>180||l21(i)>180||l22(i)>180||l23(i)>180||le(i)>170
             z(i)=z(i)+10;
             if le(i)<102
                 break
             end
             [l111 l122 l133 l211 l222 l233 theta11 fi11 r11 theta21 fi21 r21]=inversekinematics(X_P(i),Y_P(i),z(i),r);
             l11(i)=l111;l12(i)=l122;l13(i)=l133;l21(i)=l211;l22(i)=l222;l23(i)=l233;le(i)=abs(ZZ(i))-z(i);
             Z(i)=z(i)/2;
        end
    end
    if j==1
       L10=l11-107;L20=l12-107;L30=l13-107;L40=l21-107;L50=l22-107;L60=l23-107;L70=le-102;
    end
    if j==2
       L11=l11-107;L21=l12-107;L31=l13-107;L41=l21-107;L51=l22-107;L61=l23-107;L71=le-102;
    end
    if j==3
       L12=l11-107;L22=l12-107;L32=l13-107;L42=l21-107;L52=l22-107;L62=l23-107;L72=le-102;
    end
    if j==4
       L13=l11-107;L23=l12-107;L33=l13-107;L43=l21-107;L53=l22-107;L63=l23-107;L73=le-102;
    end 
end
L1=L11;L2=L21;L3=L31;L4=L41;L5=L51;L6=L61;L7=L71;  
Pre1=kringpredict(dmodel,L10,L11,L12,L13);
Pre2=kringpredict(dmodel,L20,L21,L22,L23);
Pre3=kringpredict(dmodel,L30,L31,L32,L33);
Pre4=kringpredict(dmodel,L40,L41,L42,L43);
Pre5=kringpredict(dmodel,L50,L51,L52,L53);
Pre6=kringpredict(dmodel,L60,L61,L62,L63);
Pre7=kringpredict(dmodel,L70,L71,L72,L73);
axes(handles.axes1)
plot3(X_P,Y_P,Z_P,'o')



% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have x white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function X_Callback(hObject, eventdata, handles)
% hObject    handle to X (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of X as text
%        str2double(get(hObject,'String')) returns contents of X as x double


% --- Executes during object creation, after setting all properties.
function X_CreateFcn(hObject, eventdata, handles)
% hObject    handle to X (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have x white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Y_Callback(hObject, eventdata, handles)
% hObject    handle to Y (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Y as text
%        str2double(get(hObject,'String')) returns contents of Y as x double


% --- Executes during object creation, after setting all properties.
function Y_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Y (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have x white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Z_Callback(hObject, eventdata, handles)
% hObject    handle to Z (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Z as text
%        str2double(get(hObject,'String')) returns contents of Z as x double


% --- Executes during object creation, after setting all properties.
function Z_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Z (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have x white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function Untitled_1_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_1 (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function h_Callback(hObject, eventdata, handles)
% hObject    handle to h (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of h as text
%        str2double(get(hObject,'String')) returns contents of h as x double


% --- Executes during object creation, after setting all properties.
function h_CreateFcn(hObject, eventdata, handles)
% hObject    handle to h (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have x white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function T_Callback(hObject, eventdata, handles)
% hObject    handle to T (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of T as text
%        str2double(get(hObject,'String')) returns contents of T as x double


% --- Executes during object creation, after setting all properties.
function T_CreateFcn(hObject, eventdata, handles)
% hObject    handle to T (see GCBO)
% eventdata  reserved - to be defined in x future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have x white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_reset.
function pushbutton_reset_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.sine,'value',0);
set(handles.step,'value',0);
set(handles.square,'value',0);
cla(handles.axes1);

% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over pushbutton_reset.
function pushbutton_reset_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton_reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function axes_3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes_3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
imshow(imread('01.jpg'));
% Hint: place code in OpeningFcn to populate axes_3


% --- Executes during object creation, after setting all properties.
function pushbutton_reset_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton_reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
%imshow(imread('02.jpg'))


% --- Executes on button press in pushbutton_start.
% SONG: start
function pushbutton_start_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

cla(handles.axes1);
axes(handles.axes1)
%
a=serial('COM3');
set(a,'BaudRate', 9600,'DataBits',8,'StopBits',1,'Parity','none','FlowControl','none');
set(a,'InputBufferSize',1024);
%a.BytesAvailableFcnMode='terminator';  
a.BytesAvailableFcnCount=0.6; 
a.ReadAsyncMode='continuous';
fopen(a);
%}
pressure=zeros(1,10);
%
pressure=uint16(pressure);
PressureSend(pressure);
pause(1);
%}
global x_pr;
global y_pr;
global z_pr z;
global ts;
global h_s;
global L1 L2 L3 L4 L5 L6 L7;%手臂各腔期望长度
global Pre1 Pre2 Pre3 Pre4 Pre5 Pre6 Pre7;
global X_P Y_P Z_P
global recBuff 
global DouCaY DouCaX DouCaZ;
global sCOM1
global dotcp;global datcp;
global opendo6;
global opendo7;
global opendo5;
global closedo5;
global closedo7;
global closedo6;
fwrite(dotcp,closedo7(1,:),'char');
pause(0.3);
fwrite(dotcp,closedo6(1,:),'char');
pause(0.3);
%
fwrite(dotcp,opendo7(1,:),'char');
pause(0.3);
fwrite(dotcp,opendo6(1,:),'char');
%}
ier1=0; ier2=0; ier3=0; ier4=0; ier5=0; ier6=0; ier7=0;ier=0;
xx1=0;xx2=0;yy1=0;yy2=0;zz1=0;zz2=0;r=24;ZZ=-z_pr*ones(1,ts/h_s);
eer1=0;eer2=0;eer3=0;eer4=0;eer5=0;eer6=0;eer7=0;
%%%%%%%%%强化学习卡尔曼%%%%%%%%%%%%%%%
syms kp1 ki1 kd1 kp2 ki2 kd2 kp3 ki3 kd3 kp4 ki4 kd4 kp5 ki5 kd5 kp6 ki6 kd6 kp7 ki7 kd7;
AAP1=[-kp1/kd1 -ki1/kd1 ;1 0];
AAP2=[-kp2/kd2 -ki2/kd2 ;1 0];
AAP3=[-kp3/kd3 -ki3/kd3 ;1 0];
AAP4=[-kp4/kd4 -ki4/kd4 ;1 0];
AAP5=[-kp5/kd5 -ki5/kd5 ;1 0];
AAP6=[-kp6/kd6 -ki6/kd6 ;1 0];
AAP7=[-kp7/kd7 -ki7/kd7 ;1 0];
BB1=[1 0]';
CCP1=[-1/kd1 0];CCP2=[-1/kd2 0];CCP3=[-1/kd3 0];CCP4=[-1/kd4 0];CCP5=[-1/kd5 0];CCP6=[-1/kd6 0];CCP7=[-1/kd7 0];
kp1=1;ki1=0.001;kd1=0.01;kp2=1;ki2=0.001;kd2=0.01;kp3=1;ki3=0.001;kd3=0.01;kp4=1;ki4=0.001;kd4=0.01;kp5=1;ki5=0.001;kd5=0.01;kp6=1;ki6=0.001;kd6=0.01;kp7=1;ki7=0.001;kd7=0.01;
Q_f1=zeros(7,7);state_f1=1;action_f1=1;Q_f2=zeros(7,7);state_f2=1;action_f2=1;Q_f3=zeros(7,7);state_f3=1;action_f3=1;
Q_f4=zeros(7,7);state_f4=1;action_f4=1;Q_f5=zeros(7,7);state_f5=1;action_f5=1;Q_f6=zeros(7,7);state_f6=1;action_f6=1;
Q_f7=zeros(7,7);state_f7=1;action_f7=1;
%[kp,state_f,action_f,Q_f]=GRLKPID(s_f,kp,state_f,action_f,Q_f);
AAPP1=eval(AAP1);CCPP1=eval(CCP1);AAPP2=eval(AAP2);CCPP2=eval(CCP2);AAPP3=eval(AAP3);CCPP3=eval(CCP3);
AAPP4=eval(AAP4);CCPP4=eval(CCP4);AAPP5=eval(AAP5);CCPP5=eval(CCP5);AAPP6=eval(AAP6);CCPP6=eval(CCP6);
AAPP7=eval(AAP7);CCPP7=eval(CCP7);
QQ_P1=1;R_P1=1;QQ_P2=1;R_P2=1;QQ_P3=1;R_P3=1;QQ_P4=1;R_P4=1;QQ_P5=1;R_P5=1;QQ_P6=1;R_P6=1;QQ_P7=1;R_P7=1;
P_P1=BB1*QQ_P1*BB1';P_P2=BB1*QQ_P2*BB1';P_P3=BB1*QQ_P3*BB1';P_P4=BB1*QQ_P4*BB1';P_P5=BB1*QQ_P5*BB1';P_P6=BB1*QQ_P6*BB1';P_P7=BB1*QQ_P7*BB1';
x_f1=zeros(2,1);x_f2=zeros(2,1);x_f3=zeros(2,1);x_f4=zeros(2,1);x_f5=zeros(2,1);x_f6=zeros(2,1);x_f7=zeros(2,1);
sz1=0;sz2=0;sz3=0;sz4=0;sz5=0;sz6=0;sz7=0;
ier1=0; ier2=0; ier3=0; ier4=0; ier5=0; ier6=0; ier7=0;ier=0;
xx1=0;xx2=0;yy1=0;yy2=0;zz1=0;zz2=0;r=24;ZZ=-z_pr*ones(1,ts/h_s);
%%%%%%%%%%RLKMIC基本参数初始化设定%%%%%%%%%%%%%%%%%%%
%奖惩值表RA的设定
RA=[0 -1 -1 -1 -1 -1 -1;0 0 -1 -1 -1 -1 -1; 1 1 1 0 0 0 0;10 10 10 10 10 10 10;0 0 0 0 1 1 1; -1 -1 -1 -1 -1 0 0;-1 -1 -1 -1 -1 -1 0];
%值函数矩阵Q初始化
Q1=zeros(7,7,7); Q2=zeros(7,7,7); Q3=zeros(7,7,7); Q4=zeros(7,7,7); Q5=zeros(7,7,7); Q6=zeros(7,7,7); Q7=zeros(7,7,7);
NO1=norm(Q1(:,:,1));NO2=norm(Q2(:,:,1));NO3=norm(Q3(:,:,1));NO4=norm(Q4(:,:,1));NO5=norm(Q5(:,:,1));NO6=norm(Q6(:,:,1));NO7=norm(Q7(:,:,1));
%学习因子与折扣因子
alpha=1;gama=0.8;
%可调系数lambda初始值
lambda=0;
%状态s1初始值
s1=0;
%%%%%%%%%%%%%%%%%%根据贪婪策略选择动作%%%%%%%%%%%%%%%%%%
   kexi1=0.2;c1=1-kexi1;
   beta1=rand(1,1);
   if beta1>c1
        action1=ceil(7*rand(1,1));
        if s1<-50
            state1=1;
        end
        if s1>=-50&&s1<-0.1
            state1=2;
        end
        if s1>=-0.1&&s1<-0.00001
            state1=3;
        end
        if s1>=-0.00001&&s1<=0.00001
            state1=4;
        end
        if s1>0.00001&&s1<=0.1
            state1=5;
        end
        if s1>0.1&&s1<=50
            state1=6;
        end
        if s1>50
            state1=7;
        end
    else
        if s1<-50
             [a1 action1]=find(Q1(1,:,1)==max(Q1(1,1,1)));
            state1=1;
         end
        if s1>=-50&&s1<-0.1
            [a1 action1]=find(Q1(2,:,1)==max(Q1(2,1:2,1))); 
            state1=2;
        end
        if s1>=-0.1&&s1<-0.00001
            [a1 action1]=find(Q1(3,:,1)==max(Q1(3,2:3,1)));
            state1=3;
        end
        if s1>=-0.00001&&s1<=0.00001
            [a1 action1]=find(Q1(4,:,1)==max(Q1(4,3:5,1)));
            state1=4;
        end
        if s1>0.00001&&s1<=0.1
            [a1 action1]=find(Q1(5,:,1)==max(Q1(5,5:6,1)));
            state1=5;
        end
        if s1>0.1&&s1<=50
            [a1 action1]=find(Q1(6,:,1)==max(Q1(6,6:7,1)));
            state1=6;
        end
        if s1>50
            [a1 action1]=find(Q1(7,:,1)==max(Q1(7,7,1)));
            state1=7;
        end
   end
   
   
   %%%%%%%%%%%%%%%%%%动作集的设定%%%%%%%%%%%%%%%%%% 
       if action1==1
          lambda=lambda-(0.1*exp(1-50/abs(s1)))+0.1*s1+0.1*ier;
       end
       if action1==2
          lambda=lambda+(0.1*s1*sin(pi*abs(s1)/100))+0.1*ier;
       end
       if action1==3
          lambda=lambda+(0.01*s1*sin(pi*abs(s1)/0.2))+0.1*ier;
       end
       if action1==4
          lambda=lambda+(0.0001*s1*sin(pi*abs(s1)/0.00002))*sign(s1)+0.1*ier;
       end
       if action1==5
          lambda=lambda+(0.01*s1*sin(pi*abs(s1)/0.2))+0.1*ier;
       end
       if action1==6
          lambda=lambda+(0.1*s1*sin(pi*abs(s1)/100))+0.1*ier;
       end
       if action1==7
          lambda=lambda+0.1*exp(1-50/abs(s1))+0.1*s1+0.1*ier;
       end
    %if lambda<0
       %lambda=0;
    %end
    %if lambda>5
       %lambda=5;
    %end
   %%%%%%%%%%%%%%%%%各根系数初始值%%%%%%%%%%%%%%%%%%%%%%
   lambda1=lambda;lambda2=lambda;lambda3=lambda;lambda4=lambda;lambda5=lambda;lambda6=lambda;
   lambda7=lambda;lambda8=lambda;
   LAM1=lambda;LAM2=lambda;LAM3=lambda;LAM4=lambda;LAM5=lambda;LAM6=lambda;LAM7=lambda;
   state2=state1;state3=state1;state4=state1;state5=state1;state6=state1;state7=state1; state1=state1;
   action2=action1;action3=action1;action4=action1;action5=action1;action6=action1;action7=action1; action1=action1;
   xini0=0;yini0=0;zini0=0;
   %////////////实际长度集合初始化//////////////////
   LS1=107*ones(1,ts/h_s);LS2=107*ones(1,ts/h_s);LS3=107*ones(1,ts/h_s);LS4=107*ones(1,ts/h_s);
   LS5=107*ones(1,ts/h_s);LS6=107*ones(1,ts/h_s);LS7=102*ones(1,ts/h_s);
   xini=1;yini=1;zini=214;
    %%%%%%%%%%%%%%%%%%仿真被控对象EUPI参数%%%%%%%%%%%%%%%%%%%
   %W1=[-2.22021844675873,-0.641986634334619,0.835255105659817,0.714028425537246,-2.87675157274286,-0.843894509079341,3.13845405939732,0.915365108806387,-3.83262613298571,-0.525298503899920,0.267613560786327,2.79282380496050,0.0905753561776842,0.887370755500148,0.198374983772150,2.27339816741306,5.01815378501914,3.02428151755555,-2.28503399010818,6.77508662170105,-3.80731477030946,3.75405260077398,0.265428838707564,0.641442674597030,1.49589207203980,1.87585396440132,-0.800921648497692,1.67283246775057,-2.47792406357117,-0.294630267292461,0.889104191664121,0.578357501397868,-2.06242427132022,-2.85527858978936,0.155305096783953,-0.298831868527872,4.57527985293214,0.448074783213784,0.596271206804216,2.35860808607097,-2.48856759143994,-1.19046967533148,0.313641242234930,1.63340742652537,1.17271467279212,6.44652291565462,0.504460573715024,-1.45355043961147,0.252779286512821,1.89613521617010,1.15341008677213,-0.197869554596087,1.90650203825282,0.470172402836446,9.32493478684033,1.29824776172880,-2.10352663960160,0.888006469488060,2.46874299119415,0.932083807446481,-7.76830243281942,1.98695851079901,-0.617421846407597,-3.24212294966169,1.02747346988335,0.899349306331916,-4.78514111607588,2.97415549635719,-0.777518376057248,2.05239818950669,-2.47257144965083,1.82636661824080,0.680760302311183,-1.70170315988058,-8.59715102575834,3.05493478216429,2.04569678963964,0.462229543043112,2.90989135349280,0.784718757209547,-0.823988376103487,-3.83294214772247,0.538971359685638,-0.881989852981453,-2.27048109035734,0.751599403366218,-5.05916647004845,-0.255082544290999,3.84014787550963,1.67987051360594,3.21512026748365,1.04105102554275,-2.15640359583551,0.0267051889307780,-1.10670388220477,-8.13938403896327,0.757282108822565,0.805617628983289,-3.31256838290027,1.18562251536780];
   %W2=[0.0568199678570076,-1.10375601137522,-2.94813099126420,0.0761022597751539,3.13384735793515,1.10418864203369,2.34968911823209,0.626219531034561,1.25258565388235,8.32792205299223,-2.15600559062024,-1.53060557905132,-0.121615548595415,0.0298715320691916,-0.132756742159752,-1.75539986028840,1.29726663554727,0.637389474456704,-1.56619089284471,-0.447662544464884,-0.0783275075003267,0.402486543426308,-2.14840140180063,0.276579048046998,1.60632538811143,-0.315057206756976,0.956351207730639,-0.979293061492912,-0.507409418051852,-1.38060381398132,-6.67853256494334,-0.227086552747419,-1.86924884765831,-2.42504199296709,0.382803631810499,0.0256362973965514,0.736645389578755,-0.906841256705961,-0.318605709326997,1.61121760031953,9.99999111250279,0.596117337250254,9.24478534193713,3.85976222542656,-3.17186816955872,-3.69483319963233,2.53597169288834,-0.00231323845104816,1.56122071282992,-0.700806557476776,0.0486763282193929,-9.99840754135088,3.16669901784392,0.261013239761998,0.468335003612212,-3.87104355615597,-1.96739456376273,2.62868708656364,2.13676147490345,1.10143293202864,-0.693630743784985,-3.09873524071993,-0.0530924264976952,0.126954071885016,-5.65730205607328,0.0127306815514878,1.47664872408130,-3.04349835887850,-0.0461391388247782,0.810912848744324,-3.87879341500953,5.36629614808839,2.42752122028630,0.507602021041068,0.339691538835831,3.02922736208052,-0.411787489831001,2.27151886724816,-0.810952629015356,2.27970582612959,2.22857075382504,-2.26914168583345,3.28332668455383,3.93308689877799,-2.49136927216671,0.832591732174060,-0.160076961732012,-1.70710529038937,-1.02663633783294,0.878520607521376,-0.269792897320487,-0.821935927886416,-0.594097681098450,0.286513264967286,-0.0202403323506635,-1.24411519943506,-4.37280498853710,4.14223858775809,0.000979598561111483,0.952631087207000];
   %W3=[4.58539113397914,-1.66819812796151,3.47028594305384,0.428484442871351,-4.98254592759249,4.76424420513897,1.72503598843020,0.242426479990235,1.72080774988620,2.02347455403305,-1.39153877963499,-4.87143086003907,1.23230499573022,1.80397002472888,0.409609808907849,-0.636495938178678,0.799678500932458,3.78108726648977,-1.09096129855378,-0.716616631004012,-0.485096826404239,3.88596512360001,-0.388343819227611,0.177718852093442,1.59477597502593,0.171234494033497,2.49495183319419,-0.840067509029854,-4.42635870610021,-0.899855676923782,-4.60710047290401,3.15128488232052,-1.05769563003158,-1.56754022669530,1.85811774590092,0.451653069290596,1.64510067613515,0.219928175017938,0.511796455597048,1.71612061677239,2.78331099347743,0.424828415948150,0.661447145248083,1.95987394414912,0.286465060944336,0.396088457201341,2.04358969902242,2.10096707257096,2.47586774752875,1.82024636897934,0.913319468465070,0.282961016543222,4.38733233794977,-0.739651408555436,0.198243558685412,-0.00996435484733208,0.588467971990929,3.49062640722636,1.58984256745148,1.32361842249075,0.636595408104757,3.79142410258744,1.39244458308145,0.525091063759385,-0.880593617111543,0.569446150919982,3.01213771980659,-1.84096940213880,0.189834508435664,2.19100893676952,-0.893963648313739,-0.921441272224811,4.90682415631291,1.73608612045712,-0.357500043147065,5.13160602241622,2.63754619387692,0.218568964655287,0.415032279886825,2.02376715676250,2.33237594794059,0.464589735269131,0.286668133699922,2.04904623280800,1.76171530784285,2.62724834869586,2.85535516917729,0.0700386570732575,0.519702685902996,2.79438099666980,1.02506115720260,3.29202817762273,3.17081569784617,0.411500791225221,4.90955805966055,1.62649399527506,6.59537813917385,2.93876107455599,5.27645256171583,1.13433280766069];
   %p1=0.668503789498220;
   %p2=0.338299418976386;
   W1=[-3.09479755659205,1.70048476002302,0.370414761486346,-3.38253130590779,0.431775664004444,2.35254524657906,1.84609934390742,-0.800043443802622,4.07929735400909,-0.0914602449319138,4.27271235884132,-3.57839529376426,-1.37618599677622,3.17678254455182,4.63767176695496,-3.61486974913582,-12.4914964116085,1.46056888528974,-0.606361686917978,8.00690971057119,-4.25359314347076,1.36473607342831,1.99664252003411,1.99034723694324,-2.64929724848913,1.17324534190423,-1.03367166842160,-0.556228186608643,-2.79909932964200,-2.46921434484929];
   W2=[0.889023784255756,5.31597681426150,1.48707274529788,-1.04556611632018,-3.13437516720004,0.884765361101010,-1.04978522660776,1.95219343705108,0.267542241890621,-4.99075967812481,0.392047832665825,2.74534059622179,-0.814317848386915,0.549679195986895,2.38723290945983,5.92798720159298,0.530975614871844,1.13958922739353,-0.0474886382426681,-1.54815494565088,-1.18181633669045,6.12024737107759,-1.11180072301667,-2.42769885777679,0.406861458268700,0.901299223604231,2.52545084847076,-0.0108793254379065,0.00806934861720831,-0.891265910819854];
   W3=[1.33856327829803,4.12846380127799,-0.0839218845213606,-1.04485051287188,-2.66096113597440,0.361062831824641,5.50691689576247,1.66594398397364,-2.76450693749967,-15,0.702080710593632,4.16379110400282,-14.8155746426281,1.61154827876533,5.05485130243708,7.49542912924541,0.0859678442517601,0.0713798642459240,1.61110429840560,1.50105377063121,0.750945789345602,6.66696431805256,-13.9162713605071,0.808877102371189,1.00351635980190,2.96583910591349,1.28098218964498,2.46826886709203,2.91130798918343,-2.15432142089081];
   p1=-2.09919701765775;
   p2=0.794556282051666;
   %W1=[-3.79754594135009,2.24206076039312,1.03347267229040,-2.37643765499991,1.40985217800501,3.84436463293206,-0.185259529569172,-0.143683463807759,-1.83910422700603,-4.67518974090308,-3.54787473409427,-1.43763495569546,9.26417887230630,1.28638920718333,3.34221380390271,4.13295746118179,-0.123859140319283,0.127072717313144,8.16156110480018,-3.03637960271162,-6.28993659919164,6.40900181902842,-3.39121706431312,1.21314453165951,10.6214902247237,-1.84596199874886,3.95046935166688,-1.75371554933535,-2.54358707384782,1.51050846034968];
   %W2=[4.26815271768435,-3.08085750192684,-0.848314340090781,-3.81154984524135,-2.09782815429182,-0.506960892341327,-1.40488134623908,-1.12131231779597,1.38559100346836,-1.35644030036157,-0.0304022100443678,-2.17778293291070,-0.179894203868717,3.64799443465258,2.49849816941335,2.63720053607689,-0.900259251766111,-0.179057681600981,-2.74746966554983,1.92617320325049,-1.56418084177155,-0.832778033670467,-4.55204663166798,-0.539660029727703,-0.826388876278116,4.61785295237844,-4.61978130319033,-1.05229131727001,1.22711553732711,0.258795251387357];
   %W3=[1.23906203367101,0.102588468746055,-14.1400450393985,-5.02996350973849,-1.32524009561317,0.807232540094683,8.05393391431756,-0.875764645310719,2.91929773480731,2.40026011454054,4.56257923979817,1.77162839033548,-15,4.06824621243460,1.45046149661490,2.14839126292838,-14.2149333919249,14.9999906111701,0.915102434223781,5.60824709839826,0.819647149063159,6.89688277146493,-2.98558781929827,1.43924095722313,-0.138878774005306,2.89817503259871,2.03381607258704,1.82197221446468,2.99596204821919,0.538650328525113];
   %p1=-1.200172396352898;
   %p2=1.539487319630142;
   rr=linspace(0,2,30);rr=rr';pp1=0;pp2=0;pp3=0;pp4=0;pp5=0;pp6=0;pp7=0;
   xx=zeros(30,1);tt=0.01;
   tic
   for i=1:(ts/h_s-1)
      %
       %{
       [LS1(i+1),u1]=EUPI1(lambda1,Pre1(i),W1,W2,W3,xx,rr,p1,p2,tt);
       [LS2(i+1),u2]=EUPI1(lambda2,Pre2(i),W1,W2,W3,xx,rr,p1,p2,tt);
       [LS3(i+1),u3]=EUPI1(lambda3,Pre3(i),W1,W2,W3,xx,rr,p1,p2,tt);
       [LS4(i+1),u4]=EUPI1(lambda4,Pre4(i),W1,W2,W3,xx,rr,p1,p2,tt);
       [LS5(i+1),u5]=EUPI1(lambda5,Pre5(i),W1,W2,W3,xx,rr,p1,p2,tt);
       [LS6(i+1),u6]=EUPI1(lambda6,Pre6(i),W1,W2,W3,xx,rr,p1,p2,tt);
       [LS7(i+1),u7]=EUPI1(lambda7,Pre7(i),W1,W2,W3,xx,rr,p1,p2,tt);
       %}
       pressure1 = (5555.5555555555555555555555555556*(L1(i)+107) + ((5555.5555555555555555555555555556*(L1(i)+107) - 602224.58212670832698267540517198)^2 + 93258097.726418903983160271113075)^(1/2) - 602224.58212670832698267540517198)^(1/3) - 453.48422496570644718792866941015/(5555.5555555555555555555555555556*(L1(i)+107) + ((5555.5555555555555555555555555556*(L1(i)+107) - 602224.58212670832698267540517198)^2 + 93258097.726418903983160271113075)^(1/2) - 602224.58212670832698267540517198)^(1/3) - 4.8148148148148148148148148148148;
       pressure2 = (5555.5555555555555555555555555556*(L2(i)+107) + ((5555.5555555555555555555555555556*(L2(i)+107) - 602224.58212670832698267540517198)^2 + 93258097.726418903983160271113075)^(1/2) - 602224.58212670832698267540517198)^(1/3) - 453.48422496570644718792866941015/(5555.5555555555555555555555555556*(L2(i)+107) + ((5555.5555555555555555555555555556*(L2(i)+107) - 602224.58212670832698267540517198)^2 + 93258097.726418903983160271113075)^(1/2) - 602224.58212670832698267540517198)^(1/3) - 4.8148148148148148148148148148148;
       pressure3 = (5555.5555555555555555555555555556*(L3(i)+107) + ((5555.5555555555555555555555555556*(L3(i)+107) - 602224.58212670832698267540517198)^2 + 93258097.726418903983160271113075)^(1/2) - 602224.58212670832698267540517198)^(1/3) - 453.48422496570644718792866941015/(5555.5555555555555555555555555556*(L3(i)+107) + ((5555.5555555555555555555555555556*(L3(i)+107) - 602224.58212670832698267540517198)^2 + 93258097.726418903983160271113075)^(1/2) - 602224.58212670832698267540517198)^(1/3) - 4.8148148148148148148148148148148;
       pressure4 = 0.51789321*(L7(i)+102) - 64.06856906;
       %}
       %{
       pressure1 = 0.000038897274 *(L1(i)+107)^3 - 0.022105597315*(L1(i)+107)^2 + 4.666953273827*(L1(i)+107) - 335.101861413281;
       pressure2 = 0.000038897274 *(L2(i)+107)^3 - 0.022105597315*(L2(i)+107)^2 + 4.666953273827*(L2(i)+107) - 335.101861413281;
       pressure3 = 0.000038897274 *(L3(i)+107)^3 - 0.022105597315*(L3(i)+107)^2 + 4.666953273827*(L3(i)+107) - 335.101861413281;
       pressure4 = 0.51789321*(L7(i)+102) - 64.06856906;
       %}
       %if pressure1<0
           %pressure1=0;
      % end
       %if pressure2<0
      %    pressure2=0;
      % end
       %if pressure3<0
          %pressure3=0;
       %end
       %if pressure4<0
          % pressure4=0;
      %end
       %%% pressure 1
       pressure(1)=1*pressure1+0.1*lambda1;
      % set(handles.text19, 'string', u1);
       if pressure(1)<0
          pressure(1)=0;
       end
       if pressure(1)>130
          pressure(1)=130;
       end
       %%% pressure 2
       pressure(2)=1*pressure2+0.1*lambda2;
       if pressure(2)<0
          pressure(2)=0;
       end
       if pressure(2)>130
          pressure(2)=130;
       end
       %%% pressure 3
       pressure(3)=1*pressure3+0.1*lambda3;
       if  pressure(3)<0
            pressure(3)=0;
       end
       if  pressure(3)>130
            pressure(3)=130;
       end
       
       %%% pressure 4
       pressure(4)= 1*pressure(1)+0.01*lambda4;
       if  pressure(4)>130
            pressure(4)=130;
       end
       %%% pressure 5
       pressure(5)= 1*pressure(2)+0.01*lambda5;
       if  pressure(5)>130
            pressure(5)=130;
       end
       %%% pressure 6
       pressure(6)= 1*pressure(3)+0.01*lambda6;
       if  pressure(6)>130
            pressure(6)=130;
       end
       %%% pressure 7 for elongation
       pressure(7)= 0;
       %pressure(8)= 0;
       %pressure(9)= 0;%pressure4+lambda7;
       if pressure(7)<0
          pressure(7)=0;
       end
       if pressure(7)>30
          pressure(7)=30;
       end
       %
       if i>(ts/h_s-10)/2&&i<(ts/h_s-1)
          pressure(7)=5*i/(ts/h_s-5);  
          %pressure(8)=7*i/(ts/h_s-5);
         % pressure(9)=7*i/(ts/h_s-5);
       else
           if i>(ts/h_s-3)
               pressure(7)=30;
               %pressure(8)=8;
               %pressure(9)=8;
           end
       end
       %}
       %PressureSend(pressure);
       %%% pressure 8 for gripper
       %pressure(8) = pressure5;
       %pressure(10)=90;
       pressure11=uint16(pressure*2);
       PressureSend(pressure11);
       if i==(ts/h_s-1)-2
           set(handles.text19, 'string',1);
           pause(0.1);
           fwrite(dotcp,closedo7(1,:),'char');
           pause(0.1);
           fwrite(dotcp,closedo6(1,:),'char');
           pause(0.1);
       end 
       %}
       LS7(i+1)=LS7(i+1)-5;
       %{
       rd1=(LS1(i+1)+LS2(i+1)+LS3(i+1))*r/(2*(LS1(i+1)^2+LS2(i+1)^2+LS3(i+1)^2-LS1(i+1)*LS2(i+1)-LS2(i+1)*LS3(i+1)-LS1(i+1)*LS3(i+1))^(1/2));
       fi1=(2*(LS1(i+1)^2+LS2(i+1)^2+LS3(i+1)^2-LS1(i+1)*LS2(i+1)-LS2(i+1)*LS3(i+1)-LS1(i+1)*LS3(i+1))^(1/2))/(3*r);
       theta1=atan((sqrt(3)*(LS3(i+1)-LS2(i+1)))/(LS2(i+1)+LS3(i+1)-2*LS1(i+1)));
       rd2=(LS4(i+1)+LS5(i+1)+LS6(i+1))*r/(2*(LS4(i+1)^2+LS5(i+1)^2+LS6(i+1)^2-LS4(i+1)*LS5(i+1)-LS5(i+1)*LS6(i+1)-LS4(i+1)*LS6(i+1))^(1/2));
       fi2=(2*(LS4(i+1)^2+LS5(i+1)^2+LS6(i+1)^2-LS4(i+1)*LS5(i+1)-LS5(i+1)*LS6(i+1)-LS4(i+1)*LS6(i+1))^(1/2))/(3*r);
       theta2=atan((sqrt(3)*(LS6(i+1)-LS5(i+1)))/(LS5(i+1)+LS6(i+1)-2*LS4(i+1)));
       den1=(LS2(i+1)+LS3(i+1)-2*LS1(i+1));
       num1=(sqrt(3)*(LS3(i+1)-LS2(i+1)));
       den2=(LS5(i+1)+LS6(i+1)-2*LS4(i+1));
       num2=(sqrt(3)*(LS6(i+1)-LS5(i+1))); 
       if den1>=0&&num1>=0
          theta1=theta1;
       end
       if den1<0&&num1>0
          theta1=pi+theta1;
       end
       if den1<0&&num1<=0
          theta1=pi+theta1;
       end
       if den1>=0&&num1<0
          theta1=2*pi+theta1;
       end
       if den1>=0
         theta2=theta1+pi;
       end 
       if den1<0
         theta2=theta1+pi;
       end 
       T1=[cos(theta1) -sin(theta1) 0 0; sin(theta1) cos(theta1) 0 0; 0 0 1 0;0 0 0 1]*[cos(fi1/2) 0 sin(fi1/2) 0; 0 1 0 0; -sin(fi1/2) 0 cos(fi1/2) 0; 0 0 0 1]*[1 0 0 0; 0 1 0 0; 0 0 1 2*rd1*sin(fi1/2);0 0 0 1]*[cos(fi1/2) 0 sin(fi1/2) 0; 0 1 0 0; -sin(fi1/2) 0 cos(fi1/2) 0; 0 0 0 1]*[cos(-theta1) -sin(-theta1) 0 0; sin(-theta1) cos(-theta1) 0 0; 0 0 1 0;0 0 0 1];
       T2=[cos(theta2) -sin(theta2) 0 0; sin(theta2) cos(theta2) 0 0; 0 0 1 0;0 0 0 1]*[cos(fi2/2) 0 sin(fi2/2) 0; 0 1 0 0; -sin(fi2/2) 0 cos(fi2/2) 0; 0 0 0 1]*[1 0 0 0; 0 1 0 0; 0 0 1 2*rd2*sin(fi2/2);0 0 0 1]*[cos(fi2/2) 0 sin(fi2/2) 0; 0 1 0 0; -sin(fi2/2) 0 cos(fi2/2) 0; 0 0 0 1]*[cos(-theta2) -sin(-theta2) 0 0; sin(-theta2) cos(-theta2) 0 0; 0 0 1 0;0 0 0 1];
       PO=T1*T2*[0 0 0 1]';
       %}
      %
       %pause(0.05);
       %fopen(a);
       for jj=0:0.1:1
           cc = fscanf(a);
           pause(0.02);
       end
       lcc=length(cc);
      % recBuff = fread(a, 7, 'uint8');
       while cc(1)~='#'
             cc= fscanf(a) ;
             lcc=length(cc);
       end
      %fwrite(a,'clear all');
      %fclose(a);
      %recBuff = fread(a, 7, 'uint8');
             ms=regexp(cc,'\-?\d*\.?\d*','match');
             DouCaX1=1*(str2double(ms{2}))+0*(rand(1)-0.5);
             DouCaY1=0.7*str2double(ms{1})+0*(rand(1)-0.5);
             DouCaZ1=0;
            
      yini=y_pr-DouCaY1;xini=x_pr-DouCaX1;zini=Z_P(i)-0;%}
       %}
       %xini=PO(1);%+5*tt;
       %yini=PO(2);
       %zini=PO(3);
       %rd12=rd1;
       xini0=[xini0,xini];yini0=[yini0,yini];
       [l111r l122r l133r l211r l222r l233r le theta1 fi1 r1 theta2 fi2 r2]=inversekinematics1(xini,yini,abs(zini),r);
       LS1(i+1)=l111r;LS2(i+1)=l122r;LS3(i+1)=l133r;LS4(i+1)=l211r;LS5(i+1)=l222r;LS6(i+1)=l233r;
       [l111r l122r l133r l211r l222r l233r le theta1 fi1 r1 theta2 fi2 r2]=inversekinematics1(X_P(i),Y_P(i),abs(Z_P(i)),r);
       L11=l111r;L22=l122r;L33=l133r;L44=l211r;L55=l222r;L66=l233r;L77=le;zini0=[zini0,zini+LS7(i+1)];
       [x_f1,LSS1(i+1),sz1,P_P1,pp1]=Kalman(AAPP1,BB1,CCPP1,L11,QQ_P1,P_P1,R_P1,x_f1,0.1*pressure1+40*lambda1,LS1(i+1),i,pp1);
       [x_f2,LSS2(i+1),sz2,P_P2,pp2]=Kalman(AAPP2,BB1,CCPP2,L22,QQ_P2,P_P2,R_P2,x_f2,0.1*pressure2+40*lambda2,LS2(i+1),i,pp2);
       [x_f3,LSS3(i+1),sz3,P_P3,pp3]=Kalman(AAPP3,BB1,CCPP3,L33,QQ_P3,P_P3,R_P3,x_f3,0.1*pressure3+40*lambda3,LS3(i+1),i,pp3);
       [x_f4,LSS4(i+1),sz4,P_P4,pp4]=Kalman(AAPP4,BB1,CCPP4,L44,QQ_P4,P_P4,R_P4,x_f4,0.1*pressure1+40*lambda1,LS4(i+1),i,pp4);
       [x_f5,LSS5(i+1),sz5,P_P5,pp5]=Kalman(AAPP5,BB1,CCPP5,L55,QQ_P5,P_P5,R_P5,x_f5,0.1*pressure2+40*lambda2,LS5(i+1),i,pp5);
       [x_f6,LSS6(i+1),sz6,P_P6,pp6]=Kalman(AAPP6,BB1,CCPP6,L66,QQ_P6,P_P6,R_P6,x_f6,0.1*pressure3+40*lambda3,LS6(i+1),i,pp6);
       [x_f7,LSS7(i+1),sz7,P_P7,pp7]=Kalman(AAPP7,BB1,CCPP7,L77,QQ_P7,P_P7,R_P7,x_f7,lambda7+Pre7(i),LS7(i+1),i,pp7);
       [kp1,R_P1,QQ_P1,state_f1,action_f1,action_p1,Q_f1]=GRLKPID1(-sz1,kp1,R_P1,QQ_P1,state_f1,action_f1,Q_f1);
       [kp2,R_P2,QQ_P2,state_f2,action_f2,action_p2,Q_f2]=GRLKPID1(-sz2,kp2,R_P2,QQ_P2,state_f2,action_f2,Q_f2);
       [kp3,R_P3,QQ_P3,state_f3,action_f3,action_p3,Q_f3]=GRLKPID1(-sz3,kp3,R_P3,QQ_P3,state_f3,action_f3,Q_f3);
       [kp4,R_P4,QQ_P4,state_f4,action_f4,action_p4,Q_f4]=GRLKPID1(-sz4,kp4,R_P4,QQ_P4,state_f4,action_f4,Q_f4);
       [kp5,R_P5,QQ_P5,state_f5,action_f5,action_p5,Q_f5]=GRLKPID1(-sz5,kp5,R_P5,QQ_P5,state_f5,action_f5,Q_f5);
       [kp6,R_P6,QQ_P6,state_f6,action_f6,action_p6,Q_f6]=GRLKPID1(-sz6,kp6,R_P6,QQ_P6,state_f6,action_f6,Q_f6);
       [kp7,R_P7,QQ_P7,state_f7,action_f7,action_p7,Q_f7]=GRLKPID1(-sz7,kp7,R_P7,QQ_P7,state_f7,action_f7,Q_f7);
       AAPP1=eval(AAP1);CCPP1=eval(CCP1);AAPP2=eval(AAP2);CCPP2=eval(CCP2);AAPP3=eval(AAP3);CCPP3=eval(CCP3);
       AAPP4=eval(AAP4);CCPP4=eval(CCP4);AAPP5=eval(AAP5);CCPP5=eval(CCP5);AAPP6=eval(AAP6);CCPP6=eval(CCP6);
       AAPP7=eval(AAP7);CCPP7=eval(CCP7);
        %{
       LP1(i)=LS1(i+1);LP2(i)=LS2(i+1);LP3(i)=LS3(i+1);LP4(i)=LS4(i+1);LP5(i)=LS5(i+1);LP6(i)=LS6(i+1);LP7(i)=LS7(i+1);
       if i>5
           LLL1=medfilt1(LP1,5);LLL2=medfilt1(LP2,5);LLL3=medfilt1(LP3,5);
           LLL4=medfilt1(LP4,5);LLL5=medfilt1(LP5,5);LLL6=medfilt1(LP6,5);
           LLL7=medfilt1(LP7,5);
           LSS1(i+1)=LLL1(end);
           LSS2(i+1)=LLL2(end);
           LSS3(i+1)=LLL3(end);
           LSS4(i+1)=LLL4(end);
           LSS5(i+1)=LLL5(end);
           LSS6(i+1)=LLL6(end);
           LSS7(i+1)=LLL7(end);
       else
           LSS1(i+1)=LS1(i+1);LSS2(i+1)=LS2(i+1);LSS3(i+1)=LS3(i+1);LSS4(i+1)=LS4(i+1);LSS5(i+1)=LS5(i+1);LSS6(i+1)=LS6(i+1);LSS7(i+1)=LS7(i+1);
       end
       %}
       %{
       [lambda1,Q1,ier1,state1,action1]=RLKMIC1(lambda1,LS1(i+1),i+1,Q1,RA,state1,action1,ier1,alpha,gama);
       [lambda2,Q2,ier2,state2,action2]=RLKMIC1(lambda2,LS2(i+1),i+1,Q2,RA,state2,action2,ier2,alpha,gama);
       [lambda3,Q3,ier3,state3,action3]=RLKMIC1(lambda3,LS3(i+1),i+1,Q3,RA,state3,action3,ier3,alpha,gama);
       [lambda4,Q4,ier4,state4,action4]=RLKMIC1(lambda4,LS4(i+1),i+1,Q4,RA,state4,action4,ier4,alpha,gama);
       [lambda5,Q5,ier5,state5,action5]=RLKMIC1(lambda5,LS5(i+1),i+1,Q5,RA,state5,action5,ier5,alpha,gama);
       [lambda6,Q6,ier6,state6,action6]=RLKMIC1(lambda6,LS6(i+1),i+1,Q6,RA,state6,action6,ier6,alpha,gama);
       [lambda7,Q7,ier7,state7,action7]=RLKMIC1(lambda7,LS7(i+1),i+1,Q7,RA,state7,action7,ier7,alpha,gama);
       %}
       %
       [lambda1,Q1,ier1,state1,action1,eer1]=RLKMIC2(lambda1,L11-107,LSS1(i+1)-107,sz1,i+1,Q1,RA,state1,action1,action_f1,action_p1,ier1,alpha,gama,eer1);
       [lambda2,Q2,ier2,state2,action2,eer2]=RLKMIC2(lambda2,L22-107,LSS2(i+1)-107,sz2,i+1,Q2,RA,state2,action2,action_f2,action_p2,ier2,alpha,gama,eer2);
       [lambda3,Q3,ier3,state3,action3,eer3]=RLKMIC2(lambda3,L33-107,LSS3(i+1)-107,sz3,i+1,Q3,RA,state3,action3,action_f3,action_p3,ier3,alpha,gama,eer3);
       [lambda4,Q4,ier4,state4,action4,eer4]=RLKMIC2(lambda4,L44-107,LSS4(i+1)-107,sz4,i+1,Q4,RA,state4,action4,action_f4,action_p4,ier4,alpha,gama,eer4);
       [lambda5,Q5,ier5,state5,action5,eer5]=RLKMIC2(lambda5,L55-107,LSS5(i+1)-107,sz5,i+1,Q5,RA,state5,action5,action_f5,action_p5,ier5,alpha,gama,eer5);
       [lambda6,Q6,ier6,state6,action6,eer6]=RLKMIC2(lambda6,L66-107,LSS6(i+1)-107,sz6,i+1,Q6,RA,state6,action6,action_f6,action_p6,ier6,alpha,gama,eer6);
       [lambda7,Q7,ier7,state7,action7,eer7]=RLKMIC2(lambda7,L77-102,LSS7(i+1)-102,sz7,i+1,Q7,RA,state7,action7,action_f7,action_p7,ier7,alpha,gama,eer7);
       %}
      %{
       [lambda1,Q1,ier1,state1,action1,eer1]=RLKMIC2(lambda1,L11-107,LSS1(i+1)-107,sz1,i+1,Q1,RA,state1,action1,1,1,ier1,alpha,gama,eer1);
       [lambda2,Q2,ier2,state2,action2,eer2]=RLKMIC2(lambda2,L22-107,LSS2(i+1)-107,sz2,i+1,Q2,RA,state2,action2,1,1,ier2,alpha,gama,eer2);
       [lambda3,Q3,ier3,state3,action3,eer3]=RLKMIC2(lambda3,L33-107,LSS3(i+1)-107,sz3,i+1,Q3,RA,state3,action3,1,1,ier3,alpha,gama,eer3);
       [lambda4,Q4,ier4,state4,action4,eer4]=RLKMIC2(lambda4,L44-107,LSS4(i+1)-107,sz4,i+1,Q4,RA,state4,action4,1,1,ier4,alpha,gama,eer4);
       [lambda5,Q5,ier5,state5,action5,eer5]=RLKMIC2(lambda5,L55-107,LSS5(i+1)-107,sz5,i+1,Q5,RA,state5,action5,1,1,ier5,alpha,gama,eer5);
       [lambda6,Q6,ier6,state6,action6,eer6]=RLKMIC2(lambda6,L66-107,LSS6(i+1)-107,sz6,i+1,Q6,RA,state6,action6,1,1,ier6,alpha,gama,eer6);
       [lambda7,Q7,ier7,state7,action7,eer7]=RLKMIC2(lambda7,L77-102,LSS7(i+1)-102,sz7,i+1,Q7,RA,state7,action7,1,1,ier7,alpha,gama,eer7);
       %}
       %{
       [lambda1,Q1,ier1,state1,action1]=RLKMIC1(lambda1,L11-107,LS1(i+1)-107,i+1,Q1,RA,state1,action1,ier1,alpha,gama);
       [lambda2,Q2,ier2,state2,action2]=RLKMIC1(lambda2,L22-107,LS2(i+1)-107,i+1,Q2,RA,state2,action2,ier2,alpha,gama);
       [lambda3,Q3,ier3,state3,action3]=RLKMIC1(lambda3,L33-107,LS3(i+1)-107,i+1,Q3,RA,state3,action3,ier3,alpha,gama);
       [lambda4,Q4,ier4,state4,action4]=RLKMIC1(lambda4,L44-107,LS4(i+1)-107,i+1,Q4,RA,state4,action4,ier4,alpha,gama);
       [lambda5,Q5,ier5,state5,action5]=RLKMIC1(lambda5,L55-107,LS5(i+1)-107,i+1,Q5,RA,state5,action5,ier5,alpha,gama);
       [lambda6,Q6,ier6,state6,action6]=RLKMIC1(lambda6,L66-107,LS6(i+1)-107,i+1,Q6,RA,state6,action6,ier6,alpha,gama);
       [lambda7,Q7,ier7,state7,action7]=RLKMIC1(lambda7,L77-102,LS7(i+1)-102,i+1,Q7,RA,state7,action7,ier7,alpha,gama);
       %}
        NO1=[NO1,norm(Q1(:,:,action_p1))];NO2=[NO2,norm(Q2(:,:,action_p2))];NO3=[NO3,norm(Q3(:,:,action_p3))];NO4=[NO4,norm(Q4(:,:,action_p4))];NO5=[NO5,norm(Q5(:,:,action_p5))];NO6=[NO6,norm(Q6(:,:,action_p6))];NO7=[NO7,norm(Q7(:,:,action_p7))];
        LAM1=[LAM1,lambda1];LAM2=[LAM2,lambda2];LAM3=[LAM3,lambda3];LAM4=[LAM4,lambda4];LAM5=[LAM5,lambda5];LAM6=[LAM6,lambda6];LAM7=[LAM7,lambda7];
        set(handles.text19, 'string',lambda1);
        pause(0.35);
        %u_12=u_11;u_11=lambda1+Pre1(i+1);u_22=u_21;u_21=lambda2+Pre2(i+1);u_32=u_31;u_31=lambda3+Pre3(i+1);u_42=u_41;u_41=lambda4+Pre4(i+1);u_52=u_51;u_51=lambda5+Pre5(i+1);u_62=u_61;u_61=lambda6+Pre6(i+1);u_72=u_71;u_71=lambda7+Pre7(i+1);
        %y_12=y_11;y_11=LS1(i+1)-107;y_22=y_21;y_21=LS2(i+1)-107;y_32=y_31;y_31=LS3(i+1)-107;y_42=y_41;y_41=LS4(i+1)-107;y_52=y_51;y_51=LS5(i+1)-107;y_62=y_61;y_61=LS6(i+1)-107;y_72=y_71;y_71=LS7(i+1)-102;

        %set(handles.text19, 'string',DouCaX1);
        %pause(0.02);
       %}
       tt=tt+0.01;
   end
   toc
   %pressure=zeros(1,10);
   %pressure=uint16(pressure*2);
   %PressureSend(pressure);
   %PressureSend(int2str(25), 7);
   pause(1);
   %{
   %LS1=medfilt1(LS1,50); LS2=medfilt1(LS2,50); LS3=medfilt1(LS3,50); LS4=medfilt1(LS4,50); LS5=medfilt1(LS5,50); LS6=medfilt1(LS6,50); LS7=medfilt1(LS7,50); xini0=medfilt1(xini0,50); yini0=medfilt1(yini0,50);
   rd1=(LS1+LS2+LS3)*r./(2*(LS1.^2+LS2.^2+LS3.^2-LS1.*LS2-LS2.*LS3-LS1.*LS3).^(1/2));
   fi1=(2*(LS1.^2+LS2.^2+LS3.^2-LS1.*LS2-LS2.*LS3-LS1.*LS3).^(1/2))/(3*r);
   theta1=atan((sqrt(3)*(LS3-LS2)./(LS2+LS3-2*LS1)));
   rd2=(LS4+LS5+LS6)*r./(2*(LS4.^2+LS5.^2+LS6.^2-LS4.*LS5-LS5.*LS6-LS4.*LS6).^(1/2));
   fi2=(2*(LS4.^2+LS5.^2+LS6.^2-LS4.*LS5-LS5.*LS6-LS4.*LS6).^(1/2))/(3*r);
   theta2=atan((sqrt(3)*(LS6-LS5))./(LS5+LS6-2*LS4));
   for i=1:ts/h_s
       den1=(LS2(i)+LS3(i)-2*LS1(i));
       num1=(sqrt(3)*(LS3(i)-LS2(i)));
       den2=(LS5(i)+LS6(i)-2*LS4(i));
       num2=(sqrt(3)*(LS6(i)-LS5(i)));
       if den1>=0&&num1>=0
          theta1(i)=theta1(i);
       end
       if den1<0&&num1>0
          theta1(i)=pi+theta1(i);
       end
       if den1<0&&num1<=0
          theta1(i)=pi+theta1(i);
       end
       if den1>=0&&num1<0
          theta1(i)=2*pi+theta1(i);
       end
       if den1>=0
         theta2(i)=theta1(i)+pi;
       end 
       if den1<0
         theta2(i)=theta1(i)+pi;
       end 
        T1=[cos(theta1(i)) -sin(theta1(i)) 0 0; sin(theta1(i)) cos(theta1(i)) 0 0; 0 0 1 0;0 0 0 1]*[cos(fi1(i)/2) 0 sin(fi1(i)/2) 0; 0 1 0 0; -sin(fi1(i)/2) 0 cos(fi1(i)/2) 0; 0 0 0 1]*[1 0 0 0; 0 1 0 0; 0 0 1 2*rd1(i)*sin(fi1(i)/2);0 0 0 1]*[cos(fi1(i)/2) 0 sin(fi1(i)/2) 0; 0 1 0 0; -sin(fi1(i)/2) 0 cos(fi1(i)/2) 0; 0 0 0 1]*[cos(-theta1(i)) -sin(-theta1(i)) 0 0; sin(-theta1(i)) cos(-theta1(i)) 0 0; 0 0 1 0;0 0 0 1];
        T2=[cos(theta2(i)) -sin(theta2(i)) 0 0; sin(theta2(i)) cos(theta2(i)) 0 0; 0 0 1 0;0 0 0 1]*[cos(fi2(i)/2) 0 sin(fi2(i)/2) 0; 0 1 0 0; -sin(fi2(i)/2) 0 cos(fi2(i)/2) 0; 0 0 0 1]*[1 0 0 0; 0 1 0 0; 0 0 1 2*rd2(i)*sin(fi2(i)/2);0 0 0 1]*[cos(fi2(i)/2) 0 sin(fi2(i)/2) 0; 0 1 0 0; -sin(fi2(i)/2) 0 cos(fi2(i)/2) 0; 0 0 0 1]*[cos(-theta2(i)) -sin(-theta2(i)) 0 0; sin(-theta2(i)) cos(-theta2(i)) 0 0; 0 0 1 0;0 0 0 1];
        PO=T1*T2*[0 0 LS7(i) 1]';
        xx1=[xx1,PO(1)];yy1=[yy1,PO(2)];zz1=[zz1,PO(3)];  
        i1=i/100;
        if i>10&&i1==fix(i1)
           tt=0:0.01:fi1(i);
           ZL1=rd1(i)*sin(tt);
           XL1=rd1(i)*cos(theta1(i))*(1-cos(tt));
           YL1=rd1(i)*sin(theta1(i))*(1-cos(tt));
           XL20=XL1(end);YL20=YL1(end);ZL20=ZL1(end);
           %plot3(XL1,YL1,-ZL1,'b')
           %hold on
           for tt=0:0.01:fi1(i)
               ZL2=rd1(i)*sin(tt);
               XL2=rd1(i)*cos(theta2(i))*(1-cos(tt));
               YL2=rd1(i)*sin(theta2(i))*(1-cos(tt));
               T1=[cos(theta1(i)) -sin(theta1(i)) 0 0; sin(theta1(i)) cos(theta1(i)) 0 0; 0 0 1 0;0 0 0 1]*[cos(fi1(i)/2) 0 sin(fi1(i)/2) 0; 0 1 0 0; -sin(fi1(i)/2) 0 cos(fi1(i)/2) 0; 0 0 0 1]*[1 0 0 0; 0 1 0 0; 0 0 1 2*rd1(i)*sin(fi1(i)/2);0 0 0 1]*[cos(fi1(i)/2) 0 sin(fi1(i)/2) 0; 0 1 0 0; -sin(fi1(i)/2) 0 cos(fi1(i)/2) 0; 0 0 0 1]*[cos(-theta1(i)) -sin(-theta1(i)) 0 0; sin(-theta1(i)) cos(-theta1(i)) 0 0; 0 0 1 0;0 0 0 1];
               T2=[cos(theta2(i)) -sin(theta2(i)) 0 0; sin(theta2(i)) cos(theta2(i)) 0 0; 0 0 1 0;0 0 0 1]*[cos(fi2(i)/2) 0 sin(fi2(i)/2) 0; 0 1 0 0; -sin(fi2(i)/2) 0 cos(fi2(i)/2) 0; 0 0 0 1]*[1 0 0 0; 0 1 0 0; 0 0 1 2*rd2(i)*sin(fi2(i)/2);0 0 0 1]*[cos(fi2(i)/2) 0 sin(fi2(i)/2) 0; 0 1 0 0; -sin(fi2(i)/2) 0 cos(fi2(i)/2) 0; 0 0 0 1]*[cos(-theta2(i)) -sin(-theta2(i)) 0 0; sin(-theta2(i)) cos(-theta2(i)) 0 0; 0 0 1 0;0 0 0 1];
               PO=T1*[XL2 YL2 ZL2 1]';
               ZL2=PO(3);
               XL2=PO(1);
               YL2=PO(2);
               XL20=[XL20,XL2];
               YL20=[YL20,YL2];
               ZL20=[ZL20,ZL2];
           end
            %plot(L1)
           % plot3(XL20,YL20,-ZL20,'r')
           % hold on
            PO=T1*T2*[0 0 abs(ZZ(i))-z(i) 1]';
            XL3=[XL2,XL2];YL3=[YL2,YL2];ZL3=[ZL2,PO(3)];
           % plot3(XL3,YL3,-ZL3,'b')
            hold on
           % scatter3(XL2,YL2,-PO(3),'o','b');
        end
   end
   %X_P=x_pr*ones(1,ts/h_s);
   %Y_P=y_pr*ones(1,ts/h_s);
   %Z_P=-z_pr*ones(1,ts/h_s);
   %}
   plot(Y_P,'r')
   hold on
   plot(yini0,'b')
   assignin('base','a',yini0); 
   assignin('base','b',Y_P); 
   assignin('base','c',xini0); 
   assignin('base','d',X_P); 
   assignin('base','e',zini0); 
   assignin('base','f',Z_P); 
   assignin('base','Q1',Q1);assignin('base','Q2',Q2);assignin('base','Q3',Q3);assignin('base','Q4',Q4);
   assignin('base','Q5',Q5);assignin('base','Q6',Q6);assignin('base','Q7',Q7);
   assignin('base','LAM1',LAM1);assignin('base','LAM2',LAM2);assignin('base','LAM3',LAM3);assignin('base','LAM4',LAM4);
   assignin('base','LAM5',LAM5);assignin('base','LAM6',LAM6);assignin('base','LAM7',LAM7);
   assignin('base','NO1',NO1);assignin('base','NO2',NO2);assignin('base','NO3',NO3);assignin('base','NO4',NO4);
   assignin('base','NO5',NO5);assignin('base','NO6',NO6);assignin('base','NO7',NO7);
   assignin('base','LS1',LS1);assignin('base','LS2',LS2);assignin('base','LS3',LS3);assignin('base','LS4',LS4);
   assignin('base','LS5',LS5);assignin('base','LS6',LS6);assignin('base','LS7',LS7);
   assignin('base','ZZ',ZZ);assignin('base','z',z);
   fwrite(dotcp,closedo7(1,:),'char');
   pause(0.02);
   fwrite(dotcp,closedo6(1,:),'char');
   pause(0.2);
   %
   %pause(1);
   %
   pressure= pressure11;
   %
   pressure11(10)=uint16(40*2);
   PressureSend(pressure11);
   pause(1)
   pressure11(10)=uint16(60*2);
   PressureSend(pressure11);
   pause(1)
   pressure(7:9)=zeros(1,3);
   pressure11(7:9)=uint16(pressure(7:9)*2);
   PressureSend(pressure11);
   pause(1)
   fwrite(dotcp,opendo7(1,:),'char');
   pause(0.2);
   fwrite(dotcp,opendo5(1,:),'char');
   pause(5);
   pressure(1)=0;pressure(4)=100;
   pressure(2)=120;pressure(5)=0;
   pressure(3)=120;pressure(6)=0;
   pressure11(1:6)=uint16(pressure(1:6)*2);
   PressureSend(pressure11);
   pause(5)
   fwrite(dotcp,closedo7(1,:),'char');
   pause(0.5);
   fwrite(dotcp,closedo5(1,:),'char');
   pause(0.5);
   pressure(7)=40;
   pressure11(7)=uint16(pressure(7)*2);
   PressureSend(pressure11);
   pause(4);
   
   %pressure(7)=uint16(0);
   %PressureSend(pressure);
   %pause(2)
   %}
   %
   pressure11(10)=uint16(0);
   PressureSend(pressure11);
   pause(0.5)
   fwrite(dotcp,opendo7(1,:),'char');
   pause(0.2);
   fwrite(dotcp,opendo6(1,:),'char');
   pause(3)
   %}
   fwrite(dotcp,closedo7(1,:),'char');
   pause(0.5);
   fwrite(dotcp,closedo6(1,:),'char')
   pause(0.5)
   pressure(1)=0;pressure(4)=90;
   pressure(2)=120;pressure(5)=0;
   pressure(3)=100;pressure(6)=0;
   pressure(7)=0;
   pressure11(1:7)=uint16(pressure(1:7)*2);
   PressureSend(pressure11);
   pause(1)
   %pressure(10)=uint16(0);
   %PressureSend(pressure);
   
   %} 
   %PressureSend(int2str(20), 7);
   %pause(0.02)
%   PressureSend(int2str(10), 8);
  % pause(1)
  % PressureSend(int2str(0), 7);
  % pause(0.02)
   %fwrite(dotcp,closedo7(1,:),'char');
   %pause(0.02);
   %fwrite(dotcp,closedo6(1,:),'char');
   %SendBytes( '`' );
   fclose(datcp);  
   delete(datcp);    
   fclose(dotcp);  
   delete(dotcp);
   set(handles.pushbutton_ON, 'string', 'ON');
   pause(0.2)
   fclose(a);
   set(handles.pushbutton_CON, 'string', 'ON');
   
   %scatter3(x_pr,y_pr,-z_pr,'o','r')
   %plot3(xx1(1000:end),yy1(1000:end),-zz1(1000:end),'r')
   

% --- Executes during object creation, after setting all properties.
function pushbutton_start_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called




% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over pushbutton_start.
function pushbutton_start_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


%视觉数据读取
function EveBytesAvailableFcn( handles )  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global recBuff;
global DouCaY 
global DouCaX;
global DouCaZ;
global d;
global sCOM1;
global x_pr;
global y_pr;
global z_pr z;
global ts;
global h_s;
global dmodel;%预测模型
global L1 L2 L3 L4 L5 L6 L7;%手臂各腔期望长度
global Pre1 Pre2 Pre3 Pre4 Pre5 Pre6 Pre7;
global X_P Y_P Z_P;
global opendo7;global opendo6;
global closedo7;global closedo6;
global dotcp;
fwrite(dotcp,closedo7(1,:),'char');
pause(0.5);
fwrite(dotcp,closedo6(1,:),'char')
pause(0.5);
% global handleSoftArmCtrl;
%recBuff = fread(sCOM, 7, 'uint8');
for jj=0:0.1:20
       recBuff = fscanf(sCOM1);
       if jj==1
          fwrite(dotcp,opendo7(1,:),'char');
          pause(0.3);
          fwrite(dotcp,opendo6(1,:),'char');
          pause(0.3);
          pressure1=zeros(1,10);
          pressure11(1:10)=uint16(pressure1);
          PressureSend(pressure11);
          pause(0.2);
       end
end
if recBuff(1)=='#'
   ms=regexp(recBuff,'\-?\d*\.?\d*','match');
   DouCaY=0.7*str2double(ms{2});
   DouCaX=2*str2double(ms{1});
   %DouCaZ=str2double(ms{3});%+326;
else
   DouCaX='error';
   DouCaY='error';
   %DouCaZ='error';
end
set(handles.text_DouCaX, 'string', DouCaX);
set(handles.text_DouCaY, 'string', DouCaY);
set(handles.text_DouCaZ, 'string', 400);
set(handles.X, 'string', DouCaX);
set(handles.Y, 'string', DouCaY);
set(handles.Z, 'string', 400);
set(handles.h,'string',0.01);
set(handles.T,'string',0.30);

cla(handles.axes1,'reset');
h_s=str2num(get(handles.h,'string'));%将输入的步长由字符型转为实数
x_pr=str2num(get(handles.X,'string'));%将输入的x方向的位置坐标由字符型转为实数
y_pr=str2num(get(handles.Y,'string'));%将输入的y方向的位置坐标由字符型转为实数
z_pr=str2num(get(handles.Z,'string'));%将输入的z方向的位置坐标由字符型转为实数
ts=str2num(get(handles.T,'string'));%设定手臂运动时间
%手臂运动学求取各腔长度
X_P=x_pr*ones(1,ts/h_s);
Y_P=y_pr*ones(1,ts/h_s);
Z_P=-z_pr*ones(1,ts/h_s);
for j=1:4
    z=z_pr*2/3*ones(1,ts/h_s);X=x_pr/2*ones(1,ts/h_s);Y=y_pr/2*ones(1,ts/h_s);Z=z/2;r=24;ZZ=-z_pr*ones(1,ts/h_s);
    theta1=atan(Y./X);
    for i=1:length(z)
       if X(i)>=0&&Y(i)>=0
          theta1(i)=theta1(i);
       end
       if X(i)<0&&Y(i)>0
          theta1(i)=pi+theta1(i);
       end
       if X(i)<0&&Y(i)<=0
          theta1(i)=pi+theta1(i);
       end
       if X(i)>=0&&Y(i)<0
          theta1(i)=2*pi+theta1(i);
       end
    end
    %向心角
    fi1=pi-2.*asin(Z./(X.^2+Y.^2+Z.^2).^(1/2)); 
    %曲率半径
    r1=((X.^2+Y.^2+Z.^2)./(2*(1-cos(fi1)))).^(1/2);
    %第一段充气腔长度变化
    l11=fi1.*(r1-r.*cos(theta1));
    l12=fi1.*(r1-r.*cos(2*pi/3-theta1));
    l13=fi1.*(r1-r.*cos(4*pi/3-theta1)); 
    %针对第二段手臂弯曲关节求取向心角、偏转角及曲率半径
    %偏转角
    theta2=theta1+pi;
    %向心角
    fi2=fi1;
    %曲率半径
    r2=r1;
    %第二段充气腔长度变化
    l21=abs(fi2).*(r2-r.*cos(theta2));
    l22=abs(fi2).*(r2-r.*cos(2*pi/3-theta2));
    l23=abs(fi2).*(r2-r.*cos(4*pi/3-theta2));
    le=abs(ZZ)-z;
    for i=1:length(z)
       while l11(i)>200||l12(i)>200||l13(i)>200||l21(i)>200||l22(i)>200||l23(i)>200||le(i)>170
             z(i)=z(i)+10;
             if le(i)<80
                 break
             end
             [l111 l122 l133 l211 l222 l233 theta11 fi11 r11 theta21 fi21 r21]=inversekinematics(X_P(i),Y_P(i),z(i),r);
             l11(i)=l111;l12(i)=l122;l13(i)=l133;l21(i)=l211;l22(i)=l222;l23(i)=l233;le(i)=abs(ZZ(i))-z(i);
             Z(i)=z(i)/2;
        end
    end
    if j==1
       L10=l11-107;L20=l12-107;L30=l13-107;L40=l21-107;L50=l22-107;L60=l23-107;L70=le-102;
    end
    if j==2
       L11=l11-107;L21=l12-107;L31=l13-107;L41=l21-107;L51=l22-107;L61=l23-107;L71=le-102;
    end
    if j==3
       L12=l11-107;L22=l12-107;L32=l13-107;L42=l21-107;L52=l22-107;L62=l23-107;L72=le-102;
    end
    if j==4
       L13=l11-107;L23=l12-107;L33=l13-107;L43=l21-107;L53=l22-107;L63=l23-107;L73=le-102;
    end 
end
L1=L11;L2=L21;L3=L31;L4=L41;L5=L51;L6=L61;L7=L71;  
Pre1=kringpredict(dmodel,L10,L11,L12,L13);
Pre2=kringpredict(dmodel,L20,L21,L22,L23);
Pre3=kringpredict(dmodel,L30,L31,L32,L33);
Pre4=kringpredict(dmodel,L40,L41,L42,L43);
Pre5=kringpredict(dmodel,L50,L51,L52,L53);
Pre6=kringpredict(dmodel,L60,L61,L62,L63);
Pre7=kringpredict(dmodel,L70,L71,L72,L73);
axes(handles.axes1)
plot3(X_P,Y_P,Z_P,'o')

%
%pause(0.02)
%pause(0.2)
%{
if length(recBuff) ~= 7
    return;
end

i = 1;
while i < 8
    if recBuff(i) == 191
        break;
    end
    i = i + 1;
end
recBuff = circshift( recBuff, -(i-1) );

if recBuff(1) ~= 191
    return;
end

DouCaX = 0; DouCaY = 0; DouCaZ = 0;
OnHandX = 0; OnHandY = 0;

% code e.g. BF 41 50 0F 18 39 80 
if (recBuff(1) == 191) && (recBuff(7) == 128)                           % start 0x1011 1111, end 0x1000 0000
    if isequal( bitget(recBuff(4), [8 7 6 5]), [0 0 0 0])               % X
        DouCaX = bitand(7, recBuff(4));
        if bitget(recBuff(4), 4) == 1
            DouCaX = -DouCaX;
        end
        set(handles.text_DouCaX, 'string', DouCaX);
    else
        DouCaX = 'error';
        set(handles.text_DouCaX, 'string', DouCaX);
    end
    
    if isequal( bitget(recBuff(5), [8 7 6 5]), [0 0 0 1])               % Y
        DouCaY = bitand(7, recBuff(5));
        if bitget(recBuff(5), 4) == 1
            DouCaY = -DouCaY;
        end
        set(handles.text_DouCaY, 'string', DouCaY);
    else
        DouCaY = 'error';
        set(handles.text_DouCaY, 'string', DouCaY);
    end
    
    if isequal( bitget(recBuff(6), [8 7 6 5]), [0 0 1 1])               % Z
        DouCaZ = bitand(7, recBuff(6));
        if bitget(recBuff(6), 4) == 1
            DouCaZ = -DouCaZ;
        end
        set(handles.text_DouCaZ, 'string', DouCaZ);
    else
        DouCaZ = 'error';
        set(handles.text_DouCaZ, 'string', DouCaZ);
    end
    
    if isequal( bitget(recBuff(2), [8 7 6 5]), [0 1 0 0])               % X
        OnHandX = bitand(7, recBuff(2));
        if bitget(recBuff(2), 4) == 1
            OnHandX = -OnHandX;
        end
        set(handles.text_OnHandX, 'string', OnHandX);
    elseif isequal( bitget(recBuff(2), [8 7 6 5]), [1 1 1 1])
        OnHandX = 'Null';
        set(handles.text_OnHandX, 'string', OnHandX);
    else
        OnHandX = 'error';
        set(handles.text_OnHandX, 'string', OnHandX);
    end
    
    if isequal( bitget(recBuff(3), [8 7 6 5]), [0 1 0 1])               % Y
        OnHandY = bitand(7, recBuff(3));
        if bitget(recBuff(3), 4) == 1
            OnHandY = -OnHandY;
        end
        set(handles.text_OnHandY, 'string', OnHandY);
    elseif isequal( bitget(recBuff(3), [8 7 6 5]), [1 1 1 1])
        OnHandY = 'Null';
        set(handles.text_OnHandY, 'string', OnHandY);
    else
        OnHandY = 'error';
        set(handles.text_OnHandY, 'string', OnHandY);
    end
end
%}
% process redundance data
[VisionX, VisionY, VisionZ] = getCoordinate( DouCaX, DouCaY, DouCaZ);  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% transfer to soft arm

if   strcmp( get(handles.pushbutton_VisionAble, 'string'), 'Disable')
% if strcmp( get(handles.pushbutton_VisionAble, 'string'), 'Disable')
        
    VisionX = VisionX * 5;
    VisionY = VisionY * 5;
    VisionZ = VisionZ * 5;
    set(handles.X,'string',num2str(VisionX));
    set(handles.Y,'string',num2str(VisionY));
    set(handles.Z,'string',num2str(VisionZ));
    
end
fclose(sCOM1);
% if strcmp( get(handleSoftArmCtrl.pushbutton_VisionAble, 'string'), 'Disable')
%     set(handleSoftArmCtrl.text_VisionX, 'string', x);
%     set(handleSoftArmCtrl.text_VisionY, 'string', y);
%     set(handleSoftArmCtrl.text_VisionZ, 'string', z);
%     feval(@(hObject,eventdata)SerialCOM('pushbutton_ExecuteVision_Callback',0,0,handleSoftArmCtrl));
% end
function [x, y, z] = getCoordinate( DouCaX, DouCaY, DouCaZ)  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
rangeFar = [1 0];
rangeNear = [0 1];
if ~strcmp(DouCaX, 'error') && ~strcmp(DouCaY, 'error') && ~strcmp(DouCaZ, 'error') && ...
        ~strcmp(OnHandX, 'error') && ~strcmp(OnHandY, 'error')
    if ~strcmp(OnHandX, 'Null') && ~strcmp(OnHandY, 'Null')
        cameraSelect = rangeNear;
    else
        cameraSelect = rangeFar;
        OnHandX = 0;
        OnHandY = 0;
    end  
%}
    x = DouCaX; 
    y = DouCaY;
    z = DouCaZ;


% --- Executes on button press in pushbutton_VisionAble.
function pushbutton_VisionAble_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_VisionAble (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if strcmp( get(handles.pushbutton_VisionAble, 'string'), 'Able')
        set(handles.pushbutton_VisionAble, 'String', 'Disable');
    elseif strcmp( get(handles.pushbutton_VisionAble, 'string'), 'Disable')
        set(handles.pushbutton_VisionAble, 'String', 'Able');    
 end


% --- Executes during object creation, after setting all properties.
function pushbutton_st_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton_st (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function text_DouCaX_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text_DouCaX (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on selection change in popupmenu_CCOM.
function popupmenu_CCOM_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu_CCOM (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu_CCOM contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu_CCOM
global CCOM1;  
 
val = get(hObject,'value');  
switch val  
    case 1  
        CCOM1='COM1';  
%         fprintf('COM=1\n');  
    case 2  
        CCOM1='COM2';  
    case 3  
        CCOM1='COM3';  
    case 4  
        CCOM1='COM4';  
    case 5  
        CCOM1='COM5';  
    case 6  
       CCOM1='COM6';  
    case 7  
        CCOM1='COM7';  
    case 8  
        CCOM1='COM8';  
    case 9  
        CCOM1='COM9';  
    case 10  
        CCOM1='COM10';  
    case 11  
        CCOM1='COM11';  
    case 12  
        CCOM1='COM12';  
end 


% --- Executes during object creation, after setting all properties.
function popupmenu_CCOM_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu_CCOM (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu_CBoundrate.
function popupmenu_CBoundrate_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu_CBoundrate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu_CBoundrate contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu_CBoundrate
global rate1;  
val = get(hObject,'value');  
switch val  
    case 1  
        rate1 = 9600;  
    case 2  
        rate1 =38400;  
end 

% --- Executes during object creation, after setting all properties.
function popupmenu_CBoundrate_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu_CBoundrate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_CON.
function pushbutton_CON_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_CON (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA
clc  
instrreset  
global sCOM1;
global CCOM1;
global rate1;  
global dotcp;
global datcp;
global closedo7;
global closedo6;
% global out;  
% out=1;  
sCOM1 = serial(CCOM1);

if strcmp( get(handles.pushbutton_CON, 'string'), 'ON')    
      
    set(sCOM1,'BaudRate',rate1);  %%% Baud rate
    set(sCOM1,'DataBits',8);     %%% DataBits 
    set(sCOM1,'StopBits',1);     %%% StopBits
    set(sCOM1,'InputBufferSize',102400);%%%  
    daip_address = get(handles.edit_daip,'string');
    daip_port = int16(str2num(get(handles.edit_daport,'string')));
    datcp = tcpip(daip_address, daip_port);
    fopen(datcp);
    doip_address = get(handles.edit_doip,'string');
    doip_port = int16(str2num(get(handles.edit_doport,'string')));
    dotcp = tcpip(doip_address, doip_port);
    fopen(dotcp);
    set(handles.pushbutton_ON, 'string', 'OFF');
    pause(0.1);
    pressure11=zeros(1,10);
    pressure(1)=0;pressure(4)=90;
    pressure(2)=100;pressure(5)=0;
    pressure(3)=100;pressure(6)=0;
    pressure11(1:6)=uint16(pressure(1:6)*2);
    PressureSend(pressure11);
    pause(2)
    % ????????  
    sCOM1.BytesAvailableFcnMode='terminator';  
    sCOM1.BytesAvailableFcnCount=10;                          %%% ???????10?????????
   % sCOM.BytesAvailableFcn={@EveBytesAvailableFcn,handles};  %%% ???????
    fopen(sCOM1);                %%% open serial port
%     global count;  
%     count=1;  
    fprintf('Serial port opened successfully\n');
    EveBytesAvailableFcn( handles )
    set(handles.pushbutton_CON, 'string', 'OFF');
elseif strcmp( get(handles.pushbutton_CON, 'string'), 'OFF')
     fwrite(dotcp,closedo7(1,:),'char');
     pause(0.3);
     fwrite(dotcp,closedo6(1,:),'char');
     pause(0.3);
     pressure1=zeros(1,10);
     pressure11(1:10)=uint16(pressure1);
     PressureSend(pressure11);
     pause(1);
     fclose(datcp);  
     delete(datcp);    
     fclose(dotcp);  
     delete(dotcp);
     set(handles.pushbutton_ON, 'string', 'ON');
     fclose(sCOM1);  
     delete(sCOM1);    
     fprintf('Serial port closed successfully\n');
     set(handles.pushbutton_CON, 'string', 'ON');
end

% --- Executes during object creation, after setting all properties.
function pushbutton_CON_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton_CON (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



% --- Executes on button press in pushbutton_foldseg1.
function pushbutton_foldseg1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_foldseg1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton_foldseg2.
function pushbutton_foldseg2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_foldseg2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton_foldseg3.
function pushbutton_foldseg3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_foldseg3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_ON.
function pushbutton_ON_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_ON (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global dotcp;
global datcp;

if strcmp( get(handles.pushbutton_ON, 'string'), 'ON')    

    daip_address = get(handles.edit_daip,'string');
    daip_port = int16(str2num(get(handles.edit_daport,'string')));
    datcp = tcpip(daip_address, daip_port);
    fopen(datcp);
    doip_address = get(handles.edit_doip,'string');
    doip_port = int16(str2num(get(handles.edit_doport,'string')));
    dotcp = tcpip(doip_address, doip_port);
    fopen(dotcp);
    set(handles.pushbutton_ON, 'string', 'OFF');

elseif strcmp( get(handles.pushbutton_ON, 'string'), 'OFF')
    fclose(datcp);  
    delete(datcp);    
    fclose(dotcp);  
    delete(dotcp);
    set(handles.pushbutton_ON, 'string', 'ON');
end

% --- Executes during object creation, after setting all properties.
function pushbutton_ON_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton_ON (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



function edit_daip_Callback(hObject, eventdata, handles)
% hObject    handle to edit_daip (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_daip as text
%        str2double(get(hObject,'String')) returns contents of edit_daip as a double


% --- Executes during object creation, after setting all properties.
function edit_daip_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_daip (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_doip_Callback(hObject, eventdata, handles)
% hObject    handle to edit_doip (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_doip as text
%        str2double(get(hObject,'String')) returns contents of edit_doip as a double


% --- Executes during object creation, after setting all properties.
function edit_doip_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_doip (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_daport_Callback(hObject, eventdata, handles)
% hObject    handle to edit_daport (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_daport as text
%        str2double(get(hObject,'String')) returns contents of edit_daport as a double


% --- Executes during object creation, after setting all properties.
function edit_daport_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_daport (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_doport_Callback(hObject, eventdata, handles)
% hObject    handle to edit_doport (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_doport as text
%        str2double(get(hObject,'String')) returns contents of edit_doport as a double


% --- Executes during object creation, after setting all properties.
function edit_doport_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_doport (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton14.
function pushbutton14_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton15.
function pushbutton15_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit12_Callback(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit12 as text
%        str2double(get(hObject,'String')) returns contents of edit12 as a double


% --- Executes during object creation, after setting all properties.
function edit12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton16.
function pushbutton16_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton17.
function pushbutton17_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
function PressureSend( pressure )  
% 'pressure' in uint16
global command;
global datcp;
global command_addr;
global command_registername;
global command_registeraddrH;
global command_registeraddrL;
global command_OutputNumH;
global command_OutputNumL;
global command_bytenum;
global command_DAoutput;
global command_crc16Hi;
global command_crc16Lo;
    %       pressure_hi = uint8(bitshift(pressure, -8));
    %       pressure_lo = uint8(pressure);
    command_DAoutput =  [ uint8(bitshift(pressure(1), -8)),uint8(bitand(pressure(1), 255)),...
                          uint8(bitshift(pressure(2), -8)),uint8(bitand(pressure(2), 255)),...
                          uint8(bitshift(pressure(3), -8)),uint8(bitand(pressure(3), 255)),...
                          uint8(bitshift(pressure(4), -8)),uint8(bitand(pressure(4), 255)),...
                          uint8(bitshift(pressure(5), -8)),uint8(bitand(pressure(5), 255)),...
                          uint8(bitshift(pressure(6), -8)),uint8(bitand(pressure(6), 255)),...
                          uint8(bitshift(pressure(7), -8)),uint8(bitand(pressure(7), 255)),...
                          uint8(bitshift(pressure(8), -8)),uint8(bitand(pressure(8), 255)),...
                          uint8(bitshift(pressure(9), -8)),uint8(bitand(pressure(9), 255)),...
                          uint8(bitshift(pressure(10), -8)),uint8(bitand(pressure(10), 255)) ] ;  

    command = [command_addr, command_registername, command_registeraddrH, ...
               command_registeraddrL, command_OutputNumH,command_OutputNumL,...
               command_bytenum,command_DAoutput(1,:)];
    calculateCRC(command);

    command = [command(1,:),command_crc16Hi,command_crc16Lo];
    fwrite(datcp,command(:,:),'char');

    function calculateCRC(command)
    global command_crc16Hi;
    global command_crc16Lo;
    global aucCRCHi;
    global aucCRCLo;
    uint16 iIndex;
    len = uint8( length(command));
    ucCRCHi = uint8(255); %0xFF
    ucCRCLo = uint8(255); %0xFF
    for i = 1 : len
        iIndex = bitxor( ucCRCLo,command(1,i) );
        ucCRCLo = bitxor( ucCRCHi,aucCRCHi(iIndex+1,1) );
        ucCRCHi = aucCRCLo(iIndex+1,1);
    end
    command_crc16Hi = ucCRCLo(:);
    command_crc16Lo = ucCRCHi(:);


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over pushbutton_ON.
function pushbutton_ON_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton_ON (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
