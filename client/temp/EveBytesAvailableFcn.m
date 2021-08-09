%视觉数据读取
function EveBytesAvailableFcn(handles) % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
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
    global dmodel; %预测模型
    global L1 L2 L3 L4 L5 L6 L7; %手臂各腔期望长度
    global Pre1 Pre2 Pre3 Pre4 Pre5 Pre6 Pre7;
    global X_P Y_P Z_P;
    global opendo7; global opendo6;
    global closedo7; global closedo6;
    global dotcp;
    fwrite(dotcp, closedo7(1, :), 'char');
    pause(0.5);
    fwrite(dotcp, closedo6(1, :), 'char')
    pause(0.5);
    % global handleSoftArmCtrl;
    %recBuff = fread(sCOM, 7, 'uint8');
    for jj = 0:0.1:20
        recBuff = fscanf(sCOM1);

        if jj == 1
            fwrite(dotcp, opendo7(1, :), 'char');
            pause(0.3);
            fwrite(dotcp, opendo6(1, :), 'char');
            pause(0.3);
            pressure1 = zeros(1, 10);
            pressure11(1:10) = uint16(pressure1);
            PressureSend(pressure11);
            pause(0.2);
        end

    end

    if recBuff(1) == '#'
        ms = regexp(recBuff, '\-?\d*\.?\d*', 'match');
        DouCaY = 0.7 * str2double(ms{2});
        DouCaX = 2 * str2double(ms{1});
        %DouCaZ=str2double(ms{3});%+326;
    else
        DouCaX = 'error';
        DouCaY = 'error';
        %DouCaZ='error';
    end

    set(handles.text_DouCaX, 'string', DouCaX);
    set(handles.text_DouCaY, 'string', DouCaY);
    set(handles.text_DouCaZ, 'string', 400);
    set(handles.X, 'string', DouCaX);
    set(handles.Y, 'string', DouCaY);
    set(handles.Z, 'string', 400);
    set(handles.h, 'string', 0.01);
    set(handles.T, 'string', 0.30);

    cla(handles.axes1, 'reset');
    h_s = str2num(get(handles.h, 'string')); %将输入的步长由字符型转为实数
    x_pr = str2num(get(handles.X, 'string')); %将输入的x方向的位置坐标由字符型转为实数
    y_pr = str2num(get(handles.Y, 'string')); %将输入的y方向的位置坐标由字符型转为实数
    z_pr = str2num(get(handles.Z, 'string')); %将输入的z方向的位置坐标由字符型转为实数
    ts = str2num(get(handles.T, 'string')); %设定手臂运动时间
    %手臂运动学求取各腔长度
    X_P = x_pr * ones(1, ts / h_s);
    Y_P = y_pr * ones(1, ts / h_s);
    Z_P = -z_pr * ones(1, ts / h_s);

    for j = 1:4
        z = z_pr * 2/3 * ones(1, ts / h_s); X = x_pr / 2 * ones(1, ts / h_s); Y = y_pr / 2 * ones(1, ts / h_s); Z = z / 2; r = 24; ZZ = -z_pr * ones(1, ts / h_s);
        theta1 = atan(Y ./ X);

        for i = 1:length(z)

            if X(i) >= 0 && Y(i) >= 0
                theta1(i) = theta1(i);
            end

            if X(i) < 0 && Y(i) > 0
                theta1(i) = pi + theta1(i);
            end

            if X(i) < 0 && Y(i) <= 0
                theta1(i) = pi + theta1(i);
            end

            if X(i) >= 0 && Y(i) < 0
                theta1(i) = 2 * pi + theta1(i);
            end

        end

        %向心角
        fi1 = pi - 2 .* asin(Z ./ (X.^2 + Y.^2 + Z.^2).^(1/2));
        %曲率半径
        r1 = ((X.^2 + Y.^2 + Z.^2) ./ (2 * (1 - cos(fi1)))).^(1/2);
        %第一段充气腔长度变化
        l11 = fi1 .* (r1 - r .* cos(theta1));
        l12 = fi1 .* (r1 - r .* cos(2 * pi / 3 - theta1));
        l13 = fi1 .* (r1 - r .* cos(4 * pi / 3 - theta1));
        %针对第二段手臂弯曲关节求取向心角、偏转角及曲率半径
        %偏转角
        theta2 = theta1 + pi;
        %向心角
        fi2 = fi1;
        %曲率半径
        r2 = r1;
        %第二段充气腔长度变化
        l21 = abs(fi2) .* (r2 - r .* cos(theta2));
        l22 = abs(fi2) .* (r2 - r .* cos(2 * pi / 3 - theta2));
        l23 = abs(fi2) .* (r2 - r .* cos(4 * pi / 3 - theta2));
        le = abs(ZZ) - z;

        for i = 1:length(z)

            while l11(i) > 200 || l12(i) > 200 || l13(i) > 200 || l21(i) > 200 || l22(i) > 200 || l23(i) > 200 || le(i) > 170
                z(i) = z(i) + 10;

                if le(i) < 80
                    break
                end

                [l111 l122 l133 l211 l222 l233 theta11 fi11 r11 theta21 fi21 r21] = inversekinematics(X_P(i), Y_P(i), z(i), r);
                l11(i) = l111; l12(i) = l122; l13(i) = l133; l21(i) = l211; l22(i) = l222; l23(i) = l233; le(i) = abs(ZZ(i)) - z(i);
                Z(i) = z(i) / 2;
            end

        end

        if j == 1
            L10 = l11 - 107; L20 = l12 - 107; L30 = l13 - 107; L40 = l21 - 107; L50 = l22 - 107; L60 = l23 - 107; L70 = le - 102;
        end

        if j == 2
            L11 = l11 - 107; L21 = l12 - 107; L31 = l13 - 107; L41 = l21 - 107; L51 = l22 - 107; L61 = l23 - 107; L71 = le - 102;
        end

        if j == 3
            L12 = l11 - 107; L22 = l12 - 107; L32 = l13 - 107; L42 = l21 - 107; L52 = l22 - 107; L62 = l23 - 107; L72 = le - 102;
        end

        if j == 4
            L13 = l11 - 107; L23 = l12 - 107; L33 = l13 - 107; L43 = l21 - 107; L53 = l22 - 107; L63 = l23 - 107; L73 = le - 102;
        end

    end

    L1 = L11; L2 = L21; L3 = L31; L4 = L41; L5 = L51; L6 = L61; L7 = L71;
    Pre1 = kringpredict(dmodel, L10, L11, L12, L13);
    Pre2 = kringpredict(dmodel, L20, L21, L22, L23);
    Pre3 = kringpredict(dmodel, L30, L31, L32, L33);
    Pre4 = kringpredict(dmodel, L40, L41, L42, L43);
    Pre5 = kringpredict(dmodel, L50, L51, L52, L53);
    Pre6 = kringpredict(dmodel, L60, L61, L62, L63);
    Pre7 = kringpredict(dmodel, L70, L71, L72, L73);
    axes(handles.axes1)
    plot3(X_P, Y_P, Z_P, 'o')

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

    recBuff = circshift(recBuff, -(i - 1));

    if recBuff(1) ~= 191
        return;
    end

    DouCaX = 0; DouCaY = 0; DouCaZ = 0;
    OnHandX = 0; OnHandY = 0;

    % code e.g. BF 41 50 0F 18 39 80
    if (recBuff(1) == 191) && (recBuff(7) == 128) % start 0x1011 1111, end 0x1000 0000

    if isequal(bitget(recBuff(4), [8 7 6 5]), [0 0 0 0]) % X
        DouCaX = bitand(7, recBuff(4));

        if bitget(recBuff(4), 4) == 1
            DouCaX = -DouCaX;
        end

        set(handles.text_DouCaX, 'string', DouCaX);
    else
        DouCaX = 'error';
        set(handles.text_DouCaX, 'string', DouCaX);
    end

    if isequal(bitget(recBuff(5), [8 7 6 5]), [0 0 0 1]) % Y
        DouCaY = bitand(7, recBuff(5));

        if bitget(recBuff(5), 4) == 1
            DouCaY = -DouCaY;
        end

        set(handles.text_DouCaY, 'string', DouCaY);
    else
        DouCaY = 'error';
        set(handles.text_DouCaY, 'string', DouCaY);
    end

    if isequal(bitget(recBuff(6), [8 7 6 5]), [0 0 1 1]) % Z
        DouCaZ = bitand(7, recBuff(6));

        if bitget(recBuff(6), 4) == 1
            DouCaZ = -DouCaZ;
        end

        set(handles.text_DouCaZ, 'string', DouCaZ);
    else
        DouCaZ = 'error';
        set(handles.text_DouCaZ, 'string', DouCaZ);
    end

    if isequal(bitget(recBuff(2), [8 7 6 5]), [0 1 0 0]) % X
        OnHandX = bitand(7, recBuff(2));

        if bitget(recBuff(2), 4) == 1
            OnHandX = -OnHandX;
        end

        set(handles.text_OnHandX, 'string', OnHandX);
    elseif isequal(bitget(recBuff(2), [8 7 6 5]), [1 1 1 1])
        OnHandX = 'Null';
        set(handles.text_OnHandX, 'string', OnHandX);
    else
        OnHandX = 'error';
        set(handles.text_OnHandX, 'string', OnHandX);
    end

    if isequal(bitget(recBuff(3), [8 7 6 5]), [0 1 0 1]) % Y
        OnHandY = bitand(7, recBuff(3));

        if bitget(recBuff(3), 4) == 1
            OnHandY = -OnHandY;
        end

        set(handles.text_OnHandY, 'string', OnHandY);
    elseif isequal(bitget(recBuff(3), [8 7 6 5]), [1 1 1 1])
        OnHandY = 'Null';
        set(handles.text_OnHandY, 'string', OnHandY);
    else
        OnHandY = 'error';
        set(handles.text_OnHandY, 'string', OnHandY);
    end

    %}
    % process redundance data
end

[VisionX, VisionY, VisionZ] = getCoordinate(DouCaX, DouCaY, DouCaZ); % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% transfer to soft arm

if strcmp(get(handles.pushbutton_VisionAble, 'string'), 'Disable')
    % if strcmp( get(handles.pushbutton_VisionAble, 'string'), 'Disable')

    VisionX = VisionX * 5;
    VisionY = VisionY * 5;
    VisionZ = VisionZ * 5;
    set(handles.X, 'string', num2str(VisionX));
    set(handles.Y, 'string', num2str(VisionY));
    set(handles.Z, 'string', num2str(VisionZ));

end

fclose(sCOM1);
% if strcmp( get(handleSoftArmCtrl.pushbutton_VisionAble, 'string'), 'Disable')
%     set(handleSoftArmCtrl.text_VisionX, 'string', x);
%     set(handleSoftArmCtrl.text_VisionY, 'string', y);
%     set(handleSoftArmCtrl.text_VisionZ, 'string', z);
%     feval(@(hObject,eventdata)SerialCOM('pushbutton_ExecuteVision_Callback',0,0,handleSoftArmCtrl));
% end
