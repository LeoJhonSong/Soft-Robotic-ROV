function pushbutton_start_Callback(hObject, eventdata, handles)
    cla(handles.axes1);
    axes(handles.axes1)
    a = serial('COM3');
    set(a, 'BaudRate', 9600, 'DataBits', 8, 'StopBits', 1, 'Parity', 'none', 'FlowControl', 'none');
    set(a, 'InputBufferSize', 1024);
    %a.BytesAvailableFcnMode='terminator';
    a.BytesAvailableFcnCount = 0.6;
    a.ReadAsyncMode = 'continuous';
    fopen(a);
    %}
    pressure = zeros(1, 10);
    %
    pressure = uint16(pressure);
    PressureSend(pressure);
    pause(1);
    %}
    global x_pr;
    global y_pr;
    global z_pr z;
    global ts;
    global h_s;
    global L1 L2 L3 L4 L5 L6 L7; %手臂各腔期望长度
    global Pre1 Pre2 Pre3 Pre4 Pre5 Pre6 Pre7;
    global X_P Y_P Z_P
    global recBuff
    global DouCaY DouCaX DouCaZ;
    global sCOM1
    global dotcp; global datcp;
    global opendo6;
    global opendo7;
    global opendo5;
    global closedo5;
    global closedo7;
    global closedo6;
    fwrite(dotcp, closedo7(1, :), 'char');
    pause(0.3);
    fwrite(dotcp, closedo6(1, :), 'char');
    pause(0.3);
    %
    fwrite(dotcp, opendo7(1, :), 'char');
    pause(0.3);
    fwrite(dotcp, opendo6(1, :), 'char');
    %}
    ier1 = 0; ier2 = 0; ier3 = 0; ier4 = 0; ier5 = 0; ier6 = 0; ier7 = 0; ier = 0;
    xx1 = 0; xx2 = 0; yy1 = 0; yy2 = 0; zz1 = 0; zz2 = 0; r = 24; ZZ = -z_pr * ones(1, ts / h_s);
    %# FIXME: 这好像用不上
    eer1 = 0; eer2 = 0; eer3 = 0; eer4 = 0; eer5 = 0; eer6 = 0; eer7 = 0;
    %%%%%%%%%强化学习卡尔曼%%%%%%%%%%%%%%%
    syms kp1 ki1 kd1 kp2 ki2 kd2 kp3 ki3 kd3 kp4 ki4 kd4 kp5 ki5 kd5 kp6 ki6 kd6 kp7 ki7 kd7;
    AAP1 = [-kp1 / kd1 -ki1 / kd1; 1 0];
    AAP2 = [-kp2 / kd2 -ki2 / kd2; 1 0];
    AAP3 = [-kp3 / kd3 -ki3 / kd3; 1 0];
    AAP4 = [-kp4 / kd4 -ki4 / kd4; 1 0];
    AAP5 = [-kp5 / kd5 -ki5 / kd5; 1 0];
    AAP6 = [-kp6 / kd6 -ki6 / kd6; 1 0];
    AAP7 = [-kp7 / kd7 -ki7 / kd7; 1 0];
    BB1 = [1 0]';
    CCP1 = [-1 / kd1 0]; CCP2 = [-1 / kd2 0]; CCP3 = [-1 / kd3 0]; CCP4 = [-1 / kd4 0]; CCP5 = [-1 / kd5 0]; CCP6 = [-1 / kd6 0]; CCP7 = [-1 / kd7 0];
    kp1 = 1; ki1 = 0.001; kd1 = 0.01; kp2 = 1; ki2 = 0.001; kd2 = 0.01; kp3 = 1; ki3 = 0.001; kd3 = 0.01; kp4 = 1; ki4 = 0.001; kd4 = 0.01; kp5 = 1; ki5 = 0.001; kd5 = 0.01; kp6 = 1; ki6 = 0.001; kd6 = 0.01; kp7 = 1; ki7 = 0.001; kd7 = 0.01;
    Q_f1 = zeros(7, 7); state_f1 = 1; action_f1 = 1; Q_f2 = zeros(7, 7); state_f2 = 1; action_f2 = 1; Q_f3 = zeros(7, 7); state_f3 = 1; action_f3 = 1;
    Q_f4 = zeros(7, 7); state_f4 = 1; action_f4 = 1; Q_f5 = zeros(7, 7); state_f5 = 1; action_f5 = 1; Q_f6 = zeros(7, 7); state_f6 = 1; action_f6 = 1;
    Q_f7 = zeros(7, 7); state_f7 = 1; action_f7 = 1;
    AAPP1 = eval(AAP1); CCPP1 = eval(CCP1); AAPP2 = eval(AAP2); CCPP2 = eval(CCP2); AAPP3 = eval(AAP3); CCPP3 = eval(CCP3);
    AAPP4 = eval(AAP4); CCPP4 = eval(CCP4); AAPP5 = eval(AAP5); CCPP5 = eval(CCP5); AAPP6 = eval(AAP6); CCPP6 = eval(CCP6);
    AAPP7 = eval(AAP7); CCPP7 = eval(CCP7);
    QQ_P1 = 1; R_P1 = 1; QQ_P2 = 1; R_P2 = 1; QQ_P3 = 1; R_P3 = 1; QQ_P4 = 1; R_P4 = 1; QQ_P5 = 1; R_P5 = 1; QQ_P6 = 1; R_P6 = 1; QQ_P7 = 1; R_P7 = 1;
    P_P1 = BB1 * QQ_P1 * BB1'; P_P2 = BB1 * QQ_P2 * BB1'; P_P3 = BB1 * QQ_P3 * BB1'; P_P4 = BB1 * QQ_P4 * BB1'; P_P5 = BB1 * QQ_P5 * BB1'; P_P6 = BB1 * QQ_P6 * BB1'; P_P7 = BB1 * QQ_P7 * BB1';
    x_f1 = zeros(2, 1); x_f2 = zeros(2, 1); x_f3 = zeros(2, 1); x_f4 = zeros(2, 1); x_f5 = zeros(2, 1); x_f6 = zeros(2, 1); x_f7 = zeros(2, 1);
    sz1 = 0; sz2 = 0; sz3 = 0; sz4 = 0; sz5 = 0; sz6 = 0; sz7 = 0;
    ier1 = 0; ier2 = 0; ier3 = 0; ier4 = 0; ier5 = 0; ier6 = 0; ier7 = 0; ier = 0;
    xx1 = 0; xx2 = 0; yy1 = 0; yy2 = 0; zz1 = 0; zz2 = 0; r = 24; ZZ = -z_pr * ones(1, ts / h_s);
    %%%%%%%%%%RLKMIC基本参数初始化设定%%%%%%%%%%%%%%%%%%%
    %奖惩值表RA的设定
    RA = [0 -1 -1 -1 -1 -1 -1; 0 0 -1 -1 -1 -1 -1; 1 1 1 0 0 0 0; 10 10 10 10 10 10 10; 0 0 0 0 1 1 1; -1 -1 -1 -1 -1 0 0; -1 -1 -1 -1 -1 -1 0];
    %值函数矩阵Q初始化
    Q1 = zeros(7, 7, 7); Q2 = zeros(7, 7, 7); Q3 = zeros(7, 7, 7); Q4 = zeros(7, 7, 7); Q5 = zeros(7, 7, 7); Q6 = zeros(7, 7, 7); Q7 = zeros(7, 7, 7);
    %学习因子与折扣因子
    alpha = 1; gama = 0.8;
    %可调系数lambda初始值
    lambda = 0;
    %状态s1初始值
    s1 = 0;
    %%%%%%%%%%%%%%%%%%根据贪婪策略选择动作%%%%%%%%%%%%%%%%%%
    kexi1 = 0.2; c1 = 1 - kexi1;
    beta1 = rand(1, 1);

    %#TODO: 这个有效
    if beta1 > c1
        action1 = ceil(7 * rand(1, 1));

        if s1 <- 50
            state1 = 1;
        end

        if s1 >= -50 && s1 <- 0.1
            state1 = 2;
        end

        if s1 >= -0.1 && s1 <- 0.00001
            state1 = 3;
        end

        %# TODO: 只剩这种
        if s1 >= -0.00001 && s1 <= 0.00001
            state1 = 4;
        end

        if s1 > 0.00001 && s1 <= 0.1
            state1 = 5;
        end

        if s1 > 0.1 && s1 <= 50
            state1 = 6;
        end

        if s1 > 50
            state1 = 7;
        end

    else

        if s1 <- 50
            [a1 action1] = find(Q1(1, :, 1) == max(Q1(1, 1, 1)));
            state1 = 1;
        end

        if s1 >= -50 && s1 <- 0.1
            [a1 action1] = find(Q1(2, :, 1) == max(Q1(2, 1:2, 1)));
            state1 = 2;
        end

        if s1 >= -0.1 && s1 <- 0.00001
            [a1 action1] = find(Q1(3, :, 1) == max(Q1(3, 2:3, 1)));
            state1 = 3;
        end

        %# TODO: 和这种, action = 0
        if s1 >= -0.00001 && s1 <= 0.00001
            [a1 action1] = find(Q1(4, :, 1) == max(Q1(4, 3:5, 1)));
            state1 = 4;
        end

        if s1 > 0.00001 && s1 <= 0.1
            [a1 action1] = find(Q1(5, :, 1) == max(Q1(5, 5:6, 1)));
            state1 = 5;
        end

        if s1 > 0.1 && s1 <= 50
            [a1 action1] = find(Q1(6, :, 1) == max(Q1(6, 6:7, 1)));
            state1 = 6;
        end

        if s1 > 50
            [a1 action1] = find(Q1(7, :, 1) == max(Q1(7, 7, 1)));
            state1 = 7;
        end

    end

    %%%%%%%%%%%%%%%%%%动作集的设定%%%%%%%%%%%%%%%%%%
    if action1 == 1
        lambda = lambda - (0.1 * exp(1 - 50 / abs(s1))) + 0.1 * s1 + 0.1 * ier;
    end

    if action1 == 2
        lambda = lambda + (0.1 * s1 * sin(pi * abs(s1) / 100)) + 0.1 * ier;
    end

    if action1 == 3
        lambda = lambda + (0.01 * s1 * sin(pi * abs(s1) / 0.2)) + 0.1 * ier;
    end

    if action1 == 4
        lambda = lambda + (0.0001 * s1 * sin(pi * abs(s1) / 0.00002)) * sign(s1) + 0.1 * ier;
    end

    if action1 == 5
        lambda = lambda + (0.01 * s1 * sin(pi * abs(s1) / 0.2)) + 0.1 * ier;
    end

    if action1 == 6
        lambda = lambda + (0.1 * s1 * sin(pi * abs(s1) / 100)) + 0.1 * ier;
    end

    if action1 == 7
        lambda = lambda + 0.1 * exp(1 - 50 / abs(s1)) + 0.1 * s1 + 0.1 * ier;
    end

    %%%%%%%%%%%%%%%%%各根系数初始值%%%%%%%%%%%%%%%%%%%%%%
    lambda1 = lambda; lambda2 = lambda; lambda3 = lambda; lambda4 = lambda; lambda5 = lambda; lambda6 = lambda;
    lambda7 = lambda; lambda8 = lambda;
    state2 = state1; state3 = state1; state4 = state1; state5 = state1; state6 = state1; state7 = state1; state1 = state1;
    action2 = action1; action3 = action1; action4 = action1; action5 = action1; action6 = action1; action7 = action1; action1 = action1;
    xini0 = 0; yini0 = 0; zini0 = 0;
    %////////////实际长度集合初始化//////////////////
    LS1 = 107 * ones(1, ts / h_s); LS2 = 107 * ones(1, ts / h_s); LS3 = 107 * ones(1, ts / h_s); LS4 = 107 * ones(1, ts / h_s);
    LS5 = 107 * ones(1, ts / h_s); LS6 = 107 * ones(1, ts / h_s); LS7 = 102 * ones(1, ts / h_s);
    xini = 1; yini = 1; zini = 214;
    %%%%%%%%%%%%%%%%%%仿真被控对象EUPI参数%%%%%%%%%%%%%%%%%%%
    W1 = [-3.09479755659205, 1.70048476002302, 0.370414761486346, -3.38253130590779, 0.431775664004444, 2.35254524657906, 1.84609934390742, -0.800043443802622, 4.07929735400909, -0.0914602449319138, 4.27271235884132, -3.57839529376426, -1.37618599677622, 3.17678254455182, 4.63767176695496, -3.61486974913582, -12.4914964116085, 1.46056888528974, -0.606361686917978, 8.00690971057119, -4.25359314347076, 1.36473607342831, 1.99664252003411, 1.99034723694324, -2.64929724848913, 1.17324534190423, -1.03367166842160, -0.556228186608643, -2.79909932964200, -2.46921434484929];
    W2 = [0.889023784255756, 5.31597681426150, 1.48707274529788, -1.04556611632018, -3.13437516720004, 0.884765361101010, -1.04978522660776, 1.95219343705108, 0.267542241890621, -4.99075967812481, 0.392047832665825, 2.74534059622179, -0.814317848386915, 0.549679195986895, 2.38723290945983, 5.92798720159298, 0.530975614871844, 1.13958922739353, -0.0474886382426681, -1.54815494565088, -1.18181633669045, 6.12024737107759, -1.11180072301667, -2.42769885777679, 0.406861458268700, 0.901299223604231, 2.52545084847076, -0.0108793254379065, 0.00806934861720831, -0.891265910819854];
    W3 = [1.33856327829803, 4.12846380127799, -0.0839218845213606, -1.04485051287188, -2.66096113597440, 0.361062831824641, 5.50691689576247, 1.66594398397364, -2.76450693749967, -15, 0.702080710593632, 4.16379110400282, -14.8155746426281, 1.61154827876533, 5.05485130243708, 7.49542912924541, 0.0859678442517601, 0.0713798642459240, 1.61110429840560, 1.50105377063121, 0.750945789345602, 6.66696431805256, -13.9162713605071, 0.808877102371189, 1.00351635980190, 2.96583910591349, 1.28098218964498, 2.46826886709203, 2.91130798918343, -2.15432142089081];
    p1 = -2.09919701765775;
    p2 = 0.794556282051666;
    rr = linspace(0, 2, 30); rr = rr'; pp1 = 0; pp2 = 0; pp3 = 0; pp4 = 0; pp5 = 0; pp6 = 0; pp7 = 0;
    xx = zeros(30, 1); tt = 0.01;
    tic

    for i = 1:(ts / h_s - 1)
        pressure1 = (5555.5555555555555555555555555556 * (L1(i) + 107) + ((5555.5555555555555555555555555556 * (L1(i) + 107) - 602224.58212670832698267540517198)^2 + 93258097.726418903983160271113075)^(1/2) - 602224.58212670832698267540517198)^(1/3) - 453.48422496570644718792866941015 / (5555.5555555555555555555555555556 * (L1(i) + 107) + ((5555.5555555555555555555555555556 * (L1(i) + 107) - 602224.58212670832698267540517198)^2 + 93258097.726418903983160271113075)^(1/2) - 602224.58212670832698267540517198)^(1/3) - 4.8148148148148148148148148148148;
        pressure2 = (5555.5555555555555555555555555556 * (L2(i) + 107) + ((5555.5555555555555555555555555556 * (L2(i) + 107) - 602224.58212670832698267540517198)^2 + 93258097.726418903983160271113075)^(1/2) - 602224.58212670832698267540517198)^(1/3) - 453.48422496570644718792866941015 / (5555.5555555555555555555555555556 * (L2(i) + 107) + ((5555.5555555555555555555555555556 * (L2(i) + 107) - 602224.58212670832698267540517198)^2 + 93258097.726418903983160271113075)^(1/2) - 602224.58212670832698267540517198)^(1/3) - 4.8148148148148148148148148148148;
        pressure3 = (5555.5555555555555555555555555556 * (L3(i) + 107) + ((5555.5555555555555555555555555556 * (L3(i) + 107) - 602224.58212670832698267540517198)^2 + 93258097.726418903983160271113075)^(1/2) - 602224.58212670832698267540517198)^(1/3) - 453.48422496570644718792866941015 / (5555.5555555555555555555555555556 * (L3(i) + 107) + ((5555.5555555555555555555555555556 * (L3(i) + 107) - 602224.58212670832698267540517198)^2 + 93258097.726418903983160271113075)^(1/2) - 602224.58212670832698267540517198)^(1/3) - 4.8148148148148148148148148148148;
        pressure4 = 0.51789321 * (L7(i) + 102) - 64.06856906;
        pressure(1) = 1 * pressure1 + 0.1 * lambda1;
        %%% pressure 1
        % set(handles.text19, 'string', u1);
        if pressure(1) < 0
            pressure(1) = 0;
        end

        if pressure(1) > 130
            pressure(1) = 130;
        end

        %%% pressure 2
        pressure(2) = 1 * pressure2 + 0.1 * lambda2;

        if pressure(2) < 0
            pressure(2) = 0;
        end

        if pressure(2) > 130
            pressure(2) = 130;
        end

        %%% pressure 3
        pressure(3) = 1 * pressure3 + 0.1 * lambda3;

        if pressure(3) < 0
            pressure(3) = 0;
        end

        if pressure(3) > 130
            pressure(3) = 130;
        end

        %%% pressure 4
        pressure(4) = 1 * pressure(1) + 0.01 * lambda4;

        if pressure(4) > 130
            pressure(4) = 130;
        end

        %%% pressure 5
        pressure(5) = 1 * pressure(2) + 0.01 * lambda5;

        if pressure(5) > 130
            pressure(5) = 130;
        end

        %%% pressure 6
        pressure(6) = 1 * pressure(3) + 0.01 * lambda6;

        if pressure(6) > 130
            pressure(6) = 130;
        end

        %%% pressure 7 for elongation
        pressure(7) = 0;
        %pressure(8)= 0;
        %pressure(9)= 0;%pressure4+lambda7;
        if pressure(7) < 0
            pressure(7) = 0;
        end

        if pressure(7) > 30
            pressure(7) = 30;
        end

        %
        if i > (ts / h_s - 10) / 2 && i < (ts / h_s - 1)
            pressure(7) = 5 * i / (ts / h_s - 5);
            %pressure(8)=7*i/(ts/h_s-5);
            % pressure(9)=7*i/(ts/h_s-5);
        else

            if i > (ts / h_s - 3)
                pressure(7) = 30;
                %pressure(8)=8;
                %pressure(9)=8;
            end

        end

        pressure11 = uint16(pressure * 2);
        PressureSend(pressure11);

        if i == (ts / h_s - 1) - 2
            set(handles.text19, 'string', 1);
            pause(0.1);
            fwrite(dotcp, closedo7(1, :), 'char');
            pause(0.1);
            fwrite(dotcp, closedo6(1, :), 'char');
            pause(0.1);
        end

        %}
        LS7(i + 1) = LS7(i + 1) - 5;
        %pause(0.05);
        %fopen(a);
        for jj = 0:0.1:1
            cc = fscanf(a);
            pause(0.02);
        end

        lcc = length(cc);
        % recBuff = fread(a, 7, 'uint8');
        while cc(1) ~= '#'
            cc = fscanf(a);
            lcc = length(cc);
        end

        ms = regexp(cc, '\-?\d*\.?\d*', 'match');
        DouCaX1 = 1 * (str2double(ms{2})) + 0 * (rand(1) - 0.5);
        DouCaY1 = 0.7 * str2double(ms{1}) + 0 * (rand(1) - 0.5);
        DouCaZ1 = 0;

        yini = y_pr - DouCaY1; xini = x_pr - DouCaX1; zini = Z_P(i) - 0; %}
        xini0 = [xini0, xini]; yini0 = [yini0, yini];
        [l111r l122r l133r l211r l222r l233r le theta1 fi1 r1 theta2 fi2 r2] = inversekinematics1(xini, yini, abs(zini), r);
        LS1(i + 1) = l111r; LS2(i + 1) = l122r; LS3(i + 1) = l133r; LS4(i + 1) = l211r; LS5(i + 1) = l222r; LS6(i + 1) = l233r;
        [l111r l122r l133r l211r l222r l233r le theta1 fi1 r1 theta2 fi2 r2] = inversekinematics1(X_P(i), Y_P(i), abs(Z_P(i)), r);
        L11 = l111r; L22 = l122r; L33 = l133r; L44 = l211r; L55 = l222r; L66 = l233r; L77 = le; zini0 = [zini0, zini + LS7(i + 1)];
        % LS: y_now, LSS: y_next, i+1 for new one in the list
        [x_f1, LSS1(i + 1), sz1, P_P1, pp1] = Kalman(AAPP1, BB1, CCPP1, L11, QQ_P1, P_P1, R_P1, x_f1, 0.1 * pressure1 + 40 * lambda1, LS1(i + 1), pp1);
        [x_f2, LSS2(i + 1), sz2, P_P2, pp2] = Kalman(AAPP2, BB1, CCPP2, L22, QQ_P2, P_P2, R_P2, x_f2, 0.1 * pressure2 + 40 * lambda2, LS2(i + 1), pp2);
        [x_f3, LSS3(i + 1), sz3, P_P3, pp3] = Kalman(AAPP3, BB1, CCPP3, L33, QQ_P3, P_P3, R_P3, x_f3, 0.1 * pressure3 + 40 * lambda3, LS3(i + 1), pp3);
        [x_f4, LSS4(i + 1), sz4, P_P4, pp4] = Kalman(AAPP4, BB1, CCPP4, L44, QQ_P4, P_P4, R_P4, x_f4, 0.1 * pressure1 + 40 * lambda1, LS4(i + 1), pp4);
        [x_f5, LSS5(i + 1), sz5, P_P5, pp5] = Kalman(AAPP5, BB1, CCPP5, L55, QQ_P5, P_P5, R_P5, x_f5, 0.1 * pressure2 + 40 * lambda2, LS5(i + 1), pp5);
        [x_f6, LSS6(i + 1), sz6, P_P6, pp6] = Kalman(AAPP6, BB1, CCPP6, L66, QQ_P6, P_P6, R_P6, x_f6, 0.1 * pressure3 + 40 * lambda3, LS6(i + 1), pp6);
        [x_f7, LSS7(i + 1), sz7, P_P7, pp7] = Kalman(AAPP7, BB1, CCPP7, L77, QQ_P7, P_P7, R_P7, x_f7, lambda7 + Pre7(i), LS7(i + 1), i, pp7);
        % f: future, p: present
        [kp1, R_P1, QQ_P1, state_f1, action_f1, action_p1, Q_f1] = GRLKPID1(-sz1, kp1, R_P1, QQ_P1, state_f1, action_f1, Q_f1);
        [kp2, R_P2, QQ_P2, state_f2, action_f2, action_p2, Q_f2] = GRLKPID1(-sz2, kp2, R_P2, QQ_P2, state_f2, action_f2, Q_f2);
        [kp3, R_P3, QQ_P3, state_f3, action_f3, action_p3, Q_f3] = GRLKPID1(-sz3, kp3, R_P3, QQ_P3, state_f3, action_f3, Q_f3);
        [kp4, R_P4, QQ_P4, state_f4, action_f4, action_p4, Q_f4] = GRLKPID1(-sz4, kp4, R_P4, QQ_P4, state_f4, action_f4, Q_f4);
        [kp5, R_P5, QQ_P5, state_f5, action_f5, action_p5, Q_f5] = GRLKPID1(-sz5, kp5, R_P5, QQ_P5, state_f5, action_f5, Q_f5);
        [kp6, R_P6, QQ_P6, state_f6, action_f6, action_p6, Q_f6] = GRLKPID1(-sz6, kp6, R_P6, QQ_P6, state_f6, action_f6, Q_f6);
        [kp7, R_P7, QQ_P7, state_f7, action_f7, action_p7, Q_f7] = GRLKPID1(-sz7, kp7, R_P7, QQ_P7, state_f7, action_f7, Q_f7);
        AAPP1 = eval(AAP1); CCPP1 = eval(CCP1); AAPP2 = eval(AAP2); CCPP2 = eval(CCP2); AAPP3 = eval(AAP3); CCPP3 = eval(CCP3);
        AAPP4 = eval(AAP4); CCPP4 = eval(CCP4); AAPP5 = eval(AAP5); CCPP5 = eval(CCP5); AAPP6 = eval(AAP6); CCPP6 = eval(CCP6);
        AAPP7 = eval(AAP7); CCPP7 = eval(CCP7);
        [lambda1, Q1, ier1, state1, action1, eer1] = RLKMIC2(lambda1, L11 - 107, LSS1(i + 1) - 107, sz1, i + 1, Q1, RA, state1, action1, action_f1, action_p1, ier1, alpha, gama, eer1);
        [lambda2, Q2, ier2, state2, action2, eer2] = RLKMIC2(lambda2, L22 - 107, LSS2(i + 1) - 107, sz2, i + 1, Q2, RA, state2, action2, action_f2, action_p2, ier2, alpha, gama, eer2);
        [lambda3, Q3, ier3, state3, action3, eer3] = RLKMIC2(lambda3, L33 - 107, LSS3(i + 1) - 107, sz3, i + 1, Q3, RA, state3, action3, action_f3, action_p3, ier3, alpha, gama, eer3);
        [lambda4, Q4, ier4, state4, action4, eer4] = RLKMIC2(lambda4, L44 - 107, LSS4(i + 1) - 107, sz4, i + 1, Q4, RA, state4, action4, action_f4, action_p4, ier4, alpha, gama, eer4);
        [lambda5, Q5, ier5, state5, action5, eer5] = RLKMIC2(lambda5, L55 - 107, LSS5(i + 1) - 107, sz5, i + 1, Q5, RA, state5, action5, action_f5, action_p5, ier5, alpha, gama, eer5);
        [lambda6, Q6, ier6, state6, action6, eer6] = RLKMIC2(lambda6, L66 - 107, LSS6(i + 1) - 107, sz6, i + 1, Q6, RA, state6, action6, action_f6, action_p6, ier6, alpha, gama, eer6);
        [lambda7, Q7, ier7, state7, action7, eer7] = RLKMIC2(lambda7, L77 - 102, LSS7(i + 1) - 102, sz7, i + 1, Q7, RA, state7, action7, action_f7, action_p7, ier7, alpha, gama, eer7);
        set(handles.text19, 'string', lambda1);
        pause(0.35);
        tt = tt + 0.01;
    end

    toc
    pause(1);
    pressure = pressure11;
    pressure11(10) = uint16(40 * 2);
    PressureSend(pressure11);
    pause(1)
    pressure11(10) = uint16(60 * 2);
    PressureSend(pressure11);
    pause(1)
    pressure(7:9) = zeros(1, 3);
    pressure11(7:9) = uint16(pressure(7:9) * 2);
    PressureSend(pressure11);
    pause(1)
    fwrite(dotcp, opendo7(1, :), 'char');
    pause(0.2);
    fwrite(dotcp, opendo5(1, :), 'char');
    pause(5);
    pressure(1) = 0; pressure(4) = 100;
    pressure(2) = 120; pressure(5) = 0;
    pressure(3) = 120; pressure(6) = 0;
    pressure11(1:6) = uint16(pressure(1:6) * 2);
    PressureSend(pressure11);
    pause(5)
    fwrite(dotcp, closedo7(1, :), 'char');
    pause(0.5);
    fwrite(dotcp, closedo5(1, :), 'char');
    pause(0.5);
    pressure(7) = 40;
    pressure11(7) = uint16(pressure(7) * 2);
    PressureSend(pressure11);
    pause(4);
    pressure11(10) = uint16(0);
    PressureSend(pressure11);
    pause(0.5)
    fwrite(dotcp, opendo7(1, :), 'char');
    pause(0.2);
    fwrite(dotcp, opendo6(1, :), 'char');
    pause(3)
    %}
    fwrite(dotcp, closedo7(1, :), 'char');
    pause(0.5);
    fwrite(dotcp, closedo6(1, :), 'char')
    pause(0.5)
    pressure(1) = 0; pressure(4) = 90;
    pressure(2) = 120; pressure(5) = 0;
    pressure(3) = 100; pressure(6) = 0;
    pressure(7) = 0;
    pressure11(1:7) = uint16(pressure(1:7) * 2);
    PressureSend(pressure11);
    pause(1)
    fclose(datcp);
    delete(datcp);
    fclose(dotcp);
    delete(dotcp);
    set(handles.pushbutton_ON, 'string', 'ON');
    pause(0.2)
    fclose(a);
    set(handles.pushbutton_CON, 'string', 'ON');
end
