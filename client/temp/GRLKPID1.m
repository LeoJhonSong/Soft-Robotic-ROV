function [kp, RR, QQ, state, action, action_p, Q] = GRLKPID1(e, kp1, RR1, QQ1, state1, action1, Q1)
    RA = [0 -1 -1 -1 -1 -1 -1; 0 0 -1 -1 -1 -1 -1; 1 1 1 0 0 0 0; 10 10 10 10 10 10 10; 0 0 0 0 1 1 1; -1 -1 -1 -1 -1 0 0; -1 -1 -1 -1 -1 -1 0]; %奖惩值表
    alpha = 1; %学习因子
    gama = 0.8; %折扣因子
    %Q1=zeros(7,7);%参数A值函数
    %%%%%%%%%%%%%%%%%%%%%贪婪算法动作选择策略%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    kexi1 = 0.0; cc1 = 1 - kexi1;
    beta1 = rand(1, 1);

    if beta1 > cc1
        actionnew1 = ceil(7 * rand(1, 1));

        if e <- 50 | |- isinf(e)
            prize1 = 1;
        end

        if e >= -50 && e <- 1
            prize1 = 2;
        end

        if e >= -1 && e <- 0.0001
            prize1 = 3;
        end

        if e >= -0.0001 && e <= 0.0001
            prize1 = 4;
        end

        if e > 0.0001 && e <= 1
            prize1 = 5;
        end

        if e > 1 && e <= 50
            prize1 = 6;
        end

        if e > 50 || isinf(e)
            prize1 = 7;
        end

    else

        if e <- 50
            [a1 actionnew1] = find(Q1(1, :) == max(Q1(1, 1)));
            prize1 = 1;
        end

        if e >= -50 && e <- 1
            [a1 actionnew1] = find(Q1(2, 1:2) == max(Q1(2, 1:2)));
            prize1 = 2;
        end

        if e >= -1 && e <- 0.0001
            [a1 actionnew1] = find(Q1(3, 2:3) == max(Q1(3, 2:3)));
            actionnew1 = actionnew1(1) + 1;
            prize1 = 3;
        end

        if e >= -0.0001 && e <= 0.0001
            [a1 actionnew1] = find(Q1(4, 3:5) == max(Q1(4, 3:5)));
            actionnew1 = actionnew1(1) + 2;
            prize1 = 4;
        end

        if e > 0.0001 && e <= 1
            [a1 actionnew1] = find(Q1(5, 5:6) == max(Q1(5, 5:6)));
            actionnew1 = actionnew1(1) + 4;
            prize1 = 5;
        end

        if e > 1 && e <= 50
            [a1 actionnew1] = find(Q1(6, 6:7) == max(Q1(6, 6:7)));
            actionnew1 = actionnew1(1) + 5;
            prize1 = 6;
        end

        if e > 50 || isinf(e)
            [a1 actionnew1] = find(Q1(7, :) == max(Q1(7, 7)));
            prize1 = 7;
        end

    end

    %%%%%%%%%%%%%%%%%%%%%%更新Q值表%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Q1(state1, action1) = Q1(state1, action1) + alpha * (RA(prize1, action1) + gama * Q1(prize1, actionnew1(end)) - Q1(state1, action1));
    Q = Q1;
    action_p = action1;
    state = prize1; action = actionnew1(end);
    %%%%%%%%%%%%%%%%%%%动作集%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if action == 1
        kp = kp1 + (0.2 * exp(1 - 50 / abs(e)) + 0.02 * exp(0));
        RR = RR1 + (0.1 * exp(1 - 50 / abs(e)) + 0.01 * exp(0));
        QQ = QQ1 - 0.1 * exp(1 - 50 / abs(e)) - 0.01 * exp(0.0 / abs(5));
    end

    if action == 2
        kp = kp1 + (0.02 / exp(1 - 1 / abs(50)) * exp(1 - 1 / abs(e)) + 0.002 * exp(0));
        RR = RR1 + (0.01 / exp(1 - 1 / abs(50)) * exp(1 - 1 / abs(e)) + 0.001 * exp(0));
        QQ = QQ1 - (0.01 / exp(1 - 1 / abs(50)) * exp(1 - 1 / abs(e)) + 0.001 * exp(0));
    end

    if action == 3
        kp = kp1 + (0.002 / exp(1 - 0.0001 / abs(1)) * exp(1 - 0.0001 / abs(e)) + 0.0002 * exp(0));
        RR = RR1 + (0.001 / exp(1 - 0.0001 / abs(1)) * exp(1 - 0.0001 / abs(e)) + 0.0001 * exp(0));
        QQ = QQ1 - (0.001 / exp(1 - 0.0001 / abs(1)) * exp(1 - 0.0001 / abs(e)) + 0.0001 * exp(0));
    end

    if action == 4
        kp = kp1 - (0.0002 * exp(1 - 0.0001 / abs(e))) * sign(e);
        RR = RR1 - (0.0001 * exp(1 - 0.0001 / abs(e))) * sign(e);
        QQ = QQ1 + (0.0001 * exp(1 - 0.0001 / abs(e))) * sign(e);
    end

    if action == 5
        kp = kp1 - (0.002 / exp(1 - 0.0001 / abs(1)) * exp(1 - 0.0001 / abs(e)) + 0.0002 * exp(0));
        RR = RR1 - (0.001 / exp(1 - 0.0001 / abs(1)) * exp(1 - 0.0001 / abs(e)) + 0.0001 * exp(0));
        QQ = QQ1 + (0.001 / exp(1 - 0.0001 / abs(1)) * exp(1 - 0.0001 / abs(e)) + 0.0001 * exp(0));
    end

    if action == 6
        kp = kp1 - (0.02 / exp(1 - 1 / abs(50)) * exp(1 - 1 / abs(e)) + 0.002 * exp(0.00000 / abs(1)));
        RR = RR1 - (0.01 / exp(1 - 1 / abs(50)) * exp(1 - 1 / abs(e)) + 0.001 * exp(0.00000 / abs(1)));
        QQ = QQ1 + (0.01 / exp(1 - 1 / abs(50)) * exp(1 - 1 / abs(e)) + 0.001 * exp(0.00000 / abs(1)));
    end

    if action == 7
        kp = kp1 - 0.2 * exp(1 - 50 / abs(e)) - 0.02 * exp(0.0 / abs(5));
        RR = RR1 - 0.1 * exp(1 - 50 / abs(e)) - 0.01 * exp(0.0 / abs(5));
        QQ = QQ1 + 0.1 * exp(1 - 50 / abs(e)) + 0.01 * exp(0.0 / abs(5));
    end

    if kp < 0.00001
        kp = 0.00001;
    end

    %if RR<0
    % RR=0;
    %end
