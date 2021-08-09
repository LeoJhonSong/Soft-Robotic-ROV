function [lambda, Q11, ier, state11, action11, eer] = RLKMIC2(lambda, L, LS, sz, i, Q1, RA, state1, action1, action_f, action_p, ier, alpha, gama, eer1)
    % changed L, LS

    error = -L + LS; s1 = -sqrt((error^2 + sz^2) / 2) * sign(error + sz);
    eer = -LS + L;
    %s1=eer;
    if isnan(s1)
        s1 = 0.000001; error = 0.000001;
    end

    ier = ier + eer * 0.01; %更新状态s1
    %//////根据贪婪策略选择动作///////////
    kexi1 = 0.2 - 0.01 * i;

    if kexi1 < 0.03
        kexi1 = 0.03;
    end

    c1 = 1 - kexi1;
    beta1 = 0;

    %# FIXME: 这也没用啊
    if beta1 > c1
        actionnew1 = ceil(7 * rand(1, 1));

        if s1 <- 50 || - isinf(s1)
            prize1 = 1;
        end

        if s1 >= -50 && s1 <- 1
            prize1 = 2;
        end

        if s1 >= -1 && s1 <- 0.00001
            prize1 = 3;
        end

        if s1 >= -0.00001 && s1 <= 0.00001 || isnan(s1)
            prize1 = 4;
        end

        if s1 > 0.00001 && s1 <= 1
            prize1 = 5;
        end

        if s1 > 1 && s1 <= 50
            prize1 = 6;
        end

        if s1 > 50 || isinf(s1)
            prize1 = 7;
        end

    else

        if s1 <- 50 || - isinf(s1)
            [a1 actionnew1] = find(Q1(1, :, action_p) == max(Q1(1, 1, action_p)));
            %actionnew1=1;
            prize1 = 1;
        end

        if s1 >= -50 && s1 <- 1
            [a1 actionnew1] = find(Q1(2, 1:2, action_p) == max(Q1(2, 1:2, action_p)));
            prize1 = 2;
        end

        if s1 >= -1 && s1 <- 0.00001
            [a1 actionnew1] = find(Q1(3, 3:4, action_p) == max(Q1(3, 3:4, action_p)));
            actionnew1 = actionnew1 + 1;
            prize1 = 3;
        end

        if s1 >= -0.00001 && s1 <= 0.00001 || isnan(s1)
            [a1 actionnew1] = find(Q1(4, 3:5, action_p) == max(Q1(4, 3:5, action_p)));
            actionnew1 = actionnew1 + 2;
            prize1 = 4;
        end

        if s1 > 0.00001 && s1 <= 1
            [a1 actionnew1] = find(Q1(5, 4:5, action_p) == max(Q1(5, 4:5, action_p)));
            actionnew1 = actionnew1 + 4;
            prize1 = 5;
        end

        if s1 > 1 && s1 <= 50
            [a1 actionnew1] = find(Q1(6, 6:7, action_p) == max(Q1(6, 6:7, action_p)));
            actionnew1 = actionnew1 + 5;
            prize1 = 6;
        end

        if s1 > 50 || isinf(s1)
            [a1 actionnew1] = find(Q1(7, :, action_p) == max(Q1(7, 7, action_p)));
            %actionnew1=7;
            prize1 = 7;
        end

    end

    %////////值函数表更新////////////
    %Q1(state1,action1)=Q1(state1,action1)+alpha*(RA(prize1,action1)+gama*Q1(prize1,actionnew1(end))-Q1(state1,action1));
    Q1(state1, action1, action_p) = Q1(state1, action1, action_p) + alpha * (RA(prize1, action1) + gama * Q1(prize1, actionnew1(end), action_f) - Q1(state1, action1, action_p));
    Q11 = Q1;
    state1 = prize1; action1 = actionnew1(end);
    state11 = state1; action11 = action1;
    %///////根据值函数函数表更新lambda值//////////
    %%%%%%%%%%%%%%%%%%动作集的设定%%%%%%%%%%%%%%%%%%

    if action1 == 1
        lambda = lambda - (10 * exp(1 - 50 / abs(s1)) + 1 * exp(0)) + 0.000 * ier;
    end

    if action1 == 2
        lambda = lambda - (0.1 / exp(1 - 1 / abs(50)) * exp(1 - 1 / abs(s1)) + 0.01 * exp(0)) + 0.0000 * ier;
    end

    if action1 == 3
        lambda = lambda - (0.01 / exp(1 - 0.00001 / abs(1)) * exp(1 - 0.0001 / abs(s1)) + 0.001 * exp(0)) + 0.0000 * ier;
    end

    if action1 == 4
        lambda = lambda - (0.0001 * exp(1 - 0.00001 / abs(s1))) * sign(s1);
    end

    if action1 == 5
        lambda = lambda + (0.01 / exp(1 - 0.00001 / abs(1)) * exp(1 - 0.0001 / abs(s1)) + 0.001 * exp(0)) + 0.0000 * ier;
    end

    if action1 == 6
        lambda = lambda + (0.1 / exp(1 - 1 / abs(50)) * exp(1 - 1 / abs(s1)) + 0.01 * exp(0.00000 / abs(1))) + 0.000 * ier;
    end

    if action1 == 7
        lambda = lambda + 10 * exp(1 - 50 / abs(s1)) + 1 * exp(0.0 / abs(50));
    end

    %lambda=0.02*(L-LS)+0*(eer-eer1)/0.01+2*ier;
    %lambda=lambda*s1;
    %if isnan(lambda)
    %lambda=lambda1;
    % end
    %if lambda>1
    %lambda=1;
    %end
    %if lambda<0
    %lambda=0.001;
    %end
end
