function [x22, ye, sz, P2, pp] = Kalman(AA2, BB1, CC2, Ld, QQ, P2, R2, x2, u, yv, i, pp)
    DD2 = Ld;
    [A2, B2, C2, D2] = c2dm(AA2, BB1, CC2, DD2, 0.01, 'z');
    P2 = A2 * P2 * A2' + B2 * QQ * B2';
    Mn = P2 * C2' / (C2 * P2 * C2' + R2);
    P2 = (eye(2) - Mn * C2) * P2;
    x2 = A2 * x2 + B2 * u;
    x22 = x2 + Mn * (yv - C2 * x2 - D2);
    ye = C2 * x22 + D2;
    pp1 = Ld - yv; pp2 = Ld - ye;
    pp = [pp, abs(Ld - yv)];
    pp3 = max(pp);
    sz = C2 * P2 * C2';
    %sz=-pp2;
    %
    if pp1 <- 0.001 && pp2 < (ceil(pp3 * 2)) && pp2 > 0 %
        ye = C2 * x22 + D2 - ceil(pp1);
        sz = -Ld + ye;
    end

    if pp1 > 0.001 && pp2 >- (ceil(pp3 * 2)) && pp2 < 0
        ye = C2 * x22 + D2 - ceil(pp1);
        sz = -Ld + ye;
    end

    %}
    %
    if abs(pp1) - abs(pp2) > (ceil(pp3)) && sign(pp2) == sign(pp1)
        ye = C2 * x22 + D2 - ceil(pp1);
        sz = -Ld + ye;
    end

end
