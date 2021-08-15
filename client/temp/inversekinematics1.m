function [l11 l12 l13 l21 l22 l23, le, theta1, phi1, r1, theta2, phi2, r2] = inversekinematics1(x, y, z, r)
    %syms theta1 phi1 r

    %针对第一段手臂弯曲关节求取向心角、偏转角及曲率半径
    %第一段末端原点坐标j53ezs,
    z_segBend = 100;
   %  x = x / 2;
   %  y = y / 2;
   %  z_segBend = z_segBend / 2;
    ZZ = -z;
    %偏转角
    if x >= 0 && y >= 0
        theta1 = atan(y / x);
    end

    if x < 0 && y >= 0
        theta1 = pi + atan(y / x);
    end

    if x < 0 && y < 0
        theta1 = pi + atan(y / x);
    end

    if x >= 0 && y < 0
        theta1 = 2 * pi + atan(y / x);
    end

    %向心角
    %if z_segBend>0
    phi1 = pi - 2 * asin(z_segBend / (x^2 + y^2 + z_segBend^2)^(1/2));
    %end
    %if z_segBend<0
    %phi1=0-2*asin(z_segBend/(x^2+y^2+z_segBend^2)^(1/2));
    %end
    %曲率半径
    r1 = ((x^2 + y^2 + z_segBend^2) / (8 * (1 - cos(phi1))))^(1/2);  %# FIXME:这里2改8了
    %第一段充气腔长度变化
    l11 = phi1 * (r1 - r * cos(theta1));
    l12 = phi1 * (r1 - r * cos(2 * pi / 3 - theta1));
    l13 = phi1 * (r1 - r * cos(4 * pi / 3 - theta1));

    %针对第二段手臂弯曲关节求取向心角、偏转角及曲率半径
    %偏转角
    %if x>=0
    theta2 = theta1 + pi;
    %end
    %if x<0
    theta2 = theta1 + pi;
    %end
    %向心角
    phi2 = phi1;
    %曲率半径
    r2 = r1;
    %第二段充气腔长度变化
    l21 = abs(phi2) * (r2 - r * cos(theta2));
    l22 = abs(phi2) * (r2 - r * cos(2 * pi / 3 - theta2));
    l23 = abs(phi2) * (r2 - r * cos(4 * pi / 3 - theta2));
    le = abs(ZZ) - z_segBend;

    while l11 > 200 || l12 > 200 || l13 > 200 || l21 > 200 || l22 > 200 || l23 > 200 || le > 170
        z_segBend = z_segBend + 10;

        if le < 102
            break
        end

        [l111 l122 l133 l211 l222 l233 theta11 fi11 r11 theta21 fi21 r21] = inversekinematics(x, y, z_segBend, r);
        l11 = l111; l12 = l122; l13 = l133; l21 = l211; l22 = l222; l23 = l233; le = abs(ZZ) - z_segBend;
        Z = z_segBend / 2;
    end
