function [l11 l12 l13 l21 l22 l23, le, theta1, fi1, r1, theta2, fi2, r2] = inversekinematics1(x, y, z, r)
    %syms theta1 fi1 r

    %针对第一段手臂弯曲关节求取向心角、偏转角及曲率半径
    %第一段末端原点坐标j53ezs,
    zz = 100;
    x1 = x / 2;
    y1 = y / 2;
    z1 = zz / 2;
    ZZ = -z;
    %偏转角
    if x1 >= 0 && y1 >= 0
        theta1 = atan(y1 / x1);
    end

    if x1 < 0 && y1 >= 0
        theta1 = pi + atan(y1 / x1);
    end

    if x1 < 0 && y1 < 0
        theta1 = pi + atan(y1 / x1);
    end

    if x1 >= 0 && y1 < 0
        theta1 = 2 * pi + atan(y1 / x1);
    end

    %向心角
    %if z1>0
    fi1 = pi - 2 * asin(z1 / (x1^2 + y1^2 + z1^2)^(1/2));
    %end
    %if z1<0
    %fi1=0-2*asin(z1/(x1^2+y1^2+z1^2)^(1/2));
    %end
    %曲率半径
    r1 = ((x1^2 + y1^2 + z1^2) / (2 * (1 - cos(fi1))))^(1/2);
    %第一段充气腔长度变化
    l11 = fi1 * (r1 - r * cos(theta1));
    l12 = fi1 * (r1 - r * cos(2 * pi / 3 - theta1));
    l13 = fi1 * (r1 - r * cos(4 * pi / 3 - theta1));

    %针对第二段手臂弯曲关节求取向心角、偏转角及曲率半径
    %偏转角
    %if x>=0
    theta2 = theta1 + pi;
    %end
    %if x<0
    theta2 = theta1 + pi;
    %end
    %向心角
    fi2 = fi1;
    %曲率半径
    r2 = r1;
    %第二段充气腔长度变化
    l21 = abs(fi2) * (r2 - r * cos(theta2));
    l22 = abs(fi2) * (r2 - r * cos(2 * pi / 3 - theta2));
    l23 = abs(fi2) * (r2 - r * cos(4 * pi / 3 - theta2));
    le = abs(ZZ) - zz;

    while l11 > 200 || l12 > 200 || l13 > 200 || l21 > 200 || l22 > 200 || l23 > 200 || le > 170
        zz = zz + 10;

        if le < 102
            break
        end

        [l111 l122 l133 l211 l222 l233 theta11 fi11 r11 theta21 fi21 r21] = inversekinematics(x, y, zz, r);
        l11 = l111; l12 = l122; l13 = l133; l21 = l211; l22 = l222; l23 = l233; le = abs(ZZ) - zz;
        Z = zz / 2;
    end

end
