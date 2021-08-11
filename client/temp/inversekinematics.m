%%%%%%%%%%%%%带有限制条件的软体手臂运动学逆解方程%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [l11 l12 l13 l21 l22 l23, theta1, fi1, r1, theta2, fi2, r2] = inversekinematics(x, y, z, r)
    %syms theta1 fi1 r

    %针对第一段手臂弯曲关节求取向心角、偏转角及曲率半径
    %第一段末端原点坐标j53ezs
    theta1 = 0;
    x1 = x / 2;
    y1 = y / 2;
    z1 = z / 2;
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
end
