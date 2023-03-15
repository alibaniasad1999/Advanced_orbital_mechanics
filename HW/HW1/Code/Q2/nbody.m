function  difx  = nbody(~, x, m)
    n = length(x);
    g = 6.673e-20;
    half = n/2;
    r = zeros(3,n/6);
    
    num = length(m);  % bodies' count
    
    % Seperating each bodies' position vector from input:
    for i=1:half
        if mod(i,3) == 1
            r(1,floor(i/3)+1) = x(i);
        elseif mod(i,3) == 2
            r(2,floor(i/3)+1) = x(i);
        elseif mod(i,3) == 0
            r(3,i/3) = x(i);
        end
    end
    
    % Filing the first three parameters of the output which are velocity
    % vectors:
    difx = [];
    for i=(half+1):n
        difx = [difx;x(i)];
    end
    
    % Finding and adding the applied force to each body:
    for i=1:num
        a = [0;0;0];
        for j=1:num
            if i~=j
                l = (r(1,i)-r(1,j))^2 + (r(2,i)-r(2,j))^2 + (r(3,i)-r(3,j))^2;
                l = l^(-3/2);
                u = r(1,i) - r(1,j);
                v = r(2,i) - r(2,j);
                w = r(3,i) - r(3,j);
                a = a + [(-1)*l*m(j)*g*u;(-1)*l*m(j)*g*v;(-1)*l*m(j)*g*w];
            end
        end
        difx = [difx;a];
    end
end