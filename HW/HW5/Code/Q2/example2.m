clear 
clc

m_moon = 7.34767309e22;
m_earth = 5.9742e24;
mu = m_moon/(m_moon + m_earth);
initial = [0.82411, 0, 0.05821, 0, 0.16883, 0];
T0 = 2.76301 ; 
phi = eye(6);
STM = reshape(phi,1,[]);

for hu=1:1000
    
    ode_options_yzero = odeset('Events',@Findyzero,'RelTol',1e-13,'AbsTol',1e-13);
    [t, X] = ode45(@CRTBPmodel, [0 T0], [initial STM], ode_options_yzero, mu);
    T_half = t(end) ;
    x = X(end,1) ;
    y = X(end,2) ;
    z = X(end,3) ;
    deltax = X(end,4) ;
    deltay = X(end,5) ;
    deltaz = X(end,6) ;
    r1 = [x+mu , y , z];
    r2 = [x-1+mu , y , z];
    ddx = 2*deltay + x - ((1-mu)/norm(r1)^3)*r1(1) - (mu/norm(r2)^3) * r2(1) ;
    ddz = - ((1-mu)/norm(r1)^3)*r1(3) - (mu/norm(r2)^3) * r2(3) ;
    phi = reshape(X(end,7:end),6,6);
    crm = zeros(2,2);
    crm(1,1) = phi(4,3) - (1/deltay)*ddx*phi(2,3) ;
    crm(1,2) = phi(4,5) - (1/deltay)*ddx*phi(2,5) ;
    crm(2,1) = phi(6,3) - (1/deltay)*ddz*phi(2,3) ;
    crm(2,2) = phi(6,5) - (1/deltay)*ddz*phi(2,5) ;
    imat = inv(crm);
    delta_z  = -imat(1,1)*deltax -imat(1,2)*deltaz ;
    delta_dy = -imat(2,1)*deltax -imat(2,2)*deltaz ;
    dX = [0, 0, delta_z, 0, delta_dy, 0] ;
    initial = initial + dX ;
    error = [delta_z , delta_dy];
    if norm(error) < 1e-13
        break
    end
    
end
ode_options_yzero = odeset('RelTol',1e-13,'AbsTol',1e-13);
[t, X] = ode45(@CRTBPmodel, [0:0.0001:(T0 + T0/100)], [initial STM], ode_options_yzero, mu);
[t2, X2] = ode45(@CRTBPmodel, [(T0/100):0.0001:(T0+T0/100)], [initial STM], ode_options_yzero, mu);
x = X(:,1);
x = x(277:1:end);
y = X(:,2);
y = y(277:1:end);
z = X(:,3);
z = z(277:1:end);
x2 = X2(:,1);
y2 = X2(:,2);
z2 = X2(:,3);
plot(t2, x-x2);
xlabel('Delta X')
ylabel('t')

figure
plot(t2, y-y2);
xlabel('Delta Y')
ylabel('t')

figure
plot(t2, z-z2);
xlabel('Delta Z')
ylabel('t')
