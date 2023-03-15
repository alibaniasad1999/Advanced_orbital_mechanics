%% Clearing
clear
clc
close

%% Initializing
r0 = [1600;5310;3800;0;0;0]; %km
v0 = [-7.350;0.4600;2.470;0;0;0]; % km/s
m = [227;5.972e24];
RE = 6378.137; % Earth's Equatorial Radius
G = 6.673e-20;
options = odeset('RelTol',1e-8);
[t,x] = ode45(@(t,x)nbody(t,x,m),[0 24*3600],[r0;v0],options);


%% Diagrams 
[xs, ys, zs] = sphere(50);
surf(6378*xs, 6378*ys, 6378*zs)
axis equal
colormap('gray');
xlabel('X(km)')
ylabel('Y(km)')
zlabel('Z(km)')
hold on
plot3(x(:, 1), x(:, 2), x(:, 3), 'b', 'linewidth', 4);

print -depsc ../../Figure/Q2/3Dof_view

view(0,0)

print -depsc ../../Figure/Q2/xz_view

view(-90,0)

print -depsc ../../Figure/Q2/yz_view

view(-90,-90)

print -depsc ../../Figure/Q2/xy_view

close all


%% part b %%
r = x(:, 1:3);

for i=1:length(x)
    r(i, :) = r(i, :) * tarns_matrix(t(i));
end
phi = zeros(1, length(r));
lambda = zeros(1, length(r));
for i=1:length(r)
    [phi(i), lambda(i)] = latlon(r(i, :));
end

plot((lambda-pi)*180/pi, phi*180/pi, '.')
xlabel('longitude $\lambda^{\circ}$', 'Interpreter','latex',...
    'FontSize', 20);
ylabel('latitude $\phi^{\circ}$', 'Interpreter','latex', ...
    'FontSize', 20);
axis equal
axis([-180 180 -90 90])


print -depsc ../../Figure/Q2/latlong



%% on earth fig
% initializes figure
figure('Position',[300,300,1000,500]);
% plotting options
opts_earth1.Color = [140,21,21]/255;
opts_earth1.LineWidth = 2.5;
ground_track((phi)*180/pi,(lambda-pi)*180/pi,opts_earth1,'Earth');

print -depsc ../../Figure/Q2/latlong_earth


%% bonus %%
r0 = [1600;5310;3800]; %km
v0 = [-7.350;0.4600;2.470]; % km/s

[a,e,E,I,omega,Omega,~,~,~,~] = vec2orbElem(r0,v0,398600);
[r_new, v_new] = oe2ecf(a, e, I, Omega, omega, E, 398600);
[a,e,E,I,omega,Omega,P,tau,A,B] = vec2orbElem(r_new,v_new,398600);
[r_new, v_new] = oe2ecf(a, e, 0.4, Omega, omega, E, 398600);
r0 = [r_new;0;0;0]; %km
v0 = [v_new;0;0;0]; % km/s
options = odeset('RelTol',1e-8);
[t_new,x_new] = ode45(@(t,r_new)nbody(t,r_new,m),[0 24*3600],[r0;v0],options);

%% bonus figure
close all
[xs, ys, zs] = sphere(50);


plot3(x(:, 1), x(:, 2), x(:, 3), 'b', 'linewidth', 4);
hold on
plot3(x_new(:, 1), x_new(:, 2), x_new(:, 3), 'r', 'linewidth', 4);
surf(6378*xs, 6378*ys, 6378*zs)
axis equal
colormap('gray');
xlabel('X(km)')
ylabel('Y(km)')
zlabel('Z(km)')
legend('First Orbit', 'Second Orbit')

print -depsc ../../Figure/Q2/3Dof_view_compare

view(0,0)

print -depsc ../../Figure/Q2/xz_view_compare

view(-90,0)

print -depsc ../../Figure/Q2/yz_view_compare

view(-90,-90)

print -depsc ../../Figure/Q2/xy_view_compare

close all

r_new = x_new(:, 1:3);

for i=1:length(x_new)
    r_new(i, :) = r_new(i, :) * tarns_matrix(t_new(i));
end
phi_new = zeros(1, length(r_new));
lambda_new = zeros(1, length(r_new));
for i=1:length(r_new)
    [phi_new(i), lambda_new(i)] = latlon(r_new(i, :));
end


plot((lambda-pi)*180/pi, phi*180/pi, '.')
hold on
plot((lambda_new-pi)*180/pi, phi_new*180/pi, '.')

legend('First Orbit', 'Second Orbit')
xlabel('longitude $\lambda^{\circ}$', 'Interpreter','latex',...
    'FontSize', 20);
ylabel('latitude $\phi^{\circ}$', 'Interpreter','latex', ...
    'FontSize', 20);
axis equal
axis([-180 180 -90 90])


print -depsc ../../Figure/Q2/latlong_compare







%% functions %%
function T_s =  tarns_matrix(t)
    omega = 2 * pi / 24 / 3600;
    T_s = [cos(omega*t), -sin(omega*t), 0;
           sin(omega*t),  cos(omega*t), 0;
                0              0      , 1];
end
function [phi, lambda] = latlon(r)
    phi = asin(r(3)/norm(r));
    if r(2) > 0
        lambda = acos((r(1)/norm(r))/ cos(phi));
    else
        lambda = 2*pi - acos((r(1)/norm(r)) / cos(phi));
    end
end