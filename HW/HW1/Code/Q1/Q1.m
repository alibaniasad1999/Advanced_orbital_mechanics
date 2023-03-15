clear
clc
%% Question 1 %%
mu = 3.986e5; % km^3/s^2
r = 6578.1137; % km

T = 2*pi*sqrt(r^3/mu); % period
fprintf("period is %.2f seconds\n", T)

omega = 2*pi/T;
fprintf("omega is %.5f rad/seconds\n", omega)

v = sqrt(mu/r);
fprintf("velocity is %.2f km/seconds\n", v)

v_new = v + 0.5;
fprintf("new velocity is %.2f km/seconds\n", v_new)

r_new = mu/v_new^2;

fprintf("new r is %.0f km/seconds\n", r_new)

T_new = 2*pi*sqrt(r_new^3/mu);
fprintf("new period is %.2f seconds\n", T_new)

omega_new = 2*pi/T_new;
fprintf("new omega is %.5f rad/seconds\n", omega_new)

