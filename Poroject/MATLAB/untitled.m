% Constants
G = 6.67430e-11;  % Gravitational constant (m^3 kg^(-1) s^(-2))
M_e = 5.972e24;  % Mass of Earth (kg)
M_m = 7.347e22;  % Mass of Moon (kg)
R_e = 3.844e8;  % Distance between Earth and Moon (m)
omega_m = sqrt(G * (M_e + M_m) / R_e^3);  % Angular velocity of Moon (rad/s)

% Lagrange point positions
L1_x = R_e * (M_m / (M_e + M_m));
L2_x = R_e * (M_m / (M_e + M_m));

% Plotting parameters
num_points = 1000;  % Number of points on the Lyapunov orbit
orbit_period = 2 * pi / omega_m;  % Orbital period of the Moon (s)
t = linspace(0, orbit_period, num_points);  % Time array for plotting

% Lyapunov orbit parameters
a = 500000;  % Semi-major axis (m)
e = 0.1;  % Eccentricity
omega = 0;  % Argument of periapsis (rad)
theta0 = 0;  % Initial true anomaly (rad)

% Calculate position coordinates for L1 Lyapunov orbit
L1_x_coords = L1_x - a * (1 - e^2) ./ (1 + e * cos(omega + t - theta0));
L1_y_coords = zeros(1, num_points);

% Calculate position coordinates for L2 Lyapunov orbit
L2_x_coords = L2_x + a * (1 - e^2) ./ (1 + e * cos(omega + t - theta0));
L2_y_coords = zeros(1, num_points);

% Plotting the Lyapunov orbits
figure;
hold on;
plot(L1_x_coords, L1_y_coords, 'r', 'LineWidth', 1.5);
plot(L2_x_coords, L2_y_coords, 'b', 'LineWidth', 1.5);
plot(L1_x, 0, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot(L2_x, 0, 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
title('Lyapunov Orbits around L1 and L2 Lagrange Points');
xlabel('x (m)');
ylabel('y (m)');
legend('L1 Lyapunov Orbit', 'L2 Lyapunov Orbit', 'L1 Lagrange Point', 'L2 Lagrange Point');
axis equal;
grid on;
hold off;
