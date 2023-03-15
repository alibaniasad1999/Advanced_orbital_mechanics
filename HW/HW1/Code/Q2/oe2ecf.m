function [r_ecf, v_ecf] = oe2ecf(a, e, i, raan, argp, nu, mu)
% Convert Keplerian orbital elements to ECEF position and velocity vectors

% Semi-latus rectum
p = a*(1 - e^2);

% Position in perifocal frame
r_pqw = [p*cos(nu)/(1 + e*cos(nu)); p*sin(nu)/(1 + e*cos(nu)); 0];

% Velocity in perifocal frame
v_pqw = [-sqrt(mu/p)*sin(nu); sqrt(mu/p)*(e + cos(nu)); 0];

% Rotation matrices
R3_W = [cos(raan) sin(raan) 0; -sin(raan) cos(raan) 0; 0 0 1];
R1_i = [1 0 0; 0 cos(i) sin(i); 0 -sin(i) cos(i)];
R3_w = [cos(argp) sin(argp) 0; -sin(argp) cos(argp) 0; 0 0 1];

% Perifocal to ECEF transformation matrix
C_pqw2eci = R3_W*R1_i*R3_w;

% ECEF position and velocity vectors
r_ecf = C_pqw2eci*r_pqw;
v_ecf = C_pqw2eci*v_pqw;
end