function Q3(a_data, e_data, i_data, Omega_data, theta_data, time)
%
% This function solve Example 10.9 the Gauss planetary equations for
% solar radiation pressure (Equations 10.106).
%
% User M-functions required: sv_from_coe, los, solar_position
% User subfunctions required: rates
% The M-function rsmooth may be found in Garcia, D: “Robust Smoothing of Gridded
% Data in One and Higher Dimensions with Missing Values,” Computational Statistics
% and Data Analysis, Vol. 54, 1167-1178, 2010.
% 
global JD %Julian day
%...Preliminaries:
close all
clc
%...Conversion factors:
hours = 3600; %Hours to seconds
days = 24*hours; %Days to seconds
deg = pi/180; %Degrees to radians
%...Constants;
mu = 398600; %Gravitational parameter (km^3/s^2)
RE = 6378; %Eath's radius (km)
c = 2.998e8; %Speed of light (m/s)
S = 1367; %Solar constant (W/m^2)
Psr = S/c; %Solar pressure (Pa);

%...Satellite data:
CR = 2; %Radiation pressure codfficient
m = 100; %Mass (kg)
As = 200; %Frontal area (m^2);
%...Initial orbital parameters (given):
a0 = a_data; %Semimajor axis (km)
e0 = e_data; %eccentricity
incl0 = i_data; %Inclination (radians)
RA0 = Omega_data; %Right ascencion of the node (radians)
TA0 = theta_data; %True anomaly (radians)
w0 = 227.493*deg; %Argument of perigee (radians)
%...Initial orbital parameters (inferred):
h0 = sqrt(mu*a0*(1-e0^2)); %angular momentrum (km^2/s)
T0 = 2*pi/sqrt(mu)*a0^1.5; %Period (s)
rp0 = h0^2/mu/(1 + e0); %perigee radius (km)
ra0 = h0^2/mu/(1 - e0); %apogee radius (km)
%...Store initial orbital elements (from above) in the vector coe0:
coe0 = [h0 e0 RA0 incl0 w0 TA0];
%...Use ODE45 to integrate Equations 12.106, the Gauss planetary equations
% from t0 to tf:
JD0 = 2438400.5+12122; %Initial Julian date 04/30/2022 
t0 = 0; %Initial time (s)
y0 = coe0'; %Initial orbital elements
nout = 10000; %Number of solution points to output
tspan = linspace(t0, time, nout); %Integration time interval
options = odeset(...
'reltol', 1.e-8, ...
'abstol', 1.e-8);
[t,y] = ode45(@rates, tspan, y0, options);
%...Extract or compute the orbital elements' time histories from the
% solution vector y:
h = y(:,1);
e = y(:,2);
RA = y(:,3);
incl = y(:,4);
w = y(:,5);
TA = y(:,6);
a = h.^2/mu./(1 - e.^2);
% save('data_Q2', 'y', 't');

%...Smooth the data to remove short period variations:
% h = rsmooth(h);
% e = rsmooth(e);
% RA = rsmooth(RA);
% incl = rsmooth(incl);
% w = rsmooth(w);
% a = rsmooth(a);
% plot(t/days,h - h0, 'LineWidth',2)
% set(gca, 'FontSize', 16, 'FontName', 'Times New Roman');
% xlabel('Days', 'interpreter', 'latex', 'FontSize', 24);
% ylabel('$h-h_0$', 'interpreter', 'latex', 'FontSize', 24);
% axis tight
% print('../../Figure/Q3/h_fig','-depsc');
% 
% 
% plot(t/days,e - e0, 'LineWidth',2)
% set(gca, 'FontSize', 16, 'FontName', 'Times New Roman');
% xlabel('Days', 'interpreter', 'latex', 'FontSize', 24);
% ylabel('$e-e_0$', 'interpreter', 'latex', 'FontSize', 24);
% axis tight
% print('../../Figure/Q3/e_fig','-depsc');
% 
% 
% plot(t/days,(RA - RA0)/deg, 'LineWidth',2)
% set(gca, 'FontSize', 16, 'FontName', 'Times New Roman');
% xlabel('Days', 'interpreter', 'latex', 'FontSize', 24);
% ylabel('$\Omega-Omega_0$ (deg)', 'interpreter', 'latex', 'FontSize', 24);
% axis tight
% print('../../Figure/Q3/Omega_fig','-depsc');
% 
% 
% plot(t/days,(incl - incl0)/deg, 'LineWidth',2)
% set(gca, 'FontSize', 16, 'FontName', 'Times New Roman');
% xlabel('Days', 'interpreter', 'latex', 'FontSize', 24);
% ylabel('$i-i_0$ (deg)', 'interpreter', 'latex', 'FontSize', 24);
% axis tight
% print('../../Figure/Q3/i_fig','-depsc');
% 
% 
% plot(t/days,(w - w0)/deg, 'LineWidth',2)
% set(gca, 'FontSize', 16, 'FontName', 'Times New Roman');
% xlabel('Days', 'interpreter', 'latex', 'FontSize', 24);
% ylabel('$\omega-\omega_0$ (deg)', 'interpreter', 'latex', 'FontSize', 24);
% axis tight
% print('../../Figure/Q3/omega_fig','-depsc');
% 
% 
% plot(t/days,mod((TA - TA0)/deg, 360), 'LineWidth',2)
% set(gca, 'FontSize', 16, 'FontName', 'Times New Roman');
% xlabel('Days', 'interpreter', 'latex', 'FontSize', 24);
% ylabel('$\theta-\theta_0$ (deg)', 'interpreter', 'latex', 'FontSize', 24);
% axis tight
% print('../../Figure/Q3/theta_fig','-depsc');


f = figure;
width=800;
height=900;
f.Position = [15 15 width height];
set(gca, 'FontName', 'Times New Roman');
set(gca, 'FontSize', 16)

subplot(3,2,1)
plot(t/days,h - h0)
title('Angular Momentum (km^2/s)')
xlabel('days')
axis tight
subplot(3,2,2)
plot(t/days,e - e0)
title('Eccentricity')
xlabel('days')
axis tight
subplot(3,2,4)
plot(t/days,(RA - RA0)/deg)
title('Right Ascension (deg)')
xlabel('days')
axis tight
subplot(3,2,5)
plot(t/days,(incl - incl0)/deg)
title('Inclination (deg)')
xlabel('days')
axis tight
subplot(3,2,6)
plot(t/days,(w - w0)/deg)
title('Argument of Perigee (deg)')
xlabel('days')
axis tight
subplot(3,2,3)
plot(t/days,a - a0)
title('Semimajor axis (km)')
xlabel('days')
axis tight
% ...Subfunctions:
% 

function dfdt = rates(t,f)
% 
%...Update the Julian Date at time t:
JD = JD0 + t/days;
%...Compoute the apparent position vector of the sun:
[lamda eps r_sun] = solar_position(JD);
%...Convert the ecliptic latitude and the obliquity to radians:
lamda = lamda*deg;
eps = eps*deg;
%...Extract the orbital elements at time t
h = f(1);
e = f(2);
RA = f(3);
i = f(4);
w = f(5);
TA = f(6);
u = w + TA; %Argument of latitude
%...Compute the state vector at time t:
coe = [h e RA i w TA];
[R, V] = sv_from_coe(coe,mu);
%...Calculate the manitude of the radius vector:
r = norm(R);
%...Compute the shadow function and the solar radiation perturbation:
nu = los(R, r_sun);
pSR = nu*(S/c)*CR*As/m/1000;
%...Calculate the trig functions in Equations 12.105.
sl = sin(lamda); cl = cos(lamda);
se = sin(eps); ce = cos(eps);
sW = sin(RA); cW = cos(RA);
si = sin(i); ci = cos(i);
su = sin(u); cu = cos(u);
sT = sin(TA); cT = cos(TA);
%...Calculate the earth-sun unit vector components (Equations 12.105):
ur = sl*ce*cW*ci*su + sl*ce*sW*cu - cl*sW*ci*su ...
+ cl*cW*cu + sl*se*si*su;
us = sl*ce*cW*ci*cu - sl*ce*sW*su - cl*sW*ci*cu ...
- cl*cW*su + sl*se*si*cu;

uw = - sl*ce*cW*si + cl*sW*si + sl*se*ci;
%...Calculate the time rates of the osculating elements from
% Equations 12.106:
hdot = -pSR*r*us;
edot = -pSR*(h/mu*sT*ur ...
+ 1/mu/h*((h^2 + mu*r)*cT + mu*e*r)*us);
TAdot = h/r^2 ...
- pSR/e/h*(h^2/mu*cT*ur - (r + h^2/mu)*sT*us);
RAdot = -pSR*r/h/si*su*uw;
idot = -pSR*r/h*cu*uw;
wdot = -pSR*(-1/e/h*(h^2/mu*cT*ur - (r + h^2/mu)*sT*us) ...
- r*su/h/si*ci*uw);
%...Return the rates to ode45:
dfdt = [hdot edot RAdot idot wdot TAdot]';
end %rates
end %Example_10_9
% 