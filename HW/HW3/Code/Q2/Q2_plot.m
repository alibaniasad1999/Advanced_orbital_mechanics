clear;
clc;
%% use function %%
a = 6783.34174;
e =0.0014021;
inclination = 51.27632;
Omega = 275.17058;
theta = 309.67626;
% deg ω deg Ω deg θ deg
% Value    90.69731  309.67626
Q2(a, norm(e), inclination, Omega, theta, 100);
print('../../Figure/Q2/e_100','-depsc');
close;
Q2(a, norm(e), inclination, Omega, theta, 5000);
print('../../Figure/Q2/e_5000','-depsc');
close;
Q2(a, norm(e), inclination, Omega, theta, 100000);
print('../../Figure/Q2/e_100000','-depsc');
close;
