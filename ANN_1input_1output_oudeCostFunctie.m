clear all; close all; 

a = [0.5]; %inputs
y = [1];
alpha = 1;
theta1 = [1] %startwaarden theta1
theta2 = [2]; %startwaarden theta2
for d= 1:30000
a1 = a;
Z2 = theta1*a1;
a2 = Sigmoid(Z2);
Z3 = theta2*a2;
a3 = Sigmoid(Z3);

h=a3;
cost= sum((h-y).^2)
afgeleide_cost = 2*sum(h-y);
delta1 = (a1 * ((aSigmoid(Z3) .* afgeleide_cost)' * theta2 .* (aSigmoid(Z2))'))';
delta2 = afgeleide_cost .* aSigmoid(Z3) * a2';

theta1 = theta1 .- delta1*alpha;
theta2 = theta2 .- delta2*alpha;
endfor