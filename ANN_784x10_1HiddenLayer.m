clear all; close all; 

load DATA.mat;

T = [];
K = 10;  %De hoeveelheid outputs
M = 250; %De hoeveelheid neurons in de hidden layer

for i = 1:10
  
  Ttmp = zeros(500, K);   %maak een matrix van de hoeveelheid plaatjes met i bij K
  Ttmp(:,i) = 1;          %stel in die matrix de ide rij op 1
  T = [T; Ttmp];          %voeg de labels bij T
  
endfor

clear Ttmp;
clear i;

[N, D] = size(X);
alpha = 1; %learningrate/stapgroote

theta1 = 0.2 * randn(M, D); %random waarden instellen
theta2 = 0.2 * randn(K, M); %random waarden instellen
bias = 0.2 * randn(M,1); %random waarden instellen

tau1 = 0;
tau2 = 0;

phi = 0;
Totaalcost = 0;

for b =1:100000
for i = 1:N %doorlopen van alle data
  
   a{1} = (X(1,:))';
   
   Z{2} = theta1 * a{1};
   a{2} = Sigmoid(Z{2} + bias);
   Z{3} = theta2 * a{2};
   a{3} = Sigmoid(Z{3});
   
   h = a{3};
   Cost = sum((h .- (T(i,:))') .^2); %SE
 
   afgeleide_cost = 2 .*(h .- (T(i,:))');
   Totaalcost = Totaalcost + Cost;
   
   delta1 = (a{1} * ((aSigmoid(Z{3}) .* afgeleide_cost)' * theta2 .* (aSigmoid(Z{2}+bias))'))';%dZ2/dtheta1 * da2/dZ2 * dZ3/da2 * da3/dZ3 * dCo/da3
   delta2 = afgeleide_cost .* aSigmoid(Z{3}) * a{2}'; %dZ3/dtheta2 * da3/dZ3 * dCo/da3

   kappa = (afgeleide_cost .* aSigmoid(Z{3}))' * theta2;
   
   tau1 = tau1 + delta1; %som gradienten theta1
   tau2 = tau2 + delta2; %som gradienten theta2
   
   phi = phi + kappa; %som gradienten bias
endfor

Totaalcost

tau1 = tau1 / (N);
tau2 = tau2 / (N);

phi = phi / (N);

theta1 = theta1 - alpha * tau1;
theta2 = theta2 - alpha * tau2;

bias = bias - alpha * phi';

clear i;
Totaalcost = 0;
endfor
 
 i =1;
 a{1} = (X(1))';
   
   Z{2} = theta1 * a{1};
   a{2} = Sigmoid(Z{2} + bias);
   Z{3} = theta2 * a{2};
   a{3} = Sigmoid(Z{3});
   
   h = a{3};
   Cost = sum((h .- (T(i,:))') .^2); %SE