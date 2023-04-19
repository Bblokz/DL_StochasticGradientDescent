
% deze versie liep vast na +-200 interaties op 4.4k
load DATA.mat;

K = 10;  %De hoeveelheid outputs 
M =400; %De hoeveelheid neurons in de hidden layer-1
V = 400; %De hoeveelheid neurons in de hidden layer-2
T = [];

for i = 1:10
  
  Ttmp = zeros(500, K);   %maak een matrix van de hoeveelheid plaatjes met i bij K
  Ttmp(:,i) = 1;          %stel in die matrix de ide rij op 1
  T = [T; Ttmp];          %voeg de labels bij T
  
endfor

clear Ttmp;
clear i;


[N, D] = size(X);
alpha = 0.1; %learningrate

theta1 = 0.2 * randn(M, D); %random startwaarden THETA1
theta2 = 0.2 * randn(V, M);%random startwaarden THETA2
theta3 = 0.2 * randn(K, V); %random startwaarden THETA3
bias1 = 0.2 * randn(M,1); %random startwaarden bias1
bias2 = 0.2 * randn(V,1); %random startwaarden bias2

tau1 = 0; %totaal som gradienten theta1
tau2 = 0; %totaal som gradienten theta2
tau3 = 0; %totaal som gradienten theta3
phi1 = 0; % totaal som gradienten bias1
phi2 = 0; % totaal som gradienten bias2
Totaalcost = 0; %totaal som cost training



 for b = 1:10000
for i = 1:N %doorlopen van alle data
a1 = (X(i,:))';
Z2 = theta1 * a1; 
a2 = Sigmoid(Z2 + bias1);
Z3 = theta2 * a2;
a3 = Sigmoid(Z3 + bias2);
Z4 = theta3 * a3;
a4 = Sigmoid(Z4);
h = a4;

Cost = sum((h .- (T(i,:))') .^2); %SE
Totaalcost = Totaalcost +Cost; %berekenen cost over alle data
afgeleide_cost = 2 .*(h .- (T(i,:))');

%dCo/dtheta3 = dZ4/dtheta3 * da4/dz4 * dCo/da4
dZ4_dtheta3 = a3;
da4_dZ4 = aSigmoid(Z4);
dCo_da4 = 2 .*(h .- (T(i,:))');

delta3 = (aSigmoid(Z4) .* (afgeleide_cost)) .* a3'; %berekenen van de gradients van theta3
tau3 = tau3 + delta3;

%dCo/dtheta2 = dZ3/dtheta2 * da3/dZ3 * dZ4/da3 * da4/dz4 * dCo/da4
dZ3_dtheta2 = a2;
da3_dZ3 = aSigmoid(Z3+bias2);
dZ4_da3 = theta3;

delta2 = (a2 .* aSigmoid(Z3+bias2)) * ((aSigmoid(Z4) .* (afgeleide_cost))' *theta3); %berekenen gradients theta2
tau2 = tau2 + delta2;


%dCo/dtheta1 = dZ2/dtheta1 * da2/dZ2 * dZ3/da2 * da3/dZ3 * dZ4/da3 * da4/dz4 * dCo/da4
dZ2_dtheta1 = a1;
da2_dZ2 = aSigmoid(Z2+bias1);
dZ3_da2 = theta2;

delta1 = ((((theta3 * theta2)') * (aSigmoid(Z4) .* afgeleide_cost)) .* aSigmoid(Z3+bias2) .* aSigmoid(Z2+bias1)) * a1';
tau1 = tau1 + delta1;

%dCo/dbias2 = da3/dbias2 * dZ4/da3 * da4/dZ4 * dCo/da4
deltaB2 = ((theta3') * (aSigmoid(Z4) .* afgeleide_cost)) .* aSigmoid(Z3 + bias2);
phi2 = phi2 +deltaB2;

%dCo/dbias1 = dZ3/da2 *da3/dZ3 * Z4/da3 * da4/dZ4 * dC0/da4
deltaB1 = ((theta2 * aSigmoid(Z3+bias2)) .*(theta3' * (aSigmoid(Z4) .* afgeleide_cost))) .* aSigmoid(Z2+bias1);
phi1 = phi1 + deltaB1;
endfor
Totaalcost
tau1 = tau1 / (N);
tau2 = tau2 / (N);
tau3 = tau3 / (N);

phi1 = phi1 / (N);
phi2 = phi2 / (N);

theta3 = theta3 .- (alpha*tau3); %optimalisatie
theta2 = theta2 .-(alpha*tau2); %optimalisatie
theta1 = theta1 .- (alpha*tau1); %optimalisatie

bias2 = bias2 .-(alpha*phi2); %optimalisatie
bias1 = bias1 .-(alpha*phi1); #optimalisatie
Totaalcost =0;
endfor
h