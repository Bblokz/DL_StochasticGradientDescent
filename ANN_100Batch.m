
load DATA.mat

P=100;
O=1;
U=0;
T = []; 
K = 10;  %De hoeveelheid outputs 
M = 250; %De hoeveelheid neurons in de hidden layer1
  for i = 1:10 
 
 Ttmp = zeros(500, K);   %maak een matrix van de hoeveelheid plaatjes met i bij K 

 Ttmp(:,i) = 1; %stel in die matrix de ide rij op 1 

 T = [T; Ttmp];  %voeg de labels bij T 

endfor 

clear Ttmp; 
clear i; 
 
[N, D] = size(X); 
alpha = 100; %learningrate
theta1 = 2 * randn(M, D) .- 1; 
theta2 = 2 * randn(K, M) .- 1; 
bias = 2 * randn(M,1) .- 1; 

tau1 = 0; %totaal som gradienten theta1
tau2 = 0; %totaal som gradienten theta2
phi = 0; % totaal som gradienten bias

Totaalcost = 0; %totaal som cost training

for V= 1:1
for n = 1: 1
for i = O: P   

   a{1} = (X(i,:))';   
   Z{2} = theta1 * a{1}; 
   a{2} = Sigmoid(Z{2} + bias); 
   Z{3} = theta2 * a{2}; 
   a{3} = Sigmoid(Z{3}); 
   h = a{3} ;

   Cost = sum((h .- (T(1,:))') .^2)
   aCost = 2 .* sum(h .- (T(1,:))')
   Totaalcost = Totaalcost + Cost; 
   
   delta1 = (a{1} * ((aSigmoid(Z{3}) .* aCost)' * theta2 .* (aSigmoid(Z{2}))'))'; 
   delta2 = aCost .* aSigmoid(Z{3}) * a{2}'; 
   kappa = (aCost .* aSigmoid(Z{3}))' * theta2; 

   tau1 = tau1 + delta1; 
   tau2 = tau2 + delta2;  
   phi = phi + kappa; 

endfor 

  

Totaalcost
P=P+100;
O=O+99+U;
U=1;
tau1 = tau1 / (100); %gem. gradient Theta1 over alle trainings data
tau2 = tau2 / (100); %gem. gradient Theta2 over alle trainings data
phi = phi / (100); %gem. gradient bias over alle trainings data

theta1 = theta1 .- alpha * tau1; %optimalisatie
theta2 = theta2 .- alpha * tau2; %optimalisatie
bias = bias .- alpha * phi'; %optimalisatie
clear i; 
Totaalcost = 0; 

endfor
endfor

  

