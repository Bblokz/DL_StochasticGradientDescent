
load DATA.mat


T = []; 
K = 1;  %De hoeveelheid outputs 
M = 2; %De hoeveelheid neurons in de hidden layer-1
V = 2; %De hoeveelheid neurons in de hidden layer-2
  %for i = 1:10 
 
 %Ttmp = zeros(500, K);   %maak een matrix van de hoeveelheid plaatjes met i bij K 

 %Ttmp(:,i) = 1; %stel in die matrix de ide rij op 1 

 %T = [T; Ttmp];  %voeg de labels bij T 

%endfor 

clear Ttmp; 
clear i; 
 
[N, D] = size(X); 
alpha = 10; %learningrate
theta1 = 2 * randn(M, D); %random startwaarden THETA1
theta2 = 2 * randn(V, M); %random startwaarden THETA2
theta3 = 2 * randn(K, V); %random startwaarden THETA3
bias1 = 2 * randn(M,1); %random startwaarden bias1
bias2 = 2 * randn(V,1); %random startwaarden bias2

tau1 = 0; %totaal som gradienten theta1
tau2 = 0; %totaal som gradienten theta2
tau3 = 0; %totaal som gradienten theta3
phi1 = 0; % totaal som gradienten bias1
phi2 = 0; % totaal som gradienten bias2
Totaalcost = 0; %totaal som cost training

  
for n = 1: 1000
for i = 1: (N)   % doorlopen van 1 tot 5000 plaatjes

   a{1} = (X(i,:))';   
   Z{2} = theta1 * a{1}; 
   a{2} = Sigmoid(Z{2} + bias1); 
   Z{3} = theta2 * a{2}; 
   a{3} = Sigmoid(Z{3} + bias2); 
   Z{4} = theta3 * a{3};
   a{4} = Sigmoid(Z{4});
   h = a{4} 

   Cost = sum((h .- (T(1,:))') .^2); 
   Totaalcost = Totaalcost + Cost; %cost berekenen over alle 5000 plaatjes
   
   dZ2_dtheta1 = a{1};
   da2_dZ2 = aSigmoid(Z{2}+bias1);
   d
   da3_dZ3 = 
   delta1 = (a{1} * ((aSigmoid(Z{3}) .* Cost)' * theta2 .* (aSigmoid(Z{2}))'))'; % dCo/dTHETA1
   delta2 = Cost .* aSigmoid(Z{3}) * a{2}'; % dCo/dTHETA2
   kappa = (Cost .* aSigmoid(Z{3}))' * theta2; % dCo/dbias

   tau1 = tau1 + delta1; %SOM Gradienten 
   tau2 = tau2 + delta2;  %SOM Gradienten 
   phi = phi + kappa; %SOM Gradienten 

endfor 

  

Totaalcost

  
tau1 = tau1 / (N); %gem. gradient van Theta1 over alle trainings data
tau2 = tau2 / (N); %gem. gradient van Theta2 over alle trainings data
phi = phi / (N); %gem. gradient van de bias over alle trainings data

theta1 = theta1 .- alpha * tau1; %optimalisatie
theta2 = theta2 .- alpha * tau2; %optimalisatie
bias = bias .- alpha * phi'; %optimalisatie
clear i; 
Totaalcost = 0; 
endfor


  

