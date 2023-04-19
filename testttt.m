D = 1 %De hoeveelheid inputs
K = 3;  %De hoeveelheid outputs 
M = 5; %De hoeveelheid neurons in de hidden layer-1
V = 5; %De hoeveelheid neurons in de hidden layer-2


alpha = 1; %learningrate
theta1 = 0.2 * randn(M, D) %random startwaarden THETA1
theta2 = 0.2 * randn(V, M) %random startwaarden THETA2
theta3 = 0.2 * randn(K, V) %random startwaarden THETA3
bias1 = 0.2 * randn(M,1) %random startwaarden bias1
bias2 = 0.2 * randn(V,1) %random startwaarden bias2

tau1 = 0; %totaal som gradienten theta1
tau2 = 0; %totaal som gradienten theta2
tau3 = 0; %totaal som gradienten theta3
phi1 = 0; % totaal som gradienten bias1
phi2 = 0; % totaal som gradienten bias2
Totaalcost = 0; %totaal som cost training

y =[1;0;0]
a= [1];

a1 = a;
Z2 = theta1 * a1; 
a2 = Sigmoid(Z2 + bias1);
Z3 = theta2 * a2;
a3 = Sigmoid(Z3 + bias2) ;
Z4 = theta3 * a3;
a4 = Sigmoid(Z4);
h = a4;
Cost = sum((h-y).^2)

%dCo/dtheta3 = dZ4/dtheta3 * da4/dz4 * dCo/da4
dZ4_dtheta3 = a3;
da4_dZ4 = aSigmoid(Z4)
dCo_da4 = 2 .* (h-y)

delta3 = (aSigmoid(Z4) .* (2 .* (h-y))) .* a3';

%dCo/dtheta2 = dZ3/dtheta2 * da3/dZ3 * dZ4/da3 * da4/dz4 * dCo/da4
dZ3_dtheta2 = a2;
da3_dZ3 = aSigmoid(Z3+bias2)
dZ4_da3 = theta3;

test = ((aSigmoid(Z4) .* (2 .* (h-y)))' *theta3)