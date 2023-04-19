y= 1
A1= [1;0.7;1]
THETA1= rand(3,3)
THETA2= rand(1,3)

Z2=THETA1*A1
A2= 1 ./(1+e.^-(Z2))
Z3=THETA2*A2
A3=1 ./ (1+e.^-(Z3))
h= A3
Co=(h-y)^2

%%dCo/dTHETA1 = dZ2/dTHETA1 * dA2/dZ2 * dZ3/dA2 * dA3/dZ3 * dCo/dA3
%% dCo/dTHETA2 = dZ3/dTHETA2 * dA3/dZ3 * dCo/dA3
%% afgeleide sigmoid = e^-x / ((1+e^-x)^2) <--checked met GR
dZ2dT1 = A1 %%(3,1)
dA2dZ2 = e.^-(Z2) ./ ((1+e.^-(Z2)).^2) %%(3,1)
dZ3dA2 = THETA2 %%(1,3)
dA3dZ3 = e.^-(Z3) ./ ((1+e.^-(Z3)).^2) %%(1,1)
dCodA3 = 2*(h-y) %%getal

test = dZ2dT1 .* dA2dZ2 .* dZ3dA2 .* 



