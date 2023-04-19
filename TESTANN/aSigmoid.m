function retval = aSigmoid (x)
    retval = (e.^-x)./((1 + e.^-x).^2);
  endfunction
