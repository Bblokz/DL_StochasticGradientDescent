function retval = Sigmoid (x)
    retval = 1 ./ (1 + e.^-x);
  endfunction