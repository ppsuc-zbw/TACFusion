function out=cc(A,B,F)
A=double(A);
B=double(B);
F=double(F);
W1=corrcoef(A,F);
W2=corrcoef(B,F);
out = (W1+W2)/2;
out = out(2);