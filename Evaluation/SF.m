%{
@ calculate the spatial freuency of an image
%}
function t = SF(A)

%[M N] : image size
%RF: row frequency
%CR: col frequency
A = double(A);
[M,N] = size(A);

% Calculate RF
RF = 0.0;
for i=1:M
    for j=2:N
        RF = RF + (A(i,j) - A(i,j-1))^2;
    end
end
RF = sqrt(RF/(M*N));
%RF = (RF/(M*N));

% Calculate CF
CF=0.0;
for i=2:M
    for j=1:N
        CF = CF  + (A(i,j) - A(i-1,j))^2;
    end
end
CF = sqrt(CF/(M*N));
%CF = (CF/(M*N));

% Calculate SF
SF = sqrt(RF^2 + CF^2);
%SF = (RF^2 + CF^2);
t = SF;