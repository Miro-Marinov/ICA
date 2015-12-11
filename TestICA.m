[x1, Fs1] = wavread('male.wav');
[x2, Fs2] = wavread('female.wav');
[x3, Fs3] = wavread('ENG_M.wav');

x1 = x1(1:length(x3));

x2 = x2(1:length(x3));

A = randn(3, 3);

x = A*[x1';x2';x3'];

[S,w] = myICA(x,2);

sound(S(2,:), Fs1);