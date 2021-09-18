function net=svm(p,t,spread)


% Defaults
if nargin < 3, spread = 0.1; end

% Format
if isa(p,'cell'), p = cell2mat(p); end
if isa(t,'cell'), t = cell2mat(t); end

% Error checks
if (~isa(p,'double') && ~islogical(p)) || (~isreal(p)) || (isempty(p))
  error('NNET:Arguments','Inputs are not a non-empty real matrix.')
end
if (~isa(t,'double') && ~islogical(t)) || (~isreal(t)) || (isempty(t))
  error('NNET:Arguments','Targets are not a non-empty real matrix.')
end
if (size(p,2) ~= size(t,2))
  error('NNET:Arguments','Inputs and Targets have different numbers of columns.')
end
if (~isa(spread,'double')) || ~isreal(spread) || any(size(spread) ~= 1) || (spread < 0)
  error('NNET:Arguments','Spread is not a positive or zero real value.')
end

% Dimensions
[R,Q] = size(p);
[S,Q] = size(t);

% Architecture
net = network(1,2,[1;0],[1;0],[0 0;1 0],[0 1]);

% Simulation
net.inputs{1}.size = R;
net.inputWeights{1,1}.weightFcn = 'dist';
net.layers{1}.netInputFcn = 'netprod';
net.layers{1}.transferFcn = 'radbas';
net.layers{1}.size = Q;
net.layers{2}.size = S;
net.layers{2}.transferFcn = 'compet';
net.outputs{2}.exampleOutput = t;

% Weight and Bias Values
net.b{1} = zeros(Q,1)+sqrt(-log(.5))/spread;
net.iw{1,1} = p';
net.lw{2,1} = t;
