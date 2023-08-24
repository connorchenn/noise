function nllh = static_model_llh(theta, data)
% Computes the negative log likelihood of data given parameters (theta) for a static model.
%
% Inputs:
%   - theta(1): alpha - learning rate
%   - theta(2): stickiness - stickiness parameter
%   - theta(3): epsilon - exploration parameter
%   - data: Matrix containing the choices and rewards from the data
%         Column 3: Choices (1 or 2)
%         Column 5: Rewards (0 or 1)
%
% Output:
%   - nllh: Negative log likelihood of the data given the parameters
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/28/2023

alpha = theta(1);              % learning rate - determines the weight of new information during learning
beta = 8;                      % Inverse temperature parameter, fixed - controls the exploration-exploitation trade-off
stick = theta(2);              % Stickiness parameter - controls the tendency to repeat the previous action
epsilon = theta(3);            % Epsilon parameter - controls the exploration of non-greedy actions
lapse = 0;                     % Probability of lapsing in an engaged state - not used in the static model
recover = 1;                   % Probability of returning to an engaged state after lapsing - not used in the static model

T = [1 - recover, lapse; recover, 1 - lapse];  % Transition probability matrix for latent attentional state (not used in the static model)
nA = 2;                                       % Number of actions

choices = data(:, 3);    % Get choices from data - the choices made by the participant
rewards = data(:, 5);    % Get rewards from data - the rewards received by the participant

Q = ones(1, nA) / nA;   % Initial action values [0.5 0.5]
side = 0;              % Side to stick to (1 = A1, -1 = A2)
llh = 0;               % Log-likelihood

% Iterate over trials
for k = 1:length(choices)
    choice = choices(k);   % Current choice
    r = rewards(k);        % Current reward
    % TODO: compute b, a 2x1 vector containing the probabilities of
    % choosing both actions 
    Pleft = 1/(1 + exp(beta*(Q(1)-Q(2)+(stick*side))));
    Pright = 1 - Pleft;
    Pleft = epsilon*(1/nA)+(1-epsilon)*Pleft;
    Pright = epsilon*(1/nA)+(1-epsilon)*Pright;

    b = [Pright; Pleft];
    
    % b(choice) = likelihood of making action "choice"
    
    lt = log(b(choice));   % Log-transformed probability  
    llh = llh + lt;                           % Update log-likelihood

    % TODO: Update Q values
    Q(choice) = Q(choice) + alpha * (r - Q(choice));

    if choice == 1
        side = 1;
    else
        side = -1;
    end
end

nllh = -llh;  % Return the negative log-likelihood

end
