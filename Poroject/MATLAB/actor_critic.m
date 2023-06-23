% load pretrained actor
trajectory = readmatrix('test.csv');


observationInfo = rlNumericSpec([2 1]);
actionInfo = rlNumericSpec([2, 1], "UpperLimit", 1, "LowerLimit", -1);



mdl = 'AOM';
agentBlk = [mdl '/RL Agent'];

initial_condition = readmatrix('x0.csv');
env = rlSimulinkEnv(mdl,agentBlk,observationInfo,actionInfo);
env.ResetFcn = @(in)setVariable(in,['dot_X0', 'X0'] ,...
    [initial_condition(4:5), initial_condition(1:2)],'Workspace',mdl);
