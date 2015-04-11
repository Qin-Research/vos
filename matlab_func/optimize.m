load energy.mat
addpath(genpath('external/LSA/'))
[energy.UE, energy.subPE, energy.superPE, energy.constTerm] = reparamEnergy(UE, PE);

[currLabeling, E, iteration] = LSA_TR(energy);

labels = currLabeling-1;
save('labeling.mat', 'labels');
