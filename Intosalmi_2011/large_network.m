% 1 PKCi
% 2 CaPKC
% 3 DAGPKC
% 4 DAGCaPKC
% 5 AADAGPKC
% 6 PKCbasal*
% 7 AAPKC*
% 8 CaPKCmemb*
% 9 AACaPKC*
% 10 DAGPKCmemb*
% 11 AADAGPKC*
% 12 AA
% 13 DAGCaPLA2* 
% 14 PIP2CaPLA2* 
% 15 PIP2PLA2*  
% 16 PLA2*  
% 17 CaPLA2*  
% 18 PLA2Ca*  
% 19 PLA2cytosolic 
% 20 kenz1
% 21 kenz2
% 22 kenz3
% 23 kenz4
% 24 kenz5
% 25 MAPK
% 26 MAPKtyr
% 27 MAPKK
% 28 MAPKK*
% 29 MAPKKser
% 30 RafGTPRas*
% 31 MAPKKthr
% 32 MAPKKtyr
% 33 RafGTPRas*1
% 34 RafGTPRas*2
% 35 MAPK*
% 36 MAPK*complex
% 37 cRaf1
% 38 cRaf1*
% 39 cRaf1**
% 40 PKCactraf
% 41 MAPK*feedback
% 42 craf**deph
% 43 PPhosphatase2A
% 44 GTPRas
% 45 MAPKKdephser
% 46 MAPKKdeph
% 47 MKP1thrdeph
% 48 MKP1tyrdeph
% 49 MKP1
% 50 crafdeph
% 51 PLC
% 52 CaPLC
% 53 Gq (G*GTP)
% 54 GqCaPLC
% 55 GqPLC
% 56 G*GDP
% 57 IP3
% 58 CaPLCcomplex
% 59 GqCaPLCbcomplex
% 60 Ca
% 61 DAG

% PKC active is denoted by aPKC and it is the sum of species 6-11.

% Constant inputs (to be taken into account when defining the reaction rates)
APC         = 30e-6;
tempPIP2    = 2.5e-6;
PIP2        = 2.5e-6;

names = {
    'PKCi', 'CaPKC','DAGPKC','DAGCaPKC','AADAGPKC','PKCbasal*',...
    'AAPKC*','CaPKCmemb*','AACaPKC*','DAGPKCmemb*','AADAGPKC*',...
    'AA','DAGCaPLA2*', 'PIP2CaPLA2*','PIP2PLA2*','PLA2*','CaPLA2*',...
    'PLA2Ca*','PLA2cytosolic','kenz1','kenz2','kenz3','kenz4','kenz5',...
    'MAPK','MAPKtyr','MAPKK','MAPKK*','MAPKKser','RafGTPRas*','MAPKKthr',...
    'MAPKKtyr','RafGTPRas*1','RafGTPRas*2','MAPK*','MAPK*complex','cRaf1',...
    'cRaf1*','cRaf1**','PKCactraf','MAPK*feedback','craf**deph',...
    'PPhosphatase2A','GTPRas','MAPKKdephser','MAPKKdeph','MKP1thrdeph',...
    'MKP1tyrdeph','MKP1','crafdeph','PLC','CaPLC','Gq (G*GTP)','GqCaPLC',...
    'GqPLC','G*GDP','IP3','CaPLCcomplex','GqCaPLCbcomplex','Ca','DAG'};

% stoichiometric matrix
load S;

% rate constants
load k;

for i = 1:size(S,2)
    c = S(:,i);
    reactants = find(c == -1);
    products = find(c == 1);
    % display state vector indices
    %disp(['R' num2str(i) ': ' num2str(reactants') ' ----> ' num2str(products')]);
    % names
    if(length(reactants) == 1 && length(products) == 1)
        disp(['R' num2str(i) ': ' names{reactants} ' ----> ' names{products} ', Rate constant: ' num2str(k(i))]);
    elseif(length(reactants) == 2 && length(products) == 1)
        disp(['R' num2str(i) ': ' names{reactants(1)} ' + ' names{reactants(2)} ' ----> ' names{products} ', Rate constant: ' num2str(k(i))]);
    elseif(length(reactants) == 1 && length(products) == 2)
        disp(['R' num2str(i) ': ' names{reactants(1)} ' ----> ' names{products(1)} ' + ' names{products(2)} ', Rate constant: ' num2str(k(i))]);
    elseif(length(reactants) == 1 && length(products) == 0)
        disp(['R' num2str(i) ': ' names{reactants(1)} ' ----> ', 'Rate constant: ' num2str(k(i))]);
    elseif(length(reactants) == 0 && length(products) == 1)
        disp(['R' num2str(i) ':  ----> ' names{products(1)}, ', Rate constant: ' num2str(k(i))]);
    end
end
