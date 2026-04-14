classdef SCIAMOEA < ALGORITHM
% <2025> <multi> <real/integer/label/binary/permutation> <constrained/none>
% Causal Inference Assisted Multi-objective Evoluation Algorithm

%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M); 
            T = ceil(Problem.N/10);
            Population = Problem.Initialization();
            Z = min(Population.objs,[],1);

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);
            
            %% Upper layer graph construction
            [UG,DiG] = ULG_con(Population);
            
            %% Lower layer graph construction
            [LG,LG_adjacencyMatrix] = LLG_con(Population);
            
            %% 构造双层异质图的连接
            [DoubleLayerEdges,DiG] = BG_con(DiG,LG,Population,Problem);
            UG = adjacency(DiG);
            
            %% 因果约束的遗传算法
            % 社区检测
            VV= GCDanon(LG_adjacencyMatrix);
            [Population,Fitness] = SEnvironmentalSelection(Population,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                %% 因果图更新
                if rem(Problem.FE,Problem.N*20) == 0
                    [UG,DiG] = ULG_con(Population);
                    [LG,LG_adjacencyMatrix] = LLG_con(Population);
                    [DoubleLayerEdges,DiG] = BG_con(DiG,LG,Population,Problem);
                    VV= GCDanon(LG_adjacencyMatrix);                                       
                end
                %% 基于因果的遗传算法
                geneticOffspring = [];
                while length(geneticOffspring) < 10
                    Parents = select_P(VV);
                    tempOffspring  = OperatorGACCB(Problem,Population(Parents),DoubleLayerEdges,UG,Parents);
                    geneticOffspring = [geneticOffspring tempOffspring];
                end
                
                %% 基于因果的注意力机制
                attentionOffspring = [];
                pg_ranks = centrality(LG, 'pagerank');
                degree_ranks = centrality(LG, 'degree');
                bn_ranks = centrality(LG, 'betweenness');
                %% 步骤1：筛选LG上与上层图连接最多的50个节点
                % 统计每个下层节点与上层节点的连接数
                connectionCounts = zeros(size(LG_adjacencyMatrix, 1), 1);
                for i = 1:size(DoubleLayerEdges, 1)
                    lowerNode = DoubleLayerEdges(i, 2);
                    connectionCounts(lowerNode) = connectionCounts(lowerNode) + 1;
                end
                
                % 选择连接数最多的50个节点
                scores = 0.6 * connectionCounts + 0.4 * degree_ranks;
                [~, sortedIndices] = sort(scores, 'descend');
                top50Nodes = sortedIndices(1:min(20, length(sortedIndices)));

                %% 步骤2：构建因果关联路径并更新节点特征
                % 假设每个节点的特征就是其对应的数据维度
                nodeFeatures = Population.decs;
                % 定义注意力机制的参数                
                for LA = top50Nodes'
                    % 找到与LA连接的上层节点
                    indices = find(DoubleLayerEdges(:, 2) == LA);
                    upperConnectedNodes = DoubleLayerEdges(indices, 1);
                    %upperConnectedNodes = unique(upperConnectedNodes);
                    relatedLowerNodes = [];
                    % 找到这些上层节点通过有向边链接的下层节点
                    for upperNode = upperConnectedNodes'
                        % 找到从该上层节点出发的有向边
                        outgoingEdges = find(UG(upperNode, :) == 1);
                        for edge = outgoingEdges
                            % 找到与这些有向边对应的下层节点
                            rindices = find(DoubleLayerEdges(:, 1) == edge);
                            newLowerNodes = DoubleLayerEdges(rindices, 2);
                            % 将新找到的下层节点添加到 relatedLowerNodes 中
                            relatedLowerNodes = [relatedLowerNodes; newLowerNodes];
                        end
                    end
                    
                    % 去除重复的下层节点
                    relatedLowerNodes = unique(relatedLowerNodes);
                    
                    % 步骤3：使用注意力机制更新LA节点的特征
                    if ~isempty(relatedLowerNodes)
                        weights = 0.5 * pg_ranks + 0.5 * bn_ranks;
                        rweights = weights(relatedLowerNodes);
            
                        [~, sortedWeightIndices] = sort(rweights, 'descend');
                        top3NodesIndices = sortedWeightIndices(1:min(3, length(sortedWeightIndices)));
                        top3Nodes = relatedLowerNodes(top3NodesIndices);

                        % 计算三条路径的权重距离和
                        pathDistances = zeros(3, 1);
                        for k = 1:3
                            path = shortestpath(LG, top3Nodes(k), LA);
                            if ~isempty(path)
                                for j = 1:length(path) - 1
                                    % 找到当前边在图边表中的索引
                                    edgeIdx = find(( (LG.Edges.EndNodes(:, 1) == path(j) & LG.Edges.EndNodes(:, 2) == path(j + 1)) ...
                                        | (LG.Edges.EndNodes(:, 1) == path(j + 1) & LG.Edges.EndNodes(:, 2) == path(j)) ));
                                    pathDistances(k) = pathDistances(k) + LG.Edges.Weight(edgeIdx);
                                end
                            end
                        end

                        % 归一化路径距离作为权重
                        if pathDistances == 0
                            pathWeights = ones(3, 1) / 3;
                            
                        else
                            pathWeights = pathDistances / sum(pathDistances);
                        end
                        
                        % 步骤3：使用注意力机制更新LA节点的特征
                        if ~isempty(top3Nodes)
                            % 提取这三个节点的特征
                            top3Features = nodeFeatures(top3Nodes,:);
                            % 计算这三个节点的平均特征
                            combinedFeatures = top3Features .* pathWeights;
                            combinedFeatures = sum(combinedFeatures);
                            % 更新LA节点的特征
                            newDec = 0.5 * nodeFeatures(LA,:)+ 0.5* combinedFeatures;
                            % Mutate
                            newDec = PolyMut(newDec, Problem.lower, Problem.upper);
                            newSolution = Problem.Evaluation(newDec); % 评估新解
                            Off = OperatorGA(Problem,[Population(LA) Population(top3Nodes(1))]);
                            attentionOffspring = [attentionOffspring newSolution Off];
                            % P = [LA;top3Nodes];
                            % Z = min(Z,Offspring.obj);
                            % g_old = max(abs((Population(P).objs-Z)./W(P,:)),[],2);
                            % g_new = max(abs((Offspring.obj-Z)./W(P,:)),[],2);
                            % update_idxs = (g_old>=g_new);
                            % Population(P(g_old>=g_new)) = Offspring;

                        end
                    else
                        MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                        Offspring  = OperatorGA(Problem,Population(MatingPool));
                        %[Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);
                        [Population,Fitness] = SEnvironmentalSelection([Population,Offspring],Problem.N);
                    end
                end
                Offspring = [geneticOffspring attentionOffspring];
                %[Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);
                [Population,Fitness] = SEnvironmentalSelection([Population,Offspring],Problem.N);

            end     
        end
    end
end

function [UG,DiG] = ULG_con(Population)
    % 使用全局因果发现算法构造上层图
    alg_name = 'PC';
    data = Population.decs;
    data_type='con';   
    alpha = 0.1;
    [UG,UGTime]=Causal_Learner(alg_name,data,data_type,alpha);
    DiG = digraph(UG);
    % figure('Name','UG');
    % plot(DiG);
end

function [LG,LG_adjacencyMatrix] = LLG_con(Population)
    data = Population.decs;
    % 使用欧式距离构造下层图
    LG_adjacencyMatrix = pdist2(data, data);
    % 邻接矩阵去掉本身连接
    for i = 1:size(LG_adjacencyMatrix, 1)
        LG_adjacencyMatrix(i, i) = 0;
    end
    % 计算前30%的值的阈值
    threshold = prctile(LG_adjacencyMatrix(:), 10);
    % 构建新的邻接矩阵，只保留大于阈值的值
    LG_adjacencyMatrix(LG_adjacencyMatrix < threshold) = 0;
    LG = graph(LG_adjacencyMatrix);
    %figure('Name','LG');
    %plot(LG);
end

function [DoubleLayerEdges,DiG] = BG_con(DiG,LG,Population,Problem)
    data = Population.decs;
    Edges = table2array(DiG.Edges);
    Edges = Edges(:,1:2);
    numEdges = size(Edges, 1);

    % 初始化双层图连接矩阵
    DoubleLayerEdges = [];
    DoubleLayerEdgescopy = cell(1,numEdges);
    prev_end_row = 0;
    while numEdges == 0
        Population = Problem.Initialization();
        [UG,DiG] = ULG_con(Population);
        Edges = table2array(DiG.Edges);
        Edges = Edges(:,1:2);
        numEdges = size(Edges, 1);
    end
    
    for n = 1:numEdges
        Edge = Edges(n,:);
        % 获取原因节点和结果节点的数据
        causeNode = Edge(1);
        effectNode = Edge(2);
        causeNodeData = data(:, causeNode);
        effectNodeData = data(:, effectNode);
        
        % 计算原始互信息
        originalMI = gcmi_cc(causeNodeData, effectNodeData);
        
        numSamples = size(data, 1);
        MIChanges = zeros(numSamples, 1);
        
        % 对原因节点的每个样本进行单独变化测试
        for sampleIdx = 1:numSamples
            % 复制原始数据
            newCauseNodeData = causeNodeData;
            
            % 给当前样本加上一个小的随机扰动
            newCauseValue = newCauseNodeData(sampleIdx) + 0.1 * randn();
            % 确保新值在原因节点的上下限范围内
            newCauseValue = max(Problem.lower(causeNode), min(Problem.upper(causeNode), newCauseValue));
            newCauseNodeData(sampleIdx) = newCauseValue;
            
            % 计算变化后的互信息
            newMI = gcmi_cc(newCauseNodeData, effectNodeData);
            
            % 计算互信息的变化量
            MIChanges(sampleIdx) = abs(newMI - originalMI);
        end
        
        % 选择互信息变化较小的样本索引（这里假设选择变化量最小的前 20% 的样本）
        numSelectedSamples = floor(0.5 * numSamples);
        [~, sortedIndices] = sort(MIChanges);
        selectedSampleIndices = sortedIndices(1:numSelectedSamples);

        % 建立双层图连接
        for idx = 1:length(selectedSampleIndices)
            DoubleLayerEdges = [DoubleLayerEdges; [causeNode selectedSampleIndices(idx)]];
        end

        start_row = prev_end_row + 1;
        end_row = start_row + length(selectedSampleIndices) - 1;
        DoubleLayerEdgescopy{n} = DoubleLayerEdges(start_row:end_row, :);
        prev_end_row = end_row;
    end
    if numEdges == 0
        [UG,DiG] = ULG_con(Population);
    end
end

function Parents = select_P(VV)
    % 统计社区数量
    num_communities = max(VV);
    
    % 初始化每个社区的个体索引
    community_indices = cell(num_communities, 1);
    for i = 1:num_communities
        community_indices{i} = find(VV == i);
    end

    
    % 从每个社区中随机选择一个未被选择的个体
    % 从每个社区中随机选择一个个体及其索引
    if length(community_indices) == 1 || length(community_indices) == 0
        parent1_index = datasample(VV, 1);
        parent2_index = datasample(VV, 1);
    else
        % 选择两个不同的社区
        available_communities = find(cellfun(@(x) ~isempty(x), community_indices));
        chosen_communities = datasample(available_communities, 2, 'Replace', false);
        parent1_index = datasample(community_indices{chosen_communities(1)}, 1);
        parent2_index = datasample(community_indices{chosen_communities(2)}, 1);
    end

    Parents = [parent1_index;parent2_index];
end