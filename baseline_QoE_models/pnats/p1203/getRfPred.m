function RF_prediction = getRfPred(input, forest)
    %%
    % This is an implementation of random forest regressor for p.1203.3
    % here input is a 14x1 feature vector
    % forest is a 3-d tensor where the third dimension is # of trees
    % each row in a tree represents a node in tree, consisting of 5
    % entries:
    % 1. node ID
    % 2. feature ID
    % 3. feature threshold (or MOS)
    % 4. left child node ID,
    % 5. right child node ID
    nTrees = length(forest);
    predictions = zeros(nTrees, 1);
    %% core model
    for kkk = 1:nTrees
        cur_tree = forest{kkk};
        % start at the 0-th node
        cur_node = 0;
        
        while true
            if cur_tree(cur_node + 1, 2) == -1
                break;
            end
            
            cur_value = input(cur_tree(cur_node + 1, 2) + 1);
            cur_thresh = cur_tree(cur_node + 1, 3);
            if cur_value < cur_thresh
                cur_node = cur_tree(cur_node + 1, 4);
            else
                cur_node = cur_tree(cur_node + 1, 5);
            end
        end
        predictions(kkk) = cur_tree(cur_node + 1, 3);
    end
    
    RF_prediction = mean(predictions);
end