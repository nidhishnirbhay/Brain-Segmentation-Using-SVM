function [sens,spec,acc] =sens(a,b)

%%%a = Segmented image
%%%b = Ground truth 

[r, c] =size(a);
%%%%%%%%% TruePositive Measurement
tcnt = 0;
for i =1:r
    for j=1:c
        
        if a(i,j) == 1 && b(i,j) == 1
            tcnt = tcnt+1;
        end
    end
end

%%%%%FalseNegative Measurement
fcnt = 0;
for i =1:r
    for j=1:c
        
        if a(i,j) ==  0 && b(i,j) ==  1
            fcnt = fcnt+1;
        end
    end
end

%%%%%FalsePositive Measurement
f1cnt = 0;
for i =1:r
    for j=1:c
        
        if a(i,j) ==  1 && b(i,j) ==  0
            f1cnt = f1cnt+1;
        end
    end
end

%%%%%%%%% TrueNegative Measurement
t1cnt = 0;
for i =1:r
    for j=1:c
        
        if a(i,j) == 0 && b(i,j) == 0
            t1cnt = t1cnt+1;
        end
    end
end

%%%%%%% Sensitivity Measurement
sensitivity =  (tcnt / (tcnt + fcnt));
sens = sensitivity*100;

%%%%%%% Specificity Measurement
specificity =  (t1cnt / (t1cnt + f1cnt));
spec = specificity*100;

%%%%%Total Accuracy%%%%%%%%

accuracy = (tcnt+t1cnt)./(tcnt+fcnt+t1cnt+f1cnt);

acc  = accuracy *100;

return;
