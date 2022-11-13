%%¶ÁÈ¡Êý¾Ý
filename='JAC_EDM_RL_Y-01'
%filename='JAC_EDM_FL_Y-02'
%filename='JAC_Front_X_30Hz_5mm-01'
%filename='damper1_DRV_50mm-01'
%filename='JAC_RR_X_40Hz_4mm'
data=csvread(strcat(filename,'.csv'),8,1);
t=data(:,1)';
x=data(:,2)';
y=data(:,3)';

subplot(2,1,1);
plot(t, x, 'blue');
xlabel('Time(secs)');
ylabel('Displacement(mm)');
title(strrep(filename,'_','\_'));
subplot(2,1,2);
plot(t, y, 'red');
xlabel('Time(secs)');
ylabel('Load(N)');
title(strrep(filename,'_','\_'));
saveas(gcf, strcat(filename,'_all.png'));