%%¶ÁÈ¡Êý¾Ý
%filename='JAC_EDM_RL_Y-01'
%filename='JAC_EDM_FL_Y-02'
filename='JAC_Front_X_30Hz_5mm-01'
%filename='damper1_DRV_50mm-01'
%filename='JAC_RR_X_40Hz_4mm'
data=csvread(strcat(filename,'.csv'),8,1);
t=data(:,1)';
x=data(:,2)';
y=data(:,3)';

N=length(t);
for i=2:N
    if t(i-1)<5. && t(i)>=5.
        time1=i;
    end
    if t(i-1)<20. && t(i)>=20.
        time2=i;
    end
    if t(i-1)<30. && t(i)>=30.
        time3=i;
    end
    if t(i-1)<55. && t(i)>=55.
        time4=i;
    end
end

t_train=[t(time1:time2-1), t(time3:time4-1)];
t_test=t(time2:time3-1);
x_train=[x(time1:time2-1), x(time3:time4-1)];
x_test=x(time2:time3-1);
y_train=[y(time1:time2-1), y(time3:time4-1)];
y_test=y(time2:time3-1);

poly_pre=polyfit(x_train, y_train, 3);
y_spline=polyval(poly_pre, x_test);
sqrt(sum((y_test-y_spline).^2))/sqrt(sum(y_test.^2))

for i=2:length(t_test)
    if t_test(i-1)<27. && t_test(i)>=27.
        ts1=i;
    end
    if t_test(i-1)<28. && t_test(i)>=28.
        ts2=i;
    end
end

data2=csvread(strcat(filename,'_EDMpredict.csv'));
y_edm=data2(:,1)';
sqrt(sum((y_test-y_edm).^2))/sqrt(sum(y_test.^2))

plot(t_test(ts1:ts2-1), y_test(ts1:ts2-1), 'black');
hold on;
plot(t_test(ts1:ts2-1), y_edm(ts1:ts2-1), 'blue');
plot(t_test(ts1:ts2-1), y_spline(ts1:ts2-1), 'green');
xlabel('Time(secs)');
ylabel('N,N');
title(strrep(strcat(filename, '_LOAD'),'_','\_'));
legend('Experiment', 'EDM','Spline');
hold off;
saveas(gcf, strcat(filename,'_result.png'));

plot(t_test(ts1:ts2-1), zeros(ts2-ts1), 'black');
hold on;
plot(t_test(ts1:ts2-1), y_edm(ts1:ts2-1)-y_test(ts1:ts2-1), 'blue');
plot(t_test(ts1:ts2-1), y_spline(ts1:ts2-1)-y_test(ts1:ts2-1), 'green');
xlabel('Time(secs)');
ylabel('Error(N)');
axis([27 28 -1200 1200]);
title(strrep(strcat(filename, '_error'),'_','\_'));
%legend('0-error','EDM','Spline');
hold off;
saveas(gcf, strcat(filename,'_error.png'));
