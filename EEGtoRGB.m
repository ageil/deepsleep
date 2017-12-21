cd('PhysioNet_Sleep_EEG')
dirinfo = dir();
sensor = 'fpz';

for j=4:42%42%length(dir)
    fprintf('Subject %d...\n', (j-3))
    %Create directory where to store images
    mkdir(['sub' num2str(idivide(int16(j-4),2)+1) '_n' num2str(mod(j,2)+1) '_img_' sensor])
    
    %Calculate valid intevals
    fileID = fopen([dirinfo(j).name '/info/hyp_start_time.txt']);
    hyp_start = textscan(fileID,'%{HH:mm:ss.SSS}D');
    fclose(fileID);
    
    fileID = fopen([dirinfo(j).name '/info/lights_off_time.txt']);
    lights_off = textscan(fileID,'%{HH:mm:ss}D');
    fclose(fileID);
    
    fileID = fopen([dirinfo(j).name '/info/lights_on_time.txt']);
    lights_on = textscan(fileID,'%{HH:mm:ss.SSS}D');
    fclose(fileID);
    lights_on{1}=lights_on{1}+days(1);
    
    if (lights_off{1} > datetime('00:00:00') && lights_off{1} < datetime('16:00:00'))
        lights_off{1} = lights_off{1} + days(1);
    end
    
    pre_sleep_interval = seconds(lights_off{1}-hyp_start{1});
    sleep_interval = seconds(lights_on{1}-lights_off{1});
    
    %Load eeg sleeping data
    filename = [dirinfo(j).name '/matlab/eeg_' sensor '.mat'];
    load(filename);
%    signal_fpz = signal;
    
%     filename = [dirinfo(j).name '/matlab/eeg_' 'pz' '.mat'];
%     load(filename);
%     signal_pz = signal;
    
%    signal=(signal_fpz + signal_pz)/2;
    
    %Load hynograms
    filename_hypno = [dirinfo(j).name '/matlab/hypnogram.mat'];
    load(filename_hypno)
    hypnogram = hypnogram(pre_sleep_interval/30 + 1:pre_sleep_interval/30 + sleep_interval/30);
    dlmwrite(['sub' num2str(idivide(int16(j-4),2)+1) '_n' num2str(mod(j,2)+1) '_img_' sensor '/labels.txt'],hypnogram)
    
    %Calculate sizes
    num_epochs = length(hypnogram);
    samples_x_epoch = 3000;
    frequency = 100;
    
    %select signal intervals
    signal = signal(pre_sleep_interval*frequency + 1:pre_sleep_interval*frequency + sleep_interval*frequency);
    window = 3; %3; %window = 10;
    movingwin = 0.13;  %0.13; % to get a 224 resolution at the temporal axis; was 0.67
    freq_res = 2;
    tw = (window * freq_res)/2;
    num_tapers = floor(2*tw)-1;
  
    %tapers, pad, Fs, fpass, err, trialave
    params = struct('Fs',frequency,'tapers',[tw num_tapers],'fpass',[0 30]);
    [S,t,f]=mtspecgramc(signal,[window movingwin],params);
    S=S';
    S=flipud(S);
    S=log10(S+1.0);
    
    max_val = 1.0; %0.85*max(S(:)); % mean(S(:))+std(S(:));%0.85*max(S(:)); % 1
    min_val = 0.0; %min(S(:)); %0
    C = colormap(jet(255));  % Get the figure's colormap.
    L = size(C,1);    
    
    % Print the whole night image
    % Scale the matrix to the range of the map.
    S = histeq(S,255);
    currentS=S;
%     currentS(currentS>max_val)=max_val;
%     currentS(currentS<min_val)=min_val;
    
    Gs = round(interp1(linspace(min_val,max_val,L),1:L,currentS));
    H = reshape(C(Gs,:),[size(Gs) 3]); % Make RGB image from scaled.
    imwrite(H,['sub' num2str(idivide(int16(j-4),2)+1) '_n' num2str(mod(j,2)+1) '_img_' sensor '/img_whole_night.png']);
    
    fid=fopen(['sub' num2str(idivide(int16(j-4),2)+1) '_n' num2str(mod(j,2)+1) '_img_' sensor '/params.txt'],'w');
    fprintf(fid, 'window: %d\nmoving_win: %f\nfreq_res: %d\nmax_val: %f\nmin_val: %f\n', window, movingwin,freq_res, max_val, min_val);
    fclose(fid);
    

    for ep=1:num_epochs
        % currentS = S(:,t>30*(ep-1)-60 & t<30*ep+60);
        currentS = S(:,t>30*(ep-1) & t<30*ep);
        
        % Scale the matrix to the range of the map.       
        currentS(currentS>max_val)=max_val;
        currentS(currentS<min_val)=min_val;
        Gs = round(interp1(linspace(min_val,max_val,L),1:L,currentS));
        H = reshape(C(Gs,:),[size(Gs) 3]); % Make RGB image from scaled.
        H = imresize(H,[224 224]);
       % image(H);
        imwrite(H,['sub' num2str(idivide(int16(j-4),2)+1) '_n' num2str(mod(j,2)+1) '_img_' sensor '/img_' num2str(ep) '.png']);

    end
    
%fprintf('%f\t %f\t %f\t %f\n', min(S(:)),max(S(:)),mean(S(:)),std(S(:)))
end
