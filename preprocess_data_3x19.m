%% Process dataset into mat files %%

% This version is more than x300 faster compared to legacy preprocess_data.m
% It takes less than 10 minutes to generate the same .mat files compared to 2 days and 9 hours before
% It does not require matlab parpool toolbox

clear;
clc;

%% Inputs:
% Locations of raw input files:
us101_1 = 'raw/us101-0750-0805.txt';
us101_2 = 'raw/us101-0805-0820.txt';
us101_3 = 'raw/us101-0820-0835.txt';
i80_1 = 'raw/i80-1600-1615.txt';
i80_2 = 'raw/i80-1700-1715.txt';
i80_3 = 'raw/i80-1715-1730.txt';

% Register here:
% https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj
% And download US-101-LosAngeles-CA.zip, I-80-Emeryville-CA.zip
% Extract vehicle-trajectory-data
%us101_1 = 'raw/us-101/trajectories-0750am-0805am.txt';
%us101_2 = 'raw/us-101/trajectories-0805am-0820am.txt';
%us101_3 = 'raw/us-101/trajectories-0820am-0835am.txt';
%i80_1 = 'raw/i-80/trajectories-0400-0415.txt';
%i80_2 = 'raw/i-80/trajectories-0500-0515.txt';
%i80_3 = 'raw/i-80/trajectories-0515-0530.txt';

%us101_1 = 'raw/us-101/small1.txt';
%us101_2 = 'raw/us-101/small2.txt';
%
%us101_1 = 'raw/us-101/medium1.txt';
%us101_2 = 'raw/us-101/medium2.txt';


%% Fields: 

%{ 
1: Dataset Id
2: Vehicle Id
3: Frame Number
4: Local X
5: Local Y
6: Lane Id
7: Lateral maneuver
8: Longitudinal maneuver
9-125: Neighbor Car Ids at grid location
%}



%% Load data and add dataset id
N = 6 % 6

disp('Loading data...')
traj{1} = load(us101_1);    
traj{1} = single([ones(size(traj{1},1),1),traj{1}]);
traj{2} = load(us101_2);
traj{2} = single([2*ones(size(traj{2},1),1),traj{2}]);
traj{3} = load(us101_3);
traj{3} = single([3*ones(size(traj{3},1),1),traj{3}]);
traj{4} = load(i80_1);    
traj{4} = single([4*ones(size(traj{4},1),1),traj{4}]);
traj{5} = load(i80_2);
traj{5} = single([5*ones(size(traj{5},1),1),traj{5}]);
traj{6} = load(i80_3);
traj{6} = single([6*ones(size(traj{6},1),1),traj{6}]);

for k = 1:N
	traj{k} = traj{k}(:,[1,2,3,6,7,15]);
	if k <=3
		traj{k}(traj{k}(:,6)>=6,6) = 6;
	end
end



%% Parse fields (listed above):
disp('Parsing fields...')
%poolobj = parpool(6);

%prev_vehId = -1
%parfor ii = 1:6
for ii = 1:N

	timelist = unique(traj{ii}(:,3));
	for l = 1:length(timelist)
		timetraj{timelist(l)} = traj{ii}(traj{ii}(:,3) == timelist(l),:);
	end

	vehidlist = unique(traj{ii}(:, 2));
	for l = 1:length(vehidlist)
		vehtrajlist{vehidlist(l)} = traj{ii}(traj{ii}(:,2) == vehidlist(l),:);
	end

	for k = 1:length(traj{ii}(:,1));

		if mod(k,1e5) == 0
			tnow = datestr(now,'HH:MM:SS.FFF');
			fprintf('ii=%d k=%d at %s\n', ii, k, tnow);
		end
		
		time = traj{ii}(k,3);
		%dsId = traj{ii}(k,1);
		%if dsId != ii
		%	disp('ERROR.....')
		%end
		vehId = traj{ii}(k,2);

		%vehtraj = traj{ii}(traj{ii}(:,1)==dsId & traj{ii}(:,2)==vehId,:);

		%if vehId != prev_vehId
		%	vehtraj = traj{ii}(traj{ii}(:,2)==vehId,:);
		%end
		%prev_vehId = vehId;

		vehtraj = vehtrajlist{vehId};

		ind = find(vehtraj(:,3)==time);
		ind = ind(1);
		lane = traj{ii}(k,6);
		
		
		% Get lateral maneuver:
		ub = min(size(vehtraj,1),ind+40);
		lb = max(1, ind-40);
		if vehtraj(ub,6)>vehtraj(ind,6) || vehtraj(ind,6)>vehtraj(lb,6)
			traj{ii}(k,7) = 3;
		elseif vehtraj(ub,6)<vehtraj(ind,6) || vehtraj(ind,6)<vehtraj(lb,6)
			traj{ii}(k,7) = 2;
		else
			traj{ii}(k,7) = 1;
		end
		
		
		% Get longitudinal maneuver:
		ub = min(size(vehtraj,1),ind+50);
		lb = max(1, ind-30);
		if ub==ind || lb ==ind
			traj{ii}(k,8) =1;
		else
			vHist = (vehtraj(ind,5)-vehtraj(lb,5))/(ind-lb);
			vFut = (vehtraj(ub,5)-vehtraj(ind,5))/(ub-ind);
			if vFut/vHist <0.8
				traj{ii}(k,8) =2;
			else
				traj{ii}(k,8) =1;
			end
		end
		
		
		% Get grid locations:
		%frameEgo = traj{ii}(traj{ii}(:,1)==dsId & traj{ii}(:,3)==time & traj{ii}(:,6) == lane,:);
		%frameL = traj{ii}(traj{ii}(:,1)==dsId & traj{ii}(:,3)==time & traj{ii}(:,6) == lane-1,:);
		%frameR = traj{ii}(traj{ii}(:,1)==dsId & traj{ii}(:,3)==time & traj{ii}(:,6) == lane+1,:);

		%frameEgo = traj{ii}(traj{ii}(:,3)==time & traj{ii}(:,6) == lane,:);
		%frameL = traj{ii}(traj{ii}(:,3)==time & traj{ii}(:,6) == lane-1,:);
		%frameR = traj{ii}(traj{ii}(:,3)==time & traj{ii}(:,6) == lane+1,:);
				
		%trajTime = traj{ii}(traj{ii}(:,3)==time,:);
		%frameEgo = trajTime(trajTime(:,6) == lane,:);
		%frameL = trajTime(trajTime(:,6) == lane-1,:);
		%frameR = trajTime(trajTime(:,6) == lane+1,:);

		yref = traj{ii}(k,5);

		frameEgo = timetraj{time}(timetraj{time}(:,6) == lane,:);
		frameL = timetraj{time}(timetraj{time}(:,6) == lane-1,:);
		frameR = timetraj{time}(timetraj{time}(:,6) == lane+1,:);

		frameEgo(:,5) = frameEgo(:,5) - yref;
		frameL(:,5) = frameL(:,5) - yref;
		frameR(:,5) = frameR(:,5) - yref;

		frameL = frameL( abs(frameL(:,5)) < 90, :);
		frameEgo = frameEgo( abs(frameEgo(:,5)) < 90 & frameEgo(:,5) ~= 0, :);
		frameR = frameR( abs(frameR(:,5)) < 90, :);

		if ~isempty(frameL)
			for l = 1:size(frameL,1)
				%y = frameL(l,5)-traj{ii}(k,5);
				%y = frameL(l,5)-yref;
				y = frameL(l,5);
				%if abs(y) <90

					%gridInd = 1+round((y+90)/15);
					gridInd = 1+round((y+90)/10); % 1 up to 19

					traj{ii}(k,8+gridInd) = frameL(l,2);
				%end
			end
		end
		for l = 1:size(frameEgo,1)
			%y = frameEgo(l,5)-traj{ii}(k,5);
			%y = frameEgo(l,5)-yref;
			y = frameEgo(l,5);
			%if abs(y) <90 && y~=0

				% 3x26 gridInd = 14+round((y+90)/15);
				gridInd = 20+round((y+90)/10);

				traj{ii}(k,8+gridInd) = frameEgo(l,2);
			%end
		end
		if ~isempty(frameR)
			for l = 1:size(frameR,1)
				%y = frameR(l,5)-traj{ii}(k,5);
				%y = frameR(l,5)-yref;
				y = frameR(l,5);
				%if abs(y) <90

					%gridInd = 27+round((y+90)/15);
					gridInd = 39+round((y+90)/10);

					traj{ii}(k,8+gridInd) = frameR(l,2);
				%end
			end
		end
		
	end
end

%delete(poolobj);

%% Split train, validation, test
disp('Splitting into train, validation and test sets...')

trajAll = [traj{1};traj{2};traj{3};traj{4};traj{5};traj{6}];
clear traj;

trajTr = [];
trajVal = [];
trajTs = [];
for k = 1:N
	ul1 = round(0.7*max(trajAll(trajAll(:,1)==k,2)));
	ul2 = round(0.8*max(trajAll(trajAll(:,1)==k,2)));
	
	trajTr = [trajTr;trajAll(trajAll(:,1)==k & trajAll(:,2)<=ul1, :)];
	trajVal = [trajVal;trajAll(trajAll(:,1)==k & trajAll(:,2)>ul1 & trajAll(:,2)<=ul2, :)];
	trajTs = [trajTs;trajAll(trajAll(:,1)==k & trajAll(:,2)>ul2, :)];
end

 tracksTr = {};
for k = 1:N
	trajSet = trajTr(trajTr(:,1)==k,:);
	carIds = unique(trajSet(:,2));
	for l = 1:length(carIds)
		vehtrack = trajSet(trajSet(:,2) ==carIds(l),3:5)';
		tracksTr{k,carIds(l)} = vehtrack;
	end
end

tracksVal = {};
for k = 1:N
	trajSet = trajVal(trajVal(:,1)==k,:);
	carIds = unique(trajSet(:,2));
	for l = 1:length(carIds)
		vehtrack = trajSet(trajSet(:,2) ==carIds(l),3:5)';
		tracksVal{k,carIds(l)} = vehtrack;
	end
end

tracksTs = {};
for k = 1:N
	trajSet = trajTs(trajTs(:,1)==k,:);
	carIds = unique(trajSet(:,2));
	for l = 1:length(carIds)
		vehtrack = trajSet(trajSet(:,2) ==carIds(l),3:5)';
		tracksTs{k,carIds(l)} = vehtrack;
	end
end


%% Filter edge cases: 
% Since the model uses 3 sec of trajectory history for prediction, the initial 3 seconds of each trajectory is not used for training/testing

disp('Filtering edge cases...')

indsTr = zeros(size(trajTr,1),1);
for k = 1: size(trajTr,1)
	t = trajTr(k,3);
	if tracksTr{trajTr(k,1),trajTr(k,2)}(1,31) <= t && tracksTr{trajTr(k,1),trajTr(k,2)}(1,end)>t+1
		indsTr(k) = 1;
	end
end
trajTr = trajTr(find(indsTr),:);


indsVal = zeros(size(trajVal,1),1);
for k = 1: size(trajVal,1)
	t = trajVal(k,3);
	if tracksVal{trajVal(k,1),trajVal(k,2)}(1,31) <= t && tracksVal{trajVal(k,1),trajVal(k,2)}(1,end)>t+1
		indsVal(k) = 1;
	end
end
trajVal = trajVal(find(indsVal),:);


indsTs = zeros(size(trajTs,1),1);
for k = 1: size(trajTs,1)
	t = trajTs(k,3);
	if tracksTs{trajTs(k,1),trajTs(k,2)}(1,31) <= t && tracksTs{trajTs(k,1),trajTs(k,2)}(1,end)>t+1
		indsTs(k) = 1;
	end
end
trajTs = trajTs(find(indsTs),:);

%% Save mat files:
disp('Saving mat files...')

traj = trajTr;
tracks = tracksTr;
save('TrainSetX','traj', 'tracks');

traj = trajVal;
tracks = tracksVal;
save('ValSetX','traj','tracks');

traj = trajTs;
tracks = tracksTs;
save('TestSetX','traj','tracks');
