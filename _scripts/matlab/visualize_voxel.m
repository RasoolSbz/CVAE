function out = visualize(voxel)
%face colors:
% % 'red' or 'r'	Red	[1 0 0]
% % 'green' or 'g'	Green	[0 1 0]
% % 'blue' or 'b'	Blue	[0 0 1]
% % 'yellow' or 'y'	Yellow	[1 1 0]
% % 'magenta' or 'm'	Magenta	[1 0 1]
% % 'cyan' or 'c'	Cyan	[0 1 1]
% % 'white' or 'w'	White	[1 1 1]
% % 'black' or 'k
set(fig,'Color','white', 'Visible', 'off');
set(gca,'position',[0,0,1,1],'units','normalized');
%     voxel = squeeze(voxels(i,:,:,:) > threshold);
p = patch(isosurface(voxel,0.05));
set(p,'FaceColor','y','EdgeColor','none');
daspect([1,1,1])
view(3); axis tight
camlight
lighting gouraud;
axis off;