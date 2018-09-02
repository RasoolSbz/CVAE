function out = visualize_voxel(voxel)
%face colors:
% % 'red' or 'r'	Red	[1 0 0]
% % 'green' or 'g'	Green	[0 1 0]
% % 'blue' or 'b'	Blue	[0 0 1]
% % 'yellow' or 'y'	Yellow	[1 1 0]
% % 'magenta' or 'm'	Magenta	[1 0 1]
% % 'cyan' or 'c'	Cyan	[0 1 1]
% % 'white' or 'w'	White	[1 1 1]
% % 'black' or 'k
if size(voxel , 2) < 100
    f_color = 'red';
else
    f_color = 'cyan';
end
fig_handle = figure();
set(fig_handle,'Color','white', 'Visible', 'on');
set(gca,'position',[0,0,1,1],'units','normalized');
%     voxel = squeeze(voxels(i,:,:,:) > threshold);
p = patch(isosurface(voxel,0.0005));
set(p,'FaceColor',f_color,'EdgeColor','none');
daspect([1,1,1])
%%daspect(ratio) sets the data aspect ratio for the current axes. The data aspect ratio is the relative length of the data units along the x-axis, y-axis, and z-axis. Specify ratio as a three-element vector of positive values that represent the relative lengths of data units along each axis. For example, [1 2 3] indicates that the length from 0 to 1 along the x-axis is equal to the length from 0 to 2 along the y-axis and 0 to 3 along the z-axis. For equal data unit lengths in all directions, use [1 1 1].
view(3); axis tight
camlight
lighting gouraud;
axis off;