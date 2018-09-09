data_directory_vol = 'E:\thesis_phd\msSabzi\CVAE_inverse_rendering\volumetric_data\ModelNet10';
folders_common_between_off_and_vol = dir(data_directory_vol);

data_directory_off = 'E:\thesis_phd\msSabzi\ModelNet10';
training_count = 1;
test_count = 1;
for i = 3 : length(folders_common_between_off_and_vol)
    training_vol_files_path = [data_directory_vol,'\' , folders_common_between_off_and_vol(i).name , '\32\train\'];
    test_vol_files_path = [data_directory_vol ,'\', folders_common_between_off_and_vol(i).name , '\32\test\'];
    
    training_off_files_path = [data_directory_off,'\' , folders_common_between_off_and_vol(i).name , '\train\'];
    test_off_files_path = [data_directory_off ,'\', folders_common_between_off_and_vol(i).name , '\test\'];
   
    
    test_files = dir(test_vol_files_path);
    training_files = dir(training_vol_files_path);
    
    for j = 3 : length(training_files)
        
        fprintf('[INFO] reading training voxel %d of %d from folder %s \n' , j ,length(training_data_images),[folders_common_between_off_and_vol(i).name , '/32/train/'] );
        tmp = importdata([training_vol_files_path , training_files(j).name]);
%         figure; visualize_voxel(tmp);
        
        
        fprintf('[INFO] reading training off %d of %d from folder %s \n' , j ,length(training_data_images),[folders_common_between_off_and_vol(i).name , '/train/'] );
        [vertex,face] = read_off([training_off_files_path, training_files(j).name]);
        
        for g = 1 : 8 % 8 view point for eah sample
            figure;plot3dFace_shape(struct('shape' , vertex') , face');
            view(g*(180/8) , 45);
            axis off;
            colorbar off
            saveas(gcf,'temp.png')
            input_image = double(rgb2gray(imread('temp.png')));
            input_image = input_image / 255;
            [tx , ty] = find( input_image ~= 1);
            cropped_input_image = input_image;
            cropped_input_image(: , 1:min(ty)) = [];
            cropped_input_image(: , max(ty)-min(ty):end) = [];
            cropped_input_image(1:min(tx), :) = [];
            cropped_input_image(max(tx)-min(tx):end, :) = [];
            cropped_input_image = imresize(cropped_input_image , [32 , 32]);
            cropped_input_image = 1 - cropped_input_image;
            figure;imshow(cropped_input_image);
            modelNet_training_data_X_2d_32(: , : , training_count) = cropped_input_image;
            modelnet_training_data_voxels_32(:,:,:,training_count) = tmp;
            training_count = training_count + 1;
            close all;
        end
        
        
        
%         training_count = training_count + 1;
    end
    for j = 3 : length(test_files)
        fprintf('[INFO] reading test voxel %d of %d from folder %s ' , j ,length(test_files),[folders_common_between_off_and_vol(i).name , '/32/test/'] );
        tmp = importdata([test_vol_files , test_vol_files(j).name]);
        modelnet_test_data_voxels(:,:,:,test_count) = tmp;
        
        
        fprintf('[INFO] reading test off %d of %d from folder %s \n' , j ,length(test_files),[folders_common_between_off_and_vol(i).name , '/test/'] );
        [vertex,face] = read_off([test_off_files_path, test_files(j).name]);
        
        for g = 1 : 8 % 8 view point for eah sample
            figure;plot3dFace_shape(struct('shape' , vertex') , face');
            view(g*(180/8) , 45);
            axis off;
            colorbar off
            saveas(gcf,'temp.png')
            input_image = double(rgb2gray(imread('temp.png')));
            input_image = input_image / 255;
            [tx , ty] = find( input_image ~= 1);
            cropped_input_image = input_image;
            cropped_input_image(: , 1:min(ty)) = [];
            cropped_input_image(: , max(ty)-min(ty):end) = [];
            cropped_input_image(1:min(tx), :) = [];
            cropped_input_image(max(tx)-min(tx):end, :) = [];
            cropped_input_image = imresize(cropped_input_image , [32 , 32]);
            cropped_input_image = 1 - cropped_input_image;
            figure;imshow(cropped_input_image);
            modelNet_test_data_X_2d_32(: , : , test_count) = cropped_input_image;
            modelnet_test_data_voxels_32(:,:,:,test_count) = tmp;
            test_count = test_count + 1;
            close all;
        end
    end
end
