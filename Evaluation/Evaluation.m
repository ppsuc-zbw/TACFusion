function Result=Evaluation(grey_matrixA,grey_matrixB,fusion_matrix,grey_level)
% Author:  Qu Xiao-Bo    <quxiaobo [at] xmu.edu.cn>    June 26, 2009
%          Postal address:
% Rom 509, Scientific Research Building # 2,Haiyun Campus, Xiamen University,Xiamen,Fujian, P. R. China, 361005
% Website: http://quxiaobo.go.8866.org

Result=zeros(1,2);
Result(1,1) = mutural_information(grey_matrixA,grey_matrixB,fusion_matrix,grey_level);
Result(1,2) = edge_association(grey_matrixA,grey_matrixB,fusion_matrix);
Result(1,3) =  entropy_fusion(fusion_matrix,grey_level);
% %Result(1,5) = fmi(grey_matrixA,grey_matrixB,fusion_matrix,'none',5);%111111
% % feature : 	Feature extraction method: gradient, edge, dct, wavelet, none (raw pixels) (default: image with no feature extraction)
Result(1,4) =  std2(fusion_matrix);
Result(1,5) =  ssimx(grey_matrixA,grey_matrixB,fusion_matrix);
Result(1,6) =  cc(grey_matrixA,grey_matrixB,fusion_matrix);
Result(1,7) =  SF(fusion_matrix);
Result(1,8) =  VIFF_Public(grey_matrixA,grey_matrixB,fusion_matrix);
Result(1,9) =  (psnr(grey_matrixA,fusion_matrix)+psnr(grey_matrixB,fusion_matrix))*0.5;
Result(1,10) =  avg_gradient(fusion_matrix);
Result(1,11) =  edge_intensity(fusion_matrix);
Result(1,12) =  figure_definition(fusion_matrix);
Result(1,13) =  1/2*relatively_warp(grey_matrixA,fusion_matrix) + 1/2*relatively_warp(grey_matrixB,fusion_matrix);%完全标准差
%Result(1,14) = shannon(fusion_matrix);
%Result(1,15) = space_frequency(fusion_matrix);
[Nabf,SCD,MS_SSIM] = analysis_Reference(fusion_matrix,grey_matrixB,grey_matrixA);%MS-SSIM
Result(1,14) = SCD;
Result(1,15) = Nabf;
Result(1,16) = MS_SSIM;
% Result(1,17) = Qabf;
%fmi
% Result(1,18) = fmi(grey_matrixA,grey_matrixB,fusion_matrix,'gradient',5);%11
% Result(1,19) = fmi(grey_matrixA,grey_matrixB,fusion_matrix,'edge',5);
% Result(1,20) = fmi(grey_matrixA,grey_matrixB,fusion_matrix,'dct',5);
% Result(1,21) = fmi(grey_matrixA,grey_matrixB,fusion_matrix,'wavelet',5);
disp('|| MI || ** || EN || FMI_gradient || SD || SSIM || cc || SF || VIFF || psnr ||avg_gradient||edge_intensity||figure_definition||mutinf||relatively_warp||Nabf||SCD||MS-SSIM||')

