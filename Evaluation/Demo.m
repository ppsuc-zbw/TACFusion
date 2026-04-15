close all;
clear all;
home;

NAME_IMAGE = 'LLVIP533\';
for m=1:300
    
    A=imread(strcat('D:\实验数据和文件\桌面\红外与可见光图像融合定量分析M3FD\M3FD_GRAY\',  'source_vi\',num2str(m),'.bmp'));
    %A=A(1:268,1:360,1);
    B=imread(strcat('D:\实验数据和文件\桌面\红外与可见光图像融合定量分析M3FD\M3FD_GRAY\', 'source_ir\',num2str(m),'.bmp'));
    %B=B(1:268,1:360,1);
    A=double(A);
    B=double(B);
    
    file_path=strcat('D:\实验数据和文件\桌面\红外与可见光图像融合定量分析M3FD\排序文件\',num2str(m),'\');
    file_fused=dir(strcat('D:\实验数据和文件\桌面\红外与可见光图像融合定量分析M3FD\排序文件\',num2str(m),'\*.bmp'));
    
    [k,len]=size(file_fused);
    total=zeros(4,16);
    total=num2str(total);
    results=zeros(1,16);
    n=(m-1)*6+5;
    for i=1:k
        name=file_fused(i).name;
        F=imread(fullfile(file_path,name));
        
        if size(A, 3)>2
            A = rgb2gray(A);
        end

        if size(B, 3)>2
            B = rgb2gray(B);
        end

        if size(F, 3)>2
            F = rgb2gray(F);
        end
        F=double(F);
        grey_level=256;
        [rA,cA]=size(A);
        [rF,cF]=size(F);
        if (rA~=rF)||(cA~=cF)
            %rF=rA;
            %cF=cA;
            F=imresize(F,[rA,cA]);
            F=round(F);%imresize会把整数矩阵转换成小数矩阵，后续处理会报错，需要重整为整数
        end

        F=round(F);
        Criteria=Evaluation(A,B,F,grey_level);
        %Criteria
        %disp(name);
        %disp(class(name));
        %total(i,1:length(name))=name;
        %disp(total);
        %disp(class(total));
        %xlswrite('E:\project\代码\RoadScene对比\RoadScene.xlsx',Criteria,1,'B2');
        method={name};
        xlswrite(strcat('D:\实验数据和文件\桌面\红外与可见光图像融合定量分析\','detect对比结果_M3FD.xlsx'),method,1,strcat('A',num2str(n+i+1)));
        results(i,:)=Criteria;
    end
    xlswrite(strcat('D:\实验数据和文件\桌面\红外与可见光图像融合定量分析\','detect对比结果_M3FD.xlsx'),results,1,strcat('B',num2str(2+n)));
end
