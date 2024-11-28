// ObjectMatching.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
void ChangeOneStepOffset(int width, int height, float* refC2W, float* fltW2C, float refFocal, float fltFocal, float zValue, float& offsetX, float& offsetY, int pointX, int pointY)
{
    float origin[4];
    float dir[4];
    float t_origin[4];
    float t_dir[4];
    float trans[3];
    float tr;
    int ix = pointX;
    int iy = pointY;
    int newX, newY;
    int centerX = width / 2;
    int centerY = height / 2;
    origin[0] = refC2W[12];
    origin[1] = refC2W[13];
    origin[2] = refC2W[14];
    origin[3] = refC2W[15];

    dir[0] = refC2W[0] * ((ix - centerX) / refFocal) - refC2W[4] * ((iy - centerY) / refFocal) - refC2W[8];
    dir[1] = refC2W[1] * ((ix - centerX) / refFocal) - refC2W[5] * ((iy - centerY) / refFocal) - refC2W[9];
    dir[2] = refC2W[2] * ((ix - centerX) / refFocal) - refC2W[6] * ((iy - centerY) / refFocal) - refC2W[10];
    dir[3] = refC2W[3] * ((ix - centerX) / refFocal) - refC2W[7] * ((iy - centerY) / refFocal) - refC2W[11];

    t_origin[0] = fltW2C[0] * origin[0] + fltW2C[4] * origin[1] + fltW2C[8] * origin[2] + fltW2C[12] * origin[3];
    t_origin[1] = fltW2C[1] * origin[0] + fltW2C[5] * origin[1] + fltW2C[9] * origin[2] + fltW2C[13] * origin[3];
    t_origin[2] = fltW2C[2] * origin[0] + fltW2C[6] * origin[1] + fltW2C[10] * origin[2] + fltW2C[14] * origin[3];
    t_origin[3] = fltW2C[3] * origin[0] + fltW2C[7] * origin[1] + fltW2C[11] * origin[2] + fltW2C[15] * origin[3];

    t_dir[0] = fltW2C[0] * dir[0] + fltW2C[4] * dir[1] + fltW2C[8] * dir[2] + fltW2C[12] * dir[3];
    t_dir[1] = fltW2C[1] * dir[0] + fltW2C[5] * dir[1] + fltW2C[9] * dir[2] + fltW2C[13] * dir[3];
    t_dir[2] = fltW2C[2] * dir[0] + fltW2C[6] * dir[1] + fltW2C[10] * dir[2] + fltW2C[14] * dir[3];
    t_dir[3] = fltW2C[3] * dir[0] + fltW2C[7] * dir[1] + fltW2C[11] * dir[2] + fltW2C[15] * dir[3];

    tr = (-zValue - t_origin[2]) / t_dir[2];

    trans[0] = t_origin[0] + tr * t_dir[0];
    trans[1] = t_origin[1] + tr * t_dir[1];
    trans[2] = t_origin[2] + tr * t_dir[2];

    offsetX = trans[0] / -trans[2] * fltFocal + centerX - ix;
    offsetY = -(trans[1] / -trans[2] * fltFocal) + centerY - iy;
}

void MPILoader(const char* foldername, float* c2w, float* w2c, float* cif, int numCam)
{
    FILE* fp;
    char pFileName[1024];
    int width, height, level;
    for (size_t i = 0; i < numCam; i++) {
        sprintf_s(pFileName, "%s\\mpis_360\\mpi%02d\\metadata.txt", foldername, i);
        fopen_s(&fp, pFileName, "r");
        
        fscanf_s(fp, "%d%d%d%f", &width, &height, &level, &cif[2 + i * 3]);
        for (int j = 0; j < 16; j++) {
            if (j % 4 != 3) {
                fscanf_s(fp, "%f", &c2w[j + i * 16]);
            }
            else {
                c2w[j + i * 16] = j < 15 ? 0. : 1.;
            }
        }
        fscanf_s(fp, "%f%f", &cif[1 + i * 3], &cif[i * 3]);
        fclose(fp);
    }

    float* matC2W = c2w;
    float* matW2C = w2c;
    for (int c = 0; c < numCam; c++) {
        matW2C[c * 16] = matC2W[c * 16];
        matW2C[1 + c * 16] = matC2W[4 + c * 16];
        matW2C[2 + c * 16] = matC2W[8 + c * 16];
        matW2C[3 + c * 16] = -0.0f;
        matW2C[4 + c * 16] = matC2W[1 + c * 16];
        matW2C[5 + c * 16] = matC2W[5 + c * 16];
        matW2C[6 + c * 16] = matC2W[9 + c * 16];
        matW2C[7 + c * 16] = 0.0f;
        matW2C[8 + c * 16] = matC2W[2 + c * 16];
        matW2C[9 + c * 16] = matC2W[6 + c * 16];
        matW2C[10 + c * 16] = matC2W[10 + c * 16];
        matW2C[11 + c * 16] = -0.0f;
        matW2C[15 + c * 16] = 1.0f;
        matW2C[12 + c * 16] = -(matC2W[12 + c * 16] * matC2W[c * 16] + matC2W[13 + c * 16] * matC2W[1 + c * 16] + matC2W[14 + c * 16] * matC2W[2 + c * 16]);
        matW2C[13 + c * 16] = -(matC2W[12 + c * 16] * matC2W[4 + c * 16] + matC2W[13 + c * 16] * matC2W[5 + c * 16] + matC2W[14 + c * 16] * matC2W[6 + c * 16]);
        matW2C[14 + c * 16] = -(matC2W[12 + c * 16] * matC2W[8 + c * 16] + matC2W[13 + c * 16] * matC2W[9 + c * 16] + matC2W[14 + c * 16] * matC2W[10 + c * 16]);

    }
    
}
void ComputeOffset(unsigned char* image, int width, int height, int numCam, int* boundingBox, float* c2w, float* w2c, float* cif, int refCamID, float* offsetX, float* offsetY)
{
    float zValueCurrent;
    float zstep = ((1.0 / cif[refCamID * 3]) - (1.0 / cif[1 + refCamID * 3]));
    int cnt;
    int newX;
    int newY;
    float tmpMean;
    float mean;
    
    float optMean;
    float optZ;
    float val1, val2, val3;
    int numAvailableCam;
    optZ = -1.0;
    optMean = 10000000000;
    for (float zValue = 0.0; zValue < 10.0; zValue += 0.1) {
        
        zValueCurrent = 1.0 / (zstep * zValue + 1.0 / cif[1 + refCamID * 3]);
        mean = 0.0f;
        numAvailableCam = 0;
        for (int i = 0; i < numCam; i++) {
            if (i == refCamID) continue;
            ChangeOneStepOffset(width, height, &c2w[refCamID * 16], &c2w[i * 16], cif[2 + refCamID * 3], cif[2 + i * 3], zValueCurrent, offsetX[i], offsetY[i], boundingBox[0], boundingBox[1]);
            cnt = 0;
            tmpMean = 0.0f;
            for (int y = boundingBox[1]; y <= boundingBox[3]; y++) {
                for (int x = boundingBox[0]; x <= boundingBox[2]; x++) {
                    newX = (int)(x + offsetX[i] + 0.5f);
                    newY = (int)(y + offsetY[i] + 0.5f);
                    if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                        val1 = (image[(x + y * width + refCamID * width * height) * 3] - image[(newX + newY * width + i * width * height) * 3]);
                        val2 = (image[(x + y * width + refCamID * width * height) * 3 + 1] - image[(newX + newY * width + i * width * height) * 3 + 1]);
                        val3 = (image[(x + y * width + refCamID * width * height) * 3 + 2] - image[(newX + newY * width + i * width * height) * 3 + 2]);
                        tmpMean += (val1 * val1) + (val2 * val2) + (val3 * val3);
                        cnt++;
                    }
                }
            }
            if (cnt > 0) {
                mean += tmpMean / cnt;
                numAvailableCam++;
            }
        }
        printf("%f, %f\n", zValueCurrent, mean);
        if (numAvailableCam > 0) {
            if (optMean > (mean / numAvailableCam)) {
                optMean = mean / numAvailableCam;
                optZ = zValue;
            }
        }
    }
    printf("%f\n", optZ);
    for (int i = 0; i < numCam; i++) {
        ChangeOneStepOffset(width, height, &c2w[refCamID * 16], &c2w[i * 16], cif[2 + refCamID * 3], cif[2 + i * 3], optZ, offsetX[i], offsetY[i], boundingBox[0], boundingBox[1]);
    }
}
int main()
{
    char foldername[] = "..\\Data\\Sample1";
    char filename[1024];

    int numCam = 16;
    float* pC2W = new float[numCam * 16];
    float* pW2C = new float[numCam * 16];
    float* pCIF = new float[numCam * 3];
    int boundingBox[4];
    MPILoader(foldername, pC2W, pW2C, pCIF, numCam);
    
    sprintf_s(filename, "%s\\images\\000.png", foldername);
    Mat img = imread(filename);
    int width = img.cols;
    int height = img.rows;
    int refCamID = 0;
    int fltCamID = 15;
    unsigned char* pImage = new unsigned char[width * height * numCam * 3];
    for (int c = 0; c < numCam; c++) {
        sprintf_s(filename, "%s\\images\\%03d.png", foldername, c);
        img = imread(filename);
        memcpy(&pImage[c * width * height * 3], img.data, width * height * 3 * sizeof(unsigned char));
    }
    
    img = Mat(height, width, CV_8UC3, &pImage[refCamID * width * height * 3]);
    Mat origin = img.clone();
    namedWindow("Select ROI", cv::WINDOW_NORMAL);
    resizeWindow("Select ROI", width / 4, height / 4);
    moveWindow("Select ROI", 0, 0);
    waitKey(1);
    Rect init_rect = selectROI("Select ROI", img, true, false); // ROI 박스 치기
    waitKey(0);
    destroyWindow("Select ROI");
    
    boundingBox[0] = init_rect.x;
    boundingBox[1] = init_rect.y;
    boundingBox[2] = init_rect.x + init_rect.width;
    boundingBox[3] = init_rect.y + init_rect.height;

    printf("%d, %d, %d, %d\n", boundingBox[0], boundingBox[1], boundingBox[2], boundingBox[3]);

    float* offsetX = new float[numCam];
    float* offsetY = new float[numCam];
    memset(offsetX, 0, numCam * sizeof(float));
    memset(offsetY, 0, numCam * sizeof(float));

    ComputeOffset(pImage, width, height, numCam, boundingBox, pC2W, pW2C, pCIF, refCamID, offsetX, offsetY);
    
    namedWindow("Visualize ROI", cv::WINDOW_NORMAL);
    resizeWindow("Visualize ROI", width / 4, height / 4);
    moveWindow("Visualize ROI", 0, 0);
    for (int i = 0; i < numCam; i++) {
        img = Mat(height, width, CV_8UC3, &pImage[i * width * height * 3]);
        printf("%f, %f\n", offsetX[i], offsetY[i]);
        rectangle(img, Rect((int)(boundingBox[0] + offsetX[i]), (int)(boundingBox[1] + offsetY[i]), init_rect.width, init_rect.height), Scalar(255, 0, 0), 2, 8, 0);
        imshow("Visualize ROI", img);
        getchar();
        waitKey(1);
    }
    /*img = Mat(height, width, CV_8UC3, &pImage[fltCamID * width * height * 3]);
    origin = img.clone();
    namedWindow("Visualize ROI", cv::WINDOW_NORMAL);
    resizeWindow("Visualize ROI", width / 4, height / 4);
    moveWindow("Visualize ROI", 0, 0);
    waitKey(1);
    float zstep = ((1.0 / pCIF[refCamID * 3]) - (1.0 / pCIF[1 + refCamID * 3]));
    float zValueCurrent;
    for (float zValue = 0.0; zValue < 1.0; zValue += 0.1) {
        zValueCurrent = 1.0 / (zstep * zValue + 1.0 / pCIF[1 + refCamID * 3]);

        ChangeOneStepOffset(width, height, &pC2W[refCamID * 16], &pW2C[fltCamID * 16], pCIF[2 + refCamID * 3], pCIF[2 + fltCamID * 3], zValueCurrent, offsetX[fltCamID], offsetY[fltCamID], boundingBox[0], boundingBox[1]);
        //ChangeOneStepOffset(width, height, &pC2W[refCamID * 16], &pW2C[fltCamID * 16], pCIF[2 + refCamID * 3], pCIF[2 + fltCamID * 3], zValueCurrent, offsetX[fltCamID], offsetY[fltCamID], 0, 0);
        printf("%f, %f, %f\n", zValueCurrent, offsetX[fltCamID], offsetY[fltCamID]);
        img = origin.clone();
        rectangle(img, Rect((int)(boundingBox[0] + offsetX[fltCamID]), (int)(boundingBox[1] + offsetY[fltCamID]), init_rect.width, init_rect.height), Scalar(255, 0, 0), 2, 8, 0);
        imshow("Visualize ROI", img);
        getchar();
        waitKey(1);
    }*/
    /*float zValue;
    float offsetX1, offsetX2, offsetY1, offsetY2;
    zValue = 0.0;
    zValueCurrent = 1.0 / (zstep * zValue + 1.0 / pCIF[1 + refCamID * 3]);
    ChangeOneStepOffset(width, height, &pC2W[refCamID * 16], &pW2C[fltCamID * 16], pCIF[2 + refCamID * 3], pCIF[2 + fltCamID * 3], zValueCurrent, offsetX1, offsetY1, width / 2, height / 2);

    
    zValue = 1.0;
    zValueCurrent = 1.0 / (zstep * zValue + 1.0 / pCIF[1 + refCamID * 3]);
    ChangeOneStepOffset(width, height, &pC2W[refCamID * 16], &pW2C[fltCamID * 16], pCIF[2 + refCamID * 3], pCIF[2 + fltCamID * 3], zValueCurrent, offsetX2, offsetY2, width / 2, height / 2);

    printf("%f, %f\n", boundingBox[0] + offsetX1, boundingBox[1] + offsetY1);
    printf("%f, %f\n", boundingBox[0] + offsetX2, boundingBox[1] + offsetY2);

    
    zValue = 0.0;
    zValueCurrent = 1.0 / (zstep * zValue + 1.0 / pCIF[1 + refCamID * 3]);
    ChangeOneStepOffset(width, height, &pC2W[refCamID * 16], &pW2C[fltCamID * 16], pCIF[2 + refCamID * 3], pCIF[2 + fltCamID * 3], zValueCurrent, offsetX1, offsetY1, 0, 0);

    
    zValue = 1.0;
    zValueCurrent = 1.0 / (zstep * zValue + 1.0 / pCIF[1 + refCamID * 3]);
    ChangeOneStepOffset(width, height, &pC2W[refCamID * 16], &pW2C[fltCamID * 16], pCIF[2 + refCamID * 3], pCIF[2 + fltCamID * 3], zValueCurrent, offsetX2, offsetY2, 0, 0);

    printf("%f, %f\n", boundingBox[0] + offsetX1, boundingBox[1] + offsetY1);
    printf("%f, %f\n", boundingBox[0] + offsetX2, boundingBox[1] + offsetY2);*/

    //rectangle(img, Rect((int)(boundingBox[0] + offsetX[fltCamID]), (int)(boundingBox[1] + offsetY[fltCamID]), init_rect.width, init_rect.height), Scalar(255, 0, 0), 2, 8, 0);
    //imshow("Visualize ROI", img);
    //waitKey(0);
    destroyWindow("Visualize ROI");
    delete[]pC2W;
    delete[]pW2C;
    delete[]pCIF;
    delete[]offsetX;
    delete[]offsetY;
    delete[]pImage;
    /*for (int c = 0; c < numCam; c++) {
        printf("%dth camera\n", c);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                printf("%.3f ", pC2W[j + i * 4 + c * 16]);
            }
            printf("\n");
        }
    }

    for (int c = 0; c < numCam; c++) {
        printf("%dth camera\n", c);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                printf("%.3f ", pW2C[j + i * 4 + c * 16]);
            }
            printf("\n");
        }
    }*/
}

// 프로그램 실행: <Ctrl+F5> 또는 [디버그] > [디버깅하지 않고 시작] 메뉴
// 프로그램 디버그: <F5> 키 또는 [디버그] > [디버깅 시작] 메뉴

// 시작을 위한 팁: 
//   1. [솔루션 탐색기] 창을 사용하여 파일을 추가/관리합니다.
//   2. [팀 탐색기] 창을 사용하여 소스 제어에 연결합니다.
//   3. [출력] 창을 사용하여 빌드 출력 및 기타 메시지를 확인합니다.
//   4. [오류 목록] 창을 사용하여 오류를 봅니다.
//   5. [프로젝트] > [새 항목 추가]로 이동하여 새 코드 파일을 만들거나, [프로젝트] > [기존 항목 추가]로 이동하여 기존 코드 파일을 프로젝트에 추가합니다.
//   6. 나중에 이 프로젝트를 다시 열려면 [파일] > [열기] > [프로젝트]로 이동하고 .sln 파일을 선택합니다.
