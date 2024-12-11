#include "MaskMatching.h"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

MaskMatching::MaskMatching()
{
    m_ppImage = NULL;
    m_pppMask = NULL;
    m_pppBoundingBox = NULL;
    m_ppObjectCount = NULL;
    m_pOffset = NULL;
    m_pC2W = NULL;
    m_pW2C = NULL;
    m_pFocal = NULL;
    m_pDepth = NULL;
    m_pCorrespondingID = NULL;
    m_pNumObject = NULL;
    m_numDepthPoint = 100;

}
MaskMatching::~MaskMatching()
{
    if (m_ppImage) {
        for (int i = 0; i < m_numCam; i++) {
            delete[]m_ppImage[i];
        }
        delete[]m_ppImage;
        m_ppImage = NULL;
    }
    
    if (m_pppMask) {
        for (int i = 0; i < m_numCam; i++) {
            for (int j = 0; j < m_pNumObject[i]; j++) {
                delete[]m_pppMask[i][j];
            }
            delete[]m_pppMask[i];
        }
        delete[]m_pppMask;
        m_pppMask = NULL;
    }
    if (m_pppBoundingBox) {
        for (int i = 0; i < m_numCam; i++) {
            for (int j = 0; j < m_pNumObject[i]; j++) {
                delete[]m_pppBoundingBox[i][j];
            }
            delete[]m_pppBoundingBox[i];
        }
        delete[]m_pppBoundingBox;
        m_pppBoundingBox = NULL;
    }
    if (m_ppObjectCount) {
        for (int i = 0; i < m_numCam; i++) {
            delete[]m_ppObjectCount[i];
        }
        delete[]m_ppObjectCount;
        m_ppObjectCount = NULL;
    }
    if (m_pNumObject) {
        delete[]m_pNumObject;
    }
    if (m_pOffset) {
        delete[]m_pOffset;
        m_pOffset = NULL;
    }
    if (m_pC2W) {
        delete[]m_pC2W;
        m_pC2W = NULL;
    }
    if (m_pW2C) {
        delete[]m_pW2C;
        m_pW2C = NULL;
    }
    if (m_pFocal) {
        delete[]m_pFocal;
        m_pFocal = NULL;
    }
    if (m_pDepth) {
        delete[]m_pDepth;
        m_pDepth = NULL;
    }
    if (m_pCorrespondingID) {
        delete[]m_pCorrespondingID;
        m_pCorrespondingID = NULL;
    }
}


void MaskMatching::LoadMaskImage(const char* foldername, const char* prefix, int numCam)
{
    m_numCam = numCam;
    m_pNumObject = new int[m_numCam];
    m_pppMask = new unsigned char** [m_numCam];
    m_pppBoundingBox = new int** [m_numCam];
    m_ppObjectCount = new int* [m_numCam];
    m_pOffset = new float[m_numCam * 2];
    
    Mat img;
    char filename[1024];
    for (int i = 0; i < m_numCam; i++) {
        m_pNumObject[i] = 0;
        while (true) {
            sprintf_s(filename, "%s\\masks\\%s_%03d_%02d.png", foldername, prefix, i, m_pNumObject[i]);
            img = imread(filename, IMREAD_GRAYSCALE);
            //printf("%s, %d, %d\n", filename, img.rows, img.cols);
            if (img.rows <= 0 || img.cols <= 0) break;
            m_pNumObject[i]++;
        }
        m_pppMask[i] = new unsigned char* [m_pNumObject[i]];
        m_pppBoundingBox[i] = new int* [m_pNumObject[i]];
        m_ppObjectCount[i] = new int[m_pNumObject[i]];
        for (int j = 0; j < m_pNumObject[i]; j++) {
            sprintf_s(filename, "%s\\masks\\%s_%03d_%02d.png", foldername, prefix, i, j);
            img = imread(filename, IMREAD_GRAYSCALE);
            //printf("%s, %d, %d\n", filename, img.rows, img.cols);
            m_iWidth = img.cols;
            m_iHeight = img.rows;

            m_pppBoundingBox[i][j] = new int[6];
            m_pppBoundingBox[i][j][0] = img.cols;
            m_pppBoundingBox[i][j][1] = img.rows;
            m_pppBoundingBox[i][j][2] = m_pppBoundingBox[i][j][3] = 0;
            m_ppObjectCount[i][j] = 0;
            for (int h = 0; h < img.rows; h++) {
                for (int w = 0; w < img.cols; w++) {
                    if (img.at<uchar>(h, w) == 255) {
                        m_ppObjectCount[i][j]++;
                        m_pppBoundingBox[i][j][0] = min(m_pppBoundingBox[i][j][0], w);
                        m_pppBoundingBox[i][j][1] = min(m_pppBoundingBox[i][j][1], h);
                        m_pppBoundingBox[i][j][2] = max(m_pppBoundingBox[i][j][2], w);
                        m_pppBoundingBox[i][j][3] = max(m_pppBoundingBox[i][j][3], h);
                    }
                }
            }
            m_pppBoundingBox[i][j][4] = m_pppBoundingBox[i][j][2] - m_pppBoundingBox[i][j][0] + 1;
            m_pppBoundingBox[i][j][5] = m_pppBoundingBox[i][j][3] - m_pppBoundingBox[i][j][1] + 1;
            m_pppMask[i][j] = new unsigned char[m_pppBoundingBox[i][j][4] * m_pppBoundingBox[i][j][5]];
            for (int h = 0; h < m_pppBoundingBox[i][j][5]; h++) {
                for (int w = 0; w < m_pppBoundingBox[i][j][4]; w++) {
                    m_pppMask[i][j][w + h * m_pppBoundingBox[i][j][4]] = img.at<uchar>(h + m_pppBoundingBox[i][j][1], w + m_pppBoundingBox[i][j][0]);
                }
            }
        }
    }
         
}
void MaskMatching::LoadMPI(const char* foldername, int numCam)
{
    m_numCam = numCam;
    m_pC2W = new float[m_numCam * 16];
    m_pW2C = new float[m_numCam * 16];
    m_pFocal = new float[m_numCam];
    
    FILE* fp;
    char pFileName[1024];
    int width, height, level;
    float rate;
    
    for (size_t i = 0; i < m_numCam; i++) {
        sprintf_s(pFileName, "%s\\mpis_360\\mpi%02d\\metadata.txt", foldername, i);
        fopen_s(&fp, pFileName, "r");
        if (fp != NULL) {
            fscanf_s(fp, "%d%d%d%f", &height, &width, &level, &m_pFocal[i]);
            for (int j = 0; j < 16; j++) {
                if (j % 4 != 3) {
                    fscanf_s(fp, "%f", &m_pC2W[j + i * 16]);
                }
                else {
                    m_pC2W[j + i * 16] = j < 15 ? 0. : 1.;
                }
            }
            fscanf_s(fp, "%f%f", &m_fInfDepth, &m_fCloseDepth);
            fclose(fp);
            rate = (float)m_iWidth / (float)width;
            m_pFocal[i] = m_pFocal[i] * rate;

        }
    }

    float* matC2W = m_pC2W;
    float* matW2C = m_pW2C;
    for (int c = 0; c < m_numCam; c++) {
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
void MaskMatching::ComputeOffsetByZValue(float* refC2W, float* fltW2C, float refFocal, float fltFocal, float zValue, float& offsetX, float& offsetY, int pointX, int pointY)
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
    int centerX = m_iWidth / 2;
    int centerY = m_iHeight / 2;
    //printf("%f\n", zValue);
    /*for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", refC2W[i + j * 4]);
        }
        printf("\n");
    }*/

    origin[0] = refC2W[12];
    origin[1] = refC2W[13];
    origin[2] = refC2W[14];
    origin[3] = refC2W[15];
    //printf("%f, %f, %f, %f\n", origin[0], origin[1], origin[2], origin[3]);
    //printf("%d, %d, %d, %d, %f, %f\n", ix, iy, centerX, centerY, (ix - centerX) / refFocal, (iy - centerY) / refFocal);
    dir[0] = refC2W[0] * ((ix - centerX) / refFocal) - refC2W[4] * ((iy - centerY) / refFocal) - refC2W[8];
    dir[1] = refC2W[1] * ((ix - centerX) / refFocal) - refC2W[5] * ((iy - centerY) / refFocal) - refC2W[9];
    dir[2] = refC2W[2] * ((ix - centerX) / refFocal) - refC2W[6] * ((iy - centerY) / refFocal) - refC2W[10];
    dir[3] = refC2W[3] * ((ix - centerX) / refFocal) - refC2W[7] * ((iy - centerY) / refFocal) - refC2W[11];
    //printf("%f, %f, %f, %f\n", dir[0], dir[1], dir[2], dir[3]);

    t_origin[0] = fltW2C[0] * origin[0] + fltW2C[4] * origin[1] + fltW2C[8] * origin[2] + fltW2C[12] * origin[3];
    t_origin[1] = fltW2C[1] * origin[0] + fltW2C[5] * origin[1] + fltW2C[9] * origin[2] + fltW2C[13] * origin[3];
    t_origin[2] = fltW2C[2] * origin[0] + fltW2C[6] * origin[1] + fltW2C[10] * origin[2] + fltW2C[14] * origin[3];
    t_origin[3] = fltW2C[3] * origin[0] + fltW2C[7] * origin[1] + fltW2C[11] * origin[2] + fltW2C[15] * origin[3];
    //printf("%f, %f, %f, %f\n", t_origin[0], t_origin[1], t_origin[2], t_origin[3]);

    t_dir[0] = fltW2C[0] * dir[0] + fltW2C[4] * dir[1] + fltW2C[8] * dir[2] + fltW2C[12] * dir[3];
    t_dir[1] = fltW2C[1] * dir[0] + fltW2C[5] * dir[1] + fltW2C[9] * dir[2] + fltW2C[13] * dir[3];
    t_dir[2] = fltW2C[2] * dir[0] + fltW2C[6] * dir[1] + fltW2C[10] * dir[2] + fltW2C[14] * dir[3];
    t_dir[3] = fltW2C[3] * dir[0] + fltW2C[7] * dir[1] + fltW2C[11] * dir[2] + fltW2C[15] * dir[3];
    //printf("%f, %f, %f, %f\n", t_dir[0], t_dir[1], t_dir[2], t_dir[3]);

    tr = (-zValue - t_origin[2]) / t_dir[2];

    trans[0] = t_origin[0] + tr * t_dir[0];
    trans[1] = t_origin[1] + tr * t_dir[1];
    trans[2] = t_origin[2] + tr * t_dir[2];
    //printf("%f, %f, %f, %f\n", tr, trans[0], trans[1], trans[2]);
    offsetX = trans[0] / -trans[2] * fltFocal + centerX - ix;
    offsetY = -(trans[1] / -trans[2] * fltFocal) + centerY - iy;
}
int MaskMatching::ComputeOverlapCount(unsigned char* refMask, unsigned char* fltMask, int* refBox, int* fltBox, float offsetX, float offsetY, int z, int j, int i, int k)
{
    int indexR, indexF;
    int width, height;
    int overlapCnt = 0;
    //printf("%d, %d, %d, %d, %d, %d\n", fltBox[0], fltBox[1], fltBox[2], fltBox[3], fltBox[4], fltBox[5]);

    int fltBox1[4] = { refBox[0] + (int)offsetX - fltBox[0], refBox[1] + (int)offsetY - fltBox[1], refBox[2] + (int)offsetX - fltBox[0], refBox[3] + (int)offsetY - fltBox[1] };
    int fltBox2[4] = { max(0, fltBox1[0]), max(0, fltBox1[1]), min(fltBox[4] - 1, fltBox1[2]), min(fltBox[5] - 1, fltBox1[3]) };
    int refBox1[4] = { fltBox2[0] + fltBox[0] - (int)offsetX - refBox[0], fltBox2[1] + fltBox[1] - (int)offsetY - refBox[1], fltBox2[2] + fltBox[0] - (int)offsetX - refBox[0], fltBox2[3] + fltBox[1] - (int)offsetY - refBox[1] };
    int x1, x2, y1, y2;
    if (fltBox2[3] >= fltBox2[1] && fltBox2[2] >= fltBox2[0]) {
        for (y1 = refBox1[1],y2 = fltBox2[1]; y1 <= refBox1[3]; y1++, y2++) {
            for (x1 = refBox1[0], x2 = fltBox2[0]; x1 <= refBox1[2]; x1++, x2++) {
                indexR = x1 + y1 * refBox[4];
                indexF = x2 + y2 * fltBox[4];
                if (refMask[indexR] == 255 && fltMask[indexF] == 255) {
                    overlapCnt++;
                }

            }
        }
    }
    /*if (j == 3 && k == 35) {
        printf("%d, %d, %d, %d\n", z, j, i, k);
        printf("%d, %d, %d, %d\n", refBox1[0], refBox1[1], refBox1[2], refBox1[3]);
        printf("%d, %d, %d, %d\n", fltBox2[0], fltBox2[1], fltBox2[2], fltBox2[3]);
        printf("%d\n", overlapCnt);
        getchar();
    }*/

    /*for (int y = fltBox1[1]; y <= fltBox1[3]; y++) {
        for (int x = fltBox1[0]; x <= fltBox1[2]; x++) {
            if (x >= 0 && y >= 0 && x < fltBox[4] - 1 && y < fltBox[5] - 1) {
                indexR = (x - fltBox1[0]) + (y - fltBox1[1]) * refBox[4];
                indexF = x + y * fltBox[4];
                if (refMask[indexR] == 255 && fltMask[indexF] == 255) {
                    overlapCnt++;
                }
            }
        }
    }*/
    return overlapCnt;
}

void MaskMatching::ComputeDepth(int refCamID)
{
    if (m_pCorrespondingID) {
        delete[]m_pCorrespondingID;
    }
    if (m_pDepth) {
        delete[]m_pDepth;
    }
    m_pCorrespondingID = new int[m_numCam * m_pNumObject[refCamID]];
    m_pDepth = new float[m_numCam * m_pNumObject[refCamID]];

    float zstep = (1.0f / m_fCloseDepth) - (1.0f / m_fInfDepth);
    float zValueCurrent;
    int overlapCount;
    float sim;
    float*** similarities = new float** [m_numDepthPoint];
    int*** ids = new int** [m_numDepthPoint];
    for (int z = 0; z < m_numDepthPoint; z++) {
        similarities[z] = new float* [m_pNumObject[refCamID]];
        ids[z] = new int* [m_pNumObject[refCamID]];
        zValueCurrent = 1.0f / (zstep * ((float)z / 10.0f) + 1.0 / m_fInfDepth);
        for (int j = 0; j < m_pNumObject[refCamID]; j++) {
            similarities[z][j] = new float[m_numCam];
            ids[z][j] = new int[m_numCam];
            similarities[z][j][refCamID] = 1.0f;
            ids[z][j][refCamID] = j;

            for (int i = 0; i < m_numCam; i++) {
                if (i == refCamID) continue;
                similarities[z][j][i] = 0.0f;
                ids[z][j][i] = -1;
                ComputeOffsetByZValue(&m_pC2W[refCamID * 16], &m_pW2C[i * 16], m_pFocal[refCamID], m_pFocal[i], zValueCurrent, m_pOffset[i * 2], m_pOffset[i * 2 + 1], 0, 0);
                for (int k = 0; k < m_pNumObject[i]; k++) {
                    overlapCount = ComputeOverlapCount(m_pppMask[refCamID][j], m_pppMask[i][k], m_pppBoundingBox[refCamID][j], m_pppBoundingBox[i][k], m_pOffset[i * 2], m_pOffset[i * 2 + 1], z, j, i, k);
                    sim = (float)overlapCount / (float)(m_ppObjectCount[refCamID][j] + m_ppObjectCount[i][k] - overlapCount);

                    if (sim > similarities[z][j][i]) {
                        similarities[z][j][i] = sim;
                        ids[z][j][i] = k;
                    }
                }
                //printf("%d, %d, %d, %d, %f\n", z, j, i, ids[z][j][i], similarities[z][j][i]);
            }
        }
    }
    for (int j = 0; j < m_pNumObject[refCamID]; j++) {
        for (int i = 0; i < m_numCam; i++) {
            if (i == refCamID) continue;
            sim = 0.0f;
            for (int z = 0; z < m_numDepthPoint; z++) {
                if (similarities[z][j][i] > sim) {
                    sim = similarities[z][j][i];
                    m_pCorrespondingID[i + j * m_numCam] = ids[z][j][i];
                    m_pDepth[i + j * m_numCam] = 1.0f / (zstep * ((float)z / 10.0f) + 1.0 / m_fInfDepth);
                }
            }
            printf("%d, %d, %d, %f\n", j, i, m_pCorrespondingID[i + j * m_numCam], m_pDepth[i + j * m_numCam]);
        }
    }
}

void MaskMatching::Display(int refCamID)
{
    Mat mask1;
    Mat mask2;
    for (int j = 0; j < m_pNumObject[refCamID]; j++) {
        for (int i = 0; i < m_numCam; i++) {
            if (i != refCamID) {
                //mask1 = Mat(m_pppBoundingBox[refCamID][j][5], m_pppBoundingBox[refCamID][j][4], CV_8UC1, m_pppMask[refCamID][j]);
                //mask2 = Mat(m_pppBoundingBox[i][m_pCorrespondingID[i + j * m_numCam]][5], m_pppBoundingBox[i][m_pCorrespondingID[i + j * m_numCam]][4], CV_8UC1, m_pppMask[i][m_pCorrespondingID[i + j * m_numCam]]);
                printf("%d, %d, %d\n", j, i, m_pCorrespondingID[i + j * m_numCam]);
                //imshow("Image1", mask1);
                //imshow("Image2", mask2);
                //waitKey(0);
            }
        }
    }
}