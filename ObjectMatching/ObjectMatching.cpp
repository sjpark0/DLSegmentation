// ObjectMatching.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
void ComputeInitialResult(const unsigned char* image, ImgDimension dims, int numChannelImage, unsigned char* object, ImgDimension objDims, int numChannelObject, int* startPixel, int* endPixel, int refCamID, const float* matRotation, const float* matTranslation, const float* matIntrinsic)
{	
	m_pImage = image;
	m_pObject = object;
	m_pDim = dims;
	m_pObjDim = objDims;

	m_pStartPixel = startPixel;
	m_pEndPixel = endPixel;
	m_iRefCamID = refCamID;
	//m_pRotation = matRotation;
	//m_pTranslation = matTranslation;
	//m_pIntrinsic = matIntrinsic;
	m_numChannelImage = numChannelImage;
	m_numChannelObject = numChannelObject;


	m_ptTranslation = new Point2f[m_pDim.numCam];
	for (int i = 0; i < m_pDim.numCam; i++) {
		m_ptTranslation[i].x = m_ptTranslation[i].y = 0;
		m_pStartPixel[i * 2] = m_pStartPixel[m_iRefCamID * 2];
		m_pStartPixel[i * 2 + 1] = m_pStartPixel[m_iRefCamID * 2 + 1];
		m_pEndPixel[i * 2] = m_pEndPixel[m_iRefCamID * 2];
		m_pEndPixel[i * 2 + 1] = m_pEndPixel[m_iRefCamID * 2 + 1];
	}
	int index1;
	int index2;
	// 뎁스계산 코드
	int cid = m_iRefCamID + 1;

	const float* pIntrinsicDst = &matIntrinsic[m_iRefCamID * 4];

	const float* pIntrinsicSrc = &matIntrinsic[cid * 4];
	const float* pRotation = &matRotation[cid * 9];
	const float* pTranslation = &matTranslation[cid * 3];

	float fDirX, fDirY;
	float fTxOrig3DX, fTxOrig3DY, fTxOrig3DZ;
	float xTx, yTx, zTx;
	int x, y;
	int cnt;
	float sum;
	float zValue;
	float optSum;
	int optD;

	for (int d = 0; d < 100; d += 10) {
		cnt = 0;
		sum = 0;
		zValue = d * 0.007 / 100.0;
		for (int i = m_pStartPixel[m_iRefCamID * 2 + 1]; i <= m_pEndPixel[m_iRefCamID * 2 + 1]; i++) {
			for (int j = m_pStartPixel[m_iRefCamID * 2]; j <= m_pEndPixel[m_iRefCamID * 2]; j++) {
				fDirX = ((float)j - pIntrinsicDst[2]) / pIntrinsicDst[0];
				fDirY = ((float)i - pIntrinsicDst[3]) / pIntrinsicDst[1];
				fTxOrig3DX = pRotation[0] * fDirX + pRotation[1] * fDirY + pRotation[2];
				fTxOrig3DY = pRotation[3] * fDirX + pRotation[4] * fDirY + pRotation[5];
				fTxOrig3DZ = pRotation[6] * fDirX + pRotation[7] * fDirY + pRotation[8];

				xTx = fTxOrig3DX + zValue * pTranslation[0];
				yTx = fTxOrig3DY + zValue * pTranslation[1];
				zTx = fTxOrig3DZ + zValue * pTranslation[2];

				x = (int)((xTx / zTx * pIntrinsicSrc[0]) + pIntrinsicSrc[2] + 0.5f);
				y = (int)((yTx / zTx * pIntrinsicSrc[1]) + pIntrinsicSrc[3] + 0.5f);
				if (x >= 0 && x < (int)m_pDim.width && y >= 0 && y < (int)m_pDim.height) {
					index1 = j + i * m_pDim.width + m_iRefCamID * m_pDim.height * m_pDim.width;
					index2 = x + y * m_pDim.width + cid * m_pDim.height * m_pDim.width;
					sum += abs(m_pImage[index1 * numChannelImage] - m_pImage[index2 * numChannelImage]) + abs(m_pImage[index1 * numChannelImage + 1] - m_pImage[index2 * numChannelImage + 1]) + abs(m_pImage[index1 * numChannelImage + 2] - m_pImage[index2 * numChannelImage + 2]);
					cnt++;
				}
			}
		}
		sum = sum / cnt;
		if (d == 0) {
			optSum = sum;
			optD = d;
		}
		else {
			if (optSum > sum) {
				optSum = sum;
				optD = d;
			}
		}
	}
	int startD = optD - 10;
	int endD = optD + 10;
	for (int d = startD; d <= endD; d++) {
		cnt = 0;
		sum = 0;
		zValue = d * 0.007 / 100.0;
		for (int i = m_pStartPixel[m_iRefCamID * 2 + 1]; i <= m_pEndPixel[m_iRefCamID * 2 + 1]; i++) {
			for (int j = m_pStartPixel[m_iRefCamID * 2]; j <= m_pEndPixel[m_iRefCamID * 2]; j++) {
				fDirX = ((float)j - pIntrinsicDst[2]) / pIntrinsicDst[0];
				fDirY = ((float)i - pIntrinsicDst[3]) / pIntrinsicDst[1];
				fTxOrig3DX = pRotation[0] * fDirX + pRotation[1] * fDirY + pRotation[2];
				fTxOrig3DY = pRotation[3] * fDirX + pRotation[4] * fDirY + pRotation[5];
				fTxOrig3DZ = pRotation[6] * fDirX + pRotation[7] * fDirY + pRotation[8];

				xTx = fTxOrig3DX + zValue * pTranslation[0];
				yTx = fTxOrig3DY + zValue * pTranslation[1];
				zTx = fTxOrig3DZ + zValue * pTranslation[2];

				x = (int)((xTx / zTx * pIntrinsicSrc[0]) + pIntrinsicSrc[2] + 0.5f);
				y = (int)((yTx / zTx * pIntrinsicSrc[1]) + pIntrinsicSrc[3] + 0.5f);
				if (x >= 0 && x < (int)m_pDim.width && y >= 0 && y < (int)m_pDim.height) {
					index1 = j + i * m_pDim.width + m_iRefCamID * m_pDim.height * m_pDim.width;
					index2 = x + y * m_pDim.width + cid * m_pDim.height * m_pDim.width;
					sum += abs(m_pImage[index1 * numChannelImage] - m_pImage[index2 * numChannelImage]) + abs(m_pImage[index1 * numChannelImage + 1] - m_pImage[index2 * numChannelImage + 1]) + abs(m_pImage[index1 * numChannelImage + 2] - m_pImage[index2 * numChannelImage + 2]);
					cnt++;
				}
			}
		}
		sum = sum / cnt;
		//printf("%d, %d, %f\n", d, cnt, sum);
		if (d == 0) {
			optSum = sum;
			optD = d;
		}
		else {
			if (optSum > sum) {
				optSum = sum;
				optD = d;
			}
		}
	}
	//optD = 28;
	zValue = optD * 0.007 / 100.0;
	for (int i = 0; i < m_pDim.numCam; i++) {
		if (i != m_iRefCamID) {
			pIntrinsicSrc = &matIntrinsic[i * 4];
			pRotation = &matRotation[i * 9];
			pTranslation = &matTranslation[i * 3];

			fDirX = ((float)m_pStartPixel[m_iRefCamID * 2] - pIntrinsicDst[2]) / pIntrinsicDst[0];
			fDirY = ((float)m_pStartPixel[m_iRefCamID * 2 + 1] - pIntrinsicDst[3]) / pIntrinsicDst[1];
			fTxOrig3DX = pRotation[0] * fDirX + pRotation[1] * fDirY + pRotation[2];
			fTxOrig3DY = pRotation[3] * fDirX + pRotation[4] * fDirY + pRotation[5];
			fTxOrig3DZ = pRotation[6] * fDirX + pRotation[7] * fDirY + pRotation[8];

			xTx = fTxOrig3DX + zValue * pTranslation[0];
			yTx = fTxOrig3DY + zValue * pTranslation[1];
			zTx = fTxOrig3DZ + zValue * pTranslation[2];

			m_pStartPixel[i * 2] = MAX(0, (int)((xTx / zTx * pIntrinsicSrc[0]) + pIntrinsicSrc[2] + 0.5f));
			m_pStartPixel[i * 2 + 1] = MAX(0, (int)((yTx / zTx * pIntrinsicSrc[1]) + pIntrinsicSrc[3] + 0.5f));

			m_pEndPixel[i * 2] = MIN(m_pDim.width - 1, m_pStartPixel[i * 2] + m_pObjDim.width - 1);
			m_pEndPixel[i * 2 + 1] = MIN(m_pDim.height - 1, m_pStartPixel[i * 2 + 1] + m_pObjDim.height - 1);
			/*fDirX = ((float)m_pEndPixel[m_iRefCamID * 2] - pIntrinsicDst[2]) / pIntrinsicDst[0];
			fDirY = ((float)m_pEndPixel[m_iRefCamID * 2 + 1] - pIntrinsicDst[3]) / pIntrinsicDst[1];
			fTxOrig3DX = pRotation[0] * fDirX + pRotation[1] * fDirY + pRotation[2];
			fTxOrig3DY = pRotation[3] * fDirX + pRotation[4] * fDirY + pRotation[5];
			fTxOrig3DZ = pRotation[6] * fDirX + pRotation[7] * fDirY + pRotation[8];

			xTx = fTxOrig3DX + zValue * pTranslation[0];
			yTx = fTxOrig3DY + zValue * pTranslation[1];
			zTx = fTxOrig3DZ + zValue * pTranslation[2];

			m_pEndPixel[i * 2] = MIN(m_pDim.width - 1, (int)((xTx / zTx * pIntrinsicSrc[0]) + pIntrinsicSrc[2] + 0.5f));
			m_pEndPixel[i * 2 + 1] = MIN(m_pDim.height - 1, (int)((yTx / zTx * pIntrinsicSrc[1]) + pIntrinsicSrc[3] + 0.5f));*/
		}
	}

	//


	/*for (int i = 0; i < m_pDim[0] * m_pDim[1]; i++) {
		for (int j = m_pStartPixel[i * 2 + 1]; j < m_pStartPixel[i * 2 + 1] + m_pObjDim[2]; j++) {
			for (int k = m_pStartPixel[i * 2]; k < m_pStartPixel[i * 2] + m_pObjDim[3]; k++) {
				index1 = k + j * m_pDim[3] + i * m_pDim[2] * m_pDim[3];
				index2 = (k - m_pStartPixel[i * 2]) + (j - m_pStartPixel[i * 2 + 1]) * m_pObjDim[3] + i * m_pObjDim[2] * m_pObjDim[3];

				for (int l = 0; l < m_numChannelImage - 1; l++) {
					m_pObject[index2 * m_numChannelObject + l] = m_pImage[index1 * m_numChannelImage + l];
				}
			}
		}
	}*/

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
int main()
{
    char foldername[] = "..\\Data\\Sample1";
    int numCam = 16;
    float* pC2W = new float[numCam * 16];
    float* pW2C = new float[numCam * 16];
    float* pCIF = new float[numCam * 3];
    MPILoader(foldername, pC2W, pW2C, pCIF, numCam);
    for (int c = 0; c < numCam; c++) {
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
    }
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
