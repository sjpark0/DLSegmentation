#pragma once
#include <vector>
class MaskMatching
{
private:
	unsigned char** m_ppImage;
	unsigned char*** m_pppMask;
	int *m_pNumObject;
	int m_numCam;
	int m_iWidth;
	int m_iHeight;
	int*** m_pppBoundingBox;
	int** m_ppObjectCount;

	float* m_pOffset;
	float* m_pC2W;
	float* m_pW2C;
	float* m_pFocal;

	float m_fCloseDepth;
	float m_fInfDepth;
	int   m_numDepthPoint;
	int   m_iRefCamID;

	float* m_pDepth;
	int* m_pCorrespondingID;
	void ComputeOffsetByZValue(float* refC2W, float* fltW2C, float refFocal, float fltFocal, float zValue, float& offsetX, float& offsetY, int pointX, int pointY);
	int ComputeOverlapCount(unsigned char* refMask, unsigned char* fltMask, int* refBox, int* fltBox, float offsetX, float offsetY, int z, int j, int i, int k);
public:
	MaskMatching();
	~MaskMatching();

	void LoadMaskImage(const char* foldername, const char* prefix, int numCam);
	void LoadMPI(const char* foldername, int numCam);
	void ComputeDepth(int refCamID);
	void Display(int refCamID);
};

