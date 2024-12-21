#pragma once
#include <vector>

class Camera
{
public:
	uint32_t m_camID;
	int m_modelID;
	uint64_t m_iWidth;
	uint64_t m_iHeight;
	double   m_fFocal;
	double   m_cx;
	double   m_cy;
	double   m_k;
	//std::vector<double> m_vParams;
public:
	Camera();
	~Camera();
	void Print();
};

