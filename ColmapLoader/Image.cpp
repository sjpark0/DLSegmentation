#include "Image.h"

Image::Image()
{
	m_pPoint2DX = NULL;
	m_pPoint2DY = NULL;
	m_pPoint3D_ID = NULL;
	m_pW2C = NULL;
}
Image::~Image()
{
	if (m_pPoint2DX) {
		delete[]m_pPoint2DX;
		m_pPoint2DX = NULL;
	}
	if (m_pPoint2DY) {
		delete[]m_pPoint2DY;
		m_pPoint2DY = NULL;
	}
	if (m_pPoint3D_ID) {
		delete[]m_pPoint3D_ID;
		m_pPoint3D_ID = NULL;
	}
	if (m_pW2C) {
		delete[]m_pW2C;
		m_pW2C = NULL;
	}
}
void Image::Print()
{
	printf("ImgID : %d\n", m_ImgID);
	printf("Image Filename : %s\n", m_filename);
	printf("CamID : %d\n", m_CamID);
	printf("rw : %f\n", m_rw);
	printf("rx : %f\n", m_rx);
	printf("ry : %f\n", m_ry);
	printf("rz : %f\n", m_rz);
	printf("tx : %f\n", m_tx);
	printf("ty : %f\n", m_ty);
	printf("tz : %f\n", m_tz);
	
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%f ", m_pW2C[j + i * 4]);
		}
		printf("\n");
	}
}
void Image::QuternionToMatrix()
{
	if (m_pW2C == NULL) {
		m_pW2C = new float[16];
		memset(m_pW2C, 0, 16 * sizeof(float));
		m_pW2C[15] = 1;
	}
	m_pW2C[0] = 1 - 2 * (m_ry * m_ry + m_rz * m_rz);
	m_pW2C[1] = 2 * (m_rx * m_ry - m_rz * m_rw);
	m_pW2C[2] = 2 * (m_rx * m_rz + m_ry * m_rw);
	m_pW2C[3] = m_tx;

	m_pW2C[4] = 2 * (m_rx * m_ry + m_rz * m_rw);
	m_pW2C[5] = 1 - 2 * (m_rx * m_rx + m_rz * m_rz);
	m_pW2C[6] = 2 * (m_ry * m_rz - m_rx * m_rw);
	m_pW2C[7] = m_ty;
	m_pW2C[8] = 2 * (m_rx * m_rz - m_ry * m_rw);
	m_pW2C[9] = 2 * (m_ry * m_rz + m_rx * m_rw);
	m_pW2C[10] = 1 - 2 * (m_rx * m_rx + m_ry * m_ry);
	m_pW2C[11] = m_tz;
}

void Image::ProjectPoint(float* pt, float* res)
{
	for (int i = 0; i < 3; i++) {
		res[i] = 0;
		for (int j = 0; j < 3; j++) {
			res[i] += m_pW2C[j + i * 4] * pt[j];
		}
		res[i] += m_pW2C[3 + i * 4];
	}
}