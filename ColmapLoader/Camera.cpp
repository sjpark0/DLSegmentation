#include "Camera.h"

Camera::Camera()
{

}
Camera::~Camera()
{

}
void Camera::Print()
{
	printf("CamID : %d\n", m_camID);
	printf("modelID : %d\n", m_modelID);

	printf("Width : %d\n", m_iWidth);
	printf("Height : %d\n", m_iHeight);

	printf("Focal X : %f\n", m_fFocal);
	
	printf("Principal Point X : %f\n", m_cx);
	printf("Printipal Point Y : %f\n", m_cy);

	printf("K : %f\n", m_k);

}