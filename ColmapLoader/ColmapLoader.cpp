// ColmapLoader.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
struct Camera {
	uint32_t id;
	uint32_t model;
	uint64_t width;
	uint64_t height;
	std::vector<float> params;
};
std::vector<Camera> LoadCameras(const std::string& path) {
	std::vector<Camera> cameras;

	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open cameras.bin");
	}
	printf("here!!\n");
	while (!file.eof()) {
		Camera camera;
		printf("here!!\n");

		file.read(reinterpret_cast<char*>(&camera.id), sizeof(camera.id));

		file.read(reinterpret_cast<char*>(&camera.model), sizeof(camera.model));
		file.read(reinterpret_cast<char*>(&camera.width), sizeof(camera.width));
		file.read(reinterpret_cast<char*>(&camera.height), sizeof(camera.height));

		uint64_t num_params;
		file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
		camera.params.resize(num_params);
		file.read(reinterpret_cast<char*>(camera.params.data()), num_params * sizeof(float));

		if (!file.eof()) {
			cameras.push_back(camera);
		}
	}

	file.close();
	return cameras;
}
struct Image {
	uint64_t id;
	std::vector<float> quaternion;  // 4 elements
	std::vector<float> position;    // 3 elements
	uint64_t camera_id;
	std::string name;
	std::vector<std::pair<std::vector<float>, int64_t>> points2D;
};

std::vector<Image> LoadImages(const std::string& path) {
	std::vector<Image> images;

	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open images.bin");
	}

	while (!file.eof()) {
		Image image;
		file.read(reinterpret_cast<char*>(&image.id), sizeof(image.id));

		image.quaternion.resize(4);
		file.read(reinterpret_cast<char*>(image.quaternion.data()), 4 * sizeof(float));

		image.position.resize(3);
		file.read(reinterpret_cast<char*>(image.position.data()), 3 * sizeof(float));

		file.read(reinterpret_cast<char*>(&image.camera_id), sizeof(image.camera_id));

		char name_char;
		while (file.read(&name_char, sizeof(char)) && name_char != '\0') {
			image.name += name_char;
		}

		uint64_t num_points2D;
		file.read(reinterpret_cast<char*>(&num_points2D), sizeof(num_points2D));

		for (uint64_t i = 0; i < num_points2D; ++i) {
			std::vector<float> point2D(2);
			file.read(reinterpret_cast<char*>(point2D.data()), 2 * sizeof(float));

			int64_t point3D_id;
			file.read(reinterpret_cast<char*>(&point3D_id), sizeof(point3D_id));

			image.points2D.emplace_back(point2D, point3D_id);
		}

		if (!file.eof()) {
			images.push_back(image);
		}
	}

	file.close();
	return images;
}
struct Point3D {
	uint64_t id;
	std::vector<float> xyz; // 3 elements
	std::vector<uint8_t> rgb; // 3 elements
	int64_t visibility;
};

std::vector<Point3D> LoadPoints3D(const std::string& path) {
	std::vector<Point3D> points3D;

	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open points3D.bin");
	}

	while (!file.eof()) {
		Point3D point;
		file.read(reinterpret_cast<char*>(&point.id), sizeof(point.id));

		point.xyz.resize(3);
		file.read(reinterpret_cast<char*>(point.xyz.data()), 3 * sizeof(float));

		point.rgb.resize(3);
		file.read(reinterpret_cast<char*>(point.rgb.data()), 3 * sizeof(uint8_t));

		file.read(reinterpret_cast<char*>(&point.visibility), sizeof(point.visibility));

		if (!file.eof()) {
			points3D.push_back(point);
		}
	}

	file.close();
	return points3D;
}

int main()
{
    FILE* fp;
    char foldername[] = "..\\Data\\Sample1\\sparse\\0";
    char filename[1024];
	int numCamera = 16;
	double* mat = new double[numCamera * 9];
	double* vec = new double[numCamera * 3];
	double* focal = new double[numCamera];
	double cDepth;
	double iDepth;

	sprintf_s(filename, "%s\\camears.bin", foldername);
	auto cameras = LoadCameras(filename);
	std::cout << "Loaded " << cameras.size() << " cameras." << std::endl;

	sprintf_s(filename, "%s\\images.bin", foldername);
	auto images = LoadImages(filename);
	std::cout << "Loaded " << images.size() << " images." << std::endl;

	sprintf_s(filename, "%s\\points3D.bin", foldername);
	auto points3D = LoadPoints3D(filename);
	std::cout << "Loaded " << points3D.size() << " 3D points." << std::endl;

	/*for (int i = 0; i < numCamera; i++) {
		printf("%dth Camera\n", i);
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				printf("%f ", mat[k + j * 3 + i * 9]);
			}
			printf("\n");
		}
		for (int j = 0; j < 3; j++) {
			printf("%f ", vec[j + i * 3]);
		}
		printf("%f\n", focal[i]);
	}
	printf("%f, %f\n", cDepth, iDepth);*/
    
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
