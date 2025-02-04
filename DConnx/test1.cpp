#include <vector.h>
#include <windows.h>
#include <tensor.h>
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <iostream>
#include <Infer.h>
#include <Type.h>

// 读取 ONNX 文件到 std::vector<char>
std::vector<char> LoadONNXModel1(const std::string& model_path) {
	std::ifstream file(model_path, std::ios::binary | std::ios::ate);
	if (!file) {
		throw std::runtime_error("Failed to open ONNX model file: " + model_path);
	}

	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	if (!file.read(buffer.data(), size)) {
		throw std::runtime_error("Failed to read ONNX model file: " + model_path);
	}

	return buffer;
}

void test1() {

	DC::Int64 a = std::vector<int64_t>{1,2,3,4};
	DC::Int32 b = a;
	DC::Float c = a;
	DC::Bool d = a;
	c = std::vector<float>{ 1.25,36.3,8,77.7846 };
	a = c;

	// 加载 ONNX 模型
	const std::string model_path = "C:/Users/东风谷早苗/Desktop/DiffSinger_Yoko/yoko.onnx";
	std::vector<char> onnxModelData = LoadONNXModel1(model_path);
	DC::Infer worker1(onnxModelData);


	worker1.config()
		.defaultValue<float>("depth", {}, 0.1)
		.defaultValue<int64_t>("steps", {}, 5)
	;

	std::vector<DC::Tensor> testTensor = {
		DC::Tensor("tokens", "int64", {1, -1}),
	};
	
	testTensor[0].start<int64_t>()[1].Add({1,2,3,4,5,6,7});

	try {
		worker1.Run(testTensor);
	}
	catch(std::runtime_error e){
		std::cout << e.what() << std::endl;
	}
	

	Sleep(10);
}