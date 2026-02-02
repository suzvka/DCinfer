#pragma once
#include "Infer.h"
#include "InferBase.h"
#include "TensorOrt.h"

#include <fstream>
#include <filesystem>

namespace DC {
	class InferOrt : public InferBase , public Infer{
	public:
		using InputValues = std::vector<Ort::Value>;
		using OutputValues = std::vector<Ort::Value>;
		using NameList = std::vector<const char*>;
		using ErrorCode = Infer::ErrorCode;

		// 从文件路径构造
		InferOrt(const std::filesystem::path& modelPath, size_t maxParallelCount);

		// 从内存数据构造
		InferOrt(const std::vector<std::byte>& modelData, size_t maxParallelCount);
		
		Infer::Task Run(
			Infer::Task& inputs
		);

		InferConfig& config() override {
			if (!configInfo) {
				configInfo = new InferConfig();
			}
			return *configInfo;
		}
	private:
		Ort::Env _env;						// ONNX Runtime 环境
		std::unique_ptr<Ort::Session> _session;				// ONNX Runtime 会话
		Ort::SessionOptions _options;		// ONNX Runtime 配置

		std::vector<std::string> inputNames;
		std::vector<std::string> outputNames;

		// Ort 张量数据向量
		InputValues _inputTensors;
		OutputValues _outputTensors;

	protected:
		void setTypeMap();
		bool parseONNX(const std::vector<std::byte>& onnxData);
		TensorSlot createTensorSlot(std::string name, const Ort::ConstTensorTypeAndShapeInfo& tensorInfo);

		Results runInfer(Infer::Task& inputs); // 单次推理
	};
}