#pragma once
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <mutex>

#include "TensorSlot.h"
#include "Expected.h"

namespace DC {
	// 可配置张量规则的推理器对象
	// 所有检查均在运行时进行
	// 请做好相应异常处理
	class InferBase{
	public:
		using TensorList = std::unordered_map<std::string, TensorSlot>;
		using Results = Expected<std::unordered_map<std::string, Tensor>, bool>;
		using Task = std::unordered_map<std::string, Tensor>;

		virtual ~InferBase() = default;
		class InferConfig;
		

		//仪表盘-------------------------------------------------------------
		bool isReady() const { return ready; }
		std::string getErrorMessage() const { return errorMessage; }

		//读取状态方法-------------------------------------------------------

		// 获取输入张量列表
		const TensorList& getInput() const;

		// 获取输出张量列表
		const TensorList& getOutput() const;

		// 配置推理器方法----------------------------------------------------

		// 开始配置
		virtual InferConfig& config() = 0;

		// 添加默认参数
		// 如果张量表中没有这个参数或类型不对，则不起作用
		// < 参数类型 >
		// - 张量名
		// - 张量形状
		// - 默认值
		template<typename T> InferBase& defaultValue(
			const std::string& name,
			std::vector<int64_t> shape,
			T value
		);

	protected:
		// 解析 ONNX 文件
		virtual bool parseONNX(const std::vector<std::byte>& onnxData) = 0;

		// 输入准备
		bool prepareInputs(Task& inputs) {
			// 添加输入列表
			for (auto& [name, tensor] : inputs) {
				if (inputList.find(name) != inputList.end()) {
					inputList[name].input(std::move(tensor));
				}
			}

			// 检查输入完整性
			for (auto& [name, slot] : inputList) {
				if (!slot.hasData()) return false;
			}
			return true;
		}

		bool ready = false;									// 推理器就绪标志
		std::string errorMessage;							// 错误信息
		InferConfig* configInfo;							// 配置对象

		//std::map<std::string, tensorInfo> inputTensorInfo;	// 快捷浏览张量列表
		TensorList inputList;								// 输入张量列表
		TensorList outputList;								// 输出张量列表

		// 并发控制成员
		std::mutex _mutex;
		std::condition_variable _condition;
		size_t _maxParallelCount = 0;
		size_t _activeSessions = 0;
	};

	class InferBase::InferConfig {
	public:
		enum class DeviceType {
			AUTO,
			GPU,
			CPU
		};

		virtual ~InferConfig() = default;
		DeviceType& device() { return _device; }

	private:
		DeviceType _device = DeviceType::AUTO;
	};

	template<typename T>
	inline InferBase& InferBase::defaultValue(const std::string& name, std::vector<int64_t> shape, T value) {
		//defaultList.emplace(name, Tensor::Create<T>(name, shape, value));

		return *this;
	}
}