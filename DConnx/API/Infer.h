#pragma once
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <mutex>

#include "TensorSlot.h"
#include "info.h"
#include "expected.h"

namespace DC {
	// 可配置张量规则的推理器对象
	// 所有检查均在运行时进行
	// 请做好相应异常处理
	class Infer{
	public:
		enum class ErrorCode;
		using TensorList = std::unordered_map<std::string, TensorSlot>;
		using TensorDataList = std::unordered_map<std::string, Tensor>;
		using Promise = Expected<bool, ErrorCode>;
		using Results = Expected<bool, std::unordered_map<std::string, Tensor>>;

		virtual ~Infer() = default;
		class InferConfig;
		enum class ErrorCode {
			SUCCESS,
			MISSING_TENSOR,
			ERROR_TENSOR
		};

		//仪表盘-------------------------------------------------------------
		bool isReady() const { return ready; }
		std::string getErrorMessage() const { return errorMessage; }

		//读取状态方法-------------------------------------------------------

		// 获取张量列表
		const TensorList& getInfo() const;

		// 配置推理器方法----------------------------------------------------

		// 开始配置
		virtual InferConfig& config() = 0;

		// 添加默认参数
		// 如果张量表中没有这个参数或类型不对，则不起作用
		// < 参数类型 >
		// - 张量名
		// - 张量形状
		// - 默认值
		template<typename T> Infer& defaultValue(
			const std::string& name,
			std::vector<int64_t> shape,
			T value
		);

		//推理方法-----------------------------------------------------------
		
		// 验证推理指令是否符合输入要求
		// - 待输入的指令
		ErrorCode check(
			const std::vector<Tensor>& inputs		// 待输入的指令
		);

		// 运行推理，返回结果
		// - 待输入的指令
		virtual Results Run(
			const std::vector<Tensor>& inputs		// 待输入的指令
		) = 0;

	protected:
		// 解析 ONNX 文件
		virtual Promise parseONNX(const std::vector<std::byte>& onnxData) = 0;

		// 输入准备
		Promise prepareInputs(std::vector<Tensor>& inputs) {
			// 先添加默认列表
			for (auto& value : defaultList) {
				inputList[value.first].input(value.second);
			}

			// 再添加输入列表
			for (auto& value : inputs) {
				if (inputList.find(value.name()) != inputList.end()) {
					inputList[value.name()].input(value);
				}
			}

			// 检查输入完整性
			for (auto& [name, slot] : inputList) {
				if (!slot.hasData()) return ErrorCode::MISSING_TENSOR;
			}

			return true;
		}

		bool ready = false;									// 推理器就绪标志
		std::string errorMessage;							// 错误信息
		InferConfig* configInfo;							// 配置对象

		std::map<std::string, tensorInfo> inputTensorInfo;	// 快捷浏览张量列表
		TensorList inputList;								// 输入张量列表
		TensorList outputList;								// 输出张量列表

		TensorDataList defaultList;							// 默认参数张量列表

		// 并发控制成员
		std::mutex _mutex;
		std::condition_variable _condition;
		size_t _maxParallelCount;
		size_t _activeSessions = 0;
	};

	class Infer::InferConfig {
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
	inline Infer& Infer::defaultValue(const std::string& name, std::vector<int64_t> shape, T value) {
		defaultList[name] = Tensor::Create<T>(name, shape);
		defaultList[name].start<T>()[0] = value;

		return *this;
	}
}