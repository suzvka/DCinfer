#pragma once
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <sstream>
#include <functional>
#include <stdexcept>

#include "TensorSlot.h"

namespace DC {
	template<typename ExternalTensor, typename InferEngine>
	struct InferTools {
		using TensorList = std::vector<ExternalTensor>;
		using TensorType = Tensor::TensorType;
		using Shape = Tensor::Shape;

		// Optional runtime lifetime token (e.g., Ort::Env wrapped in shared_ptr).
		// The framework/node stores this token to ensure external runtime objects stay alive
		// for as long as the Infer instance is alive.
		std::shared_ptr<void> runtime;

		std::function<bool(const std::vector<std::byte>&, InferEngine&)> loadModelFunc;
		std::function<TensorList(TensorList&, InferEngine&)> inferFunc;
		std::function<Tensor(const ExternalTensor&)> toInternal;
		std::function<ExternalTensor(const Tensor&)> toExternal;
		std::function<std::string(const ExternalTensor&)> getName;
		std::function<TensorType(const ExternalTensor&)> getType;
		std::function<Shape(const ExternalTensor&)> getShape;
		std::function<std::vector<std::string>(const InferEngine&)> getInputNames;
		std::function<std::vector<std::string>(const InferEngine&)> getOutputNames;
	};

	// 单线程推理器
	// 通过依赖注入运行
	class InferBase{
	public:

		using TensorList = std::unordered_map<std::string, TensorSlot>;
		using Shape = Tensor::Shape;

		using Task = std::unordered_map<std::string, Tensor>;
		using Results = std::unordered_map<std::string, Tensor>;
		struct Config {
			enum class DeviceType {
				AUTO,
				GPU,
				CPU
			};
			DeviceType device = DeviceType::AUTO;
		};

		// 构造函数
		// - 推理引擎类型（如 TensorRT、ONNX Runtime 等）
		// - 推理器名称（用户自定义）
		// - 可选配置（如设备选择、性能调优参数等）
		InferBase(
			const std::string& type,
			const std::string& name,
			const Config& config = Config()
		) {
			// 构造函数逻辑
			 ready = false; // 初始状态为未就绪
			 configInfo = Config(config); // 存储配置
			 _logStream << "Infer created: " << type << " - " << name << std::endl;
		}
		virtual ~InferBase() = default;
		//仪表盘-------------------------------------------------------------
		bool isReady() const { return ready; }
		std::ostringstream& log() { return _logStream; }

		// 配置推理器方法----------------------------------------------------

		// 开始配置
		// virtual Config& config() = 0;

		// 添加默认参数
		// 如果张量表中没有这个参数或类型不对，则不起作用
		// < 参数类型 >
		// - 张量名
		// - 张量形状
		// - 默认值
		template<typename T> InferBase& defaultValue(
			const std::string& name,
			Shape shape,
			Tensor value
		);

		

	private:
		bool ready = false;									// 推理器就绪标志
		std::ostringstream _logStream;						// 日志流
		Config configInfo;							// 配置对象

		
	};

	template<typename ExternalTensor, typename InferEngine>
	class Infer : public InferBase {
	public:
		using SlotList = std::unordered_map<std::string, CurrencyTensorSlot<ExternalTensor>>;
		using TensorList = std::vector<ExternalTensor>;
		using TensorType = Tensor::TensorType;
		using Shape = Tensor::Shape;
		using Tools = DC::InferTools<ExternalTensor, InferEngine>;
		using InferTools = Tools;

		Infer(
			const std::string& type,
			const std::string& name,
			const Tools& tools,
			const Config& config = Config()
		) : InferBase(type, name, config), inferTools(tools), runtimeToken(tools.runtime) {}

		bool input(const std::string& name, const ExternalTensor& tensor) {
			if (!inferTools.toInternal) {
				throw std::logic_error("Infer::toInternal not injected");
			}
			if (!inferTools.getName || !inferTools.getType || !inferTools.getShape) {
				throw std::logic_error("Infer::getName/getType/getShape not injected");
			}
			
			inputList[name] << tensor;
			return true;
		}

		SlotList run() {
			if (!inferTools.inferFunc) {
				throw std::logic_error("Infer::inferFunc not injected");
			}

			std::vector<ExternalTensor> inputTensors;
			for (auto& slot : inputList) {
				inputTensors.push_back(slot.second.read());
			}

			auto outputTensors = inferTools.inferFunc(inputTensors, *engine);
			for (auto& slot : outputList) {
				auto it = std::find_if(outputTensors.begin(), outputTensors.end(),
					[&](const ExternalTensor& t) { return inferTools.getName(t) == slot.first; });
				if (it != outputTensors.end()) {
					slot.second << *it;
				}
			}

			return outputList;
		}

		TensorList run(TensorList& inputTensors) {
			if (!inferTools.inferFunc) {
				throw std::logic_error("Infer::inferFunc not injected");
			}

			return inferTools.inferFunc(inputTensors, *engine);
		}	

		bool loadModel(const std::vector<std::byte>& modelData) {
			if (!inferTools.loadModelFunc) {
				throw std::logic_error("Infer::loadModelFunc not injected");
			}

			if (!engine) {
				engine = std::make_unique<InferEngine>();
			}

			if (!inferTools.loadModelFunc(modelData, *engine)) {
				throw std::runtime_error("Failed to load model");
			}

			auto inputNames = inferTools.getInputNames(*engine);
			for (auto& name : inputNames) {
				inputList.emplace(name, CurrencyTensorSlot<ExternalTensor>(
					name,
					inferTools.getType(ExternalTensor(name)),
					inferTools.getShape(ExternalTensor(name)),
					inferTools.toInternal,
					inferTools.toExternal
				));
			}

			auto outputNames = inferTools.getOutputNames(*engine);
			for (auto& name : outputNames) {
				outputList.emplace(name, CurrencyTensorSlot<ExternalTensor>(
					name,
					inferTools.getType(ExternalTensor(name)),
					inferTools.getShape(ExternalTensor(name)),
					inferTools.toInternal,
					inferTools.toExternal
				));
			}

			return true;
		}
		
		

		std::unique_ptr<InferEngine> engine; // 推理引擎实例
		InferTools inferTools; // 推理工具集
		std::shared_ptr<void> runtimeToken; // Keeps external runtime objects alive (e.g., Ort::Env)
		SlotList inputList;								// 输入张量列表
		SlotList outputList;								// 输出张量列表
	};

	template<typename ExternalTensor, typename InferEngine>
	static std::unique_ptr<InferBase> CreateInfer(
		const std::string& type,
		const std::string& name,
		const typename Infer<ExternalTensor, InferEngine>::Tools& tools,
		const InferBase::Config& config = InferBase::Config()
	) {
		return std::make_unique<Infer<ExternalTensor, InferEngine>>(type, name, tools, config);
	}

	// Fully deduced: both ExternalTensor and InferEngine are deduced from `tools`.
	template<typename ExternalTensor, typename InferEngine>
	static std::unique_ptr<InferBase> CreateInfer(
		const std::string& type,
		const std::string& name,
		const InferTools<ExternalTensor, InferEngine>& tools,
		const InferBase::Config& config = InferBase::Config()
	) {
		return std::make_unique<Infer<ExternalTensor, InferEngine>>(type, name, tools, config);
	}

}