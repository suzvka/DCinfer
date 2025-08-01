#pragma once
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "tensor.h"
#include "info.h"
namespace DC {
	// 定义错误码
	enum ValidationError {
		VALIDATION_SUCCESS = 0,
		MISSING_TENSOR = 1,
		TYPE_MISMATCH = 2,
		SHAPE_MISMATCH = 3
	};
	// 可配置张量规则的推理器对象
	// 所有检查均在运行时进行
	// 请做好相应异常处理
	class Infer;
	class Infer{
	public:
		// 构造函数
		// - ONNX 模型文件
		Infer(
			const std::vector<char>& onnxData,				// - ONNX模型文件
			Ort::Env* env = nullptr							// onnxruntime 运行环境
		);
		//仪表盘-------------------------------------------------------------
		bool isReady() const { return ready; }
		std::string getErrorMessage() const { return errorMessage; }

		//读取状态方法-------------------------------------------------------

		// 获取张量列表
		const std::map<std::string, tensorInfo>& getInfo() const;

		// 配置推理器方法----------------------------------------------------

		// 开始配置
		Infer& config();

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
		) {
			defaultList[name] = Tensor(name, getName.at(typeid(T).name()), shape);
			defaultList[name].start<T>()[0] = value;
			defaultList[name].load();
			return *this;
		}

		// 添加锚定动态维度的默认参数
		// 如果目标张量不存在，则不起作用
		// < 参数类型 >
		// - 张量名
		// - 目标张量名
		// - 张量形状
		// - 默认值
		//template<typename T> Infer& defaultValue(
		//	const std::string& name,
		//	const std::string& targetName,
		//	int dim,
		//	T value
		//) {
		//	defaultList[name] = Tensor(name, getName.at(typeid(T).name()), shape);
		//	return *this;
		//}

		//推理方法-----------------------------------------------------------
		
		// 验证推理指令是否符合输入要求
		// - 待输入的指令
		int check(
			const std::vector<Tensor>& inputs		// 待输入的指令
		);

		// 运行推理，返回结果
		// - 待输入的指令
		std::unordered_map<std::string,Tensor> Run(
			const std::vector<Tensor>& inputs		// 待输入的指令
		);

	private:
		bool ready = false;									// 推理器就绪标志
		std::string errorMessage;							// 错误信息

		std::shared_ptr<Ort::Env> env;						// ONNX Runtime 环境
		std::unique_ptr<Ort::Session> session;				// ONNX Runtime 会话
		std::unique_ptr<Ort::SessionOptions>_options;		// ONNX Runtime 配置

		std::map<std::string, tensorInfo> inputTensorInfo;	// 快捷浏览张量列表
		std::unordered_map<std::string, Tensor> tensorList;	// 工作用张量列表
		std::unordered_map<std::string, Tensor> defaultList;// 默认参数张量列表


		// 解析 ONNX 文件
		void parseInputTensorInfo();

		void recordTensor(std::string name, std::string type, std::vector<int64_t> shape, bool IOtype);

		// 整合输入张量
		// - 推理指令
		std::vector<Ort::Value> prepareInputs(
			const std::vector<Tensor>& inputs
		);
	};
}