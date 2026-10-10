#pragma once
#include <type_traits>
#include <stdexcept>
#include <optional>
#include <functional>
#include <memory>
#include <unordered_map>

#include "Tensor.hpp"
#include "Exception.h"
#include "DCtype.h"
#include "SlotType.h"
#include "Value.h"

namespace DC {

/// @brief 张量数据槽位：Node 输入/输出端口的基础存储单元（类型擦除，store() 经 ValidatorRegistry 校验）。
class TensorSlot {
	using TensorType = TensorMeta::TensorType;
	using ErrorType = TensorException::ErrorType;

public:
	using Shape = Tensor::Shape;

	/// @brief 同节点内所有槽位的映射表（供 DefaultProvider 查阅锚定数据）。
	using SlotMap = std::unordered_map<std::string, TensorSlot>;

	/// @brief 懒求值默认值工厂：槽位无显式输入且无静态默认值时按需生成 Tensor（返回 nullptr 保持空）。
	using DefaultProvider = std::function<std::unique_ptr<Tensor>(const SlotMap&)>;

	/// @brief 槽位配置。
	class Config {
	public:
		enum class Position {
			Input,
			Output,
			Auto
		};

		Config() : position(Position::Auto) {}

		Config& setPosition(Position p);

		Position position;
	};

	TensorSlot(const TensorSlot&) = delete;
	TensorSlot& operator=(const TensorSlot&) = delete;

	/// @brief 析构：释放类型擦除的运行时数据（RAII）。
	~TensorSlot() { releaseBlob(); }

	/// @brief 移动构造：接管数据；源只 reset 不走 deleter（防已转移指针二次释放）。
	TensorSlot(TensorSlot&& other) noexcept
		: _rule(std::move(other._rule)),
		  _defaultData(std::move(other._defaultData)),
		  _defaultProvider(std::move(other._defaultProvider)),
		  _blob(std::move(other._blob)),
		  _config(std::move(other._config)) {
		other._blob.reset();
	}

	/// @brief 移动赋值：先释放自身数据再接管。
	TensorSlot& operator=(TensorSlot&& other) noexcept {
		if (this != &other) {
			releaseBlob();
			_rule = std::move(other._rule);
			_defaultData = std::move(other._defaultData);
			_defaultProvider = std::move(other._defaultProvider);
			_blob = std::move(other._blob);
			other._blob.reset(); // 仅清空源，不调用 deleter
			_config = std::move(other._config);
		}
		return *this;
	}

	TensorSlot(const std::string& name, TensorMeta::TensorType type, size_t size, const Shape& shape,
			   const Config& config = Config());

	/// @brief 设置默认张量数据（输入槽位 fallback）。
	TensorSlot& setDefaultTensor(const Tensor& data);

	/// @brief 设置懒求值默认值工厂（输入槽位 fallback；典型：形状锚定到同节点另一端口）。
	TensorSlot& setDefaultProvider(DefaultProvider fn);

	/// @brief 无运行时数据但有 DefaultProvider 时调用工厂填充。
	void resolveDefaultIfNeeded(const SlotMap& peers);

	const std::string& name() const;
	TensorType type() const;
	size_t typeSize() const;
	Shape shape() const;

	bool isInput() const;
	bool hasDefaultData() const;

	template <typename T>
	bool isType() const;

	/// @brief 类型擦除存储；校验失败抛 TensorException（InvalidShape / TypeMismatch / ShapeMismatch）。
	template <typename T>
	TensorSlot& store(T&& data);

	/// @brief 移动取出数据；空槽位抛 NotData，类型不匹配抛 TypeMismatch。
	template <typename T>
	T take();

	/// @brief 只读指针访问；类型不匹配返回 nullptr。
	template <typename T>
	const T* peek() const;

	/// @brief 以 const Tensor& 获取数据（仅 DCTensor 类型有效；无数据抛 NotData）。
	const Tensor& view() const;

	bool hasData() const;
	SlotDataType storedType() const;

	void clear();

	static Config CreateConfig();

private:
	struct TypedBlob {
		void* ptr = nullptr;
		std::function<void(void*)> deleter;
		SlotDataType type = SlotDataTypeUnknown;
	};

	TensorMeta _rule;
	std::unique_ptr<Tensor> _defaultData;
	DefaultProvider _defaultProvider;
	std::optional<TypedBlob> _blob;
	Config _config;

	/// @brief 释放类型擦除数据（幂等）。
	void releaseBlob() {
		if (_blob.has_value() && _blob->deleter && _blob->ptr) {
			_blob->deleter(_blob->ptr);
			_blob->ptr = nullptr;
		}
	}

	[[noreturn]] void abort(ErrorType errorType = ErrorType::Other, const std::string& message = "") const;
};

// Template method definitions
template <typename T>
bool TensorSlot::isType() const {
	return type() == Type::getType<TensorMeta::TensorType, T>();
}

template <typename T>
TensorSlot& TensorSlot::store(T&& data) {
	ValidatorRegistry::ensureDefaults(); // 保证默认注册已执行（std::call_once）
	auto typeEnum = ensureSlotType<std::decay_t<T>>();

	auto status = ValidatorRegistry::instance().validate(std::addressof(data), typeEnum, _rule);

	if (!status.ready()) {
		if (status.invalid) {
			abort(ErrorType::InvalidShape, "Input data is invalid");
		}
		if (status.needConvert) {
			abort(ErrorType::TypeMismatch, "Type mismatch and conversion not allowed");
		}
		if (status.needAlign) {
			abort(ErrorType::ShapeMismatch, "Shape mismatch and alignment not allowed");
		}
	}

	// 强异常安全：先在局部构造新值，成功后再释放旧数据（避免 UAF/双重释放窗口）。
	TypedBlob blob;
	blob.type = typeEnum;
	blob.ptr = new std::decay_t<T>(std::forward<T>(data));
	blob.deleter = [](void* p) { delete static_cast<std::decay_t<T>*>(p); };

	if (_blob.has_value() && _blob->deleter && _blob->ptr) {
		_blob->deleter(_blob->ptr);
	}
	_blob = std::move(blob);

	return *this;
}

template <typename T>
T TensorSlot::take() {
	if (!_blob.has_value() || !_blob->ptr) {
		abort(ErrorType::NotData, "Slot is empty");
	}

	auto expectedType = ensureSlotType<T>();
	if (_blob->type != expectedType) {
		abort(ErrorType::TypeMismatch,
			  "take<T>: type mismatch, stored=" + std::to_string(_blob->type) +
				  " expected=" + std::to_string(expectedType));
	}

	auto* typed = static_cast<T*>(_blob->ptr);
	T result = std::move(*typed);

	// 释放存储（不调用 deleter：已移动）
	typed->~T();
	operator delete(typed);
	_blob.reset();

	return result;
}

template <typename T>
const T* TensorSlot::peek() const {
	if (!_blob.has_value() || !_blob->ptr) {
		return nullptr;
	}

	auto expectedType = ensureSlotType<T>();
	if (_blob->type != expectedType) {
		return nullptr;
	}

	return static_cast<const T*>(_blob->ptr);
}
} // namespace DC