#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <optional>
#include <cassert>
#include <atomic>

#if defined(__GXX_RTTI) || defined(_CPPRTTI)
#define DC_RTTI_ENABLED 1
#else
#define DC_RTTI_ENABLED 0
#endif

namespace DC::Type {

#if DC_RTTI_ENABLED
/// @brief RTTI 启用：以 std::type_index 作类型标识符；多模块（DLL/SO）环境下识别稳定。
using TypeId = std::type_index;
#else
/// @brief RTTI 禁用：以静态对象地址作类型标识符；同一类型跨模块可能得到不同 ID，可为相关类型特化 CustomTypeKey。
using TypeId = const void*;
#endif

/// @brief 特化此结构体可为类型提供自定义 TypeId 生成策略（用于禁用 RTTI 的跨模块场景）。
/// 约束：CustomTypeKey<T>::get() 必须返回跨模块稳定的 DC::TypeId。
template <typename T>
struct CustomTypeKey {};

namespace detail {
template <typename T>
concept CustomKeyAvailable = requires {
	{ CustomTypeKey<T>::get() } -> std::same_as<TypeId>;
};
} // namespace detail

template <typename T>
TypeId getTypeId() {
	if constexpr (detail::CustomKeyAvailable<T>) {
		return CustomTypeKey<T>::get();
	} else {
#if DC_RTTI_ENABLED
		return typeid(T);
#else
		static const char id = 0;
		return &id;
#endif
	}
}

/// @brief 类型擦除基类：使注册表容器能以统一指针持有异构 TypeRegistry。
struct ITypeRegistry {
	virtual ~ITypeRegistry() = default;
	[[nodiscard]] virtual std::string getEnumTypeName() const = 0;
	virtual void freeze() = 0;
	[[nodiscard]] virtual bool isFrozen() const = 0;
};

/// @brief 线程安全的类型到枚举的映射存储。
template <class Enum>
class TypeRegistry final : public ITypeRegistry {
private:
	using TypeEnumMap = std::unordered_map<TypeId, Enum>;
	struct EnumHash {
		std::size_t operator()(Enum e) const noexcept {
			return std::hash<std::underlying_type_t<Enum>>{}(static_cast<std::underlying_type_t<Enum>>(e));
		}
	};
	using EnumSizeMap = std::unordered_map<Enum, std::size_t, EnumHash>;

	mutable std::mutex mutex_;
	TypeEnumMap mappings_;
	mutable std::atomic<bool> frozen_{false};
	std::optional<Enum> fallback_;
	EnumSizeMap sizes_;

	void ensureFrozen() const {
		if (!frozen_.load(std::memory_order_acquire)) {
			std::unique_lock lock(mutex_);
			if (!frozen_.load(std::memory_order_relaxed)) {
				frozen_.store(true, std::memory_order_release);
			}
		}
	}

public:
	void freeze() override {
		std::unique_lock lock(mutex_);
		frozen_.store(true, std::memory_order_release);
	}

	[[nodiscard]] bool isFrozen() const override {
		return frozen_.load(std::memory_order_acquire);
	}

	/// @brief 设置查询未命中时的回退值；freeze 后设置会触发断言。
	void setFallback(Enum value) {
		std::unique_lock lock(mutex_);
		assert(!frozen_.load(std::memory_order_relaxed) && "Cannot set fallback after freeze.");
		fallback_ = value;
	}

	[[nodiscard]] std::optional<Enum> tryGetFallback() const {
		if (isFrozen()) {
			return fallback_;
		}
		std::unique_lock lock(mutex_);
		return fallback_;
	}

	/// @brief 注册类型 T 到枚举值；已冻结时返回 false。
	template <class T>
	bool registerType(Enum value) {

		if (frozen_.load(std::memory_order_acquire)) {
			return false;
		}
		std::unique_lock lock(mutex_);

		if (frozen_.load(std::memory_order_relaxed)) {
			return false;
		}
		mappings_[getTypeId<T>()] = value;
		auto& slot = sizes_[value];
		if (slot < sizeof(T)) {
			slot = sizeof(T);
		}

		return true;
	}

	[[nodiscard]] std::size_t getSize(Enum value) const {
		ensureFrozen();
		std::scoped_lock lock(mutex_);
		auto it = sizes_.find(value);
		return it != sizes_.end() ? it->second : 0;
	}

	[[nodiscard]] std::size_t getSizeOr(Enum value, std::size_t fallback) const {
		ensureFrozen();
		std::scoped_lock lock(mutex_);
		auto it = sizes_.find(value);
		return it != sizes_.end() ? it->second : fallback;
	}

	/// @brief 查询类型 T 的枚举值；未命中时依次回退到 fallback 与 Enum{}。
	template <class T>
	[[nodiscard]] Enum getType() const {
		ensureFrozen();

		// Lock-free read
		auto it = mappings_.find(getTypeId<T>());
		if (it != mappings_.end()) {
			return it->second;
		}

		if (fallback_.has_value()) {
			return *fallback_;
		}

		return Enum{};
	}

	template <class T>
	[[nodiscard]] Enum getTypeOr(Enum fallback) const {
		ensureFrozen();

		auto it = mappings_.find(getTypeId<T>());
		return it != mappings_.end() ? it->second : fallback;
	}

	template <class T>
	[[nodiscard]] std::optional<Enum> tryGetType() const {
		ensureFrozen();

		auto it = mappings_.find(getTypeId<T>());
		return it != mappings_.end() ? std::optional<Enum>(it->second) : std::nullopt;
	}

	template <class T>
	[[nodiscard]] std::size_t getSize() const {
		return sizeof(T);
	}

	template <class T>
	[[nodiscard]] std::size_t getSizeOr(std::size_t fallback) const {
		(void)fallback;
		return sizeof(T);
	}

	[[nodiscard]] std::string getEnumTypeName() const override {
#if DC_RTTI_ENABLED
		return typeid(Enum).name();
#else
		return "Unknown (RTTI disabled)";
#endif
	}
};

/// @brief 一组 TypeRegistry 的管理器；可用全局单例 instance()，也可独立实例化用于局部上下文。
class TypeEnvironment {
private:
	using RegistryMap = std::unordered_map<TypeId, std::unique_ptr<ITypeRegistry>>;

	std::shared_mutex mutex_;
	RegistryMap registries_;

public:
	TypeEnvironment() = default;
	TypeEnvironment(const TypeEnvironment&) = delete;
	TypeEnvironment& operator=(const TypeEnvironment&) = delete;

	static TypeEnvironment& instance() {
		static TypeEnvironment inst;
		return inst;
	}

	/// @brief 获取或创建指定枚举类型的注册表。
	template <class Enum>
	TypeRegistry<Enum>& getRegistry() {
		const auto key = getTypeId<Enum>();

		{
			std::shared_lock lock(mutex_);
			auto it = registries_.find(key);
			if (it != registries_.end()) {
				return *static_cast<TypeRegistry<Enum>*>(it->second.get());
			}
		}

		std::unique_lock lock(mutex_);
		auto it = registries_.find(key);
		if (it != registries_.end()) {
			return *static_cast<TypeRegistry<Enum>*>(it->second.get());
		}

		auto registry = std::make_unique<TypeRegistry<Enum>>();
		auto* ptr = registry.get();
		registries_[key] = std::move(registry);
		return *ptr;
	}

	template <class Enum>
	void freeze() {
		getRegistry<Enum>().freeze();
	}
};

template <class T, class Enum>
bool registerType(Enum value) {
	return TypeEnvironment::instance().getRegistry<Enum>().template registerType<T>(value);
}

template <class Enum>
void setFallback(Enum fallback) {
	TypeEnvironment::instance().getRegistry<Enum>().setFallback(fallback);
}

/// @brief 冻结某个枚举的注册表；冻结后禁止再注册。
template <class Enum>
void freeze() {
	TypeEnvironment::instance().freeze<Enum>();
}

template <class Enum, class T>
[[nodiscard]] Enum getType(const T&) {
	return TypeEnvironment::instance().getRegistry<Enum>().template getType<T>();
}

template <class Enum, class T>
[[nodiscard]] Enum getType() {
	return TypeEnvironment::instance().getRegistry<Enum>().template getType<T>();
}

template <class Enum, class T>
[[nodiscard]] Enum getTypeOr(const T&, Enum fallback) {
	return TypeEnvironment::instance().getRegistry<Enum>().template getTypeOr<T>(fallback);
}

template <class Enum, class T>
[[nodiscard]] Enum getTypeOr(Enum fallback) {
	return TypeEnvironment::instance().getRegistry<Enum>().template getTypeOr<T>(fallback);
}

template <class Enum, class T>
[[nodiscard]] std::optional<Enum> tryGetType(const T&) {
	return TypeEnvironment::instance().getRegistry<Enum>().template tryGetType<T>();
}

template <class Enum, class T>
[[nodiscard]] std::optional<Enum> tryGetType() {
	return TypeEnvironment::instance().getRegistry<Enum>().template tryGetType<T>();
}

template <class Enum, class T>
[[nodiscard]] std::size_t getSize(const T&) {
	return TypeEnvironment::instance().getRegistry<Enum>().template getSize<T>();
}

/// @brief 查询枚举值对应的注册大小；同一枚举值注册多个类型时取最大 sizeof(T)。
template <class Enum>
[[nodiscard]] std::size_t getSize(Enum value) {
	return TypeEnvironment::instance().getRegistry<Enum>().getSize(value);
}

template <class Enum, class T>
[[nodiscard]] std::size_t getSize() {
	return TypeEnvironment::instance().getRegistry<Enum>().template getSize<T>();
}

template <class Enum, class T>
[[nodiscard]] std::size_t getSizeOr(const T&, std::size_t fallback) {
	return TypeEnvironment::instance().getRegistry<Enum>().template getSizeOr<T>(fallback);
}

template <class Enum>
[[nodiscard]] std::size_t getSizeOr(Enum value, std::size_t fallback) {
	return TypeEnvironment::instance().getRegistry<Enum>().getSizeOr(value, fallback);
}

} // namespace DC::Type